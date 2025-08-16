import torch
import tqdm
import os
import argparse
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from autosmoothquant.models import QuantizedLlamaForCausalLM, QuantizedOPTForCausalLM
from autosmoothquant.utils import parse_quant_config
# ==== add imports (top of file) ====
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# ==== config (원하면 argparse로 뺄 것) ====
PROBE_LAYER = 6          # 측정할 레이어 index
PROBE_KIND  = "attn_q"   # "attn_q", "attn_k", "attn_v", "attn_out", "mlp_fc1"
SEQ_LEN     = 512
USE_P99     = True       # True: 채널별 p99, False: 채널별 max
SAMPLE_IDX  = 0          # 데이터셋에서 고정 샘플 하나 뽑기
OUTDIR      = "./probe_out"

os.makedirs(OUTDIR, exist_ok=True)

def _target_suffix(layer_idx, kind):
    base = f"model.decoder.layers.{layer_idx}"
    if kind == "attn_q":   return f"{base}.self_attn.q_proj"
    if kind == "attn_k":   return f"{base}.self_attn.k_proj"
    if kind == "attn_v":   return f"{base}.self_attn.v_proj"
    if kind == "attn_out": return f"{base}.self_attn.out_proj"
    if kind == "mlp_fc1":  return f"{base}.fc1"
    raise ValueError(f"Unknown kind: {kind}")

def _find_module_by_suffix(model, suffix):
    for name, m in model.named_modules():
        if name.endswith(suffix):
            return name, m
    raise RuntimeError(f"Module not found: endswith('{suffix}')")

def _pick_one_batch(dataset, tokenizer, device, seq_len=512, sample_idx=0):
    text = dataset[sample_idx]["text"]
    enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
    return {k: v.to(device) for k, v in enc.items()}

@torch.no_grad()
def _collect_ln_after_once(model, module_name, module, batch, use_p99=True):
    """
    module의 입력(= LN 뒤, 다음 Linear로 들어가는 텐서)을 한 번 패스해서 채널별 통계로 축약
    OPT Linear 입력: [..., H] -> (H,)
    """
    store = {}

    def pre_hook(mod, inputs):
        x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        H = x.shape[-1]
        flat = x.reshape(-1, H).abs()
        vec = torch.quantile(flat, 0.99, dim=0) if use_p99 else flat.max(dim=0)[0]
        store["vec"] = vec.float().cpu()

    h = module.register_forward_pre_hook(pre_hook)
    model.eval()
    _ = model(**batch)
    h.remove()

    if "vec" not in store:
        raise RuntimeError(f"Hook did not fire for {module_name}")
    return store["vec"]  # (H,)

def _save_hist(vec, title, outfile, bins=80):
    v = vec.numpy()
    logv = np.log10(v + 1e-12)
    plt.figure(figsize=(6,4))
    plt.hist(logv, bins=bins, alpha=0.9)
    plt.xlabel("log10(channel scale)")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []

        seq_len = 2048
        total_len = int(self.dataset.size(1))
        max_chunks = total_len // seq_len
        if max_chunks == 0:
            raise ValueError(f"Not enough tokens ({total_len}) for a {seq_len}-token chunk.")

        n_samples = min(self.n_samples if self.n_samples else max_chunks, max_chunks)

        with torch.inference_mode():
            for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
                start = i * seq_len
                end = min((i + 1) * seq_len, total_len)
                if end <= start:
                    break

                batch = self.dataset[:, start:end].to(model.device)  # [1, L]
                if batch.size(1) < 2:
                    continue

                outputs = model(batch)
                lm_logits = outputs.logits  # [1, L, V]

                shift_logits = lm_logits[:, :-1, :].contiguous().float()          # [1, L-1, V]
                shift_labels = batch[:, 1:].contiguous()                           # [1, L-1]

                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),  # [(L-1), V]
                    shift_labels.view(-1)                          # [(L-1)]
                )
                neg_log_likelihood = loss.float() * (shift_labels.numel())
                nlls.append(neg_log_likelihood)

        if not nlls:
            raise RuntimeError("No valid chunks were evaluated (dataset may be too short).")

        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seq_len))
        return ppl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='/workspace/AutoSmoothQuant/autosmoothquant/examples/quantized_model/opt-6.7b/a45aa65bbeb77c1558bc99bedc6779195462dab0-smoothquant-int8', help='path contains model weight and quant config')
    parser.add_argument('--tokenizer-path', type=str,
                        default='facebook/opt-6.7b', help='path contains tokenizer')

    parser.add_argument("--origin-model", action="store_true", default=False)
    parser.add_argument("--non-smoothing", action="store_true", default=False)
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    model_path = "facebook/opt-6.7b"
    if not args.origin_model:
        config_path = os.path.join(args.model_path, "quant_config.json")
        quant_config = parse_quant_config(config_path)
        if args.non_smoothing:
            ns_model_path = "/workspace/cache/opt-6.7b_non_smooth/Non_smoothquant_int8"
            model = QuantizedOPTForCausalLM.from_pretrained(ns_model_path, quant_config,
                                                            attn_implementation="eager", device_map="sequential")
        else:
            model = QuantizedOPTForCausalLM.from_pretrained(args.model_path, quant_config,
                                                                attn_implementation="eager", device_map="sequential")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir='/workspace/cache', attn_implementation="eager", device_map="sequential", torch_dtype=torch.float16)

    n_samples = 100
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset(
    "wikitext", "wikitext-2-raw-v1",  
    split="train[:512]"              
            )

    evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples)
    ppl = evaluator.evaluate(model)
    print(f"Perplexity: {ppl}")
    
        # ---------- (A) 동일 데이터셋에서 하나만 뽑기 ----------
    device = next(model.parameters()).device
    batch = _pick_one_batch(dataset, tokenizer, device, seq_len=SEQ_LEN, sample_idx=SAMPLE_IDX)

    # ---------- (B) 측정 지점(OPT: LN 뒤 → q_proj 입력 등) 찾기 ----------
    suffix = _target_suffix(PROBE_LAYER, PROBE_KIND)
    mod_name, mod = _find_module_by_suffix(model, suffix)

    # ---------- (C) 한 번 패스해서 분포 수집 ----------
    vec = _collect_ln_after_once(model, mod_name, mod, batch, use_p99=USE_P99)

    # ---------- (D) 저장 ----------
    label = "origin" if args.origin_model else ("nosq" if args.non_smoothing else "sq")
    npy_path = os.path.join(OUTDIR, f"act_{label}_{PROBE_KIND}_L{PROBE_LAYER}_i{SAMPLE_IDX}.npy")
    png_path = os.path.join(OUTDIR, f"act_{label}_{PROBE_KIND}_L{PROBE_LAYER}_i{SAMPLE_IDX}.png")
    np.save(npy_path, vec.numpy())
    _save_hist(vec, f"{label} | {mod_name} | {'p99' if USE_P99 else 'max'}", png_path)

    print(f"[Probe] saved: {npy_path}")
    print(f"[Probe] saved: {png_path}")