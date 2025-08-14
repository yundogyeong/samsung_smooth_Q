import torch
import tqdm
import os
import argparse
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from autosmoothquant.models import QuantizedLlamaForCausalLM, QuantizedOPTForCausalLM
from autosmoothquant.utils import parse_quant_config

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
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    model_path = "facebook/opt-6.7b"
    if not args.origin_model:
        config_path = os.path.join(args.model_path, "quant_config.json")
        quant_config = parse_quant_config(config_path)
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