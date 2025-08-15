import copy
import gc
import re
from typing import List

import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from autosmoothquant.layers.functional.quantization import per_tensor_quantize_fp8
from autosmoothquant.models import _MODEL_TYPE



def get_act_scales(model, tokenizer, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    # Only support pretraining_tp=1 when capturing activation for now
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1 
    act_scales = {}
    
    def stat_input_hook(m, input, y, name):
        import pdb; pdb.set_trace()
        """
        GOAL: 모듈 입력의 채널별(마지막 차원 기준) 최대 절대값을 수집해 act_scales[name]에 누적 저장.
        """
        
        input = input[0]

        # 1) 마지막 차원을 hidden 채널로 정의한다. (예: [B, T, H]에서 H)
        # hidden_dim = input.shape[_____]

        # 2) 배치/시퀀스 축을 모두 펴서 [N*, H]로 만든 뒤, 절대값만 사용.
        #    통계 수집이므로 그래프에서 분리(detach)한다.
        # x_flat = input.view(_____)._____( )._____( )

        # 3) 열(채널) 기준 최대값을 구한다. 반환 튜플의 첫 요소가 실제 값.
        #    이후 계산/저장을 단순화하기 위해 float + CPU로 옮긴다.
        # col_max = torch.max(x_flat, dim=_____)[0]._____( )._____( )

        # 4) 같은 모듈 이름이 이미 있으면 요소별 최대값으로 누적, 없으면 신규로 저장.
        # act_scales[name] = _____

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset(
    "wikitext", "wikitext-2-raw-v1",  # WikiText-2 raw 버전
    split="train[:512]"               # train split에서 앞 512개
            )
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()
        
    return act_scales

@torch.no_grad()
def collect_transformers_layer_scales(model, act_dict):
    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.q_proj"]['input'] / 127
        scale_dict["q_output_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.q_proj"]['output'] / 127
        scale_dict["k_output_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.k_proj"]['output'] / 127
        scale_dict["v_output_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.v_proj"]['output'] / 127
        scale_dict["out_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.out_proj"]['input'] / 127
        scale_dict["fc1_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.fc1"]['input'] / 127
        scale_dict["fc2_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.fc2"]["input"] / 127
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales


@torch.no_grad()
def collect_llama_layer_scales(model, act_dict):
    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.q_proj"]['input'] / 127
        scale_dict["q_output_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.q_proj"]['output'] / 127
        scale_dict["k_output_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.k_proj"]['output'] / 127
        scale_dict["v_output_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.v_proj"]['output'] / 127
        scale_dict["out_input_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.o_proj"]['input'] / 127
        # mlp scales
        scale_dict["gate_input_scale"] = act_dict[
            f"model.layers.{idx}.mlp.gate_proj"]['input'] / 127
        scale_dict["down_input_scale"] = act_dict[
            f"model.layers.{idx}.mlp.down_proj"]["input"] / 127
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales

@torch.no_grad()
def get_static_decoder_layer_scales(model,
                                    tokenizer,
                                    num_samples=512,
                                    seq_len=512,
                                    model_type = "transformers"
                                    ):
    model.eval()
    device = next(model.parameters()).device
    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item())
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item())

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(
                partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset(
    "wikitext", "wikitext-2-raw-v1",  # WikiText-2 raw 버전
    split="train[:512]"               # train split에서 앞 512개
        )
    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    decoder_layer_scales = []
    if model_type == "transformers":
        decoder_layer_scales = collect_transformers_layer_scales(model, act_dict)
    elif model_type == "llama":
        decoder_layer_scales = collect_llama_layer_scales(model, act_dict)
    else:
        raise ValueError(f"unsupport model type: {model_type}")

    return decoder_layer_scales, act_dict


def replace_module(model: AutoModelForCausalLM, name: str, new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1:]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name
    setattr(parent, child_name, new_module)


def get_layers_to_ignore(model, ignore_patterns) -> List[str]:
    ignored_layers = set()

    for name, linear in model.named_modules():
        if not isinstance(linear, torch.nn.Linear):
            continue

        for ignore_pattern in ignore_patterns:
            regex_prefix = "re:"
            if ignore_pattern.startswith(regex_prefix):
                # check if name matches regex and add to set if true
                regex_pattern = ignore_pattern[len(regex_prefix):]
                if re.search(regex_pattern, name):
                    ignored_layers.add(name)
            else:
                # else, exact match
                if ignore_pattern == name:
                    ignored_layers.add(name)

    return list(ignored_layers)


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


"""
 Inspired by https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py#L249
"""
