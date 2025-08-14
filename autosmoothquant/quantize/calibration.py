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
        input = input[0]
        hidden_dim = input.shape[-1]
        input = input.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(input, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

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
