import os
import torch
import argparse
import json

from autosmoothquant.models import QuantizedLlamaForCausalLM, QuantizedOPTForCausalLM, Int8BaichuanForCausalLM, Int8MixtralForCausalLM
from autosmoothquant.utils import parse_quant_config
from transformers import AutoTokenizer, AutoModelForCausalLM

def print_model_size(model, state):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'{state} Model size: {size_all_mb:.3f}MB')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='/workspace/AutoSmoothQuant/autosmoothquant/examples/quantized_model/opt-6.7b/a45aa65bbeb77c1558bc99bedc6779195462dab0-smoothquant-int8', help='path contains model weight and quant config')
    parser.add_argument('--tokenizer-path', type=str,
                        default='facebook/opt-6.7b', help='path contains tokenizer')
    parser.add_argument('--model-class', type=str,
                        default='opt', help='currently support: llama, baichuan, opt, mixtral')
    parser.add_argument('--prompt', type=str,
                        default='In recent years, quantization has become', help='prompts')
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    config_path = os.path.join(args.model_path, "quant_config.json")
    quant_config = parse_quant_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Except GEMM uses int8, the default data type is torch.float32 for quant now.
    # Consider setting the default data type to torch.float16 to speed up, but this may decrease model performance.
    # torch.set_default_dtype(torch.float16)
    if args.model_class == "llama":
      model = QuantizedLlamaForCausalLM.from_pretrained(args.model_path, quant_config, attn_implementation="eager", device_map="sequential")
    elif args.model_class == "opt":
      model = QuantizedOPTForCausalLM.from_pretrained(args.model_path, quant_config, attn_implementation="eager", device_map="sequential")
    else:
      raise ValueError(
        f"Model type {args.model_class} are not supported for now.")
      
    inputs = tokenizer(
      args.prompt,
      padding=True,
      truncation=True,
      max_length=2048,
      return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=20)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(f"Quantized model output : {outputs}")
    print_model_size(model, "Quantized")
    
    base_model = AutoModelForCausalLM.from_pretrained(args.tokenizer_path, device_map="sequential", cache_dir='/workspace/cache')
    inputs = tokenizer(
      args.prompt,
      padding=True,
      truncation=True,
      max_length=2048,
      return_tensors="pt").to("cuda")
    output_ids = base_model.generate(**inputs, max_new_tokens=20)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(f"Base model output : {outputs}")
    print_model_size(base_model, "Base")



if __name__ == '__main__':
    main()