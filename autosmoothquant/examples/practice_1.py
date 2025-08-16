import torch
import argparse
import os
import json
from pathlib import Path

from autosmoothquant.quantize.smooth import smooth_lm
from autosmoothquant.quantize.prac1 import get_act_scales
from autosmoothquant.utils import get_config, get_model_architecture, build_model_and_tokenizer, parse_quant_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='/workspace/cache/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0', help='model path contains weights and config etc')
    parser.add_argument('--quantize-model', action="store_true",
                        help='whether to quant model or not', default=True)
    parser.add_argument('--generate-scale', action="store_true",
                        help='whether to generate scale or not', default=True)
    parser.add_argument('--scale-output', type=str, default='scales/opt-6.7b.pt',
                        help='where to save the act scales, activate when generating scales')
    parser.add_argument("--scale-input", type=str, default='scales/opt-6.7b.pt',
                        help='where to save the act scales, activate when quantizing models')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--type', type=str, default="int8", 
                        help='quantization type')
    parser.add_argument('--activation-scheme', type=str, default="dynamic", help='dynamic or static, just for fp8')
    parser.add_argument('--ignore-patterns', type=str, default="re:.*lm_head", help='ignore layer, just for fp8')
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument("--model-output", type=str, default='quantized_model/opt-6.7b',
                        help='where to save the quantized models, activate when quantizing models')
    parser.add_argument("--smooth-strength", type=float, default=0.8,
                        help='migration strength of smoothquant, should be in a range of (0, 1)')
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_path)

    if args.generate_scale:
        act_scales = get_act_scales(model, tokenizer,
                                    args.num_samples, args.seq_len)
        os.makedirs(os.path.dirname(args.scale_output), exist_ok=True)
        torch.save(act_scales, args.scale_output)
        
    orig = torch.load("/workspace/AutoSmoothQuant/autosmoothquant/examples/scales/opt-6.7b_org.pt")
    saved = torch.load(args.scale_output)
    
    rtol, atol = 1e-4, 1e-6
    all_equal = True
    for key in orig:
        if key not in saved:
            print(f"Missing key in saved: {key}")
            all_equal = False
            continue
        if not torch.allclose(orig[key], saved[key], rtol=rtol, atol=atol):
            print(f"Mismatch in {key}")
            all_equal = False

    if all_equal:
        print("All tensors match within tolerance.")
    else:
        print("Some tensors differ.")
        


if __name__ == '__main__':
    main()
