import torch
import argparse
import os
import json
from pathlib import Path

from autosmoothquant.quantize.smooth import smooth_lm
from autosmoothquant.quantize.calibration import get_act_scales, get_static_decoder_layer_scales
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

    if args.quantize_model:
        act_scales = torch.load(args.scale_input)
        smooth_lm(model, act_scales, args.smooth_strength)
        config = get_config(args.model_path)
        quant_model_class, model_type = get_model_architecture(config)
        config_path = os.path.join(args.model_path, "quant_config.json")
        quant_config = parse_quant_config(config_path)

        output_path = Path(args.model_output) / (Path(args.model_path).name + "-smoothquant-int8")
        decoder_layer_scales, _ = get_static_decoder_layer_scales(model,
                                                                    tokenizer,
                                                                    
                                                                    num_samples=args.num_samples,
                                                                    seq_len=args.seq_len,
                                                                    model_type=model_type)
        quant_model = quant_model_class.from_float_to_int8(model, decoder_layer_scales, quant_config)

        quant_model.save_pretrained(output_path)
        config_output = os.path.join(output_path, "quant_config.json")
        with open(config_output, 'w') as json_file:
            json.dump(quant_config, json_file, indent=4)


if __name__ == '__main__':
    main()
