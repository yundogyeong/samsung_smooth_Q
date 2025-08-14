import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRMSNorm
from autosmoothquant.thirdparty.baichuan.modeling_baichuan import RMSNorm, BaichuanLayer

temp_save = {}
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, model_type = "transformers", alpha=0.5, org_smooth_params=None, name="model.layers.0", mode="self_attn"):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    if model_type == "llama":
        assert isinstance(ln, LlamaRMSNorm)
    else:
        assert isinstance(ln, nn.LayerNorm)

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    ## TODO
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    smoothing_factor = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    temp_save[name + '.' + mode] = smoothing_factor.detach().cpu()
    
    ln.weight.div_(smoothing_factor)
    if model_type == "transformers":
        ln.bias.div_(smoothing_factor)

    for fc in fcs:
        fc.weight.mul_(smoothing_factor.view(1, -1))

@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    # smooth_params_org = torch.load('smooth_params.pt')
    org_smooth_params=None
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, "transformers", alpha, org_smooth_params, name, 'self_attn')

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, "transformers", alpha, org_smooth_params, name, 'mlp')
            
        elif isinstance(module, LlamaDecoderLayer):
            print(f"smooth llama model: {name}")
            attn_ln = module.input_layernorm #attention forward norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, "llama", alpha)

            ffn_ln = module.post_attention_layernorm #feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + '.mlp.gate_proj']
            smooth_ln_fcs(ffn_ln, fcs, fcs_input_scales, "llama", alpha)

    torch.save(temp_save, 'smooth_factor_opt_6.7b.pt')
