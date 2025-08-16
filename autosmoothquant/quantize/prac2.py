import torch
import torch.nn as nn
import sys

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

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
    import pdb; pdb.set_trace()
    """
    목적: 목적: 여러 FC의 가중치 스케일(열 기준 max)과 사전 측정된 활성 스케일을 결합해 
    채널별 스무딩 팩터 s(= act^a / weight^(1-a))를 계산한다.
    """
    
    # 1) 모든 FC의 weight에서 "입력 채널(in_features)별" 스케일 후보를 모은다.
    #    - 각 fc.weight의 shape: [out_features, in_features]
    #    - '출력 축(out_features)'을 따라 감소 연산 → [1, in_features] 형태의 한 줄 요약으로 만든다.
    #    - 절댓값 기준으로 채널별 최대치(열 기준)를 구한다.
    # w_cols = [fc.weight._____( )._____(dim=_____, keepdim=True)[0] for fc in fcs]
    
    # 2) FC들에서 나온 줄 요약들을 한데 모은 뒤(행 결합), 채널별로 최댓값만 남긴다.
    #    - 결과 shape 흐름: list[[1, H], [1, H], ...] → [N_fc, H] → [H]
    # weight_scales = torch._____(w_cols, dim=_____)
    # weight_scales = weight_scales._____(dim=_____)[0]._____(min=1e-5)
    
    # 3) 스무딩 팩터 s 계산 (채널별):
    #    s = (act_scales^alpha) / (weight_scales^(1 - alpha))
    #    - 0 분모/언더플로 방지용 하한값 적용 (1e-5)
    #    - 이후 연산 대상 파라미터와 동일한 device/dtype으로 맞춘다.
    # smoothing_factor = _____
    smoothing_factor = None
    if org_smooth_params is not None:
        if not torch.allclose(smoothing_factor, org_smooth_params.cuda(), rtol=1e-4, atol=1e-6):
            print(f"[Error] smoothing_factor mismatch at {name}.{mode}")
            sys.exit(1)
    
    
    ln.weight.div_(smoothing_factor)
    if model_type == "transformers":
        ln.bias.div_(smoothing_factor)

    for fc in fcs:
        fc.weight.mul_(smoothing_factor.view(1, -1))

@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    org_smooth = torch.load('/workspace/AutoSmoothQuant/autosmoothquant/examples/smooth_factor_opt_6.7b.pt')
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            org_smooth_params = org_smooth[name + '.self_attn']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, "transformers", alpha, org_smooth_params, name, 'self_attn')

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']
            org_smooth_params = org_smooth[name + '.mlp']
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
