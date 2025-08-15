import torch
from torch import nn
from typing import Optional, Tuple, List

from transformers.models.opt.modeling_opt import (
    OPTConfig,
    OPTForCausalLM,
    OPTModel,
    OPTPreTrainedModel,
    OPTLearnedPositionalEmbedding,
    OPTAttention,
    OPTDecoderLayer,
    OPTDecoder,
)
from transformers.utils import logging
from transformers.activations import ACT2FN

from autosmoothquant.layers.nn.linear import (
    W8A8BFP32OFP32Linear,
    W8A8BFP32OFP32LinearWithQuantScale,
)

logger = logging.get_logger(__name__)


# -----------------------------
# LayerNorm (per-tensor scale 적용)
# -----------------------------
class QuantizedOPTLayerNorm(nn.LayerNorm):
    @staticmethod
    def from_float(module: nn.LayerNorm, input_scale: float):
        assert module.normalized_shape[0] == module.weight.numel()
        assert module.normalized_shape[0] == module.bias.numel()
        q_module = QuantizedOPTLayerNorm(module.normalized_shape[0], module.eps,
                                         elementwise_affine=True)
        # per-tensor 입력스케일일 때: y = LN(x) 를 생각하면, x' = x / s 로 바꿔치기 위해
        # 가중치/바이어스를 1/s 로 스케일링(입력에 곱해질 s를 보정)
        q_module.weight = nn.Parameter(module.weight / input_scale)
        q_module.bias   = nn.Parameter(module.bias   / input_scale)
        return q_module


# -----------------------------
# Attention
# -----------------------------
class QuantizedOPTAttention(nn.Module):
    """OPT Multi-head self-attention (양자화 래퍼)"""

    def __init__(
        self,
        config: OPTConfig,
        quant_config: dict[str, str],
        is_decoder: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        # handle deprecated kwargs compat (HF와 동일한 경로)
        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            if fn_arg_name in kwargs:
                logging.warning(
                    f"Passing in {fn_arg_name} to {self.__class__.__name__} is deprecated and won't be supported from v4.38. "
                    "Please set it in the config instead"
                )
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        self.embed_dim = _handle_deprecated_argument("hidden_size", config, "embed_dim", kwargs)
        self.num_heads = _handle_deprecated_argument("num_attention_heads", config, "num_heads", kwargs)
        self.dropout   = _handle_deprecated_argument("attention_dropout",   config, "dropout", kwargs)
        self.enable_bias = _handle_deprecated_argument("enable_bias", config, "bias", kwargs)

        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={self.num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.is_causal = True

        # quantization selections
        self.qkv_quant_type = quant_config["qkv"]
        self.o_quant_type   = quant_config["out"]

        qtype = quant_config["type"]


        # int8 path
        self.q_proj  = W8A8BFP32OFP32Linear(self.embed_dim, self.embed_dim, use_bias=self.enable_bias, act_quant=self.qkv_quant_type)
        self.k_proj  = W8A8BFP32OFP32Linear(self.embed_dim, self.embed_dim, use_bias=self.enable_bias, act_quant=self.qkv_quant_type)
        self.v_proj  = W8A8BFP32OFP32Linear(self.embed_dim, self.embed_dim, use_bias=self.enable_bias, act_quant=self.qkv_quant_type)
        self.out_proj= W8A8BFP32OFP32LinearWithQuantScale(self.embed_dim, self.embed_dim, use_bias=self.enable_bias, act_quant=self.o_quant_type)

    _shape  = OPTAttention._shape
    forward = OPTAttention.forward

    @staticmethod
    @torch.no_grad()
    def from_float_to_int8(module: OPTAttention,
                           config: OPTConfig,
                           quant_config: dict[str, str],
                           attn_input_scale: float,
                           q_output_scale: float,
                           k_output_scale: float,
                           v_output_scale: float,
                           out_input_scale: float):
        qmod = QuantizedOPTAttention(config, quant_config, is_decoder=True)
        # per-layer 가중치/act 양자화 파라미터 주입
        qmod.q_proj = W8A8BFP32OFP32Linear.from_float(module.q_proj, attn_input_scale, act_quant=qmod.qkv_quant_type)
        qmod.k_proj = W8A8BFP32OFP32Linear.from_float(module.k_proj, attn_input_scale, act_quant=qmod.qkv_quant_type)
        qmod.v_proj = W8A8BFP32OFP32Linear.from_float(module.v_proj, attn_input_scale, act_quant=qmod.qkv_quant_type)
        qmod.out_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(module.out_proj, out_input_scale, act_quant=qmod.o_quant_type)
        return qmod


# -----------------------------
# Decoder Layer (FFN 포함)
# -----------------------------
class QuantizedOPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, quant_config: dict[str, str]):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = QuantizedOPTAttention(config=config, quant_config=quant_config, is_decoder=True)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout   = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        # LN (per-tensor일 경우 입력스케일 적용 가능)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

        # FFN
        self.fc1_quant_type = quant_config["fc1"]
        self.fc2_quant_type = quant_config["fc2"]

        qtype = quant_config["type"]

        if qtype == "int8":
            self.fc1 = W8A8BFP32OFP32Linear(self.embed_dim, config.ffn_dim, use_bias=config.enable_bias, act_quant=self.fc1_quant_type)
            self.fc2 = W8A8BFP32OFP32LinearWithQuantScale(config.ffn_dim, self.embed_dim, use_bias=config.enable_bias, act_quant=self.fc2_quant_type)
        else:
            raise ValueError(f"Unsupported quant type: {qtype}")

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    forward = OPTDecoderLayer.forward

    @staticmethod
    def from_float_to_int8(module: OPTDecoderLayer,
                           config: OPTConfig,
                           quant_config: dict[str, str],
                           attn_input_scale: float,
                           q_output_scale: float,
                           k_output_scale: float,
                           v_output_scale: float,
                           out_input_scale: float,
                           fc1_input_scale: float,
                           fc2_input_scale: float):
        qlayer = QuantizedOPTDecoderLayer(config, quant_config)
        qlayer.self_attn = QuantizedOPTAttention.from_float_to_int8(
            module.self_attn, config, quant_config,
            attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale
        )
        qlayer.fc1 = W8A8BFP32OFP32Linear.from_float(module.fc1, fc1_input_scale, act_quant=qlayer.fc1_quant_type)
        qlayer.fc2 = W8A8BFP32OFP32LinearWithQuantScale.from_float(module.fc2, fc2_input_scale, act_quant=qlayer.fc2_quant_type)
        # per-tensor일 때에만 LN 스케일 적용
        if quant_config["qkv"] == "per-tensor":
            qlayer.self_attn_layer_norm = QuantizedOPTLayerNorm.from_float(module.self_attn_layer_norm, attn_input_scale)
        else:
            qlayer.self_attn_layer_norm = module.self_attn_layer_norm

        if quant_config["fc1"] == "per-tensor":
            qlayer.final_layer_norm = QuantizedOPTLayerNorm.from_float(module.final_layer_norm, fc1_input_scale)
        else:
            qlayer.final_layer_norm = module.final_layer_norm

        return qlayer

# -----------------------------
# Decoder
# -----------------------------
class QuantizedOPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    """

    def __init__(self, config: OPTConfig, quant_config: dict[str, str]):
        super().__init__(config)
        self.dropout   = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self._use_flash_attention_2 = (getattr(config, "_attn_implementation", None) == "flash_attention_2")
        self._use_sdpa = (getattr(config, "_attn_implementation", None) == "sdpa")

        # FlashAttention2/SDPA 미지원이면 강제로 끄기
        self._use_flash_attention_2 = False
        self._use_sdpa = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([QuantizedOPTDecoderLayer(config, quant_config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        # HF init
        self.post_init()

    get_input_embeddings = OPTDecoder.get_input_embeddings
    set_input_embeddings = OPTDecoder.set_input_embeddings
    forward = OPTDecoder.forward

    @staticmethod
    def from_float_to_int8(module: OPTDecoder, decoder_layer_scales: List[dict], quant_config: dict[str, str]):
        qdec = QuantizedOPTDecoder(module.config, quant_config)
        qdec.embed_tokens    = module.embed_tokens
        qdec.embed_positions = module.embed_positions
        qdec.project_out     = module.project_out
        qdec.project_in      = module.project_in
        qdec.final_layer_norm= module.final_layer_norm

        for i, layer in enumerate(module.layers):
            qdec.layers[i] = QuantizedOPTDecoderLayer.from_float_to_int8(
                layer, module.config, quant_config, **decoder_layer_scales[i]
            )
        return qdec



# -----------------------------
# Model
# -----------------------------
class QuantizedOPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig, quant_config: dict[str, str]):
        super().__init__(config)
        self.decoder = QuantizedOPTDecoder(config, quant_config)
        self.post_init()

    get_input_embeddings = OPTModel.get_input_embeddings
    set_input_embeddings = OPTModel.set_input_embeddings
    get_decoder = OPTModel.get_decoder
    forward = OPTModel.forward

    @staticmethod
    def from_float_to_int8(module: OPTModel, decoder_layer_scales: List[dict], quant_config: dict[str, str]):
        qmodel = QuantizedOPTModel(module.config, quant_config)
        qmodel.decoder = QuantizedOPTDecoder.from_float_to_int8(
            module.decoder, decoder_layer_scales, quant_config
        )
        return qmodel



# -----------------------------
# ForCausalLM
# -----------------------------
class QuantizedOPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: OPTConfig, quant_config: dict[str, str]):
        super().__init__(config)
        self.model = QuantizedOPTModel(config, quant_config)

        # lm_head는 OPT에서 embed_tokens.weight와 tie
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        self.post_init()

    get_input_embeddings = OPTForCausalLM.get_input_embeddings
    set_input_embeddings = OPTForCausalLM.set_input_embeddings
    get_output_embeddings = OPTForCausalLM.get_output_embeddings
    set_output_embeddings = OPTForCausalLM.set_output_embeddings
    set_decoder = OPTForCausalLM.set_decoder
    get_decoder = OPTForCausalLM.get_decoder
    forward = OPTForCausalLM.forward
    prepare_inputs_for_generation = OPTForCausalLM.prepare_inputs_for_generation
    _reorder_cache = OPTForCausalLM._reorder_cache

    @staticmethod
    def from_float_to_int8(module: OPTForCausalLM, decoder_layer_scales: List[dict], quant_config: dict[str, str]):
        qlm = QuantizedOPTForCausalLM(module.config, quant_config)
        print("start perform weight quantization (INT8), this might take a while")
        qlm.model = QuantizedOPTModel.from_float_to_int8(
            module.model, decoder_layer_scales, quant_config
        )
        qlm.lm_head = module.lm_head
        return qlm