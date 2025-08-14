from .baichuan import Int8BaichuanForCausalLM
from .llama import QuantizedLlamaForCausalLM
from .mixtral import Int8MixtralForCausalLM
from .opt import QuantizedOPTForCausalLM
from autosmoothquant.thirdparty.baichuan.configuration_baichuan import BaichuanConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": QuantizedLlamaForCausalLM,
    "LLaMAForCausalLM": QuantizedLlamaForCausalLM,
    "BaichuanForCausalLM": Int8BaichuanForCausalLM,
    "OPTForCausalLM": QuantizedOPTForCausalLM,
    "MixtralForCausalLM": Int8MixtralForCausalLM
}

_MODEL_TYPE = {
    "LlamaForCausalLM": "llama",
    "LLaMAForCausalLM": "llama",
    "BaichuanForCausalLM": "baichuan",
    "OPTForCausalLM": "transformers",
    "MixtralForCausalLM": "mixtral"
}

_CONFIG_REGISTRY = {
    "baichuan": BaichuanConfig,
}
