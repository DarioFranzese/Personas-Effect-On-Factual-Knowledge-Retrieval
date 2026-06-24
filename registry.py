from models import BackendType, ModelArchitecture, ModelAlignment, QuantizationType

# Internal registry mapping supported model IDs to default configurations
MODEL_REGISTRY = {

    # --- DeepSeek Family ---
    # "deepseek-ai/DeepSeek-V4-Flash-Base": { # NO 4-bit QUANTIZATION AVAILABLE
    #     "size": "248B (13B)",
    #     "backend": BackendType.TRANSFORMER,
    #     "architecture": [ModelArchitecture.BASE],
    #     "alignment": ModelAlignment.GRPO,
    #     "quantization": QuantizationType.NONE,
    # },
    # "sgl-project/DeepSeek-V4-Flash-FP8": { # NO 4-bit QUANTIZATION AVAILABLE
    #     "size": "248B (13B)",
    #     "backend": BackendType.TRANSFORMER,
    #     "architecture": [ModelArchitecture.MOE, ModelArchitecture.REASONING],
    #     "alignment": ModelAlignment.GRPO,
    #     "quantization" : QuantizationType.FP8
    # },
    "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ": {
        "size": "70B",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.REASONING],
        "alignment": ModelAlignment.OTHER,
        "quantization" : QuantizationType.AWQ
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "size": "8B",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.REASONING],
        "alignment": ModelAlignment.OTHER,
        "quantization"  : QuantizationType.NONE
    },

    # --- Qwen Family ---
    # "Qwen/Qwen3.5-397B-A17B-FP8": { # Too Big for now
    #     "size": "379B (17B)",
    #     "backend": BackendType.VLLM,
    #     "architecture": [ModelArchitecture.MOE, ModelArchitecture.REASONING],
    #     "alignment": ModelAlignment.OTHER,
    #     "quantization" : QuantizationType.FP8
    # },
    "Qwen/Qwen3.5-35B-A3B-FP8": {
        "size": "35B (3B)",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.MOE, ModelArchitecture.REASONING],
        "alignment": ModelAlignment.OTHER,
        "quantization" : QuantizationType.FP8
    },
    # "mradermacher/Qwen3.5-35B-A3B-Base-GGUF": { # Even if GGUF is supported, repositories do not have config files which fails loading those models
    #     "size": "35B (3B)",
    #     "backend": BackendType.TRANSFORMER,
    #     "max_model_len": 32768,
    #     "architecture": [ModelArchitecture.MOE, ModelArchitecture.REASONING],
    #     "alignment": ModelAlignment.NONE,
    #     "quantization": QuantizationType.GGUF # Name of the file is Qwen3.5-35B-A3B-Base.Q8_0.gguf
    # },
    "Qwen/Qwen3.5-9B": {

        "size": "9B",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.REASONING],
        "alignment": ModelAlignment.OTHER,
        "quantization" : QuantizationType.NONE
    },
    "Qwen/Qwen3.5-9B-Base": {
        "size": "9B",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.BASE, ModelArchitecture.REASONING],
        "alignment": ModelAlignment.NONE,
        "quantization" : QuantizationType.NONE

    },

    # --- Llama Family ---
    # "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": { # Too big as of right now
    #     "size": "400B (17B)",
    #     "backend": BackendType.VLLM,
    #     "max_model_len": 32768,
    #     "architecture": [ModelArchitecture.MOE],
    #     "alignment": ModelAlignment.DPO,
    #     "quantization" : QuantizationType.FP8

    # },

    # "mradermacher/Llama-4-Maverick-17B-128E-GGUF": { # Too big as of right now
    #     "size": "400B (17B)",
    #     "backend": BackendType.VLLM,
    #     "max_model_len": 32768,
    #     "architecture": [ModelArchitecture.MOE],
    #     "alignment": ModelAlignment.NONE,
    #     "quantization" : QuantizationType.GGUF
    # },
    "kosbu/Llama-3.3-70B-Instruct-AWQ": {
        "size": "70B",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.INSTRUCTION_TUNED],
        "alignment": ModelAlignment.RLHF,
        "quantization" : QuantizationType.AWQ
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "size": "8B",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.INSTRUCTION_TUNED],
        "alignment": ModelAlignment.OTHER,
        "quantization" : QuantizationType.NONE
    },
    "meta-llama/Llama-3.1-8B": { # Pure Base
        "size": "8B",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.BASE],
        "alignment": ModelAlignment.NONE,
        "quantization" : QuantizationType.NONE
    },

    # --- GPT OSS ---
    "openai/gpt-oss-20b": {
        "size": "20B",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.MOE, ModelArchitecture.REASONING],
        "alignment": ModelAlignment.OTHER,
        "quantization": QuantizationType.NONE
    },
    "mlx-community/gpt-oss-120b-MXFP4-Q4": { # cannot load on one GPU
        "size": "120B",
        "backend": BackendType.VLLM,
        "max_model_len": 32768,
        "architecture": [ModelArchitecture.MOE, ModelArchitecture.REASONING],
        "alignment": ModelAlignment.OTHER,
        "quantization" : QuantizationType.FP8
    },

}

def get_model_defaults(model_id: str) -> dict:
    """
    Retrieves default configuration parameters for a specific model_id from the registry.
    
    Args:
        model_id (str): The unique identifier of the model.
        
    Returns:
        dict: A dictionary containing the base configuration for the model.
   """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not supported, available models: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_id]