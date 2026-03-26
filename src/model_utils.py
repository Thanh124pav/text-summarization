"""Model loading utilities for decoder-only summarization models."""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Small decoder-only models suitable for summarization finetuning
SUPPORTED_MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "phi-2": "microsoft/phi-2",
    "phi-1.5": "microsoft/phi-1_5",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma-2b": "google/gemma-2b",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "bloom-560m": "bigscience/bloom-560m",
    "bloom-1b1": "bigscience/bloom-1b1",
}


def get_model_name(model_key: str) -> str:
    """Resolve model key to HuggingFace model name."""
    if model_key in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_key]
    # Assume it's a direct HuggingFace model path
    return model_key


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer and set pad token if needed."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(
    model_name: str,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device_map: str = "auto",
):
    """Load a decoder-only model with optional LoRA and quantization.

    Args:
        model_name: HuggingFace model name or path.
        use_lora: Whether to apply LoRA adapters.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout.
        load_in_4bit: Use 4-bit quantization.
        load_in_8bit: Use 8-bit quantization.
        device_map: Device mapping strategy.

    Returns:
        Model (with LoRA if enabled).
    """
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model
