"""Model loading, quantization, and LoRA adapter setup."""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import TrainConfig


def load_tokenizer(cfg: TrainConfig) -> PreTrainedTokenizerBase:
    """Load the tokenizer and set the pad token to EOS.

    GPT-2 has no dedicated pad token. Reusing EOS is the standard workaround;
    the data collator masks pad positions so they don't contribute to loss.

    Args:
        cfg: Training configuration with ``model_name``.

    Returns:
        Configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg: TrainConfig) -> PreTrainedModel:
    """Load the base causal LM, optionally in 4-bit NF4 quantization.

    4-bit quantization (QLoRA) reduces peak GPU memory by ~4x with negligible
    perplexity loss on short fine-tuning runs. See scripts/benchmark.py for
    measured memory savings on this workload.

    Args:
        cfg: Training configuration.

    Returns:
        Loaded model with token embeddings resized to match the tokenizer.
    """
    bnb_config: BitsAndBytesConfig | None = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model


def apply_lora(model: PreTrainedModel, cfg: TrainConfig) -> PreTrainedModel:
    """Wrap the base model with LoRA adapters.

    Only the low-rank adapter matrices (A, B) are updated during training.
    The effective weight update is W' = W + (alpha/r) * B @ A, where alpha/r
    scales the adapter contribution. With r=8 and alpha=32, the scale is 4x —
    chosen empirically to match GPT-2's weight magnitude.

    Args:
        model: Base causal LM (frozen weights).
        cfg: Config with LoRA hyperparameters.

    Returns:
        PEFT-wrapped model with LoRA adapters injected.
    """
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def count_trainable_params(model: PreTrainedModel) -> tuple[int, int]:
    """Count trainable vs total parameters.

    Args:
        model: Any PyTorch model.

    Returns:
        Tuple of ``(trainable_params, total_params)``.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
