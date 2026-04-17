"""Centralized training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """All hyperparameters and paths for a single fine-tuning run.

    Keeping everything in one place makes ablation studies mechanical:
    swap one field, rerun, compare results.

    Args:
        model_name: HuggingFace model ID for the base model.
        dataset: HuggingFace dataset name (wikitext variant).
        block_size: Token block length for causal LM chunking.
        train_samples: Number of training examples to sample.
        eval_samples: Number of evaluation examples to sample.
        seed: Global random seed for reproducibility.
        lora_rank: LoRA decomposition rank (r). Lower = fewer params, higher = more capacity.
        lora_alpha: LoRA scaling factor. Effective LR scale = lora_alpha / lora_rank.
        lora_dropout: Dropout applied inside LoRA adapter layers.
        target_modules: Attention weight names to inject LoRA into.
        use_4bit: Whether to load the base model in NF4 4-bit quantization.
        batch_size: Per-device training batch size.
        num_epochs: Number of full passes over the training set.
        weight_decay: L2 regularization strength for the AdamW optimizer.
        num_eval_pairs: Number of prompt/reference pairs for generation eval.
        max_new_tokens: Maximum tokens generated per prompt during evaluation.
        output_dir: Directory for checkpoints, plots, and text outputs.
        mlflow_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment label.
    """

    # Model and data
    model_name: str = "gpt2"
    dataset: str = "wikitext-2-raw-v1"
    block_size: int = 128
    train_samples: int = 1000
    eval_samples: int = 1000
    seed: int = 42

    # LoRA — rank=8 balances capacity vs parameter count (~0.1% trainable).
    # See scripts/ablate_rank.py for a full rank sweep.
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["c_attn"])

    # Quantization
    use_4bit: bool = True

    # Training
    batch_size: int = 4
    num_epochs: int = 3
    weight_decay: float = 0.01

    # Evaluation
    num_eval_pairs: int = 20
    max_new_tokens: int = 64
    repetition_penalty: float = 1.3

    # Output
    output_dir: str = "./results_llm"
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "llm-finetune-gpt2"
