"""Evaluation utilities: generation metrics, perplexity, and plots."""

from __future__ import annotations

import math
from typing import Any

import evaluate
import matplotlib.pyplot as plt
from transformers import PreTrainedModel, PreTrainedTokenizerBase, pipeline


def generate_continuations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.3,
) -> list[str]:
    """Generate text continuations for a list of prompts.

    Args:
        model: Fine-tuned or base causal LM.
        tokenizer: Matching tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Maximum tokens to generate per prompt.
        repetition_penalty: Penalizes repeated tokens; >1 reduces loops.

    Returns:
        List of generated continuations (prompt text stripped).
    """
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    results = [
        pipe(p, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=repetition_penalty)[0][
            "generated_text"
        ][len(p) :]
        for p in prompts
    ]
    del pipe
    return results


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute ROUGE and BERTScore between generated and reference continuations.

    ROUGE measures n-gram overlap and is fast but lexical — it misses
    paraphrases. BERTScore uses DistilBERT embeddings to capture semantic
    similarity and correlates better with human judgement on fluent text.

    Args:
        predictions: Model-generated continuations.
        references: Ground-truth reference continuations.

    Returns:
        Dict with keys ``rouge1``, ``rouge2``, ``rougeL``, ``bertscore_f1``.
    """
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=references)

    bertscore = evaluate.load("bertscore")
    bs_result = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
        verbose=False,
    )

    return {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bertscore_f1": sum(bs_result["f1"]) / len(bs_result["f1"]),
    }


def perplexity_from_loss(eval_loss: float) -> float:
    """Convert cross-entropy eval loss to perplexity.

    Args:
        eval_loss: Mean cross-entropy loss from Trainer.evaluate().

    Returns:
        Perplexity score (lower is better; random baseline ~vocab_size).
    """
    return math.exp(eval_loss)


def plot_loss(log_history: list[dict[str, Any]], save_path: str) -> None:
    """Plot training and evaluation loss curves.

    Args:
        log_history: Trainer's ``state.log_history`` list.
        save_path: File path for the saved figure.
    """
    train_entries = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    eval_entries = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    steps_train, losses_train = zip(*train_entries) if train_entries else ([], [])
    steps_eval, losses_eval = zip(*eval_entries) if eval_entries else ([], [])

    plt.figure(figsize=(8, 5))
    plt.plot(steps_train, losses_train, label="Train loss")
    if steps_eval:
        plt.plot(steps_eval, losses_eval, label="Eval loss", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metric_comparison(
    baseline_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
    save_path: str,
) -> None:
    """Grouped bar chart comparing baseline vs fine-tuned metric scores.

    Args:
        baseline_metrics: Scores from the base (pre-LoRA) model.
        finetuned_metrics: Scores from the LoRA fine-tuned model.
        save_path: File path for the saved figure.
    """
    metric_names = list(baseline_metrics.keys())
    x = list(range(len(metric_names)))
    width = 0.35

    _, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [i - width / 2 for i in x],
        [baseline_metrics[m] for m in metric_names],
        width,
        label="Baseline GPT-2",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [finetuned_metrics[m] for m in metric_names],
        width,
        label="LoRA fine-tuned",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Fine-tuned: Generation Quality")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
