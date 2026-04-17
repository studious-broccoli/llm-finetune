"""LoRA rank ablation study.

Sweeps LoRA rank across [1, 2, 4, 8, 16, 32] and reports perplexity,
trainable parameter count, and training time for each. Results are saved
to results_llm/ablation_rank.csv and plotted to results_llm/ablation_rank.png.

Usage:
    python scripts/ablate_rank.py [--epochs 1] [--train_samples 500]

Typical runtime: ~3-5 min per rank on a consumer GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from llm_finetune.config import TrainConfig
from llm_finetune.data import load_wikitext, sample_splits, tokenize_and_chunk
from llm_finetune.evaluate import perplexity_from_loss
from llm_finetune.model import apply_lora, count_trainable_params, load_base_model, load_tokenizer

RANKS = [1, 2, 4, 8, 16, 32]
CSV_PATH = "results_llm/ablation_rank.csv"
PLOT_PATH = "results_llm/ablation_rank.png"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="LoRA rank ablation study.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per rank")
    parser.add_argument("--train_samples", type=int, default=500, help="Training examples per rank")
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    return parser.parse_args()


def run_rank(rank: int, cfg: TrainConfig) -> dict[str, float | int]:
    """Train with a single LoRA rank and return evaluation metrics.

    Args:
        rank: LoRA decomposition rank to evaluate.
        cfg: Base training configuration (``lora_rank`` will be overridden).

    Returns:
        Dict with ``rank``, ``trainable_params``, ``trainable_pct``,
        ``perplexity``, ``train_time_sec``.
    """
    cfg.lora_rank = rank

    tokenizer = load_tokenizer(cfg)
    raw = load_wikitext(cfg)
    tokenized = tokenize_and_chunk(raw, tokenizer, cfg)
    train_ds, eval_ds = sample_splits(tokenized, cfg)

    model = load_base_model(cfg)
    model.resize_token_embeddings(len(tokenizer))
    model = apply_lora(model, cfg)

    trainable, total = count_trainable_params(model)
    pct = 100 * trainable / total

    training_args = TrainingArguments(
        output_dir=f"./results_llm/rank_{rank}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        save_total_limit=0,
        logging_steps=50,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0

    eval_results = trainer.evaluate()
    ppl = perplexity_from_loss(eval_results["eval_loss"])

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "rank": rank,
        "trainable_params": trainable,
        "trainable_pct": round(pct, 4),
        "perplexity": round(ppl, 2),
        "train_time_sec": round(elapsed, 1),
    }


def plot_results(rows: list[dict[str, Any]]) -> None:
    """Plot perplexity vs rank on a dual-axis chart.

    Args:
        rows: List of result dicts from ``run_rank``.
    """
    ranks = [r["rank"] for r in rows]
    ppls = [r["perplexity"] for r in rows]
    params = [r["trainable_params"] / 1e6 for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ranks, ppls, "o-", color="steelblue", label="Perplexity (↓)")
    ax1.set_xlabel("LoRA Rank (r)")
    ax1.set_ylabel("Perplexity", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.bar(ranks, params, alpha=0.3, color="orange", label="Trainable params (M)")
    ax2.set_ylabel("Trainable Parameters (M)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    ax1.set_title("LoRA Rank Ablation: Perplexity vs Parameter Count")
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.88))
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Plot saved to {PLOT_PATH}")


def main() -> None:
    """Run the full rank ablation sweep."""
    args = parse_args()
    os.makedirs("results_llm", exist_ok=True)

    base_cfg = TrainConfig(
        num_epochs=args.epochs,
        train_samples=args.train_samples,
        eval_samples=200,
    )

    rows: list[dict[str, Any]] = []
    for rank in args.ranks:
        print(f"\n{'='*50}\nRank = {rank}\n{'='*50}")
        result = run_rank(rank, base_cfg)
        rows.append(result)
        print(f"  Perplexity:       {result['perplexity']}")
        print(f"  Trainable params: {result['trainable_params']:,} ({result['trainable_pct']:.3f}%)")
        print(f"  Train time:       {result['train_time_sec']:.1f}s")

    # Save CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {CSV_PATH}")

    # Print table
    print(f"\n{'Rank':>6}  {'Trainable':>12}  {'Pct':>7}  {'Perplexity':>10}  {'Time (s)':>9}")
    print("-" * 55)
    for r in rows:
        print(
            f"{r['rank']:>6}  {r['trainable_params']:>12,}  {r['trainable_pct']:>6.3f}%"
            f"  {r['perplexity']:>10.2f}  {r['train_time_sec']:>9.1f}"
        )

    plot_results(rows)


if __name__ == "__main__":
    main()
