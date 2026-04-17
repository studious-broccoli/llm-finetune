"""Main training entrypoint for QLoRA fine-tuning."""

from __future__ import annotations

import os
from collections import Counter

import matplotlib.pyplot as plt
import mlflow
import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline as hf_pipeline

from llm_finetune.config import TrainConfig
from llm_finetune.data import build_eval_pairs, load_wikitext, sample_splits, tokenize_and_chunk
from llm_finetune.evaluate import (
    compute_generation_metrics,
    generate_continuations,
    perplexity_from_loss,
    plot_loss,
    plot_metric_comparison,
)
from llm_finetune.model import apply_lora, count_trainable_params, load_base_model, load_tokenizer

os.environ["WANDB_DISABLED"] = "true"


def main(cfg: TrainConfig | None = None) -> None:
    """Run the full fine-tuning pipeline.

    Args:
        cfg: Training configuration. Defaults to ``TrainConfig()`` if not provided.
    """
    if cfg is None:
        cfg = TrainConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # --- Data ---
    tokenizer = load_tokenizer(cfg)
    raw = load_wikitext(cfg)
    tokenized = tokenize_and_chunk(raw, tokenizer, cfg)
    train_ds, eval_ds = sample_splits(tokenized, cfg)
    eval_pairs = build_eval_pairs(eval_ds, tokenizer, n=cfg.num_eval_pairs)
    eval_prompts = [p for p, _ in eval_pairs]
    eval_refs = [r for _, r in eval_pairs]

    # --- Baseline generation (before LoRA) ---
    base_model = load_base_model(cfg)
    base_model.resize_token_embeddings(len(tokenizer))
    baseline_gens = generate_continuations(
        base_model, tokenizer, eval_prompts, cfg.max_new_tokens, cfg.repetition_penalty
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Apply LoRA ---
    model = apply_lora(base_model, cfg)
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=10,
        push_to_hub=False,
        report_to="mlflow",
    )

    mlflow.log_params({
        "model_name": cfg.model_name,
        "batch_size": cfg.batch_size,
        "num_epochs": cfg.num_epochs,
        "weight_decay": cfg.weight_decay,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "use_4bit": cfg.use_4bit,
        "block_size": cfg.block_size,
        "train_samples": cfg.train_samples,
        "dataset": cfg.dataset,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 4),
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(os.path.join(cfg.output_dir, "fine-tuned-model"))
    tokenizer.save_pretrained(os.path.join(cfg.output_dir, "fine-tuned-model"))

    # --- Perplexity ---
    eval_results = trainer.evaluate()
    ppl = perplexity_from_loss(eval_results["eval_loss"])
    print(f"Perplexity: {ppl:.2f}")

    # --- Loss curve ---
    loss_path = os.path.join(cfg.output_dir, "loss_curve.png")
    plot_loss(trainer.state.log_history, save_path=loss_path)

    # --- ROUGE + BERTScore ---
    finetuned_gens = generate_continuations(
        model, tokenizer, eval_prompts, cfg.max_new_tokens, cfg.repetition_penalty
    )
    baseline_metrics = compute_generation_metrics(baseline_gens, eval_refs)
    finetuned_metrics = compute_generation_metrics(finetuned_gens, eval_refs)
    print("\nBaseline metrics:", baseline_metrics)
    print("Fine-tuned metrics:", finetuned_metrics)

    comparison_path = os.path.join(cfg.output_dir, "metric_comparison.png")
    plot_metric_comparison(baseline_metrics, finetuned_metrics, save_path=comparison_path)

    # --- Token frequency ---
    all_ids = [tok for row in train_ds["input_ids"] for tok in row]
    counts = Counter(all_ids)
    top_ids, top_freqs = zip(*counts.most_common(20))
    tokens = [tokenizer.decode([i]) for i in top_ids]
    freq_path = os.path.join(cfg.output_dir, "token_frequencies.png")
    plt.figure(figsize=(10, 4))
    plt.bar(tokens, top_freqs)
    plt.title("Top-20 Token Frequencies (training set)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(freq_path)
    plt.close()

    # --- Sample outputs ---
    pipe = hf_pipeline("text-generation", model=os.path.join(cfg.output_dir, "fine-tuned-model"), tokenizer=tokenizer)
    prompts = [
        "The future of AI in medicine is",
        "Quantum computing will",
        "Education systems are evolving because",
    ]
    out_path = os.path.join(cfg.output_dir, "multiple_generations.txt")
    with open(out_path, "w") as f:
        for p in prompts:
            text = pipe(p, max_new_tokens=50)[0]["generated_text"]
            print(f"\nPrompt: {p}\nGenerated: {text}\n")
            f.write(f"Prompt: {p}\nGenerated: {text}\n\n")

    # --- MLflow logging ---
    mlflow.log_metric("final_perplexity", ppl)
    for name, score in baseline_metrics.items():
        mlflow.log_metric(f"baseline_{name}", score)
    for name, score in finetuned_metrics.items():
        mlflow.log_metric(f"finetuned_{name}", score)
    for path in [loss_path, comparison_path, freq_path, out_path]:
        mlflow.log_artifact(path)

    mlflow.set_tracking_uri("file:./mlruns")
    print("\nDone. Run: mlflow ui")


if __name__ == "__main__":
    main()
