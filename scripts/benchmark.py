"""Memory and latency benchmark: 4-bit QLoRA vs FP16 full precision.

Measures peak GPU memory and inference throughput for both loading modes,
then prints a comparison table and saves results to results_llm/benchmark.json.

Usage:
    python scripts/benchmark.py

Requires a CUDA-capable GPU. On CPU-only machines the memory measurements
will be skipped and only latency is reported.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

MODEL_NAME = "gpt2"
PROMPT = "The history of artificial intelligence begins with"
MAX_NEW_TOKENS = 64
WARMUP_RUNS = 2
TIMED_RUNS = 5
OUTPUT_PATH = "results_llm/benchmark.json"


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return None


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(
    model_name: str,
    use_4bit: bool,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, float | str | None]:
    """Load a model and measure peak memory + inference latency.

    Args:
        model_name: HuggingFace model ID.
        use_4bit: If True, loads in NF4 4-bit; otherwise loads in FP32/FP16.
        prompt: Text prompt for inference timing.
        max_new_tokens: Tokens to generate per run.

    Returns:
        Dict with keys: ``precision``, ``peak_memory_mb``, ``avg_latency_ms``,
        ``tokens_per_sec``, ``model_size_params``.
    """
    label = "4-bit NF4" if use_4bit else "FP16"
    print(f"\n[{label}] Loading {model_name} ...")

    _reset_peak_memory()

    bnb_config: BitsAndBytesConfig | None = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"device_map": "auto"}
    if use_4bit:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    load_memory = _peak_memory_mb()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)

    _reset_peak_memory()
    latencies: list[float] = []
    for _ in range(TIMED_RUNS):
        _sync()
        t0 = time.perf_counter()
        result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        _sync()
        latencies.append(time.perf_counter() - t0)

    peak_memory = _peak_memory_mb()
    continuation = result[0]["generated_text"][len(prompt):]
    token_count = len(tokenizer.encode(continuation))
    avg_latency = sum(latencies) / len(latencies)

    del model, pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "precision": label,
        "load_peak_memory_mb": round(load_memory, 1) if load_memory else None,
        "inference_peak_memory_mb": round(peak_memory, 1) if peak_memory else None,
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "tokens_per_sec": round(token_count / avg_latency, 1),
        "total_params": total_params,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table.

    Args:
        results: List of benchmark result dicts.
    """
    headers = ["Precision", "Load mem (MB)", "Infer mem (MB)", "Latency (ms)", "Tokens/sec"]
    rows = [
        [
            r["precision"],
            str(r["load_peak_memory_mb"] or "N/A (CPU)"),
            str(r["inference_peak_memory_mb"] or "N/A (CPU)"),
            str(r["avg_latency_ms"]),
            str(r["tokens_per_sec"]),
        ]
        for r in results
    ]
    col_widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print("\n" + fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))

    if results[0]["load_peak_memory_mb"] and results[1]["load_peak_memory_mb"]:
        savings = 100 * (1 - results[1]["load_peak_memory_mb"] / results[0]["load_peak_memory_mb"])
        print(f"\nMemory reduction (4-bit vs FP16): {savings:.1f}%")
    speedup = results[0]["avg_latency_ms"] / results[1]["avg_latency_ms"]
    print(f"Latency ratio (FP16 / 4-bit): {speedup:.2f}x")


def main() -> None:
    """Run benchmarks for FP16 and 4-bit and save results."""
    os.makedirs("results_llm", exist_ok=True)

    fp16_result = benchmark_model(MODEL_NAME, use_4bit=False, prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS)
    quant_result = benchmark_model(MODEL_NAME, use_4bit=True, prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS)

    results = [fp16_result, quant_result]
    print_table(results)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
