"""Standalone inference script for the fine-tuned model.

Usage:
    python scripts/infer.py --model_dir results_llm/fine-tuned-model \
        --prompt "The future of AI in medicine is" \
        --max_new_tokens 100
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Run inference on the fine-tuned LoRA model.")
    parser.add_argument("--model_dir", type=str, default="results_llm/fine-tuned-model")
    parser.add_argument("--prompt", type=str, default="The future of AI in medicine is")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=5, help="Runs for latency averaging")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--do_sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load the fine-tuned model and generate text for the given prompt."""
    args = parse_args()

    print(f"Loading model from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto")
    model.eval()

    device = next(model.parameters()).device
    gen_kwargs: dict = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": 1.3,
    }
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Warmup
    _ = pipe(args.prompt, **gen_kwargs)

    # Timed runs
    latencies: list[float] = []
    for _ in range(args.num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pipe(args.prompt, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    generated = result[0]["generated_text"]
    continuation = generated[len(args.prompt):]
    token_count = len(tokenizer.encode(continuation))
    avg_latency = sum(latencies) / len(latencies)
    tokens_per_sec = token_count / avg_latency

    print(f"\n{'='*60}")
    print(f"Prompt:      {args.prompt}")
    print(f"Continuation:{continuation}")
    print(f"{'='*60}")
    print(f"Device:           {device}")
    print(f"Tokens generated: {token_count}")
    print(f"Avg latency:      {avg_latency*1000:.1f} ms  (n={args.num_runs})")
    print(f"Throughput:       {tokens_per_sec:.1f} tokens/sec")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory:  {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
