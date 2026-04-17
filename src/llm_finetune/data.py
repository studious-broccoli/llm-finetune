"""Dataset loading and tokenization for causal language modeling."""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import TrainConfig


def load_wikitext(cfg: TrainConfig) -> DatasetDict:
    """Download and return the WikiText-2 dataset splits.

    Args:
        cfg: Training configuration specifying which dataset variant to load.

    Returns:
        Raw HuggingFace DatasetDict with train/validation/test splits.
    """
    return load_dataset("wikitext", cfg.dataset)


def tokenize_and_chunk(
    raw: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    cfg: TrainConfig,
) -> DatasetDict:
    """Tokenize text and chunk into fixed-length blocks for causal LM training.

    Text sequences are first tokenized, then concatenated into a continuous
    stream and split into non-overlapping blocks of ``cfg.block_size`` tokens.
    This avoids padding and ensures every token in the batch is a real signal.

    Args:
        raw: Raw dataset with a ``text`` column.
        tokenizer: Tokenizer matching the base model.
        cfg: Config with ``block_size`` for chunking.

    Returns:
        Tokenized DatasetDict with ``input_ids``, ``attention_mask``, and
        ``labels`` columns. Labels are a copy of ``input_ids`` (standard CLM).
    """

    def _tokenize(examples: dict) -> dict:
        return tokenizer(examples["text"], truncation=True, return_special_tokens_mask=True)

    def _chunk(examples: dict) -> dict:
        concatenated = {k: sum(examples[k], []) for k in examples}
        total = (len(concatenated["input_ids"]) // cfg.block_size) * cfg.block_size
        result = {
            k: [t[i : i + cfg.block_size] for i in range(0, total, cfg.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = raw.map(_tokenize, batched=True, remove_columns=["text"])
    return tokenized.map(_chunk, batched=True)


def sample_splits(tokenized: DatasetDict, cfg: TrainConfig) -> tuple[Dataset, Dataset]:
    """Subsample train and eval splits for fast iteration.

    Args:
        tokenized: Chunked tokenized dataset.
        cfg: Config with ``train_samples``, ``eval_samples``, and ``seed``.

    Returns:
        Tuple of ``(train_dataset, eval_dataset)``.
    """
    train = tokenized["train"].shuffle(seed=cfg.seed).select(range(cfg.train_samples))
    eval_ = tokenized["test"].shuffle(seed=cfg.seed).select(range(cfg.eval_samples))
    return train, eval_


def build_eval_pairs(
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    n: int = 20,
) -> list[tuple[str, str]]:
    """Build prompt/reference continuation pairs from the eval split.

    Each block is split at its midpoint. The first half becomes the prompt;
    the second half is the ground-truth reference used for ROUGE/BERTScore.

    Args:
        eval_dataset: Tokenized HuggingFace dataset with ``input_ids``.
        tokenizer: Tokenizer for decoding token IDs back to text.
        n: Maximum number of pairs to return.

    Returns:
        List of ``(prompt_text, reference_continuation)`` tuples.
    """
    pairs: list[tuple[str, str]] = []
    for sample in eval_dataset.select(range(min(n * 3, len(eval_dataset)))):
        ids = sample["input_ids"]
        if len(ids) < 32:
            continue
        mid = len(ids) // 2
        prompt = tokenizer.decode(ids[:mid], skip_special_tokens=True).strip()
        reference = tokenizer.decode(ids[mid:], skip_special_tokens=True).strip()
        if len(prompt) > 10 and len(reference) > 10:
            pairs.append((prompt, reference))
        if len(pairs) >= n:
            break
    return pairs
