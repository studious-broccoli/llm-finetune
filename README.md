# llm-finetune

Fine-tunes GPT-2 on WikiText-2 using QLoRA (4-bit quantization + LoRA adapters), with MLflow experiment tracking and a pre/post evaluation suite covering perplexity, ROUGE, and BERTScore.

---

## How it works

### Model and dataset

The base model is **GPT-2** (124M parameters), trained on a 1,000-sample subset of **WikiText-2** — a clean collection of Wikipedia articles. The training objective is causal language modeling (CLM): predict the next token given all previous tokens. WikiText-2 was chosen because it is small enough to train quickly while being representative of formal, factual English prose.

### Efficiency: QLoRA

Training a full LLM is memory-intensive. This pipeline combines two techniques to make it tractable on consumer hardware:

**4-bit quantization (QLoRA)** loads the base model weights in NormalFloat4 (NF4) format using `bitsandbytes`, reducing GPU memory by ~4x compared to full float32. Compute is performed in float16 during forward and backward passes.

**LoRA (Low-Rank Adapters)** freezes the base model weights entirely and injects small trainable rank-decomposition matrices into the attention layers (`c_attn` for GPT-2). With `r=8` and `lora_alpha=32`, only ~0.1% of the model's parameters are updated during training. This makes fine-tuning fast, memory-efficient, and less prone to catastrophic forgetting.

### Tokenization

Text is tokenized with the GPT-2 BPE tokenizer. Because GPT-2 has no dedicated pad token, the EOS token is reused for padding. After tokenization, sequences are concatenated and chunked into fixed 128-token blocks to simulate a continuous text stream — the standard approach for causal LM training.

### Training

The Hugging Face `Trainer` runs for 3 epochs with a batch size of 4, weight decay of 0.01, and evaluation after each epoch. All hyperparameters and run metadata are logged to MLflow at the start of training.

---

## Evaluation

Evaluation is designed to measure both language modeling quality and generation quality before and after fine-tuning.

**Perplexity** is computed from the held-out eval loss: `exp(eval_loss)`. It measures how confidently the model assigns probability to unseen WikiText sequences — lower is better.

**Pre/post generation comparison** captures outputs from the base GPT-2 (before LoRA is applied) and the fine-tuned model on the same set of 20 continuation prompts drawn from the validation split. Each prompt is the first half of a tokenized WikiText block; the second half serves as the ground-truth reference. This makes the evaluation grounded in the actual training distribution.

The following metrics are computed for both baseline and fine-tuned generations:

| Metric | What it measures |
|---|---|
| ROUGE-1 | Unigram overlap between generation and reference |
| ROUGE-2 | Bigram overlap — rewards fluent two-word sequences |
| ROUGE-L | Longest common subsequence — captures sentence-level structure |
| BERTScore F1 | Semantic similarity via DistilBERT embeddings — catches paraphrases ROUGE misses |

All metrics are logged to MLflow as `baseline_<metric>` and `finetuned_<metric>`, and a grouped bar chart (`metric_comparison.png`) visualizes the delta.

---

## Outputs

**MLflow metrics**

| Metric | Description |
|---|---|
| `final_perplexity` | `exp(eval_loss)` — how surprised the model is by held-out text |
| `eval_loss` | Validation cross-entropy loss |
| `train_loss` | Average training loss across the run |
| `grad_norm` | Gradient magnitude — high values may indicate instability |
| `learning_rate` | Logged per step |
| `total_flos` | Total floating point operations — proxy for compute cost |
| `baseline_rouge1/2/L` | ROUGE scores for base GPT-2 on continuation eval set |
| `finetuned_rouge1/2/L` | ROUGE scores for fine-tuned model on continuation eval set |
| `baseline_bertscore_f1` | Semantic similarity score for base GPT-2 |
| `finetuned_bertscore_f1` | Semantic similarity score for fine-tuned model |

**Local files (`results_llm/`)**

| File | Description |
|---|---|
| `loss_curve.png` | Train and eval loss over training steps |
| `metric_comparison.png` | Grouped bar chart: baseline vs fine-tuned on ROUGE + BERTScore |
| `sample_output.txt` | Single generation from the fine-tuned model |
| `multiple_generations.txt` | Generations for three open-ended prompts |
| `token_frequencies.png` | Top-20 token frequency histogram — sanity check on tokenized input |

---

## Setup

```bash
pip install torch transformers peft datasets evaluate bitsandbytes bert-score mlflow matplotlib
mlflow server --host 127.0.0.1 --port 5000  # in a separate terminal
python llm-finetune.py
mlflow ui                                    # view results
```

---

<figure>
    <img src="results_llm/loss_curve.png" alt="Loss curve">
    <figcaption>Figure 1: Training and evaluation loss over steps.</figcaption>
</figure>
