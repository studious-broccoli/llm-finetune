"""
    Script: llm-finetune.py
    Description: complete LLM fine-tuning pipeline using Hugging Face Transformers
                + PEFT (LoRA) + quantization + MLflow for tracking and evaluation
"""

import os
import math
# import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset  # list_datasets
import evaluate
from transformers import pipeline
from peft import LoraConfig, get_peft_model, TaskType

# Begin MLFlow Experiment
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("llm-finetune-gpt2")

# disable Weights and Biases
os.environ['WANDB_DISABLED'] = "true"

output_dir = "./results_llm"
os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------
# Import transformer libraries
# --------------------------------------------
"""
    Bitsandbytes: An excellent package that provides a lightweight wrapper 
    around custom CUDA functions that make LLMs go faster — optimizers, 
    matrix multiplication, and quantization. In this tutorial, we’ll be using 
    this library to load our model as efficiently as possible.
"""
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# --------------------------------------------
# Load dataset from https://huggingface.co/datasets
# --------------------------------------------
# WikiText is a clean dump of WikiPedia articles
# Using Causal Language Modeling (CLM) objective — predicting the next word/token.
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print(dataset["train"][0])


# --------------------------------------------
# Load Tokenizer and model
# --------------------------------------------
model_name = "gpt2"  # GPT-2 tokenizer converts raw text into tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # no pad token, add EOS token for padding to maintain format


# --------------------------------------------
# Tokenize the dataset
# --------------------------------------------
# Chunking: concatenate text into blocks to simulate a continuous stream.
def group_texts(examples):
    block_size = 128
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Convert text into Tokens
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        return_special_tokens_mask=True  # Optional but useful
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.map(group_texts, batched=True)


# --------------------------------------------
# Small Dataset
# --------------------------------------------
small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))


# --------------------------------------------
# Create Bitsandbytes configuration
# Quantization (4-bit) to reduce memory and speed up training
# --------------------------------------------
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )


# --------------------------------------------
# LoRA (Low-Rank Adapters) training only a small part of the model - helps efficiency
# --------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # For GPT2; for LLaMA use ["q_proj", "v_proj"]
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)


# --------------------------------------------
# Load model
# --------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# --------------------------------------------
# Training arguments
# --------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,
    report_to="mlflow",
)

# --------------------------------------------
# Log experiment parameters to MLflow
# --------------------------------------------
mlflow.log_params({
    "model_name": model_name,
    "batch_size": training_args.per_device_train_batch_size,
    "num_epochs": training_args.num_train_epochs,
    "weight_decay": training_args.weight_decay,
    "use_lora": True,
    "quantized_4bit": True,
    "block_size": 128,
    "dataset": "wikitext-2-raw-v1"
})


# --------------------------------------------
# Trainer
# --------------------------------------------
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
)


# --------------------------------------------
# Train
# --------------------------------------------
trainer.train()


# --------------------------------------------
# Save model
# --------------------------------------------
trainer.save_model(os.path.join(output_dir, "fine-tuned-model"))
tokenizer.save_pretrained(os.path.join(output_dir, "fine-tuned-model"))


# --------------------------------------------
# Evaluate Perplexity (how well model predicts a sequence of words)
# --------------------------------------------
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss'])}")


# --------------------------------------------
# Plot Loss
# --------------------------------------------
def plot_loss(log_history, save_path="loss_curve.png"):
    train_loss = [entry["loss"] for entry in log_history if "loss" in entry]
    eval_loss = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
    steps = [entry["step"] for entry in log_history if "loss" in entry]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, label="Train Loss")
    if eval_loss:
        eval_steps = [entry["step"] for entry in log_history if "eval_loss" in entry]
        plt.plot(eval_steps, eval_loss, label="Eval Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


plot_loss(trainer.state.log_history, save_path=os.path.join(output_dir, "loss_curve.png"))


# --------------------------------------------
# Pipeline
# --------------------------------------------
pipe = pipeline("text-generation", model="./fine-tuned-model", tokenizer=tokenizer)
prompt = "The future of AI in medicine is"
print(pipe(prompt, max_new_tokens=50)[0]["generated_text"])

with open(os.path.join(output_dir, "sample_output.txt"), "w") as f:
    f.write(pipe(prompt, max_new_tokens=50)[0]["generated_text"])

# --------------------------------------------
# Token Frequency Histogram
# --------------------------------------------
all_ids = [token for row in small_train_dataset["input_ids"] for token in row]
counts = Counter(all_ids)
top_ids = counts.most_common(20)

ids, freqs = zip(*top_ids)
tokens = [tokenizer.decode([i]) for i in ids]

plt.figure(figsize=(10, 4))
plt.bar(tokens, freqs)
plt.title("Top 20 Token Frequencies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("token_frequencies.png")
plt.close()


# --------------------------------------------
# BLEU
# --------------------------------------------
from evaluate import load
bleu = load("bleu")

sample_output = pipe(prompt, max_new_tokens=50)[0]["generated_text"]
reference_text = "The future of AI in medicine is bright and full of potential."
result = bleu.compute(predictions=[sample_output], references=[reference_text])
print("BLEU Score:", result)


prompts = [
    "The future of AI in medicine is",
    "Quantum computing will",
    "Education systems are evolving because"
]

with open(os.path.join(output_dir, "multiple_generations.txt"), "w") as f:
    for p in prompts:
        output = pipe(p, max_new_tokens=50)[0]["generated_text"]
        print(f"\nPrompt: {p}\nGenerated: {output}\n")
        f.write(f"Prompt: {p}\nGenerated: {output}\n\n")


# --------------------------------------------
# Final MLFlow adds
# --------------------------------------------
mlflow.log_metric("final_perplexity", math.exp(eval_results['eval_loss']))
mlflow.log_artifact(os.path.join(output_dir, "loss_curve.png"))
mlflow.log_artifact(os.path.join(output_dir, "sample_output.txt"))
mlflow.log_metric("bleu_score", result["bleu"])
mlflow.log_artifact("token_frequencies.png")
mlflow.log_artifact(os.path.join(output_dir, "multiple_generations.txt"))
mlflow.set_tracking_uri("file:./mlruns")
print("[-] Run: $ mlflow ui")


"""
Output figures:
- bleu_score: how close model generated text is to a reference (higher is better, 0–1).
- eval_loss: how well the model did on the validation set (lower = better).
- eval_runtime: how long evaluation took (in seconds).
- eval_samples_per_second: how many examples were evaluated per second — shows efficiency.
- eval_steps_per_second: how fast eval steps are processed (higher = faster)
- final_perplexity: measures how "surprised" the model is by the data — lower = better, ideal is ~1
- grad_norm: size of gradients during training — helps detect vanishing/exploding gradients
- learning_rate: step size the model takes during training — how fast it learns.
- loss: training loss at a particular step — lower is better.
- total_flos: total number of floating point operations — a rough measure of compute cost
- train_loss: average loss the model saw while training — should trend down
- train_runtime: ttal time training took (in seconds)
- train_samples_per_second: speed of training in terms of examples per second
- train_steps_per_second: speed of training in terms of steps per second.
- loss_curve: how loss (train & eval) changed over time — helps visualize learning progress
- multiple_generations.txt: file with generated text samples for several different prompts
- sample_output.txt: generated example from model, typically from a main prompt
- token_frequencies.png: bar chart of the most common tokens in training data — good for sanity-checking inputs

"""

# --------------------------------------------
# Tutorial Sources
# --------------------------------------------
# https://www.datacamp.com/tutorial/fine-tuning-large-language-models
# https://www.reddit.com/r/LocalLLaMA/comments/1fm59kg/how_do_you_actually_finetune_a_llm_on_your_own/
# https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07
# huggingface_dataset_name = "neil-code/dialogsum-test"
# dataset = load_dataset(huggingface_dataset_name)
