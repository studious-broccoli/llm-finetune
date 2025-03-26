import os
import torch
from datasets import load_dataset  # list_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Step 1: Load tokenizer and model
model_name = "gpt2"  # You can replace with EleutherAI/gpt-neo-125M, etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT2 doesn't have padding token by default
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Load or create dataset
# https://huggingface.co/datasets
# all_datasets = list_datasets()
# print(all_datasets[:10])  # Show some available datasets

# Choose one
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print(dataset["train"][0])


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to True if using a masked language model like BERT
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,  # Set True if you want to upload to Hugging Face Hub
)

# Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,  # still OK for now despite deprecation
    data_collator=data_collator,
)

# Train
trainer.train()

#  Save model
trainer.save_model("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
