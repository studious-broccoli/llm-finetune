# llm-finetune
Basic LLM fine-tuning

Complete LLM fine-tuning pipeline using Hugging Face Transformers + PEFT (LoRA) + quantization + MLflow for tracking and evaluation

<figure>
    <img src="results_llm/loss_curve.png" alt="Screenshot">
    <figcaption>Figure 1: Loss Curve.</figcaption>
</figure>

Training and Evaluation Loss (TBD)
- Consider changing learning rate, weight decay, batch size, and dropout to stabilize curves.
- Confirm x-axis


Outputs (MLFlow):
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


Outputs (local):
- loss_curve: how loss (train & eval) changed over time — helps visualize learning progress
- multiple_generations.txt: file with generated text samples for several different prompts
- sample_output.txt: generated example from model, typically from a main prompt
- token_frequencies.png: bar chart of the most common tokens in training data — good for sanity-checking inputs
