# -*- coding: utf-8 -*-
"""VLM_Book_Title.ipynb

 
Original file is located at
    https://colab.research.google.com/drive/1T5UjKlfQR2oSVUxhTwgSOeKIpQx9SrwF

# Chapter four, mini project:
Persian Book Cover Title Recognition using Vision-Language Models

Fine-tuning VLM to extract Persian book titles from cover images

Installation
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# import os, re
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # Do this only in Colab notebooks! Otherwise use pip install unsloth
#     import torch; v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
#     xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
#     !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
#     !pip install --no-deps unsloth
# !pip install transformers==4.55.4
# !pip install --no-deps trl==0.22.2

!pip install -q Levenshtein

"""Step 1: Load Dataset and Prepare Data"""

from datasets import load_dataset

dataset = load_dataset("shenasa/bookroom-persian-book-covers-and-titles")

dataset

dataset['test'][112]['text']

dataset['test'][112]['image']

dataset['test'][1]['image']

dataset['train'][150]['text']

dataset['train'][150]['image']

# Select first 1000 samples from train
train_subset = dataset['train'].select(range(min(1000, len(dataset['train']))))

# Select first 200 samples from test
test_subset = dataset['test'].select(range(min(200, len(dataset['test']))))

print(f"\nTrain samples: {len(train_subset)}")
print(f"Test samples: {len(test_subset)}")

test_subset[0]

"""Step 2: Load VLM Model"""

from unsloth import FastVisionModel
import torch

print("\nLoading VLM model...")

# Using Gemma 3 which supports multilingual text including Persian
model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3-4b-pt",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Add LoRA adapters for fine-tuning
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules="all-linear",
)

# Set chat template
from unsloth import get_chat_template
processor = get_chat_template(processor, "gemma-3")

print("Model loaded successfully!")

"""Step 3: Define Metrics"""

import Levenshtein

def exact_match(pred, true):
    """Exact Match: Does prediction exactly match ground truth?"""
    return 1 if pred.strip() == true.strip() else 0

def levenshtein_accuracy(pred, true):
    """Accuracy based on Levenshtein distance (0 to 1)"""
    if len(true) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    distance = Levenshtein.distance(pred, true)
    max_len = max(len(pred), len(true))
    return 1 - (distance / max_len)

def word_level_f1(pred, true):
    """F1-Score at word level"""
    pred_words = set(pred.split())
    true_words = set(true.split())

    if len(true_words) == 0:
        return 1.0 if len(pred_words) == 0 else 0.0

    tp = len(pred_words & true_words)
    fp = len(pred_words - true_words)
    fn = len(true_words - pred_words)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def calculate_metrics(predictions, ground_truths):
    """Calculate all metrics"""
    em_scores = []
    lev_scores = []
    f1_scores = []

    for pred, true in zip(predictions, ground_truths):
        em_scores.append(exact_match(pred, true))
        lev_scores.append(levenshtein_accuracy(pred, true))
        f1_scores.append(word_level_f1(pred, true))

    return {
        'exact_match': sum(em_scores) / len(em_scores) * 100,
        'levenshtein_accuracy': sum(lev_scores) / len(lev_scores) * 100,
        'word_f1': sum(f1_scores) / len(f1_scores) * 100,
    }

"""Step 4: Initial Evaluation (Before Fine-tuning)"""

def evaluate_model(model, processor, test_data, max_samples=None):
    """Evaluate model on test data"""
    FastVisionModel.for_inference(model)

    predictions = []
    ground_truths = []

    instruction = "Write the title of this book."

    samples_to_eval = test_data if max_samples is None else test_data.select(range(max_samples))

    print(f"Evaluating {len(samples_to_eval)} samples...")

    import time
    start_time = time.time()

    for idx, sample in enumerate(samples_to_eval):
        if idx % 10 == 0 and idx > 0:
            elapsed = (time.time() - start_time) / 60
            avg_per_sample = elapsed / idx
            remaining = (len(samples_to_eval) - idx) * avg_per_sample
            print(f"Processing sample {idx}/{len(samples_to_eval)}... (Elapsed: {elapsed:.1f}min, Est. remaining: {remaining:.1f}min)")

        image = sample['image']
        true_text = sample['text']

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ],
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=16,  # CHANGED from 128
                use_cache=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64
            )

        # Extract predicted text
        pred_text = processor.decode(output[0], skip_special_tokens=True)
        pred_text = pred_text.split("model\n")[-1].strip()

        predictions.append(pred_text)
        ground_truths.append(true_text)

        # Show running metrics every 20 samples
        if (idx + 1) % 20 == 0:
            temp_metrics = calculate_metrics(predictions, ground_truths)
            print(f"  → Current metrics after {idx+1} samples:")
            print(f"     Exact Match: {temp_metrics['exact_match']:.2f}%")
            print(f"     Levenshtein: {temp_metrics['levenshtein_accuracy']:.2f}%")

    metrics = calculate_metrics(predictions, ground_truths)
    return metrics, predictions, ground_truths

print("\n" + "="*60)
print("Initial Evaluation (Before Fine-tuning)")
print("="*60)

initial_metrics, initial_preds, initial_truths = evaluate_model(
    model, processor, test_subset, max_samples=200 #max_samples=20 for test
)

print("\nInitial evaluation results:")
print(f"Exact Match: {initial_metrics['exact_match']:.2f}%")
print(f"Levenshtein Accuracy: {initial_metrics['levenshtein_accuracy']:.2f}%")
print(f"Word-level F1: {initial_metrics['word_f1']:.2f}%")

# Show some examples
print("\nSample predictions:")
for i in range(min(5, len(initial_preds))):
    print(f"\nSample {i+1}:")
    print(f"  Ground truth: {initial_truths[i]}")
    print(f"  Prediction: {initial_preds[i]}")

"""The results are very far from the real book titles. Without fine tuning we can not expect to extract titles from the books properly.

Step 5: Fine-tune Model
"""

# Prepare Data for Fine-tuning
instruction_finetune = "Write the title of this book."

def convert_to_conversation(sample):
    """Convert sample to conversation format"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_finetune},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["text"]}]
        },
    ]
    return {"messages": conversation}

# Convert dataset
print("\nPreparing data for fine-tuning...")
converted_dataset = [convert_to_conversation(sample) for sample in train_subset]

print(f"Number of prepared samples: {len(converted_dataset)}")
print("\nExample of prepared data:")
print(converted_dataset[0])

# Fine-tune Model

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

print("\n" + "="*70)
print("Starting Fine-tuning")
print("="*70)

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1,
        warmup_ratio=0.1,
        #num_train_epochs=2,  # Can increase for better results
        num_train_epochs=0.8,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs_persian_books",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=32,
    )
)

# Show memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Start training
print("\nStarting training...")
trainer_stats = trainer.train()

print("\nTraining completed!")
print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")

# Plot Training Loss

import matplotlib.pyplot as plt

logs = trainer.state.log_history
steps = [log["step"] for log in logs if "loss" in log]
losses = [log["loss"] for log in logs if "loss" in log]

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, marker="o", linewidth=2)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.title("Training Loss over Steps", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""Step 6: Final Evaluation (After Fine-tuning)"""

print("\n" + "="*60)
print("Final Evaluation (After Fine-tuning)")
print("="*60)

final_metrics, final_preds, final_truths = evaluate_model(
    model, processor, test_subset, max_samples=200
)

print("\nFinal evaluation results:")
print(f"Exact Match: {final_metrics['exact_match']:.2f}%")
print(f"Levenshtein Accuracy: {final_metrics['levenshtein_accuracy']:.2f}%")
print(f"Word-level F1: {final_metrics['word_f1']:.2f}%")

# Show some examples
print("\nSample predictions after fine-tuning:")
for i in range(min(5, len(final_preds))):
    print(f"\nSample {i+1}:")
    print(f"  Ground truth: {final_truths[i]}")
    print(f"  Prediction: {final_preds[i]}")

"""Step 7: Comparison: Before vs After Fine-tuning"""

print("\n" + "="*55)
print("Comparison: Before vs After Fine-tuning")
print("="*55)

comparison = {
    'Exact Match': [initial_metrics['exact_match'], final_metrics['exact_match']],
    'Levenshtein Acc': [initial_metrics['levenshtein_accuracy'], final_metrics['levenshtein_accuracy']],
    'Word F1': [initial_metrics['word_f1'], final_metrics['word_f1']],
}

for metric_name, values in comparison.items():
    improvement = values[1] - values[0]
    print(f"\n{metric_name}:")
    print(f"  Before: {values[0]:.2f}%")
    print(f"  After: {values[1]:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")

# Plot comparison
import numpy as np

metrics_names = list(comparison.keys())
before_values = [comparison[m][0] for m in metrics_names]
after_values = [comparison[m][1] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, before_values, width, label='Before Fine-tuning', color='skyblue')
bars2 = ax.bar(x + width/2, after_values, width, label='After Fine-tuning', color='lightcoral')

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Performance Comparison: Before vs After Fine-tuning', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Save Model

print("\n" + "="*55)
print("Saving Model")
print("="*55)

# Save locally
model.save_pretrained("persian_book_title_model")
processor.save_pretrained("persian_book_title_model")
print("Model saved locally: ./persian_book_title_model")