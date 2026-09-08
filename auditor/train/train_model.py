"""auditor/train/train_model.py

Trains CodeBERT on the deduplicated SolidiFI + SmartBugs datasets.
This script is designed to be run on a Colab T4 GPU.

Key features (per ADRs):
- Multi-label classification (sigmoid + BCE)
- Class weighting (pos_weight) for the rare Access Control class
- Emphasizes Recall and per-class metrics
"""

import json
import numpy as np
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from sklearn.metrics import precision_recall_fscore_support

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
NUM_CLASSES = 4
EPOCHS = 4     
BATCH_SIZE = 8
LR = 3e-5       # Slightly higher learning rate


# ── Data Loading & Weights ────────────────────────────────────────────────────

def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def get_pos_weights(train_data: list) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    Formula: pos_weight = (number of negative examples) / (number of positive examples)
    This severely penalizes the model for missing the rare access control class.
    """
    counts = np.zeros(NUM_CLASSES)
    for item in train_data:
        counts += np.array(item["label"])
        
    counts = np.maximum(counts, 1)  # Prevent division by zero
    neg_counts = len(train_data) - counts
    
    pos_weights = neg_counts / counts
    return torch.tensor(pos_weights, dtype=torch.float)


# ── Custom Trainer ────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Overrides the default loss function to use our computed pos_weights."""
    def __init__(self, pos_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Move weights to the correct device (GPU if available)
        self.pos_weights = pos_weights.to(self.args.device)
        self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(p: EvalPrediction):
    """Calculate P/R/F1 individually for all 4 classes."""
    logits = p.predictions
    labels = p.label_ids
    
    # Apply sigmoid to convert raw logits into probabilities [0, 1]
    probs = 1.0 / (1.0 + np.exp(-logits))
    # Threshold at 0.5
    preds = (probs > 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    class_names = ["SWC_107", "SWC_115", "SWC_104", "SWC_105_106"]
    metrics = {}
    
    for i, name in enumerate(class_names):
        metrics[f"{name}_precision"] = precision[i]
        metrics[f"{name}_recall"] = recall[i]
        metrics[f"{name}_f1"] = f1[i]
        
    # We track macro recall to tell HuggingFace which model checkpoint is "best"
    metrics["macro_recall"] = np.mean(recall)
    metrics["macro_f1"] = np.mean(f1)
    
    return metrics


# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    # Make path absolute relative to the script to avoid Colab working directory issues
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "processed_data"
    
    if not data_dir.exists():
        print(f"Error: Could not find {data_dir}. Run prepare_data.py first.")
        print(f"Current working directory was: {Path.cwd()}")
        return

    print("Loading datasets...")
    train_data = load_jsonl(data_dir / "train.jsonl")
    eval_data = load_jsonl(data_dir / "eval.jsonl")
    
    pos_weights = get_pos_weights(train_data)
    print(f"Calculated pos_weights for BCE Loss: {pos_weights.tolist()}")
    
    # Convert to HuggingFace Dataset format
    train_ds = Dataset.from_list([{"text": d["text"], "labels": d["label"]} for d in train_data])
    eval_ds = Dataset.from_list([{"text": d["text"], "labels": d["label"]} for d in eval_data])
    
    print(f"Loading {MODEL_NAME} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        
    print("Tokenizing (this takes a minute)...")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)
    
    print(f"Loading {MODEL_NAME} model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_CLASSES,
        problem_type="multi_label_classification"
    )
    
    # We use standard evaluation_strategy="epoch"
    # Use standard load_best_model_at_end targeting macro_recall
    training_args = TrainingArguments(
        output_dir="auditor/train/model_output",
        eval_strategy="epoch",        # Evaluate at the end of every epoch
        save_strategy="epoch",        # Save at the end of every epoch
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2, # Added: effective batch size = 16 for better gradients
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_recall",  # Recall is more important than precision for auditors
        greater_is_better=True,
    )
    
    trainer = WeightedTrainer(
        pos_weights=pos_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    
    print("\nStarting training loop...")
    trainer.train()
    
    print("\nFinal Evaluation on SmartBugs:")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        if k.startswith("eval_SWC"):
            print(f"  {k}: {v:.4f}")
    
    # Save the absolute best model checkpoint
    final_dir = "auditor/train/final_classifier"
    print(f"\nSaving best model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done! You can now download `final_classifier/` from Colab to your local machine.")

if __name__ == "__main__":
    main()
