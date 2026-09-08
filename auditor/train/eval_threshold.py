"""auditor/train/eval_threshold.py

Re-evaluates the saved model at a custom sigmoid threshold.
Run this locally after downloading final_classifier/ from Colab.

Usage:
  .venv/bin/python -m auditor.train.eval_threshold --threshold 0.15
"""

import argparse
import json
import numpy as np
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support

CLASS_NAMES = ["SWC-107 Reentrancy", "SWC-115 tx.origin", "SWC-104 Unchecked Call", "SWC-105/106 Access Control"]
MAX_LENGTH = 512
BATCH_SIZE = 8


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(threshold: float):
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "processed_data"
    model_dir = script_dir / "final_classifier"

    if not model_dir.exists():
        print(f"Error: {model_dir} not found. Download final_classifier/ from Colab first.")
        return

    print(f"Loading eval data from {data_dir}...")
    eval_data = load_jsonl(data_dir / "eval.jsonl")
    texts = [d["text"] for d in eval_data]
    labels = np.array([d["label"] for d in eval_data])

    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running on: {device}")

    all_logits = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, truncation=True, padding="max_length",
                           max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    preds = (probs > threshold).astype(int)

    print(f"\n{'='*55}")
    print(f"  Eval Results @ threshold = {threshold}")
    print(f"{'='*55}")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    for i, name in enumerate(CLASS_NAMES):
        pos = int(labels[:, i].sum())
        flagged = int(preds[:, i].sum())
        print(f"\n  {name}")
        print(f"    Positives in eval set : {pos}")
        print(f"    Flagged by model      : {flagged}")
        print(f"    Precision : {precision[i]:.4f}")
        print(f"    Recall    : {recall[i]:.4f}")
        print(f"    F1        : {f1[i]:.4f}")

    print(f"\n  Macro F1     : {np.mean(f1):.4f}")
    print(f"  Macro Recall : {np.mean(recall):.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Sigmoid probability threshold (default: 0.15)")
    args = parser.parse_args()
    evaluate(args.threshold)
