"""Evaluate the fine-tuned BioBERT model on the held-out test set.

Reports entity-level precision / recall / F1 (padding and the "O" tag are
excluded so the numbers reflect real NER performance, not the dominant
non-entity class).
"""
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForTokenClassification

from config import BATCH_SIZE, MODEL_OUTPUT_DIR, TEST_NPZ
from device import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open(MODEL_OUTPUT_DIR / "id2label.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    model = AutoModelForTokenClassification.from_pretrained(MODEL_OUTPUT_DIR).to(device)
    model.eval()

    d = np.load(TEST_NPZ)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(d["input_ids"]),
            torch.tensor(d["attention_mask"]),
            torch.tensor(d["labels"]),
        ),
        batch_size=BATCH_SIZE,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions = torch.argmax(logits, dim=-1)
            for i in range(labels.shape[0]):
                true = labels[i].cpu().numpy()
                pred = predictions[i].cpu().numpy()
                m = true != -100
                all_preds.extend(pred[m])
                all_labels.extend(true[m])

    used = sorted(set(all_labels) | set(all_preds))
    used = [i for i in used if i != -100 and i in id2label and id2label[i] != "O"]

    report = classification_report(
        all_labels, all_preds, labels=used,
        target_names=[id2label[i] for i in used],
        output_dict=True, zero_division=0,
    )
    print("\nPer-entity report:")
    print(classification_report(
        all_labels, all_preds, labels=used,
        target_names=[id2label[i] for i in used], zero_division=0,
    ))

    print("\nModel Evaluation Summary (entities only):")
    print("-" * 55)
    print(f"{'Metric':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
    print("-" * 55)
    for name in ("micro avg", "macro avg", "weighted avg"):
        r = report[name]
        print(f"{name:<15}{r['precision']:<12.4f}{r['recall']:<12.4f}{r['f1-score']:<12.4f}")
    print("-" * 55)
    print(f"{'Token Accuracy':<15}{accuracy_score(all_labels, all_preds):<12.4f}")


if __name__ == "__main__":
    main()
