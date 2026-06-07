"""Fine-tune BioBERT for token-level medical NER.

Runs on Apple Silicon GPU (MPS) when available, otherwise CUDA or CPU.
Loads the .npz tensors produced by prepare_data.py, trains, plots loss,
and saves the fine-tuned model + tokenizer to models/biobert_ner.
"""
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, get_scheduler,
)

from config import (
    BASE_MODEL, BATCH_SIZE, ID2LABEL_JSON, LABEL2ID_JSON, LEARNING_RATE,
    MODEL_OUTPUT_DIR, NUM_EPOCHS, TEST_NPZ, TRAIN_NPZ, VAL_NPZ,
)
from device import get_device


def load_split(path):
    d = np.load(path)
    return TensorDataset(
        torch.tensor(d["input_ids"]),
        torch.tensor(d["attention_mask"]),
        torch.tensor(d["labels"]),
    )


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open(LABEL2ID_JSON) as f:
        label2id = json.load(f)
    with open(ID2LABEL_JSON) as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    train_loader = DataLoader(load_split(TRAIN_NPZ), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(load_split(VAL_NPZ), batch_size=BATCH_SIZE)

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label2id), id2label=id2label, label2id=label2id
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * NUM_EPOCHS
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    train_losses, val_losses = [], []
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}\n" + "-" * 30)
        model.train()
        total_train = 0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        total_val, preds, golds = 0, [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                for i in range(labels.shape[0]):
                    true = labels[i].cpu().numpy()
                    pred = predictions[i].cpu().numpy()
                    m = true != -100
                    preds.extend(pred[m])
                    golds.extend(true[m])
        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)
        print(f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
              f"Val Acc: {accuracy_score(golds, preds):.4f}")

    # Loss curve
    MODEL_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, NUM_EPOCHS + 1), train_losses, "o-", label="Train")
        plt.plot(range(1, NUM_EPOCHS + 1), val_losses, "x--", label="Val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(MODEL_OUTPUT_DIR.parent / "training_loss_curve.png")
        print("Saved loss curve.")
    except Exception as e:
        print(f"(skipped plot: {e})")

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUTPUT_DIR)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(MODEL_OUTPUT_DIR)
    with open(MODEL_OUTPUT_DIR / "label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    with open(MODEL_OUTPUT_DIR / "id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)
    print(f"\nModel saved to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
