"""Run the fine-tuned BioBERT NER model on free text.

Usage:
    python src/predict.py "The patient complained of fatigue and was admitted."
    python src/predict.py              # runs the built-in demo sentences
"""
import json
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import MODEL_OUTPUT_DIR
from device import get_device


def load_model(model_path=MODEL_OUTPUT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    with open(f"{model_path}/id2label.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    return tokenizer, model, id2label


def get_entities(words, tags):
    entities, entity, entity_type = [], [], None
    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if entity:
                entities.append((" ".join(entity), entity_type))
            entity, entity_type = [word], tag[2:]
        elif tag.startswith("I-") and entity_type == tag[2:]:
            entity.append(word)
        else:
            if entity:
                entities.append((" ".join(entity), entity_type))
            entity, entity_type = [], None
    if entity:
        entities.append((" ".join(entity), entity_type))
    return entities


def predict(text, tokenizer, model, id2label, device):
    model.eval()
    words = text.split()
    tokens = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        logits = model(**tokens).logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()
    word_ids = tokenizer(words, is_split_into_words=True, truncation=True).word_ids()

    tags, prev = [], None
    for idx, wid in enumerate(word_ids):
        if wid is None or wid == prev:
            continue
        tags.append(id2label.get(predictions[idx], "O"))
        prev = wid
    return get_entities(words, tags)


def main():
    device = get_device()
    print(f"Using device: {device}\n")
    tokenizer, model, id2label = load_model()
    model.to(device)

    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    else:
        texts = [
            "The patient complained of fatigue and was admitted for observation.",
            "The patient was extremely dehydrated due to immense summer heat.",
        ]

    for text in texts:
        print(f"Text: {text}")
        ents = predict(text, tokenizer, model, id2label, device)
        if ents:
            for ent_text, ent_type in ents:
                print(f"  - {ent_text!r:40} -> {ent_type}")
        else:
            print("  (no entities detected)")
        print()


if __name__ == "__main__":
    main()
