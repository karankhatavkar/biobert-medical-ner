"""Convert BIO-tagged files into BioBERT-tokenised train/val/test .npz tensors.

Reads ``data/bio_data_files/*.bio`` and produces:
    data/train.npz, data/val.npz, data/test.npz   (input_ids, attention_mask, labels)
    data/label2id.json, data/id2label.json

This is the PyTorch/HuggingFace-only replacement for the old TensorFlow
preprocessing step. Run it after notebook 02 has created the .bio files.
"""
import json
import os
import re
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config import (
    BASE_MODEL, BIO_DIR, ID2LABEL_JSON, IGNORE_LABEL_ID, LABEL2ID_JSON,
    MAX_LENGTH, SEED, TEST_NPZ, TRAIN_NPZ, VAL_NPZ,
)


def read_bio_file(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Parse a single .bio file into lists of (sentence, labels)."""
    sentences, labels = [], []
    curr_sentence, curr_labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_sentence:
                    sentences.append(curr_sentence)
                    labels.append(curr_labels)
                    curr_sentence, curr_labels = [], []
                continue
            splits = line.split()
            if len(splits) != 2:
                continue
            word, tag = splits
            word = re.sub(r"[^\w\s]", "", word).strip().lower()
            if word:
                curr_sentence.append(word)
                curr_labels.append(tag)
    if curr_sentence:
        sentences.append(curr_sentence)
        labels.append(curr_labels)
    return sentences, labels


def load_all_bio_files(directory: str) -> Tuple[List[List[str]], List[List[str]]]:
    all_sentences, all_labels = [], []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".bio"):
            s, l = read_bio_file(os.path.join(directory, filename))
            all_sentences.extend(s)
            all_labels.extend(l)
    return all_sentences, all_labels


def create_label_map(label_lists: List[List[str]]) -> Tuple[dict, dict]:
    unique = {lab for sent in label_lists for lab in sent}
    label2id = {lab: i for i, lab in enumerate(sorted(unique))}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def encode(sentences, labels, label2id, tokenizer):
    """Tokenise with sub-word alignment; non-initial sub-tokens get IGNORE id."""
    input_ids, attention_masks, label_ids = [], [], []
    for words, tags in zip(sentences, labels):
        toks, labs = [], []
        for word, tag in zip(words, tags):
            wp = tokenizer.tokenize(word)
            if not wp:
                continue
            toks.extend(wp)
            labs.extend([label2id[tag]] + [IGNORE_LABEL_ID] * (len(wp) - 1))
        toks = toks[: MAX_LENGTH - 2]
        labs = labs[: MAX_LENGTH - 2]
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + toks + ["[SEP]"])
        labs = [IGNORE_LABEL_ID] + labs + [IGNORE_LABEL_ID]
        mask = [1] * len(ids)
        pad = MAX_LENGTH - len(ids)
        ids += [tokenizer.pad_token_id] * pad
        mask += [0] * pad
        labs += [IGNORE_LABEL_ID] * pad
        input_ids.append(ids)
        attention_masks.append(mask)
        label_ids.append(labs)
    return (np.array(input_ids), np.array(attention_masks), np.array(label_ids))


def main():
    print(f"Reading .bio files from {BIO_DIR} ...")
    sentences, labels = load_all_bio_files(str(BIO_DIR))
    print(f"Loaded {len(sentences)} sentences")

    label2id, id2label = create_label_map(labels)
    print(f"Found {len(label2id)} labels")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_s, temp_s, train_l, temp_l = train_test_split(
        sentences, labels, test_size=0.2, random_state=SEED
    )
    val_s, test_s, val_l, test_l = train_test_split(
        temp_s, temp_l, test_size=0.5, random_state=SEED
    )

    for name, (s, l), path in [
        ("train", (train_s, train_l), TRAIN_NPZ),
        ("val", (val_s, val_l), VAL_NPZ),
        ("test", (test_s, test_l), TEST_NPZ),
    ]:
        ids, mask, labs = encode(s, l, label2id, tokenizer)
        np.savez_compressed(path, input_ids=ids, attention_mask=mask, labels=labs)
        print(f"  saved {name}.npz  ({ids.shape[0]} samples)")

    with open(LABEL2ID_JSON, "w") as f:
        json.dump(label2id, f, indent=2)
    with open(ID2LABEL_JSON, "w") as f:
        json.dump(id2label, f, indent=2)
    print("Saved label mappings.")


if __name__ == "__main__":
    main()
