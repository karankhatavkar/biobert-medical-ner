# BioBERT Medical NER

Fine-tuning **BioBERT** (`dmis-lab/biobert-base-cased-v1.1`) for token-level
**Named Entity Recognition** on biomedical case reports. The model tags clinical
entities — signs/symptoms, diseases, diagnostic procedures, medications, lab
values, dosages, and more — in free-form medical text.

Built for **Apple Silicon**: training, evaluation, and inference automatically
use the Mac GPU (Metal / MPS) when available, falling back to CUDA or CPU.

## Dataset

[MACCROBAT2018](https://figshare.com/articles/dataset/MACCROBAT2018/9764942) —
~200 annotated biomedical case reports (brat `.ann` + `.txt`), converted to BIO
tagging across ~40 entity types (78 BIO labels + `O`).

## Project layout

```
biobert-medical-ner/
├── environment.yml          # conda env (Apple Silicon / MPS ready)
├── requirements.txt         # pip fallback
├── data/
│   ├── MACCROBAT/           # raw .txt/.ann (gitignored, re-downloadable)
│   ├── bio_data_files/      # BIO-formatted files
│   ├── train/val/test.npz   # tokenised tensors (input_ids, attention_mask, labels)
│   └── label2id.json / id2label.json
├── notebooks/
│   ├── 00_data_download.ipynb
│   ├── 01_eda_and_json_export.ipynb   # EDA + parse .ann -> annotated_data.json
│   ├── 02_convert_to_bio.ipynb        # json -> BIO files
│   ├── 03_preprocessing.ipynb         # BIO -> BioBERT tensors (.npz)
│   └── 04_train_biobert.ipynb         # fine-tune + evaluate + inference demo
├── src/
│   ├── config.py            # paths & hyper-parameters
│   ├── device.py            # CUDA / MPS / CPU selection
│   ├── prepare_data.py      # BIO -> .npz (script form of notebook 03)
│   ├── train.py             # fine-tune BioBERT (script form of notebook 04)
│   ├── evaluate.py          # entity-level P/R/F1 on the test set
│   └── predict.py           # run NER on free text
└── models/biobert_ner/      # fine-tuned weights (gitignored)
```

## Setup

```bash
conda env create -f environment.yml
conda activate biobert-ner

# spaCy model — only needed for notebooks 01–02
python -m spacy download en_core_web_sm

# Verify the GPU backend
python src/device.py        # -> "Selected device: mps" on Apple Silicon
```

## Quick start

The repo ships the processed tensors (`data/*.npz`), so you can train straight
away — no data prep needed. Train first, then evaluate / run inference:

```bash
python src/train.py        # fine-tunes BioBERT -> models/biobert_ner/ (~minutes on MPS)
python src/evaluate.py     # entity-level precision / recall / F1 on the test set
python src/predict.py "The patient complained of fatigue and severe chest pain."
```

> The fine-tuned weights (`models/`) are not committed — they are produced by
> `src/train.py`. `evaluate.py` and `predict.py` require a trained model.

## Reproduce end-to-end (from raw data)

Run the notebooks in order (`jupyter lab`), or use the scripts:

| Step | Notebook | Script | Output |
|------|----------|--------|--------|
| 1 | `00_data_download.ipynb` | — | `data/MACCROBAT/` |
| 2 | `01_eda_and_json_export.ipynb` | — | `data/annotated_json_data/annotated_data.json` |
| 3 | `02_convert_to_bio.ipynb` | — | `data/bio_data_files/*.bio` |
| 4 | `03_preprocessing.ipynb` | `python src/prepare_data.py` | `data/{train,val,test}.npz`, label maps |
| 5 | `04_train_biobert.ipynb` | `python src/train.py` | `models/biobert_ner/` |
| 6 | (in notebook 04) | `python src/evaluate.py` | test metrics |
| 7 | (in notebook 04) | `python src/predict.py "<text>"` | predicted entities |

## Notes

- Reported metrics are **entity-level** (padding and the `O` tag are excluded),
  so they reflect real NER quality rather than the dominant non-entity class.
- If you hit an unsupported-op error on MPS, run with a CPU fallback for that op:
  `PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py`.
