"""Central configuration for the BioBERT medical NER project."""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

BIO_DIR = DATA_DIR / "bio_data_files"
TRAIN_NPZ = DATA_DIR / "train.npz"
VAL_NPZ = DATA_DIR / "val.npz"
TEST_NPZ = DATA_DIR / "test.npz"
LABEL2ID_JSON = DATA_DIR / "label2id.json"
ID2LABEL_JSON = DATA_DIR / "id2label.json"

MODEL_OUTPUT_DIR = MODELS_DIR / "biobert_ner"

# --- Model / training hyper-parameters --------------------------------------
BASE_MODEL = "dmis-lab/biobert-base-cased-v1.1"
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
SEED = 42
IGNORE_LABEL_ID = -100
