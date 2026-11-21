"""Shared paths and settings for the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset: OffensEval-2019 (English), distributed as the "offensive" subset of TweetEval.
DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "offensive"
LABELS = ["clean", "offensive"]

# Model artifact locations
LEXICON_PATH = MODELS_DIR / "lexicon.json"
TFIDF_PATH = MODELS_DIR / "tfidf_lr.joblib"
W2V_PATH = MODELS_DIR / "word2vec.kv"
FFN_PATH = MODELS_DIR / "w2v_ffn.pt"
BERT_BASE = "bert-base-uncased"
BERT_DIR = MODELS_DIR / "bert_fp32"
BERT_INT8_PATH = MODELS_DIR / "bert_int8.pt"
# Knowledge-distillation student: 4-layer BERT ("BERT-small", Turc et al. 2019), quantized for deployment
STUDENT_BASE = "google/bert_uncased_L-4_H-512_A-8"
STUDENT_DIR = MODELS_DIR / "student_fp32"
STUDENT_INT8_PATH = MODELS_DIR / "student_int8.pt"

# Latency budget for the CPU-only moderation server (per message, end to end).
LATENCY_BUDGET_MS = 50.0
# The server runs one worker per CPU core, so every model is benchmarked with 1 thread.
CPU_THREADS = 1
MAX_LEN = 128
SEED = 42

# Hugging Face Hub repo that holds the trained weights (fill in after uploading; see src/download.py)
HF_MODEL_REPO = "mahmut-ornek/realtime-chat-moderation-models"
