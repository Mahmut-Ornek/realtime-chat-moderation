"""One registry for all models so the evaluation script and the app use identical code paths.
Every model exposes predict_proba(raw_text) -> P(offensive); normalization happens inside."""
import torch

from .config import (BERT_DIR, BERT_INT8_PATH, CPU_THREADS, FFN_PATH, LEXICON_PATH, STUDENT_DIR, STUDENT_INT8_PATH,
                     TFIDF_PATH, W2V_PATH)

# display name -> (loader, files that make up the deployed model, precision)
MODELS = {
    "Rule: edit-distance lexicon": ("lexicon", [LEXICON_PATH], "rules"),
    "Baseline: tf-idf + LogReg": ("tfidf", [TFIDF_PATH], "FP64"),
    "NN: own word2vec + FFN": ("w2v_ffn", [W2V_PATH, FFN_PATH], "FP32"),
    "Transformer: BERT (FP32)": ("bert_fp32", [BERT_DIR / "model.safetensors"], "FP32"),
    "Transformer: BERT (INT8 quantized)": ("bert_int8", [BERT_INT8_PATH], "INT8 weights"),
    "Distilled: BERT-small (FP32)": ("student_fp32", [STUDENT_DIR / "model.safetensors"], "FP32"),
    "Distilled: BERT-small (INT8 quantized)": ("student_int8", [STUDENT_INT8_PATH], "INT8 weights"),
}
# the deployed model: distilled + quantized Transformer (the only one that meets the latency budget)
DEFAULT_MODEL = "Distilled: BERT-small (INT8 quantized)"


def load_model(name):
    torch.set_num_threads(CPU_THREADS)
    kind = MODELS[name][0]
    if kind == "lexicon":
        from .lexicon import LexiconMatcher
        return LexiconMatcher.load()
    if kind == "tfidf":
        from . import tfidf_model
        return tfidf_model.load()
    if kind == "w2v_ffn":
        from . import w2v_ffn
        return w2v_ffn.load()
    from . import bert_model
    if kind == "bert_fp32":
        return bert_model.load_fp32(BERT_DIR)
    if kind == "bert_int8":
        return bert_model.load_int8(BERT_DIR, BERT_INT8_PATH)
    if kind == "student_fp32":
        return bert_model.load_fp32(STUDENT_DIR)
    return bert_model.load_int8(STUDENT_DIR, STUDENT_INT8_PATH)


def model_size_mb(name) -> float:
    files = MODELS[name][1]
    total = 0
    for f in files:
        total += f.stat().st_size
        npy = f.with_name(f.name + ".vectors.npy")  # gensim stores large arrays next to the .kv file
        if npy.exists():
            total += npy.stat().st_size
    return total / 1e6
