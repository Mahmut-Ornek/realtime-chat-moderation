"""Trains every model of the ladder, quantizes BERT, then runs the evaluation.

Run:  python -m src.train_all        (about 10-15 minutes with a GPU)
"""
import time

from . import bert_model, distill, evaluate, lexicon, tfidf_model, w2v_ffn
from .config import STUDENT_DIR, STUDENT_INT8_PATH
from .data import load_splits


def main():
    t0 = time.time()
    splits = load_splits()
    print({k: len(v[0]) for k, v in splits.items()})
    print("\n[1/6] Edit-distance lexicon (rule baseline)")
    lexicon.train(splits)
    print("\n[2/6] tf-idf + logistic regression (classical baseline)")
    tfidf_model.train(splits)
    print("\n[3/6] Own word2vec + feedforward network (SGD + backprop)")
    w2v_ffn.train(splits)
    print("\n[4/6] Fine-tuning BERT")
    bert_model.train(splits)
    print("\n[5/6] INT8 dynamic quantization of BERT-base")
    bert_model.quantize()
    print("\n[6/6] Knowledge distillation into BERT-small, then INT8 quantization")
    distill.train(splits)
    bert_model.quantize(STUDENT_DIR, STUDENT_INT8_PATH)
    print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min. Evaluating on the test set...")
    evaluate.main()


if __name__ == "__main__":
    main()
