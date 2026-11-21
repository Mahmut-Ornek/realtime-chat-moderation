"""Knowledge distillation (technique area D): BERT-base teacher -> 4-layer BERT-small student.

Teacher : our fine-tuned bert-base-uncased (12 layers, 110M parameters).
Student : google/bert_uncased_L-4_H-512_A-8 (4 layers, hidden 512, ~29M parameters).
Loss    : alpha * CE(student, gold label)
          + (1 - alpha) * T^2 * KL( softmax(teacher/T) || softmax(student/T) )      (Hinton et al., 2015)
          with temperature T = 2 and alpha = 0.5.
The best epoch on validation macro-F1 is saved, then quantized to INT8 (see bert_model.quantize).

Run:  python -m src.distill
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .bert_model import _batches, _evaluate, quantize
from .config import BERT_DIR, LABELS, SEED, STUDENT_BASE, STUDENT_DIR, STUDENT_INT8_PATH

T, ALPHA = 2.0, 0.5


@torch.no_grad()
def teacher_logits(texts, device):
    teacher = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(device).eval()
    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    out = []
    for enc, _ in _batches(tok, texts, [0] * len(texts), 128, False, device):
        out.append(teacher(**enc).logits.float().cpu())
    del teacher
    return torch.cat(out)


def train(splits, epochs=6, lr=1e-4, bs=32):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_x, train_y = splits["train"]
    t_logits = teacher_logits(train_x, device)
    print(f"  teacher logits computed for {len(train_x)} training messages")

    # the student shares the BERT uncased WordPiece vocabulary with the teacher
    tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_BASE, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}
    ).to(device)
    steps = epochs * ((len(train_x) + bs - 1) // bs)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best = -1
    for epoch in range(1, epochs + 1):
        student.train()
        perm = np.random.permutation(len(train_x))
        total, n = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            enc, y = next(_batches(tok, [train_x[j] for j in idx], [train_y[j] for j in idx], bs, False, device))
            s_logits = student(**enc).logits
            t = t_logits[idx].to(device)
            hard = F.cross_entropy(s_logits, y)
            soft = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t / T, -1), reduction="batchmean") * T * T
            loss = ALPHA * hard + (1 - ALPHA) * soft
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        f1 = _evaluate(student, tok, *splits["validation"], device)
        print(f"  student epoch {epoch}  loss={total / n:.4f}  val macro-F1={f1:.4f}")
        if f1 > best:
            best = f1
            student.save_pretrained(STUDENT_DIR)
            tok.save_pretrained(STUDENT_DIR)
    print(f"  -> saved best student (val macro-F1={best:.4f}) to {STUDENT_DIR}")


def main():
    from .data import load_splits
    train(load_splits())
    quantize(STUDENT_DIR, STUDENT_INT8_PATH)


if __name__ == "__main__":
    main()
