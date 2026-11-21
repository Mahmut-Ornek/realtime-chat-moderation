"""BERT fine-tuning (technique area C) and INT8 dynamic quantization (technique area D).

Fine-tuning: bert-base-uncased (WordPiece tokenizer, 12 self-attention layers) with a
classification head on [CLS]; AdamW, lr 2e-5, linear warmup/decay, 3 epochs, mixed
precision on GPU. The epoch with the best validation macro-F1 is saved.

Quantization: PyTorch post-training *dynamic* quantization. Every nn.Linear layer
(attention Q/K/V/output, feed-forward, classifier) gets INT8 weights; activations are
quantized on the fly at inference time. Embeddings and LayerNorm stay FP32.
Runs on CPU (x86 fbgemm kernels).
"""
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import BERT_BASE, BERT_DIR, BERT_INT8_PATH, LABELS, MAX_LEN, SEED
from .normalize import normalize


class BertModel:
    def __init__(self, model, tokenizer):
        self.model, self.tok = model.eval(), tokenizer

    @torch.no_grad()
    def predict_proba(self, text: str) -> float:
        enc = self.tok(normalize(text), truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return float(torch.softmax(self.model(**enc).logits, dim=-1)[0, 1])


def _batches(tok, texts, labels, bs, shuffle, device):
    idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(idx), bs):
        b = idx[i : i + bs]
        enc = tok([normalize(texts[j]) for j in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        yield enc, torch.tensor([labels[j] for j in b], device=device)


@torch.no_grad()
def _evaluate(model, tok, texts, labels, device):
    model.eval()
    preds = []
    for enc, _ in _batches(tok, texts, labels, 64, False, device):
        preds.extend(model(**enc).logits.argmax(-1).tolist())
    return f1_score(labels, preds, average="macro")


def train(splits, epochs=3, lr=2e-5, bs=32):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}
    ).to(device)
    train_x, train_y = splits["train"]
    steps = epochs * ((len(train_x) + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    best = -1
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for step, (enc, y) in enumerate(_batches(tok, train_x, train_y, bs, True, device), 1):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = nn.functional.cross_entropy(model(**enc).logits, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        f1 = _evaluate(model, tok, *splits["validation"], device)
        print(f"  BERT epoch {epoch}  train loss={total / step:.4f}  val macro-F1={f1:.4f}")
        if f1 > best:
            best = f1
            model.save_pretrained(BERT_DIR)
            tok.save_pretrained(BERT_DIR)
    print(f"  -> saved best epoch (val macro-F1={best:.4f}) to {BERT_DIR}")


def quantize(model_dir=BERT_DIR, out_path=BERT_INT8_PATH):
    """Dynamic INT8 quantization of a fine-tuned model; saves the quantized state dict."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(qmodel.state_dict(), out_path)
    print(f"  -> saved INT8 model to {out_path}")
    return qmodel


def load_fp32(model_dir=BERT_DIR):
    return BertModel(AutoModelForSequenceClassification.from_pretrained(model_dir), AutoTokenizer.from_pretrained(model_dir))


def load_int8(model_dir=BERT_DIR, int8_path=BERT_INT8_PATH):
    # rebuild the architecture from the config (no FP32 weights needed), quantize it, then load INT8 weights
    config = AutoConfig.from_pretrained(model_dir)
    skeleton = AutoModelForSequenceClassification.from_config(config).eval()
    qmodel = torch.ao.quantization.quantize_dynamic(skeleton, {nn.Linear}, dtype=torch.qint8)
    qmodel.load_state_dict(torch.load(int8_path, map_location="cpu", weights_only=False))
    return BertModel(qmodel, AutoTokenizer.from_pretrained(model_dir))
