"""Own word2vec + feedforward network (technique area A, with word2vec from area B).

1. Skip-gram word2vec (gensim) is trained from scratch on the *training* messages only.
2. A message is represented by the average of its word vectors (unknown words skipped).
3. A feedforward network (100 -> 128 -> ReLU -> dropout -> 2) is trained with plain SGD
   (+momentum) and backpropagation on cross-entropy loss; the best epoch on the
   validation split is kept (early stopping).
"""
import numpy as np
import torch
from gensim.models import KeyedVectors, Word2Vec
from sklearn.metrics import f1_score
from torch import nn

from .config import FFN_PATH, SEED, W2V_PATH
from .normalize import normalize, tokenize

DIM = 100


class FFN(nn.Module):
    def __init__(self, dim=DIM, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def embed(kv: KeyedVectors, text: str) -> np.ndarray:
    vecs = [kv[w] for w in tokenize(normalize(text)) if w in kv.key_to_index]
    return np.mean(vecs, axis=0) if vecs else np.zeros(kv.vector_size, dtype=np.float32)


class W2VFFNModel:
    def __init__(self, kv, ffn):
        self.kv, self.ffn = kv, ffn.eval()

    @torch.no_grad()
    def predict_proba(self, text: str) -> float:
        x = torch.from_numpy(embed(self.kv, text)).unsqueeze(0)
        return float(torch.softmax(self.ffn(x), dim=-1)[0, 1])


def train(splits, epochs=60):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_x, train_y = splits["train"]
    sentences = [tokenize(normalize(t)) for t in train_x]
    w2v = Word2Vec(sentences, vector_size=DIM, window=5, min_count=2, sg=1, negative=10, epochs=20, workers=4, seed=SEED)
    kv = w2v.wv
    kv.save(str(W2V_PATH))
    print(f"  word2vec: {len(kv)} words x {DIM} dims")
    print(f"  most similar to 'idiot': {[w for w, _ in kv.most_similar('idiot', topn=6)]}")

    def to_tensor(texts):
        return torch.tensor(np.stack([embed(kv, t) for t in texts]), dtype=torch.float32)

    Xtr, ytr = to_tensor(train_x), torch.tensor(train_y)
    Xva, yva = to_tensor(splits["validation"][0]), splits["validation"][1]

    ffn = FFN()
    # class weights compensate for the 2:1 clean/offensive imbalance
    counts = torch.bincount(ytr).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
    opt = torch.optim.SGD(ffn.parameters(), lr=0.05, momentum=0.9)
    best_f1, best_state = -1, None
    for epoch in range(1, epochs + 1):
        ffn.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            loss = loss_fn(ffn(Xtr[idx]), ytr[idx])
            loss.backward()  # backpropagation
            opt.step()  # SGD update
        ffn.eval()
        with torch.no_grad():
            pred = ffn(Xva).argmax(-1).numpy()
        f1 = f1_score(yva, pred, average="macro")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in ffn.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  FFN epoch {epoch:3d}  loss={loss.item():.4f}  val macro-F1={f1:.4f}")
    ffn.load_state_dict(best_state)
    torch.save(ffn.state_dict(), FFN_PATH)
    print(f"  -> best val macro-F1={best_f1:.4f}")
    return W2VFFNModel(kv, ffn)


def load():
    kv = KeyedVectors.load(str(W2V_PATH))
    ffn = FFN()
    ffn.load_state_dict(torch.load(FFN_PATH, map_location="cpu"))
    return W2VFFNModel(kv, ffn)
