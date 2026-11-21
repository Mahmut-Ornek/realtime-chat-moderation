"""Classical baseline: tf-idf (word 1-2 grams + char 2-5 grams) + logistic regression (area B)."""
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .config import SEED, TFIDF_PATH
from .normalize import normalize


class TfidfModel:
    def __init__(self, C=1.0):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"[a-z*']+")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100_000)
        self.clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEED)

    def _features(self, texts, fit=False):
        norm = [normalize(t) for t in texts]
        if fit:
            return hstack([self.word.fit_transform(norm), self.char.fit_transform(norm)]).tocsr()
        return hstack([self.word.transform(norm), self.char.transform(norm)]).tocsr()

    def fit(self, texts, labels):
        self.clf.fit(self._features(texts, fit=True), labels)
        return self

    def predict_proba(self, text: str) -> float:
        return float(self.clf.predict_proba(self._features([text]))[0, 1])

    def predict_proba_batch(self, texts):
        return self.clf.predict_proba(self._features(texts))[:, 1]


def train(splits):
    """Picks the regularization strength C on validation, then saves the model."""
    best = (-1, None)
    for C in (0.5, 1.0, 2.0, 4.0, 8.0):
        model = TfidfModel(C).fit(*splits["train"])
        val_x, val_y = splits["validation"]
        f1 = f1_score(val_y, (model.predict_proba_batch(val_x) >= 0.5).astype(int), average="macro")
        print(f"  tf-idf C={C}  val macro-F1={f1:.4f}")
        if f1 > best[0]:
            best = (f1, model)
    joblib.dump(best[1], TFIDF_PATH)
    return best[1]


def load():
    return joblib.load(TFIDF_PATH)
