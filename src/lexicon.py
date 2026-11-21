"""Edit-distance lexicon matcher: the rule baseline (technique area B).

The offensive-word lexicon is learned from the *training* split only: every word that
appears >= MIN_COUNT times is scored with a smoothed log-odds ratio (offensive vs clean
tweets), and the top-K words become the lexicon. K is chosen on the validation split.

A message is flagged when one of its tokens is within minimum edit distance 1 of a lexicon
word (distance 0 for words shorter than 5 letters). '*' in a token is a wildcard, so
masked spellings like "f**k" or "sh*t" also match.
"""
import json
import math
from collections import Counter
from functools import lru_cache

from sklearn.metrics import f1_score

from .config import LEXICON_PATH
from .normalize import normalize, tokenize

MIN_COUNT = 5


def edit_distance(a: str, b: str, wildcard: str = "*", max_dist: int | None = None) -> int:
    """Levenshtein distance (insert/delete/substitute cost 1) with dynamic programming.
    A wildcard character in `a` substitutes for any character at cost 0.
    If max_dist is given, stops early once every cell in a row exceeds it."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            sub = 0 if (ca == cb or ca == wildcard) else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def _collapse_doubles(word: str) -> str:
    out = []
    for c in word:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def score_vocabulary(texts, labels):
    """Smoothed log-odds of each word appearing in an offensive vs a clean tweet."""
    off, clean = Counter(), Counter()
    for text, y in zip(texts, labels):
        words = set(tokenize(normalize(text)))
        (off if y == 1 else clean).update(words)
    n_off, n_clean = sum(labels), len(labels) - sum(labels)
    scores = {}
    for w in set(off) | set(clean):
        if off[w] + clean[w] < MIN_COUNT or "*" in w or len(w) < 3:
            continue
        p_off = (off[w] + 1) / (n_off + 2)
        p_clean = (clean[w] + 1) / (n_clean + 2)
        scores[w] = math.log(p_off / p_clean)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class LexiconMatcher:
    def __init__(self, words):
        self.words = sorted(set(words))
        self.by_len = {}
        for w in self.words:
            self.by_len.setdefault(len(w), []).append(w)
        self._match = lru_cache(maxsize=100_000)(self._match_uncached)

    def _match_uncached(self, token: str):
        for cand in (token, _collapse_doubles(token)):
            if cand in self.by_len.get(len(cand), ()):
                return cand
            if len(cand) < 5 and "*" not in cand:
                continue
            for L in (len(cand) - 1, len(cand), len(cand) + 1):
                for w in self.by_len.get(L, ()):
                    if len(w) >= 4 and edit_distance(cand, w, max_dist=1) <= 1:
                        return w
        return None

    def find(self, text: str):
        """Returns [(token, matched_lexicon_word), ...] for a raw message."""
        hits = []
        for tok in tokenize(normalize(text)):
            m = self._match(tok)
            if m:
                hits.append((tok, m))
        return hits

    def predict_proba(self, text: str) -> float:
        """Rule model: no hits -> 0.0, one hit -> 0.75, two or more -> 0.95."""
        n = len(self.find(text))
        return 0.0 if n == 0 else (0.75 if n == 1 else 0.95)

    def save(self, path=LEXICON_PATH):
        path.write_text(json.dumps(self.words, indent=0))

    @classmethod
    def load(cls, path=LEXICON_PATH):
        return cls(json.loads(path.read_text()))


def train(splits):
    """Builds the lexicon on train and picks its size K on validation (macro-F1)."""
    ranked = score_vocabulary(*splits["train"])
    val_x, val_y = splits["validation"]
    best = (-1, None)
    for k in (25, 50, 75, 100, 150, 200, 300, 400):
        matcher = LexiconMatcher([w for w, _ in ranked[:k]])
        pred = [int(matcher.predict_proba(t) >= 0.5) for t in val_x]
        f1 = f1_score(val_y, pred, average="macro")
        print(f"  lexicon K={k:4d}  val macro-F1={f1:.4f}")
        if f1 > best[0]:
            best = (f1, k)
    k = best[1]
    matcher = LexiconMatcher([w for w, _ in ranked[:k]])
    matcher.save()
    print(f"  -> kept K={k} words, top 15: {[w for w, _ in ranked[:15]]}")
    return matcher
