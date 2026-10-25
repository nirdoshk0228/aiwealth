"""Okapi BM25 over chunk texts (in memory; a few thousand chunks).

    score(q, d) = sum_{t in q} idf(t) * f(t,d) (k1 + 1) / (f(t,d) + k1 (1 - b + b |d| / avgdl))
    idf(t)      = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))        (always > 0)

f(t,d) = count of term t in d, |d| = tokens in d, N = documents, n(t) = documents containing t.
k1 = 1.5, b = 0.75 (the usual defaults). Tokens: lower-cased alphanumeric runs; numbers are kept
because users ask about "exit load 1%" or "Section 10". No stemming or stop words: the corpus
is small and specialised, and exact terms (scheme names, "riskometer") matter most.

Written out instead of a dependency so the formula is visible and tested by hand.
"""

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(d)) for d in docs]
        self._len = [sum(c.values()) for c in self._tf]
        self.n = len(docs)
        self.avgdl = (sum(self._len) / self.n) if self.n else 0.0
        df: Counter[str] = Counter()
        for c in self._tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.n - k + 0.5) / (k + 0.5)) for t, k in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, dl in zip(self._tf, self._len, strict=True):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def top(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """(index, score) of the k best documents with a positive score."""
        ranked = [
            (i, s)
            for i, s in enumerate(self.scores(query))
            if s > 0 and (allowed is None or i in allowed)
        ]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked[:k]
