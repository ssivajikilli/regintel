"""Hybrid retrieval: BM25 + dense (LSA) -> Reciprocal Rank Fusion -> rerank.
Offline by default. In production swap dense -> Azure OpenAI embeddings / Qdrant,
rerank -> ColBERT or a cross-encoder (same interface)."""
import math, re, pathlib
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

STOP = set("the a an of to and or in on for is are be by with as at that this it from shall must what how does do "
           "which who when should can i we our my about there their any if".split())
def tokens(t): return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]

def load_chunks(folder):
    out = []
    for p in sorted(pathlib.Path(folder).glob("*.md")):
        for i, para in enumerate(re.split(r"\n\s*\n", p.read_text(encoding="utf-8"))):
            if len(para.strip()) > 40:
                out.append({"id": f"{p.stem}#{i}", "doc": p.stem.replace("_", " ").title(), "text": para.strip()})
    return out

class HybridIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.docs = [tokens(c["text"]) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.df = Counter(w for d in self.docs for w in set(d))
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        self.vec = TfidfVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None, sublinear_tf=True)
        X = self.vec.fit_transform([c["text"] for c in chunks])
        self.svd = TruncatedSVD(max(2, min(64, X.shape[0] - 1, X.shape[1] - 1)), random_state=0)
        Z = self.svd.fit_transform(X)
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def bm25(self, q, k1=1.5, b=0.75):
        n, out = len(self.docs), np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            for w in tokens(q):
                if w in tf:
                    idf = math.log(1 + (n - self.df[w] + .5) / (self.df[w] + .5))
                    out[i] += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / self.avg))
        return out

    def dense(self, q):
        v = self.svd.transform(self.vec.transform([q]))
        return (self.Z @ (v / (np.linalg.norm(v) + 1e-9)).T).ravel()

    def search(self, q, k=4, pool=10):
        b, d = self.bm25(q), self.dense(q)
        rank = lambda s: {i: r for r, i in enumerate(np.argsort(-s))}
        rb, rd = rank(b), rank(d)
        fused = {i: 1 / (60 + rb[i]) + 1 / (60 + rd[i]) for i in range(len(self.chunks))}   # RRF
        qs, res, n = set(tokens(q)), [], len(self.docs)
        idf = lambda w: math.log(1 + (n - self.df[w] + .5) / (self.df[w] + .5))
        total = sum(idf(w) for w in qs) or 1.0
        for i in sorted(fused, key=fused.get, reverse=True)[:pool]:
            cov = sum(idf(w) for w in qs & set(self.docs[i])) / total                        # idf-weighted rerank signal
            res.append({**self.chunks[i], "bm25": round(float(b[i]), 2), "dense": round(float(d[i]), 2),
                        "score": round(0.6 * cov + 0.4 * max(float(d[i]), 0), 3)})
        return sorted(res, key=lambda x: -x["score"])[:k]
