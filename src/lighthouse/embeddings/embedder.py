"""Deterministic local embedders.

From *Retrieval-Augmented Generation (RAG) in Production*, Chapter 7.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from lighthouse.retrieval.keyword import tokenize


class LsaEmbedder:
    """Deterministic dense embedder. Not the manuscript's model."""
    name = "tfidf-svd-256"
    dim = 256

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.pipe = None

    def fit_transform(self, texts):
        vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False,
                              token_pattern=None, min_df=1)
        matrix = vec.fit_transform(texts)
        n_feat = matrix.shape[1]
        self.dim = min(256, max(2, n_feat - 1))
        self.pipe = make_pipeline(
            vec, TruncatedSVD(n_components=self.dim,
                              random_state=self.seed),
            Normalizer(copy=False))
        return self.pipe.fit_transform(texts)

    def encode(self, texts):
        return self.pipe.transform(texts)


class SpacyEmbedder:
    """Trained static word vectors, mean-pooled. Fixed vocabulary,
    so out-of-vocabulary tokens (identifiers) contribute nothing.
    NOT a transformer bi-encoder; see the Phase 2.6 report."""
    name = "en_core_web_md-3.8.0"
    dim = 300

    def __init__(self):
        import spacy
        self._nlp = spacy.load("en_core_web_md",
                               exclude=["parser", "ner", "tagger",
                                        "lemmatizer", "attribute_ruler"])

    def _vec(self, text):
        doc = self._nlp.make_doc(text[:20000])
        vecs = [t.vector for t in doc if t.has_vector and not t.is_space]
        if not vecs:
            return np.zeros(self.dim, dtype="float32")
        v = np.mean(vecs, axis=0)
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v

    def fit_transform(self, texts):
        return np.vstack([self._vec(t) for t in texts]).astype("float32")

    def encode(self, texts):
        return np.vstack([self._vec(t) for t in texts]).astype("float32")

    def oov_fraction(self, text) -> float:
        doc = self._nlp.make_doc(text)
        toks = [t for t in doc if not t.is_space and not t.is_punct]
        if not toks:
            return 1.0
        return sum(1 for t in toks if not t.has_vector) / len(toks)
