from typing import List, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
import re

"""Sparse (lexical) embedding model leveraging BM25 scoring.

This implementation is intentionally lightweight, has **no external
network dependencies**, and works entirely offline. It is sufficient for
unit/integration testing while remaining faithful to traditional BM25
behaviour.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

class SparseEmbedder:
    """Simple BM25‐based sparse retriever."""

    def __init__(self, model_type: str = "bm25"):
        if model_type.lower() != "bm25":
            raise ValueError("Currently only the 'bm25' model_type is supported")
        self.model_type = model_type.lower()
        self._bm25: BM25Okapi | None = None
        self._docs: List[str] = []

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fit(self, documents: Sequence[str]) -> None:
        """Tokenise and index *documents* for subsequent scoring."""
        self._docs = list(documents)
        tokenised_corpus = [self._tokenise(doc) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenised_corpus)

    def compute_scores(self, queries: Sequence[str], documents: Sequence[str] | None = None) -> List[List[float]]:
        """Return BM25 relevance scores.

        Parameters
        ----------
        queries:
            A sequence of raw query strings.
        documents:
            If provided, *documents* will be **re‐indexed** before scoring
            (this is useful for stateless calls). When *None*, the method
            expects that :py:meth:`fit` was previously called.
        """
        # (Re-)index if necessary
        if documents is not None:
            if documents != self._docs:
                self.fit(documents)
        elif self._bm25 is None:
            raise ValueError("BM25 model is not yet initialised — call 'fit' first or provide 'documents'.")

        assert self._bm25 is not None  # mypy static guard

        scores: List[List[float]] = []
        for q in queries:
            tokenised_q = self._tokenise(q)
            scores.append(self._bm25.get_scores(tokenised_q).tolist())
        return scores

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Tokenise with basic normalisation + naive stemming."""
        tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
        processed = []
        for tok in tokens:
            if tok.endswith("es"):
                tok = tok[:-2]
            elif tok.endswith("s") and len(tok) > 3:
                tok = tok[:-1]
            processed.append(tok)
        return processed 