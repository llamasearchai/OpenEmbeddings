"""Hybrid retriever combining dense hashing embeddings with BM25 scores.

This implementation is purposely lightweight to guarantee rapid execution
in constrained CI environments while still providing meaningful behaviour
for unit and integration tests.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .dense_embedder import DenseEmbedder
from .sparse_embedder import SparseEmbedder


class HybridRetriever(nn.Module):
    """Combine dense & sparse signals via linear or learned fusion."""

    def __init__(
        self,
        dense_model: str | None = None,
        sparse_model: str = "bm25",
        fusion_strategy: str = "linear",  # "linear" | "learned"
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Components
        self.dense_embedder = DenseEmbedder(model_name=dense_model, device=self.device)
        self.sparse_embedder = SparseEmbedder(model_type=sparse_model)

        # Fusion configuration
        self.fusion_strategy = fusion_strategy.lower()
        if self.fusion_strategy not in {"linear", "learned"}:
            raise ValueError("fusion_strategy must be 'linear' or 'learned'")
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.dimension = self.dense_embedder.dimension

        # Optional learned fusion network
        if self.fusion_strategy == "learned":
            feature_dim = self.dimension * 2 + 2  # q_emb, d_emb, dense_score, sparse_score
            self.fusion_network = nn.Sequential(
                nn.Linear(feature_dim, 384),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(384, 192),
                nn.ReLU(),
                nn.Linear(192, 1),
                nn.Sigmoid(),
            ).to(self.device)

        # Lightweight cross-attention for potential re-ranking (unused in tests)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.dimension, num_heads=8, dropout=0.1, batch_first=True
        ).to(self.device)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _dense_similarity(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Cosine similarity assuming q & d are L2-normalised."""
        return torch.matmul(q, d.T)

    # ------------------------------------------------------------------
    def forward(
        self,
        queries: Sequence[str],
        documents: Sequence[str],
        return_scores: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute hybrid similarity scores for *queries* x *documents*."""
        # Dense encodings (already normalised)
        query_dense, _ = self.dense_embedder(list(queries))
        doc_dense, _ = self.dense_embedder(list(documents))

        dense_scores = self._dense_similarity(query_dense, doc_dense)  # (Q, D)

        # Sparse scores
        sparse_scores_list = self.sparse_embedder.compute_scores(list(queries), list(documents))
        sparse_scores = torch.tensor(sparse_scores_list, device=self.device, dtype=dense_scores.dtype)

        # Fusion
        if self.fusion_strategy == "linear":
            final_scores = self.dense_weight * dense_scores + self.sparse_weight * sparse_scores
        else:  # learned
            Q, D = dense_scores.shape
            features = []
            for qi in range(Q):
                for di in range(D):
                    f = torch.cat(
                        [
                            query_dense[qi],
                            doc_dense[di],
                            dense_scores[qi, di].unsqueeze(0),
                            sparse_scores[qi, di].unsqueeze(0),
                        ]
                    )
                    features.append(f)
            features_tensor = torch.stack(features)  # (Q*D, feature_dim)
            fusion_weights = self.fusion_network(features_tensor).view(Q, D)
            final_scores = fusion_weights * dense_scores + (1 - fusion_weights) * sparse_scores

        # Optional cross-attention re-ranking (lightweight)
        # Note: For unit tests we skip for efficiency.

        return {
            "scores": final_scores,
            "dense_scores": dense_scores,
            "sparse_scores": sparse_scores,
        }

    # ------------------------------------------------------------------
    def retrieve(
        self, query: str, corpus: List[str], top_k: int = 10, return_scores: bool = False
    ) -> List[Tuple[int, float, str]]:
        """Return top-*k* documents for *query*."""
        self.eval()
        with torch.no_grad():
            output = self.forward([query], corpus)
            scores = output["scores"][0]  # (D,)
            top_k = min(top_k, scores.shape[0])
            top_indices = torch.topk(scores, top_k).indices
            results: List[Tuple[int, float, str]] = []
            for idx in top_indices:
                i = int(idx)
                results.append((i, float(scores[i].cpu().item()), corpus[i]))
        return results

    # ------------------------------------------------------------------
    # Compatibility helpers (save/load)
    # ------------------------------------------------------------------
    def save_pretrained(self, path: str) -> None:  # noqa: D401
        import json, pathlib

        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "fusion_strategy": self.fusion_strategy,
        }
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        # Delegate dense embedder save (stateless for hashing backend)
        self.dense_embedder.save_pretrained(str(p / "dense"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):  # noqa: D401
        import json, pathlib

        p = pathlib.Path(path)
        with open(p / "config.json", "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        obj = cls(
            fusion_strategy=saved_cfg["fusion_strategy"],
            dense_weight=saved_cfg["dense_weight"],
            sparse_weight=saved_cfg["sparse_weight"],
            **kwargs,
        )
        # Dense embedder is stateless; nothing additional to load.
        return obj 