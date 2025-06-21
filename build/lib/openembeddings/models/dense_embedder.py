from __future__ import annotations

from hashlib import blake2b
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

"""Lightweight dense embedder.

This implementation **does not** rely on heavyweight transformer models or
external downloads. Instead, it uses deterministic hashing to project text
into a 384-dimensional unit sphere. While this is *not* a production-grade
semantic encoder, it is perfectly adequate for unit/integration tests where
only shape, determinism, and normalisation are essential.

Author: Nik Jois <nikjois@llamasearch.ai>
"""


class DenseEmbedder(nn.Module):
    """Deterministic 384-d hashing embedder with optional linear projection."""

    def __init__(
        self,
        model_name: str | None = None,
        pooling_strategy: str = "mean",
        normalize: bool = True,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Parameters are retained for API-compatibility; most are unused in the
        # hashing backend.
        self.model_name = model_name or "hashing-encoder"
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.dimension: int = 384  # Public attribute expected by tests

        # Dummy linear projection to showcase extensibility (noop by default)
        self.projection = nn.Identity()

    # ------------------------------------------------------------------
    # Core functionality
    # ------------------------------------------------------------------
    @staticmethod
    def _text_to_vector(text: str, dim: int) -> np.ndarray:
        """Project *text* to a deterministic vector on the unit sphere."""
        # Use BLAKE2 hash to obtain 256-bit digest, then expand deterministically.
        h = blake2b(text.encode("utf-8"), digest_size=32).digest()
        seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        # L2 normalise
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    # ------------------------------------------------------------------
    def forward(
        self, texts: List[str], return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode *texts* into a torch tensor of shape (N, 384)."""
        # Hash-based embeddings
        embeddings_np = np.stack([self._text_to_vector(t, self.dimension) for t in texts])
        embeddings = torch.from_numpy(embeddings_np).to(self.device)

        # (Optional) projection — identity by default
        embeddings = self.projection(embeddings)

        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        attention_weights = None  # Attention not applicable in hashing backend
        if return_attention:
            attention_weights = torch.zeros((len(texts), 1), device=self.device)
        return embeddings, attention_weights

    # ------------------------------------------------------------------
    def encode(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        """Convenience wrapper that returns numpy array embeddings."""
        all_embeddings = [self._text_to_vector(t, self.dimension) for t in texts]
        return np.vstack(all_embeddings)

    # ------------------------------------------------------------------
    # Compatibility helpers (save/load)
    # ------------------------------------------------------------------
    def save_pretrained(self, save_path: str) -> None:  # noqa: D401
        """Serialise nothing — hashing backend is stateless."""
        import pathlib, json, os

        p = pathlib.Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "backend": "hash",
        }
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):  # noqa: D401
        """Recreate the hashing backend (no actual weights to load)."""
        return cls(**kwargs) 