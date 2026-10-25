"""Text embeddings behind a small interface.

- FastEmbedEmbedder: BAAI/bge-small-en-v1.5 via fastembed (ONNX Runtime, no torch), 384
  dimensions, model files cached in a gitignored folder. BGE models expect a query instruction
  prefix for short queries; fastembed's `query_embed` applies the model's convention.
- HashingEmbedder: deterministic bag-of-words hashing, for tests (no model download, no network).
Vectors are L2-normalised, so cosine similarity = dot product.
"""

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class HashingEmbedder:
    """Each token is hashed to a dimension (with a sign); texts sharing words get similar
    vectors. Not semantic, but deterministic and enough to test the pipeline."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array([self._one(t) for t in texts], dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._one(text)[None, :].astype(np.float32))[0]


class FastEmbedEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding

        self.name = model
        self._model = TextEmbedding(
            model_name=model, cache_dir=str(cache_dir) if cache_dir else None
        )
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.passage_embed(texts, batch_size=64)), dtype=np.float32)
        return _normalise(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)
        return _normalise(vec[None, :])[0]


@lru_cache
def get_fastembed(model: str, cache_dir: str | None) -> FastEmbedEmbedder:
    """One loaded model per process (loading takes a few seconds and ~100 MB of RAM)."""
    return FastEmbedEmbedder(model, Path(cache_dir) if cache_dir else None)
