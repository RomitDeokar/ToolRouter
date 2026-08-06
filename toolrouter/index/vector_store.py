"""Vector store: FAISS when available, brute-force numpy otherwise.

Both code paths return identical output shapes, and both are exercised by the
test suite. faiss is deliberately *not* a hard dependency -- for a tool index
of a few dozen to a few thousand entries, brute-force numpy is exact and fast
enough, and a hard faiss dependency would be an unreasonable install tax on a
library this size.

Vectors are assumed L2-normalised (``EmbeddingModel`` guarantees this), so the
inner product equals cosine similarity. We normalise defensively anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

__all__ = ["VectorStore", "FAISS_AVAILABLE"]

logger = logging.getLogger(__name__)

try:  # pragma: no cover - availability is environment-dependent
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:  # noqa: BLE001  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


class VectorStore:
    """Cosine-similarity vector index keyed by string IDs.

    Examples
    --------
    >>> import numpy as np
    >>> store = VectorStore(dim=3)
    >>> store.add(["a", "b"], np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float32))
    >>> store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)[0][0]
    'a'
    """

    def __init__(self, dim: int, *, use_faiss: bool | None = None) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"VectorStore dim must be a positive int, got {dim!r}.")
        self.dim = dim
        self._ids: list[str] = []
        self._id_set: set[str] = set()
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)

        if use_faiss is None:
            use_faiss = FAISS_AVAILABLE
        self.use_faiss = bool(use_faiss and FAISS_AVAILABLE)
        if use_faiss and not FAISS_AVAILABLE:
            logger.warning("faiss requested but not importable; using numpy backend.")

        self._index = faiss.IndexFlatIP(dim) if self.use_faiss else None
        self.backend = "faiss:IndexFlatIP" if self.use_faiss else "numpy:brute-force"
        logger.debug("VectorStore backend=%s dim=%d", self.backend, dim)

    # -- mutation ---------------------------------------------------------- #
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Add vectors under the given IDs.

        Raises
        ------
        ValueError
            On length mismatch, wrong dimensionality, non-finite values, or a
            duplicate ID (which would make search results ambiguous).
        """
        ids = list(ids)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be a 2-D array, got shape {vectors.shape}.")
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Got {len(ids)} ids but {vectors.shape[0]} vectors -- they must align."
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dim {vectors.shape[1]} does not match store dim {self.dim}."
            )
        if not ids:
            return
        if not np.all(np.isfinite(vectors)):
            raise ValueError("vectors contain NaN or inf values.")

        duplicates = [i for i in ids if i in self._id_set]
        if duplicates:
            raise ValueError(f"Duplicate ids already present in store: {duplicates}.")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate ids within the same add() call.")

        vectors = _l2_normalize(vectors)
        self._ids.extend(str(i) for i in ids)
        self._id_set.update(str(i) for i in ids)
        self._matrix = (
            vectors.copy()
            if self._matrix.shape[0] == 0
            else np.vstack([self._matrix, vectors])
        )
        if self._index is not None:
            self._index.add(vectors)

    def reset(self) -> None:
        """Drop all vectors."""
        self._ids.clear()
        self._id_set.clear()
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        if self._index is not None:
            self._index.reset()

    # -- query -------------------------------------------------------------- #
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the ``k`` nearest IDs as ``[(id, score), ...]``, score descending.

        Scores are **raw cosine similarities clamped to** ``[0, 1]``. An earlier
        version rescaled via ``(1 + cos) / 2`` to guarantee the 0-1 range that
        downstream components expect; that was measurably wrong. Text embeddings
        essentially never produce negative cosines here, so the rescale only
        compressed every score into ``[0.5, 1.0]`` -- which halved every score
        gap and pushed genuinely out-of-domain queries above the confidence
        gate's absolute floor (a "what's the weather in Tokyo" query scored
        0.77). Clamping instead preserves the true scale, where in-domain
        queries measure ~0.59-0.83 and out-of-domain ones ~0.45-0.51.
        """
        if k is None or k <= 0:
            return []
        if len(self._ids) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(
                f"Query dim {query.shape[0]} does not match store dim {self.dim}."
            )
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm

        k = min(int(k), len(self._ids))

        if self._index is not None:
            scores, indices = self._index.search(query.reshape(1, -1), k)
            pairs = [
                (self._ids[int(idx)], float(score))
                # strict=True: FAISS always returns parallel index/score rows;
                # a mismatch would pair a tool with another tool's score.
                for idx, score in zip(indices[0], scores[0], strict=True)
                if int(idx) >= 0
            ]
        else:
            sims = self._matrix @ query
            # argpartition for O(n) top-k, then sort just the slice.
            if k < sims.shape[0]:
                top = np.argpartition(-sims, k - 1)[:k]
            else:
                top = np.arange(sims.shape[0])
            top = top[np.argsort(-sims[top], kind="stable")]
            pairs = [(self._ids[int(i)], float(sims[int(i)])) for i in top]

        # Clamp into [0, 1] (negative cosines are treated as "no similarity")
        # and stabilise the ordering for reproducible benchmarks.
        scaled = [(tool_id, max(0.0, min(1.0, cos))) for tool_id, cos in pairs]
        scaled.sort(key=lambda pair: (-pair[1], pair[0]))
        return scaled

    # -- introspection ------------------------------------------------------ #
    @property
    def ids(self) -> list[str]:
        """IDs in insertion order."""
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, item: object) -> bool:
        return isinstance(item, str) and item in self._id_set

    def __repr__(self) -> str:
        return f"VectorStore(backend={self.backend!r}, dim={self.dim}, size={len(self)})"
