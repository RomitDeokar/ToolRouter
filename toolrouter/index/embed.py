"""Embedding model wrapper with a deterministic offline fallback.

Primary path is `fastembed <https://github.com/qdrant/fastembed>`_ (ONNX, no
torch, small download). If fastembed or its weights cannot be loaded -- no
network, no local cache, air-gapped CI -- we fall back to a deterministic
hash-based vectoriser so the *rest of the pipeline stays testable*.

The fallback is loudly announced. Fallback-quality embeddings would quietly
wreck benchmark numbers if someone forgot they were active, so:

* a ``logging.warning`` is emitted on construction,
* :attr:`EmbeddingModel.is_fallback` is a public flag,
* :attr:`EmbeddingModel.backend` names the active backend, and
* the benchmark writes the backend name into every result file.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections.abc import Iterable, Sequence

import numpy as np

__all__ = ["EmbeddingModel", "HashEmbeddingBackend", "DEFAULT_MODEL_NAME"]

logger = logging.getLogger(__name__)

#: Small, fast, high-quality-for-its-size English embedding model.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

#: Dimensionality used by the hash fallback. Chosen to be large enough that
#: random token collisions don't dominate, small enough to stay fast.
FALLBACK_DIM = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation, plus ``snake_case``/``camelCase`` splits."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return _TOKEN_RE.findall(spaced.lower())


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows are left as zeros (no NaNs)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


# --------------------------------------------------------------------------- #
# Fallback backend
# --------------------------------------------------------------------------- #
class HashEmbeddingBackend:
    """Deterministic hashed bag-of-features vectoriser.

    Not a semantic model. It embeds *lexical* overlap: each token (and each
    character 3-gram of each token, so typos degrade gracefully instead of
    falling off a cliff) is hashed into a bucket with a signed, sublinear
    weight. Cosine similarity over these vectors behaves like a crude
    normalised term-overlap score.

    This exists purely so the pipeline runs offline. Any benchmark run using
    it must be reported as such.
    """

    def __init__(self, dim: int = FALLBACK_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"Embedding dim must be positive, got {dim}.")
        self.dim = int(dim)

    # -- internals --------------------------------------------------------- #
    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    @staticmethod
    def _char_ngrams(token: str, n: int = 3) -> Iterable[str]:
        padded = f"#{token}#"
        if len(padded) <= n:
            yield padded
            return
        for i in range(len(padded) - n + 1):
            yield padded[i : i + n]

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text or "")
        if not tokens:
            return vector
        for token in tokens:
            index, sign = self._bucket(f"w:{token}")
            vector[index] += sign * 1.0
            for gram in self._char_ngrams(token):
                g_index, g_sign = self._bucket(f"g:{gram}")
                vector[g_index] += g_sign * 0.35
        # Sublinear damping so long descriptions don't dominate short queries.
        vector = np.sign(vector) * np.log1p(np.abs(vector))
        return vector

    # -- public API -------------------------------------------------------- #
    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)
        return _l2_normalize(matrix)


# --------------------------------------------------------------------------- #
# Public wrapper
# --------------------------------------------------------------------------- #
class EmbeddingModel:
    """Text embedding model with a guaranteed-available offline fallback.

    Parameters
    ----------
    model_name:
        fastembed model identifier. Defaults to :data:`DEFAULT_MODEL_NAME`, or
        ``$TOOLROUTER_EMBED_MODEL`` when set.
    allow_fallback:
        When ``False``, a failure to load the real model raises instead of
        silently degrading. Use this in CI that must measure real quality.
    force_fallback:
        Skip the real model entirely. Left at ``None`` (the default) the value
        is read from ``TOOLROUTER_FORCE_FALLBACK``, which is how the test suite
        stays fast and hermetic. Passing ``True``/``False`` explicitly
        **overrides** the environment -- an explicit ``force_fallback=False`` is
        how a ``@pytest.mark.semantic`` test opts back into a real model inside
        a run that globally forced the fallback.

    Attributes
    ----------
    dim:
        Vector dimensionality.
    backend:
        ``"fastembed:<model>"``, ``"sentence-transformers:<model>"``, or
        ``"hash-fallback"``.
    is_fallback:
        ``True`` when embeddings are *not* semantic.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        allow_fallback: bool = True,
        force_fallback: bool | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "TOOLROUTER_EMBED_MODEL", DEFAULT_MODEL_NAME
        )
        self._cache_dir = cache_dir or os.environ.get("TOOLROUTER_EMBED_CACHE")
        self._lock = threading.Lock()
        self._cache: dict[str, np.ndarray] = {}

        # An explicit argument beats the environment. Previously this was
        # `force_fallback or env_force`, which meant a caller passing
        # force_fallback=False could never escape a global
        # TOOLROUTER_FORCE_FALLBACK=1 -- so the semantic tests silently skipped
        # even with a real model installed, and nothing ever measured real
        # retrieval quality.
        if force_fallback is None:
            env_force = os.environ.get("TOOLROUTER_FORCE_FALLBACK", "").strip().lower()
            force_fallback = env_force in {"1", "true", "yes"}

        self._impl: object | None = None
        self._kind = "hash"
        self.backend = "hash-fallback"
        self.is_fallback = True
        self.load_error: str | None = None

        if force_fallback:
            self.load_error = "fallback forced by caller/environment"
        else:
            self._try_load_real_model()

        if self.is_fallback:
            if not allow_fallback:
                raise RuntimeError(
                    "Could not load a real embedding model and allow_fallback=False. "
                    f"Reason: {self.load_error}"
                )
            self._impl = HashEmbeddingBackend(FALLBACK_DIM)
            self.dim = FALLBACK_DIM
            logger.warning(
                "EmbeddingModel is using the DETERMINISTIC HASH FALLBACK "
                "(dim=%d), not a semantic embedding model. Reason: %s. "
                "Retrieval quality will be lexical, not semantic -- do not "
                "report benchmark numbers from this mode without labelling it.",
                FALLBACK_DIM,
                self.load_error,
            )
        else:
            logger.info("EmbeddingModel backend=%s dim=%d", self.backend, self.dim)

    # -- loading ----------------------------------------------------------- #
    def _try_load_real_model(self) -> None:
        errors: list[str] = []

        # Primary: fastembed (ONNX runtime, no torch).
        try:
            from fastembed import TextEmbedding  # type: ignore

            kwargs = {"model_name": self.model_name}
            if self._cache_dir:
                kwargs["cache_dir"] = self._cache_dir
            model = TextEmbedding(**kwargs)
            probe = np.asarray(list(model.embed(["dimension probe"]))[0], dtype=np.float32)
            self._impl = model
            self._kind = "fastembed"
            self.dim = int(probe.shape[0])
            self.backend = f"fastembed:{self.model_name}"
            self.is_fallback = False
            return
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            errors.append(f"fastembed: {type(exc).__name__}: {exc}")

        # Secondary: sentence-transformers, if the user happens to have it.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(self.model_name)
            self._impl = model
            self._kind = "sentence-transformers"
            self.dim = int(model.get_sentence_embedding_dimension())
            self.backend = f"sentence-transformers:{self.model_name}"
            self.is_fallback = False
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sentence-transformers: {type(exc).__name__}: {exc}")

        self.load_error = " | ".join(errors)

    # -- embedding --------------------------------------------------------- #
    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many texts. Returns an L2-normalised ``(len(texts), dim)`` array."""
        if texts is None:
            raise ValueError("embed_batch() requires a sequence of strings, got None.")
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(
                    f"embed_batch() expects strings, got {type(text).__name__}."
                )

        if self._kind == "fastembed":
            vectors = list(self._impl.embed(texts))  # type: ignore[union-attr]
            matrix = np.asarray(vectors, dtype=np.float32)
        elif self._kind == "sentence-transformers":
            matrix = np.asarray(
                self._impl.encode(texts, convert_to_numpy=True),  # type: ignore[union-attr]
                dtype=np.float32,
            )
        else:
            matrix = self._impl.embed_batch(texts)  # type: ignore[union-attr]

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return _l2_normalize(matrix)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text. Returns a 1-D ``(dim,)`` L2-normalised vector.

        Results are memoised: the benchmark embeds the same query repeatedly
        across baselines, and caching keeps latency measurements about
        *retrieval*, not about re-running the encoder.
        """
        if not isinstance(text, str):
            raise TypeError(f"embed_text() expects a string, got {type(text).__name__}.")
        with self._lock:
            cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = self.embed_batch([text])[0]
        with self._lock:
            self._cache[text] = vector
        return vector

    # -- dunder ------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"EmbeddingModel(backend={self.backend!r}, dim={self.dim}, "
            f"is_fallback={self.is_fallback})"
        )
