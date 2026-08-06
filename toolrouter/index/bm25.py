"""Lexical BM25 index -- optional, used only when ``hybrid=True``.

Wraps ``rank_bm25.BM25Okapi`` when installed, and falls back to a compact
in-house BM25-Okapi implementation otherwise. The fallback exists for the same
reason as the numpy vector store: hybrid retrieval is a *baseline in the
benchmark*, so it must be runnable without optional wheels being present.

BM25 complements dense retrieval on exact-token matches -- a query naming a
tool almost verbatim ("place_order") scores highly lexically even when the
embedding model is lukewarm about it.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence

__all__ = ["BM25Index", "RANK_BM25_AVAILABLE", "tokenize"]

logger = logging.getLogger(__name__)

try:  # pragma: no cover - availability is environment-dependent
    from rank_bm25 import BM25Okapi  # type: ignore

    RANK_BM25_AVAILABLE = True
except Exception:  # noqa: BLE001  # pragma: no cover
    BM25Okapi = None  # type: ignore[assignment]
    RANK_BM25_AVAILABLE = False


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokeniser that also splits identifiers.

    ``get_menu_items``/``getMenuItems`` -> ``["get", "menu", "items"]``, so a
    query like "show me the menu" matches the tool name lexically and not only
    through its description.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text or "")
    return _TOKEN_RE.findall(spaced.lower())


class _PureBM25:
    """Minimal BM25-Okapi. Used when ``rank_bm25`` is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / self.corpus_size) if self.corpus_size else 0.0
        self.doc_freqs: list[Counter] = [Counter(doc) for doc in corpus]

        df: Counter = Counter()
        for doc in corpus:
            df.update(set(doc))
        # Robertson/Sparck-Jones IDF, floored at zero to avoid negative weights
        # for terms appearing in more than half the corpus (common with tiny
        # tool corpora, where "search" might be in most descriptions).
        self.idf: dict[str, float] = {
            term: max(
                0.0,
                math.log(
                    1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
                ),
            )
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for index in range(self.corpus_size):
            freqs = self.doc_freqs[index]
            doc_len = self.doc_lens[index]
            total = 0.0
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl else 0.0)
                )
                total += idf * (tf * (self.k1 + 1.0)) / (denom or 1.0)
            scores[index] = total
        return scores


class BM25Index:
    """BM25 lexical index over tool texts, keyed by tool name.

    Examples
    --------
    >>> index = BM25Index()
    >>> index.build(["a", "b"], ["book a restaurant table", "buy milk and bread"])
    >>> index.search("restaurant table", k=1)[0][0]
    'a'
    """

    def __init__(self, *, use_rank_bm25: bool | None = None) -> None:
        self._ids: list[str] = []
        self._impl: object | None = None
        if use_rank_bm25 is None:
            use_rank_bm25 = RANK_BM25_AVAILABLE
        self._use_rank_bm25 = bool(use_rank_bm25 and RANK_BM25_AVAILABLE)
        self.backend = "rank_bm25:BM25Okapi" if self._use_rank_bm25 else "builtin:BM25Okapi"

    # -- build -------------------------------------------------------------- #
    def build(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        """Fit the index over ``texts``, addressable by the aligned ``ids``."""
        ids = list(ids)
        texts = list(texts)
        if len(ids) != len(texts):
            raise ValueError(
                f"Got {len(ids)} ids but {len(texts)} texts -- they must align."
            )
        self._ids = [str(i) for i in ids]
        if not texts:
            self._impl = None
            return
        corpus = [tokenize(text) for text in texts]
        # A document with zero tokens breaks BM25 length normalisation.
        corpus = [doc if doc else ["__empty__"] for doc in corpus]
        self._impl = BM25Okapi(corpus) if self._use_rank_bm25 else _PureBM25(corpus)
        logger.debug("BM25Index built: backend=%s docs=%d", self.backend, len(corpus))

    # -- query -------------------------------------------------------------- #
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return ``[(id, raw_bm25_score), ...]`` sorted descending.

        Scores are *raw* BM25 and are not comparable to cosine similarity;
        :class:`~toolrouter.router.retrieve.Retriever` min-max normalises them
        before any fusion.
        """
        if self._impl is None or not self._ids or k is None or k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = list(self._impl.get_scores(tokens))  # type: ignore[union-attr]
        # strict=True: the backend must return exactly one score per indexed
        # document. A length mismatch would silently misalign every score with
        # the wrong tool, so fail loudly instead of truncating.
        pairs = list(zip(self._ids, (float(s) for s in scores), strict=True))
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs[: min(int(k), len(pairs))]

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return f"BM25Index(backend={self.backend!r}, docs={len(self)})"
