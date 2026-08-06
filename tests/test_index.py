"""Tests for embeddings, the vector store, and the BM25 index."""

from __future__ import annotations

import numpy as np
import pytest

from toolrouter.index.bm25 import BM25Index, _PureBM25, tokenize
from toolrouter.index.embed import EmbeddingModel, HashEmbeddingBackend
from toolrouter.index.vector_store import FAISS_AVAILABLE, VectorStore


# --------------------------------------------------------------------------- #
# EmbeddingModel
# --------------------------------------------------------------------------- #
def test_fallback_embedder_reports_itself_clearly():
    """Fallback mode must never be silent -- it would wreck benchmark numbers."""
    model = EmbeddingModel(force_fallback=True)
    assert model.is_fallback is True
    assert model.backend == "hash-fallback"
    assert model.dim > 0


def test_fallback_warning_is_logged(caplog):
    with caplog.at_level("WARNING"):
        EmbeddingModel(force_fallback=True)
    assert any("FALLBACK" in record.message.upper() for record in caplog.records)


def test_allow_fallback_false_raises_instead_of_degrading():
    with pytest.raises(RuntimeError, match="allow_fallback=False"):
        EmbeddingModel(force_fallback=True, allow_fallback=False)


def test_env_forces_fallback_when_argument_is_unset(monkeypatch):
    """Default (None) defers to the environment -- how the suite stays hermetic."""
    monkeypatch.setenv("TOOLROUTER_FORCE_FALLBACK", "1")
    assert EmbeddingModel().is_fallback is True


def test_explicit_force_fallback_false_overrides_the_environment(monkeypatch):
    """Regression: an explicit opt-out must beat TOOLROUTER_FORCE_FALLBACK=1.

    The precedence used to be ``force_fallback or env``, so ``force_fallback=False``
    was indistinguishable from the default and the environment always won. That
    made every ``@pytest.mark.semantic`` test skip unconditionally under the
    suite-wide force flag -- real retrieval quality was never actually measured,
    and CI's "real embeddings" job silently asserted nothing.
    """
    monkeypatch.setenv("TOOLROUTER_FORCE_FALLBACK", "1")
    try:
        model = EmbeddingModel(force_fallback=False, allow_fallback=False)
    except RuntimeError as exc:
        # No real model installed here; the point is that the attempt was made
        # rather than short-circuited by the environment.
        assert "fallback forced by caller/environment" not in str(exc)
        pytest.skip("No real embedding model available in this environment.")
    assert model.is_fallback is False


def test_embed_text_shape_and_normalisation():
    model = EmbeddingModel(force_fallback=True)
    vector = model.embed_text("book a table for four tonight")
    assert vector.shape == (model.dim,)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_embed_batch_shape():
    model = EmbeddingModel(force_fallback=True)
    matrix = model.embed_batch(["first text", "second text", "third text"])
    assert matrix.shape == (3, model.dim)


def test_embed_batch_empty_returns_empty_matrix():
    model = EmbeddingModel(force_fallback=True)
    assert model.embed_batch([]).shape == (0, model.dim)


def test_embeddings_are_deterministic():
    """Same input, same vector -- required for reproducible benchmarks."""
    a = EmbeddingModel(force_fallback=True)
    b = EmbeddingModel(force_fallback=True)
    np.testing.assert_allclose(a.embed_text("order paneer"), b.embed_text("order paneer"))


def test_embed_text_is_cached():
    model = EmbeddingModel(force_fallback=True)
    first = model.embed_text("same query")
    second = model.embed_text("same query")
    assert first is second, "repeated queries should hit the memo cache"


def test_embed_rejects_non_string():
    model = EmbeddingModel(force_fallback=True)
    with pytest.raises(TypeError):
        model.embed_text(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        model.embed_batch(["ok", 5])  # type: ignore[list-item]


def test_hash_backend_similar_text_scores_higher_than_unrelated():
    """The fallback is lexical, but must still rank overlap above non-overlap."""
    backend = HashEmbeddingBackend(256)
    matrix = backend.embed_batch(
        ["book restaurant table dine in", "book a table at a restaurant", "scale kubernetes pods"]
    )
    assert float(matrix[0] @ matrix[1]) > float(matrix[0] @ matrix[2])


def test_hash_backend_handles_empty_text():
    backend = HashEmbeddingBackend(64)
    assert backend.embed_batch([""]).shape == (1, 64)


def test_hash_backend_rejects_bad_dim():
    with pytest.raises(ValueError):
        HashEmbeddingBackend(0)


def test_typo_degrades_gracefully_via_char_ngrams():
    """Character n-grams mean a typo shouldn't fall off a cliff."""
    backend = HashEmbeddingBackend(512)
    matrix = backend.embed_batch(["restaurant reservation", "restaurnat reservtion", "milk bread"])
    assert float(matrix[0] @ matrix[1]) > float(matrix[0] @ matrix[2])


@pytest.mark.semantic
def test_real_model_ranks_semantic_similarity(semantic_embedder):
    """Paraphrases should beat lexical overlap once real embeddings are in play."""
    vectors = semantic_embedder.embed_batch(
        ["reserve a table for dinner", "book a restaurant seat for tonight", "scale a kubernetes deployment"]
    )
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


# --------------------------------------------------------------------------- #
# VectorStore
# --------------------------------------------------------------------------- #
def _basis(dim: int, index: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_search_returns_exact_match_first():
    """The contract test from PROMPTS.md prompt 4: 5 vectors, query one of them."""
    store = VectorStore(dim=8)
    ids = [f"tool_{i}" for i in range(5)]
    vectors = np.vstack([_basis(8, i) for i in range(5)])
    store.add(ids, vectors)

    results = store.search(_basis(8, 2), k=5)
    assert results[0][0] == "tool_2"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_search_results_sorted_descending():
    store = VectorStore(dim=4)
    store.add(["a", "b", "c"], np.vstack([_basis(4, 0), _basis(4, 1), _basis(4, 2)]))
    scores = [score for _, score in store.search(_basis(4, 0), k=3)]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k():
    store = VectorStore(dim=4)
    store.add(["a", "b", "c"], np.vstack([_basis(4, 0), _basis(4, 1), _basis(4, 2)]))
    assert len(store.search(_basis(4, 0), k=2)) == 2


def test_k_larger_than_corpus_is_clamped():
    store = VectorStore(dim=4)
    store.add(["a"], _basis(4, 0).reshape(1, -1))
    assert len(store.search(_basis(4, 0), k=50)) == 1


def test_scores_are_within_unit_range():
    """Downstream code (the gate's floor, hybrid fusion) assumes [0, 1]."""
    store = VectorStore(dim=4)
    store.add(["a", "b"], np.vstack([_basis(4, 0), -_basis(4, 0)]))
    for _, score in store.search(_basis(4, 0), k=2):
        assert 0.0 <= score <= 1.0


def test_raw_cosine_is_preserved_not_rescaled():
    """Orthogonal vectors must score ~0, not 0.5.

    Regression test: an earlier version mapped cosine via (1+cos)/2, which
    compressed every score into [0.5, 1.0], halved all gaps, and pushed
    out-of-domain queries above the confidence gate's absolute floor.
    """
    store = VectorStore(dim=4)
    store.add(["a", "b"], np.vstack([_basis(4, 0), _basis(4, 1)]))
    scores = dict(store.search(_basis(4, 0), k=2))
    assert scores["a"] == pytest.approx(1.0, abs=1e-5)
    assert scores["b"] == pytest.approx(0.0, abs=1e-5)


def test_search_on_empty_store_returns_empty():
    assert VectorStore(dim=4).search(_basis(4, 0), k=5) == []


def test_non_positive_k_returns_empty():
    store = VectorStore(dim=4)
    store.add(["a"], _basis(4, 0).reshape(1, -1))
    assert store.search(_basis(4, 0), k=0) == []


def test_duplicate_ids_rejected():
    store = VectorStore(dim=4)
    store.add(["a"], _basis(4, 0).reshape(1, -1))
    with pytest.raises(ValueError, match="Duplicate ids"):
        store.add(["a"], _basis(4, 1).reshape(1, -1))


def test_duplicate_ids_within_one_call_rejected():
    store = VectorStore(dim=4)
    with pytest.raises(ValueError, match="Duplicate ids"):
        store.add(["x", "x"], np.vstack([_basis(4, 0), _basis(4, 1)]))


def test_id_vector_length_mismatch_rejected():
    store = VectorStore(dim=4)
    with pytest.raises(ValueError, match="must align"):
        store.add(["a", "b"], _basis(4, 0).reshape(1, -1))


def test_wrong_dimension_rejected():
    store = VectorStore(dim=4)
    with pytest.raises(ValueError, match="does not match store dim"):
        store.add(["a"], np.zeros((1, 8), dtype=np.float32))


def test_query_dimension_mismatch_rejected():
    store = VectorStore(dim=4)
    store.add(["a"], _basis(4, 0).reshape(1, -1))
    with pytest.raises(ValueError, match="does not match store dim"):
        store.search(np.zeros(8, dtype=np.float32), k=1)


def test_nan_vectors_rejected():
    store = VectorStore(dim=4)
    bad = np.full((1, 4), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or inf"):
        store.add(["a"], bad)


def test_invalid_dim_rejected():
    with pytest.raises(ValueError, match="positive int"):
        VectorStore(dim=0)


def test_reset_clears_the_index():
    store = VectorStore(dim=4)
    store.add(["a"], _basis(4, 0).reshape(1, -1))
    store.reset()
    assert len(store) == 0
    assert store.search(_basis(4, 0), k=1) == []


def test_membership_and_len():
    store = VectorStore(dim=4)
    store.add(["a", "b"], np.vstack([_basis(4, 0), _basis(4, 1)]))
    assert len(store) == 2
    assert "a" in store and "zzz" not in store


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
def test_faiss_and_numpy_backends_agree():
    """Both code paths must produce the same output -- the fallback is not a
    second-class citizen."""
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(12, 16)).astype(np.float32)
    ids = [f"t{i}" for i in range(12)]
    query = rng.normal(size=16).astype(np.float32)

    faiss_store = VectorStore(dim=16, use_faiss=True)
    numpy_store = VectorStore(dim=16, use_faiss=False)
    faiss_store.add(ids, vectors)
    numpy_store.add(ids, vectors)

    faiss_hits = faiss_store.search(query, k=5)
    numpy_hits = numpy_store.search(query, k=5)
    assert [i for i, _ in faiss_hits] == [i for i, _ in numpy_hits]
    for (_, a), (_, b) in zip(faiss_hits, numpy_hits, strict=True):
        assert a == pytest.approx(b, abs=1e-5)


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
def test_tokenize_splits_identifiers():
    assert tokenize("get_menu_items") == ["get", "menu", "items"]
    assert tokenize("getMenuItems") == ["get", "menu", "items"]


def test_bm25_ranks_lexical_overlap_first():
    index = BM25Index()
    index.build(
        ["book", "buy"],
        ["book a restaurant table for dining", "buy milk bread and grocery items"],
    )
    assert index.search("restaurant table", k=1)[0][0] == "book"


def test_bm25_exact_tool_name_match():
    index = BM25Index()
    index.build(["place_order", "track_order"], ["place order checkout", "track order delivery"])
    assert index.search("place_order", k=1)[0][0] == "place_order"


def test_bm25_empty_index_returns_empty():
    assert BM25Index().search("anything", k=5) == []


def test_bm25_length_mismatch_rejected():
    with pytest.raises(ValueError, match="must align"):
        BM25Index().build(["a", "b"], ["only one text"])


def test_bm25_empty_query_returns_empty():
    index = BM25Index()
    index.build(["a"], ["some text"])
    assert index.search("!!!", k=5) == []


def test_bm25_handles_empty_documents_without_crashing():
    """An empty document must not break BM25 length normalisation.

    Note the two backends legitimately differ here: ``rank_bm25`` uses unsmoothed
    IDF, so a term appearing in half of a 2-document corpus gets ``log(1) == 0``
    and *both* documents score 0.0. The built-in implementation uses smoothed,
    zero-floored IDF and ranks the matching document first. Neither crashes,
    which is what this test guards; see
    ``test_pure_bm25_fallback_agrees_with_rank_bm25_ordering`` for the agreement
    check on realistic corpora, where the two rank identically.
    """
    for use_rank_bm25 in (True, False):
        index = BM25Index(use_rank_bm25=use_rank_bm25)
        index.build(["a", "b"], ["", "real content here"])
        results = index.search("content", k=2)
        assert len(results) == 2
        assert {i for i, _ in results} == {"a", "b"}


def test_builtin_bm25_idf_smoothing_beats_unsmoothed_on_tiny_corpora():
    """Why the built-in implementation smooths IDF.

    Tool corpora are inherently tiny (tens of tools, not millions of documents),
    so terms routinely appear in a large fraction of documents. Unsmoothed IDF
    zeroes those terms out entirely.
    """
    builtin = BM25Index(use_rank_bm25=False)
    builtin.build(["a", "b"], ["", "real content here"])
    top_id, top_score = builtin.search("content", k=2)[0]
    assert top_id == "b"
    assert top_score > 0.0, "smoothed IDF must keep a real match non-zero"


def test_bm25_results_sorted_descending():
    index = BM25Index()
    index.build(["a", "b", "c"], ["order food now", "order", "unrelated text"])
    scores = [s for _, s in index.search("order food", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_pure_bm25_fallback_agrees_with_rank_bm25_ordering():
    """The built-in implementation must rank like the library one."""
    ids = ["a", "b", "c", "d"]
    texts = [
        "book a restaurant table for dinner tonight",
        "search restaurants by cuisine and rating",
        "buy grocery milk bread eggs",
        "track my delivery order status",
    ]
    library = BM25Index(use_rank_bm25=True)
    builtin = BM25Index(use_rank_bm25=False)
    library.build(ids, texts)
    builtin.build(ids, texts)
    for query in ["restaurant table", "grocery milk", "track delivery"]:
        assert [i for i, _ in library.search(query, k=4)][0] == [
            i for i, _ in builtin.search(query, k=4)
        ][0], f"top hit disagreed for {query!r}"


def test_pure_bm25_idf_is_never_negative():
    """A term in most documents must not get a negative weight.

    Common with tiny tool corpora where e.g. "search" appears in most
    descriptions; negative IDF would actively penalise matching documents.
    """
    corpus = [["search", "food"], ["search", "grocery"], ["search", "table"]]
    model = _PureBM25(corpus)
    assert all(value >= 0.0 for value in model.idf.values())
