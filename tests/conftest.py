"""Shared pytest fixtures.

The test suite runs against the **hash fallback** embedder by default
(``TOOLROUTER_FORCE_FALLBACK=1``, set in :func:`pytest_configure`). That keeps
the suite hermetic and fast -- no model download, no network. Tests that assert
on *semantic* behaviour rather than plumbing are marked ``@pytest.mark.semantic``
and skipped unless a real model is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from toolrouter.parser.manifest_parser import Tool

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_MANIFEST = REPO_ROOT / "examples" / "swiggy_manifest.json"


def pytest_configure(config: pytest.Config) -> None:
    """Force the offline embedder and register custom markers."""
    os.environ.setdefault("TOOLROUTER_FORCE_FALLBACK", "1")
    config.addinivalue_line(
        "markers",
        "semantic: needs a real embedding model (skipped when unavailable).",
    )


@pytest.fixture
def sample_tools() -> list[Tool]:
    """Three tools with deliberately overlapping surface area."""
    return [
        Tool(
            name="search_restaurants",
            description="Search restaurants available for food delivery by cuisine.",
            parameters={
                "type": "object",
                "properties": {
                    "cuisine": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
            server="food",
            examples=["find italian food near me"],
        ),
        Tool(
            name="search_products",
            description="Search the grocery catalogue for packaged products.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "brand": {"type": "string"},
                },
                "required": ["query"],
            },
            server="instamart",
        ),
        Tool(
            name="book_table",
            description="Reserve a table at a restaurant for dine-in.",
            parameters={
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string"},
                    "party_size": {"type": "integer"},
                },
                "required": ["restaurant_id", "party_size"],
            },
            server="dineout",
        ),
    ]


@pytest.fixture
def minimal_manifest_dict() -> dict:
    """The canonical single-server manifest shape from ``ARCHITECTURE.md``."""
    return {
        "server": "food",
        "tools": [
            {
                "name": "search_restaurants",
                "description": "Search restaurants by cuisine.",
                "parameters": {
                    "type": "object",
                    "properties": {"cuisine": {"type": "string"}},
                },
            },
            {
                "name": "place_order",
                "description": "Place a food delivery order.",
                "parameters": {
                    "type": "object",
                    "properties": {"cart_id": {"type": "string"}},
                },
            },
        ],
    }


@pytest.fixture
def write_manifest(tmp_path: Path):
    """Write a manifest dict to a temp file and return its path."""

    def _write(document: object, name: str = "manifest.json") -> str:
        target = tmp_path / name
        target.write_text(json.dumps(document), encoding="utf-8")
        return str(target)

    return _write


@pytest.fixture(scope="session")
def mock_manifest_path() -> str:
    """Path to the repo's mock multi-server manifest."""
    if not MOCK_MANIFEST.is_file():
        pytest.skip(f"Mock manifest missing at {MOCK_MANIFEST}")
    return str(MOCK_MANIFEST)


@pytest.fixture
def router(mock_manifest_path: str):
    """A router over the mock manifest, hybrid enabled, offline embedder."""
    from toolrouter import ToolRouter

    return ToolRouter.from_manifest(mock_manifest_path, use_hybrid=True)


@pytest.fixture(scope="session")
def semantic_embedder():
    """A real embedding model, or skip the test if one cannot be loaded."""
    from toolrouter.index.embed import EmbeddingModel

    try:
        model = EmbeddingModel(force_fallback=False, allow_fallback=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"No real embedding model available: {exc}")
    return model
