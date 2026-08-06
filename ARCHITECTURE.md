# ARCHITECTURE.md

This document is the source of truth for how `toolrouter` is built. If
you're using an AI coding assistant, paste the relevant section (see
`PROMPTS.md`) along with this file so it implements against the same
contracts instead of inventing its own.

## Data flow

```
MCP manifest (JSON)
      │
      ▼
[parser.manifest_parser]  ──►  list[Tool]
      │
      ▼
[parser.tool_registry]    ──►  ToolRegistry (queryable store)
      │
      ▼
[index.embed]              ──►  embeddings per tool
      │
      ▼
[index.vector_store]       ──►  FAISS/np index, add() + search()
      │                         (optional) [index.bm25] for hybrid
      ▼
[router.retrieve]          ──►  ranked ScoredTool candidates
      │
      ▼
[router.confidence_gate]   ──►  adaptive-k filtered candidates
      │
      ▼
[router.explain]           ──►  human-readable reasons per candidate
      │
      ▼
[router.prompt_builder]    ──►  minimal tool-schema string for the LLM
      │
      ▼
Agent runtime (OpenAI Agents SDK / LangGraph / your choice)
      │
      ▼
Real MCP tool call
```

## Core types

Define these once in `parser/manifest_parser.py` and `router/retrieve.py`
respectively. Every other module imports them — don't redefine them
elsewhere.

```python
# parser/manifest_parser.py
from dataclasses import dataclass, field

@dataclass
class Tool:
    name: str                  # e.g. "search_restaurants"
    description: str
    parameters: dict            # raw JSON schema for params
    server: str                  # e.g. "food", "instamart", "dineout"
    examples: list[str] = field(default_factory=list)  # optional sample calls

    def to_embedding_text(self) -> str:
        """
        Text used to generate the tool's embedding. Per the design spec,
        this must include name + description + parameter names (not
        description alone) - parameter names carry disambiguating signal
        that pure descriptions miss (e.g. distinguishing search_restaurants
        from search_products when descriptions are similarly worded).
        """
        param_names = ", ".join(self.parameters.get("properties", {}).keys())
        parts = [self.name, self.description, param_names]
        if self.examples:
            parts.append(" | ".join(self.examples))
        return " - ".join(p for p in parts if p)
```

```python
# router/retrieve.py
from dataclasses import dataclass

@dataclass
class ScoredTool:
    tool: Tool
    score: float          # similarity score, 0-1 range
    source: str            # "dense" | "bm25" | "hybrid"

@dataclass
class RouteResult:
    query: str
    tools: list[Tool]              # final selected tools, post-gate
    scored: list[ScoredTool]         # full scored list, pre/post gate
    explanation: list[dict]          # one dict per tool, see explain.py
```

## Module contracts

### `parser/manifest_parser.py`

```python
def parse_manifest(path_or_url: str) -> list[Tool]:
    """
    Accepts a local file path or a URL to an MCP manifest JSON.
    Must be generic - do not hardcode Swiggy-specific field names.
    Expected input shape (adapt defensively; real MCP manifests vary):
        {
          "server": "food",
          "tools": [
            {"name": ..., "description": ..., "parameters": {...}}
          ]
        }
    Raise a clear error if required fields are missing rather than
    silently producing a malformed Tool.
    """
```

### `parser/tool_registry.py`

```python
class ToolRegistry:
    def __init__(self, tools: list[Tool]): ...
    def by_server(self, server: str) -> list[Tool]: ...
    def by_name(self, name: str) -> Tool | None: ...
    @property
    def tools(self) -> list[Tool]: ...
```

### `index/embed.py`

```python
class EmbeddingModel:
    """
    Wraps a real embedding model (fastembed / sentence-transformers).
    Must expose `.dim` (int) and support batch embedding.
    Include an offline-safe fallback (e.g. a deterministic hash-based
    vector) so the rest of the pipeline is testable without network
    access or an API key - clearly log a warning when the fallback is
    active so nobody mistakes it for a real embedding model.
    """
    dim: int
    def embed_text(self, text: str) -> "np.ndarray": ...
    def embed_batch(self, texts: list[str]) -> "np.ndarray": ...
```

### `index/vector_store.py`

```python
class VectorStore:
    """
    FAISS-backed if faiss is installed, else a brute-force numpy
    cosine-similarity fallback. The fallback must exist - don't make
    faiss a hard dependency for a project this size.
    """
    def __init__(self, dim: int): ...
    def add(self, ids: list[str], vectors: "np.ndarray") -> None: ...
    def search(self, query_vector: "np.ndarray", k: int) -> list[tuple[str, float]]:
        """Returns [(tool_name, similarity_score), ...] sorted desc."""
```

### `index/bm25.py`

```python
class BM25Index:
    """Thin wrapper around rank_bm25. Optional - only used when hybrid=True."""
    def build(self, ids: list[str], texts: list[str]) -> None: ...
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...
```

### `router/retrieve.py`

```python
class Retriever:
    def __init__(self, registry, embedder, vector_store, bm25=None): ...
    def retrieve(self, query: str, k: int = 5, hybrid: bool = False) -> list[ScoredTool]:
        """
        Dense-only by default. If hybrid=True and bm25 is configured,
        combine dense + BM25 scores (normalize both to 0-1 before
        combining - do not average raw BM25 and cosine scores directly,
        their scales are not comparable).
        """
```

### `router/confidence_gate.py`

```python
def confidence_gate(
    candidates: list[ScoredTool],
    min_k: int = 1,
    max_k: int = 5,
    gap_threshold: float = 0.15,
) -> list[ScoredTool]:
    """
    Core differentiator - implement this carefully, it's the thing
    that turns 'retrieval' into 'routing'.

    Logic:
      gap = candidates[0].score - candidates[1].score
      if gap >= gap_threshold:
          return candidates[:min_k]      # confident -> narrow
      else:
          return candidates[:max_k]      # ambiguous -> widen

    Edge cases to handle explicitly:
      - fewer than 2 candidates (can't compute a gap)
      - all scores below some absolute floor (no good match at all -
        this should probably return an empty list or a "no confident
        match" sentinel, not force a top-1 guess)
    """
```

### `router/explain.py`

```python
def explain_candidates(query: str, scored: list[ScoredTool], registry) -> list[dict]:
    """
    Returns one dict per candidate:
        {"tool": name, "score": 0.96, "reason": "high semantic similarity
         to query terms 'restaurant' and 'book'"}
    Keep the reason generation simple and honest for v1 - a template
    that cites the score and matched terms is fine. Don't fabricate
    reasons the retrieval process didn't actually use (e.g. don't claim
    "previously successful" if there's no historical-success tracking
    implemented).
    """
```

### `router/prompt_builder.py`

```python
def build_tool_prompt(tools: list[Tool]) -> str:
    """
    Produces the minimal tool-schema text block to inject into the
    agent's system prompt - only the routed subset, not all tools.
    """
```

## Non-goals for v1 (do not implement these - see README Limitations)

- Policy engine / OPA integration
- Adaptive learning from execution feedback
- Next.js/web dashboard, Postgres, Redis, Docker deployment
- Multi-provider (GPT/Claude/Gemini) comparative benchmarking
- ToolGraph workflow planning

If an AI coding assistant suggests adding any of these, decline and
point it back to this file.
