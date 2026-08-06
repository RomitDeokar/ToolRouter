# CI configuration

`github-actions-ci.yml` is the project's GitHub Actions workflow. It lives here
rather than at `.github/workflows/ci.yml` because the automation account that
opened the initial pull request does not hold GitHub's `workflows` permission,
and GitHub rejects any push that creates or edits a workflow file without it.

To activate CI, move it into place from a normal (human) checkout:

```bash
mkdir -p .github/workflows
git mv ci/github-actions-ci.yml .github/workflows/ci.yml
git commit -m "ci: enable GitHub Actions workflow"
git push
```

## What the workflow checks

Two jobs, deliberately different, because each catches a class of failure the
other cannot see.

**`test` — core install, no optional extras** (Python 3.10–3.13)

Installs only `numpy` plus dev tooling, then runs `ruff` and the full suite.
Because none of the optional backends are present, this job exercises every
in-tree fallback: the hash embedder, brute-force numpy cosine search, the
in-tree BM25-Okapi, and the ~4-chars-per-token heuristic. If a fallback path
regresses, this job fails instead of a user discovering it after
`pip install toolrouter`.

**`test-with-extras` — real embeddings**

Installs `.[bench,dev]` (fastembed, faiss, rank_bm25, tiktoken). With a real
model loadable, the six tests that skip on a core install become real
assertions — including the four `@pytest.mark.semantic` tests that measure
retrieval *quality* rather than plumbing.

This job then re-runs CommerceBench and compares the result against the
accuracy figures published in the README, failing if they drift by more than a
small tolerance. That guard exists because a benchmark number in a README is
otherwise unfalsifiable: this makes the published claim reproducible from the
committed dataset rather than something that was true once on someone's laptop.

The tolerance absorbs float and backend jitter, not real regressions —
accuracy is a deterministic function of the committed dataset and the pinned
model, so it should match closely. Latency is deliberately *not* asserted: it
is machine-dependent, and the README says so where the numbers are reported.
