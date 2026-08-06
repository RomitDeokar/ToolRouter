"""CommerceBench: dataset generation, baselines, metrics, and evaluation."""

from .baselines import (
    BASELINES,
    BaselineOutcome,
    run_all_tools,
    run_baseline,
    run_confidence_gate,
    run_dense,
    run_hybrid,
)
from .evaluate import evaluate, evaluate_all, render_summary
from .generate_dataset import BenchQuery, generate_dataset, load_dataset, write_dataset

__all__ = [
    "BASELINES",
    "BaselineOutcome",
    "BenchQuery",
    "evaluate",
    "evaluate_all",
    "generate_dataset",
    "load_dataset",
    "render_summary",
    "run_all_tools",
    "run_baseline",
    "run_confidence_gate",
    "run_dense",
    "run_hybrid",
    "write_dataset",
]
