"""
geodesic_thrml.types — Shared data classes and utilities
=========================================================

Pure data structures and scheduling utilities used across the
geodesic controller, curriculum cascade, and bridges.

No inference logic here — only types and deterministic math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Curriculum types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CascadeResult:
    """Result of a coarse-to-fine cascade."""
    strength: float
    confidence: float
    posterior: np.ndarray          # final K-bin posterior
    level_results: list            # per-level (k, posterior, s, c)
    error_per_level: list[float]   # |s_cascade - s_direct| per level


@dataclass
class ResolutionLevel:
    """One level in a resolution ladder."""
    k: int
    label: str = ""


def build_resolution_ladder(
    k_levels: Sequence[int] = (4, 8, 16),
    labels: Sequence[str] | None = None,
) -> list[ResolutionLevel]:
    """Build a resolution ladder for coarse-to-fine cascade.

    Args:
        k_levels: bin counts in ascending order (e.g., [4, 8, 16])
        labels: optional human-readable labels per level

    Returns:
        List of ResolutionLevel objects.
    """
    if labels is None:
        labels = [f"K={k}" for k in k_levels]
    if len(labels) != len(k_levels):
        raise ValueError(f"labels length {len(labels)} != k_levels length {len(k_levels)}")
    for i in range(1, len(k_levels)):
        if k_levels[i] <= k_levels[i - 1]:
            raise ValueError(f"k_levels must be strictly ascending: {k_levels}")
    return [ResolutionLevel(k=k, label=lab) for k, lab in zip(k_levels, labels)]


# ═══════════════════════════════════════════════════════════════════════════
#  Controller types
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_TEMPERATURE = 1.0
DEFAULT_COST_WEIGHT = 0.1


@dataclass
class SelectionResult:
    """Result of geodesic step selection.

    Attributes:
        selected_idx: index of the selected rule
        selected_name: name of the selected rule
        rule_probs: Boltzmann probabilities for all rules
        rho: reinforcement ρ = f·g for all rules
        energy: geodesic energy for all rules
        f_scores: forward factors
        g_scores: backward factors
    """
    selected_idx: int
    selected_name: str
    rule_probs: np.ndarray
    rho: np.ndarray
    energy: np.ndarray
    f_scores: np.ndarray
    g_scores: np.ndarray


@dataclass
class BatchResult:
    """Result of batch parallel selection.

    Attributes:
        primary: SelectionResult for the best step
        parallel_group: indices of steps that can run in parallel with primary
        parallel_names: names of parallel steps
    """
    primary: SelectionResult
    parallel_group: list[int]
    parallel_names: list[str]


# ═══════════════════════════════════════════════════════════════════════════
#  Annealing schedule
# ═══════════════════════════════════════════════════════════════════════════

def annealing_schedule(
    t_start: float,
    t_end: float,
    n_steps: int,
    strategy: str = "exponential",
) -> list[float]:
    """Generate a temperature annealing schedule.

    Strategies:
        - "exponential": T(t) = T_start * (T_end/T_start)^(t/(n-1))
        - "linear": T(t) = T_start + (T_end - T_start) * t / (n-1)
        - "cosine": T(t) = T_end + 0.5*(T_start-T_end)*(1+cos(πt/(n-1)))

    Args:
        t_start: initial temperature (high = exploratory)
        t_end: final temperature (low = greedy)
        n_steps: number of steps in the schedule
        strategy: annealing strategy

    Returns:
        List of temperatures, length n_steps.
    """
    if n_steps <= 0:
        return []
    if n_steps == 1:
        return [t_start]

    if strategy == "exponential":
        ratio = max(t_end / max(t_start, 1e-30), 1e-30)
        return [t_start * (ratio ** (i / (n_steps - 1))) for i in range(n_steps)]

    elif strategy == "linear":
        return [t_start + (t_end - t_start) * i / (n_steps - 1)
                for i in range(n_steps)]

    elif strategy == "cosine":
        return [t_end + 0.5 * (t_start - t_end) *
                (1 + math.cos(math.pi * i / (n_steps - 1)))
                for i in range(n_steps)]

    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Use 'exponential', 'linear', or 'cosine'.")
