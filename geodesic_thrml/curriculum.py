"""
geodesic_thrml.curriculum — Schrödinger Bridge Learning (coarse-to-fine)
========================================================================

Implements the "bridging" approach from Hyperon whitepaper: find models
balancing accuracy with simplicity by following geodesics from very
simple to very accurate models.

For PLN-THRML, this means cascading K=4 → K=8 → K=16: each level's
posterior becomes the next level's prior, producing a smooth interpolation
path through resolution space.

Theory: each level is a simple EBM (short mixing time, unimodal), and the
cascade builds distributional complexity without ever constructing a single
hard-to-sample monolithic model — the same principle as DTM.

References:
    - Hyperon whitepaper §5.2: "bridging" approach
    - genenergy-logic §3.3: Schrödinger bridge geometry
    - tsu-architecture §II: mixing-expressivity tradeoff
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class CascadeResult:
    """Result of a coarse-to-fine cascade."""
    strength: float
    confidence: float
    posterior: np.ndarray          # final K-bin posterior
    level_results: list            # per-level (k, posterior, s, c)
    error_per_level: list[float]   # |s_cascade - s_direct| per level (if direct available)


@dataclass
class ResolutionLevel:
    """One level in a resolution ladder."""
    k: int
    label: str = ""


def rebin_posterior(posterior: np.ndarray, source_k: int,
                    target_k: int) -> np.ndarray:
    """Rebin a K-bin posterior histogram to a different resolution.

    Uses piecewise-linear interpolation over bin centers to transfer
    probability mass between resolutions.

    Args:
        posterior: normalized histogram of shape [source_k]
        source_k: number of bins in the source
        target_k: number of bins in the target

    Returns:
        Normalized histogram of shape [target_k]
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    posterior = posterior / (posterior.sum() + 1e-30)

    # Bin centers for source and target
    src_centers = np.linspace(0.5 / source_k, 1.0 - 0.5 / source_k, source_k)
    tgt_centers = np.linspace(0.5 / target_k, 1.0 - 0.5 / target_k, target_k)

    # Interpolate log-probabilities for smoothness
    log_src = np.log(np.clip(posterior, 1e-30, None))
    log_tgt = np.interp(tgt_centers, src_centers, log_src)

    # Back to probabilities and normalize
    result = np.exp(log_tgt)
    result = result / result.sum()
    return result


def posterior_to_prior_weights(posterior: np.ndarray, k: int) -> np.ndarray:
    """Convert a posterior histogram into log-space prior weights.

    These weights can be used as a warm-start prior for the next
    resolution level.  Centered for numerical stability.

    Args:
        posterior: normalized histogram of shape [k]
        k: number of bins

    Returns:
        Log-space weight vector of shape [k], centered.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    posterior = posterior / (posterior.sum() + 1e-30)
    w = np.log(np.clip(posterior, 1e-30, None))
    w = w - w.mean()  # center for stability
    return w


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


def cascade_solve(
    solve_fn: Callable[[int, np.ndarray | None], tuple[np.ndarray, float, float]],
    ladder: list[ResolutionLevel],
    direct_results: dict[int, tuple[float, float]] | None = None,
) -> CascadeResult:
    """Run coarse-to-fine cascade along a resolution ladder.

    At each level, the previous level's posterior is rebinned and passed
    as a warm-start prior to the solver.  This implements the Schrödinger
    Bridge "bridging" approach: geodesic from simple (low-K) to accurate
    (high-K) models.

    Args:
        solve_fn: callable(k, prior_weights_or_None) -> (posterior, s, c)
            - k: bin count for this level
            - prior_weights: log-space weight vector [k] or None (first level)
            - returns: (posterior_histogram[k], strength, confidence)
        ladder: resolution levels in ascending order
        direct_results: optional dict {k: (s, c)} for error tracking

    Returns:
        CascadeResult with final (s, c), per-level history, and errors.
    """
    if not ladder:
        raise ValueError("ladder must have at least one level")

    level_results = []
    error_per_level = []
    prior_weights = None

    for level in ladder:
        posterior, s, c = solve_fn(level.k, prior_weights)
        level_results.append((level.k, np.array(posterior), s, c))

        # Track error if direct results available
        if direct_results and level.k in direct_results:
            s_direct, _ = direct_results[level.k]
            error_per_level.append(abs(s - s_direct))
        else:
            error_per_level.append(float('nan'))

        # Prepare prior for next level (if there is one)
        if level != ladder[-1]:
            next_k = ladder[ladder.index(level) + 1].k
            rebinned = rebin_posterior(posterior, level.k, next_k)
            prior_weights = posterior_to_prior_weights(rebinned, next_k)

    # Final result from the highest resolution
    _, final_posterior, final_s, final_c = level_results[-1]

    return CascadeResult(
        strength=final_s,
        confidence=final_c,
        posterior=final_posterior,
        level_results=level_results,
        error_per_level=error_per_level,
    )
