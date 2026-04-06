"""
geodesic_thrml.curriculum_thrml — SB cascade compiled to unified THRML factor graph
====================================================================================

Compiles the Schrödinger Bridge coarse-to-fine cascade to a single
thermodynamic factor graph executable on TSU hardware.

The entire K=4→K=8→K=16 cascade becomes ONE factor graph with:
- One CategoricalNode per resolution level (each with its own K)
- Non-square CategoricalEBMFactor coupling adjacent levels
- Per-block CategoricalGibbsConditional with matching n_categories

Joint Gibbs sampling propagates information across all levels
simultaneously — no CPU-side rebinning needed.

Key insight: CategoricalEBMFactor (not SquareCategoricalEBMFactor)
accepts non-square weight tensors [b, K_src, K_tgt].  thrml's
_batch_gather_with_k correctly computes conditional logits for each
block regardless of the other block's K.

Mapping to DTM (tsu-architecture §II):
    Level i ←→ DTM denoising step t
    Non-square coupling W[K_i, K_{i+1}] ←→ ℰ^f(x^{t-1}, x^t)
    Unary prior on level 0 ←→ ℰ^θ (learned EBM term)
    Joint Gibbs over all levels ←→ DTM reverse process

References:
    - tsu-architecture §II: DTM = chain of simple EBMs bypassing MET
    - tsu-architecture §III: ℰ(x) = -β(Σ J_ij x_i x_j + Σ h_i x_i)
    - genenergy-logic §3.3: SB geodesic from simple to accurate models
    - genenergy-logic §4.4: Noether evidence conservation along geodesics
    - hyperon-whitepaper §5.2: bridging approach for inference control
    - evidence-conservation-theorems Thm 3.1: ρ conservation
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import jax
import jax.numpy as jnp
from thrml.pgm import CategoricalNode
from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.models.discrete_ebm import CategoricalEBMFactor
from thrml.factor import FactorSamplingProgram

from geodesic_thrml.types import CascadeResult, ResolutionLevel
from geodesic_thrml.sampling import (
    SamplingConfig,
    PLN_DEFAULT_CONFIG,
    assemble_sampling_program,
    run_gibbs_sampling,
    extract_posterior,
    posterior_to_stv,
)


# Default sampling parameters (matching PLN-THRML conventions)
DEFAULT_THRML_SCHEDULE = PLN_DEFAULT_CONFIG.to_schedule()
DEFAULT_THRML_N_BATCHES = PLN_DEFAULT_CONFIG.n_batches


# ═══════════════════════════════════════════════════════════════════════════
#  Inter-level coupling weights
# ═══════════════════════════════════════════════════════════════════════════

def rebin_coupling_weights(
    source_k: int,
    target_k: int,
    precision: float | None = None,
) -> jnp.ndarray:
    """Build K_src × K_tgt inter-level coupling weight matrix.

    Encodes the proximity energy ℰ^f(x^{t-1}, x^t) from DTM
    (tsu-architecture §II):

        W[i, j] = -precision * (src_center[i] - tgt_center[j])²

    Uses CategoricalEBMFactor (NOT SquareCategoricalEBMFactor) which
    accepts non-square weight tensors [b, K_src, K_tgt].

    The quadratic form matches QuantiMORK's pc_prediction_weights
    and DTM's Gaussian forward-process coupling.

    Args:
        source_k: bin count at the source (coarser) level
        target_k: bin count at the target (finer) level
        precision: coupling tightness (default: min(source_k, target_k)).
            Higher → tighter coupling → more faithful transfer.
            Lower → more exploration at the new resolution.

    Returns:
        Weight matrix of shape [K_src, K_tgt], centered per row.

    References:
        - tsu-architecture §II: ℰ^f enforces proximity between steps
        - genenergy-logic §4.3: quantale divergence D_Q(ρ_t ‖ p_t)
    """
    if precision is None:
        precision = float(min(source_k, target_k))

    src_centers = jnp.linspace(0.5 / source_k, 1.0 - 0.5 / source_k, source_k)
    tgt_centers = jnp.linspace(0.5 / target_k, 1.0 - 0.5 / target_k, target_k)

    # Quadratic proximity: W[i,j] = -precision * (s_i - t_j)²
    diff = src_centers[:, None] - tgt_centers[None, :]  # [K_src, K_tgt]
    W = -precision * diff ** 2

    # Center each row for numerical stability
    W = W - jnp.mean(W, axis=1, keepdims=True)
    return W


# ═══════════════════════════════════════════════════════════════════════════
#  Unified cascade factor graph
# ═══════════════════════════════════════════════════════════════════════════

def build_unified_cascade_graph(
    ladder: list[ResolutionLevel],
    base_prior: tuple[float, float] | None = None,
    coupling_precision: float | None = None,
) -> dict:
    """Build ONE factor graph for the entire K cascade.

    Structure (e.g., K=4→8→16):

        [node_0 (K=4)] --CategoricalEBMFactor[1,4,8]-- [node_1 (K=8)]
              |                                              |
        prior_factor(K=4)                    CategoricalEBMFactor[1,8,16]
                                                             |
                                                       [node_2 (K=16)]

    All nodes are free.  Each block has its own
    CategoricalGibbsConditional(n_categories=K_level).  Joint Gibbs
    sampling propagates information bidirectionally across all levels.

    This is the direct hardware implementation of the SB "bridging"
    approach (hyperon-whitepaper §5.2) as a unified action functional
    (genenergy-logic §4.3).

    Args:
        ladder: resolution levels in ascending K order
        base_prior: (strength, confidence) for the first level's Beta prior
        coupling_precision: precision for inter-level coupling factors

    Returns:
        Dict with keys: program, nodes, free_blocks, factors, ladder.
    """
    if not ladder:
        raise ValueError("ladder must have at least one level")

    nodes = [CategoricalNode() for _ in ladder]
    factors = []

    # Unary Beta prior for the first level
    if base_prior is not None:
        pw = _beta_prior_weights_np(base_prior[0], base_prior[1], ladder[0].k)
        prior_factor = CategoricalEBMFactor(
            [Block([nodes[0]])], jnp.array(pw)[None, :])
        factors.append(prior_factor)

    # Inter-level coupling factors (NON-SQUARE CategoricalEBMFactor)
    for i in range(len(ladder) - 1):
        W = rebin_coupling_weights(
            ladder[i].k, ladder[i + 1].k, coupling_precision)
        coupling_factor = CategoricalEBMFactor(
            [Block([nodes[i]]), Block([nodes[i + 1]])],
            W[None, :, :])  # shape [1, K_i, K_{i+1}]
        factors.append(coupling_factor)

    # Each level is its own block with its own sampler
    free_blocks = [Block([n]) for n in nodes]
    k_per_block = [level.k for level in ladder]
    program = assemble_sampling_program(free_blocks, [], factors, k=k_per_block)

    return {
        "program": program,
        "nodes": nodes,
        "free_blocks": free_blocks,
        "factors": factors,
        "ladder": ladder,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Sampling and posterior extraction
# ═══════════════════════════════════════════════════════════════════════════

def _run_unified_sampling(
    graph: dict,
    schedule: SamplingSchedule | None = None,
    n_batches: int = DEFAULT_THRML_N_BATCHES,
    seed: int = 42,
):
    """Run joint Gibbs sampling on the unified cascade factor graph.

    Returns raw samples — a list of per-block arrays.
    """
    if schedule is None:
        schedule = DEFAULT_THRML_SCHEDULE

    ladder = graph["ladder"]
    k_per_block = [level.k for level in ladder]

    config = SamplingConfig(
        n_warmup=schedule.n_warmup,
        n_samples=schedule.n_samples,
        steps_per_sample=schedule.steps_per_sample,
        n_batches=n_batches,
    )

    return run_gibbs_sampling(
        graph["program"],
        graph["free_blocks"],
        config=config,
        k=k_per_block,
        seed=seed,
    )


def _extract_level_posterior(
    samples,
    level_idx: int,
    k: int,
) -> np.ndarray:
    """Extract K-bin posterior for a specific level from unified samples."""
    return extract_posterior(samples, level_idx, 0, k)


# Re-export for backward compatibility (test_curriculum_thrml imports this)
_posterior_to_stv = posterior_to_stv


# ═══════════════════════════════════════════════════════════════════════════
#  Public API: cascade solve
# ═══════════════════════════════════════════════════════════════════════════

def cascade_solve_thrml(
    ladder: list[ResolutionLevel],
    base_prior: tuple[float, float] | None = None,
    schedule: SamplingSchedule | None = None,
    n_batches: int = DEFAULT_THRML_N_BATCHES,
    coupling_precision: float | None = None,
    seed: int = 42,
) -> CascadeResult:
    """Run coarse-to-fine cascade as a unified THRML factor graph.

    Builds ONE factor graph with a CategoricalNode per resolution level,
    coupled by non-square CategoricalEBMFactor weights.  Joint Gibbs
    sampling propagates information across all levels simultaneously.

    This is the direct hardware implementation of the Schrödinger Bridge
    "bridging" approach (hyperon-whitepaper §5.2):
    - The unified graph embodies a single action functional S_Q(π)
    - Evidence conservation (Thm 3.1) holds across the full cascade
    - No CPU-side rebinning — all transfer is via factor coupling

    Args:
        ladder: resolution levels in ascending K order
        base_prior: (strength, confidence) for the first level's Beta prior.
            If None, first level uses uniform prior.
        schedule: THRML SamplingSchedule (default: 500 warmup, 2000 samples)
        n_batches: number of parallel sampling batches
        coupling_precision: precision for inter-level coupling.
            None = auto (min(source_k, target_k)).
        seed: random seed

    Returns:
        CascadeResult with final (s, c) and per-level history.

    References:
        - tsu-architecture §II: DTM chain of simple EBMs
        - genenergy-logic §3.3, §4.3: SB geodesic, unified action
        - genenergy-logic §4.4: evidence conservation along geodesics
    """
    if not ladder:
        raise ValueError("ladder must have at least one level")

    # Build unified factor graph
    graph = build_unified_cascade_graph(
        ladder, base_prior, coupling_precision)

    # Joint Gibbs sampling across all levels
    samples = _run_unified_sampling(graph, schedule, n_batches, seed)

    # Extract posterior from each level
    level_results = []
    for i, level in enumerate(ladder):
        posterior = _extract_level_posterior(samples, i, level.k)
        s, c = _posterior_to_stv(posterior, level.k)
        level_results.append((level.k, posterior, s, c))

    _, final_posterior, final_s, final_c = level_results[-1]

    return CascadeResult(
        strength=final_s,
        confidence=final_c,
        posterior=final_posterior,
        level_results=level_results,
        error_per_level=[],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _beta_prior_weights_np(
    strength: float, confidence: float, k: int,
) -> np.ndarray:
    """Compute Beta prior log-weights using numpy.

    Mirrors pln_thrml.beta.beta_prior_weights:
        W[j] = (alpha-1)*log(c_j) + (beta-1)*log(1-c_j), centered.
    """
    c_clamped = min(confidence, 0.9999)
    if c_clamped <= 0:
        w = 0.0
    else:
        w = c_clamped / (1.0 - c_clamped)
    n = w + 2.0
    alpha = max(strength * n, 1e-7)
    beta_param = max((1.0 - strength) * n, 1e-7)

    centers = np.linspace(0.5 / k, 1.0 - 0.5 / k, k)
    eps = 1e-7
    log_c = np.log(np.clip(centers, eps, 1.0 - eps))
    log_1mc = np.log(np.clip(1.0 - centers, eps, 1.0 - eps))

    weights = (alpha - 1.0) * log_c + (beta_param - 1.0) * log_1mc
    weights = weights - weights.mean()
    return weights
