"""
geodesic_thrml.bridges.quantimork — QuantiMORK-THRML bridge
=============================================================

Extracts wavelet level specifications from QuantiMORK's WaveletLinear
layers for use with the geodesic curriculum (coarse-to-fine cascade)
and controller (level-aware scheduling).

QuantiMORK's wavelet hierarchy is a natural resolution ladder:
    Level 3 (64d, coarse)  → Level 2 (128d, medium) → Level 1 (256d, fine)

This maps directly to curriculum.build_resolution_ladder for SB Learning.

References:
    - quantimork_thrml.wavelet_linear: WaveletLinear.extract_energy_params()
    - quantimork_thrml.thrml_verify: build_single_level_graph()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from geodesic_thrml.types import ResolutionLevel


@dataclass
class WaveletLevelSpec:
    """Specification for one wavelet level.

    Attributes:
        level: wavelet decomposition level (1 = finest, n = coarsest)
        band: 'detail' or 'approx'
        dim: feature dimension at this level
        weight_shape: shape of the per-level weight matrix
        max_connections: max connections per node at this level
        energy_params: raw energy parameters (from extract_energy_params)
    """
    level: int
    band: str
    dim: int
    weight_shape: tuple[int, ...]
    max_connections: int
    energy_params: dict[str, Any] | None = None


def collect_wavelet_level_specs(
    model: Any,
    block_idx: int = 0,
) -> list[WaveletLevelSpec]:
    """Extract wavelet level specs from a WaveletPCTransformer.

    Traverses the model's wavelet layers and collects per-level
    information needed for geodesic scheduling and curriculum cascade.

    Args:
        model: a WaveletPCTransformer instance (or any model with
               .blocks[i].mlp.wavelet attribute)
        block_idx: which transformer block to extract from

    Returns:
        List of WaveletLevelSpec, ordered coarsest-first.
    """
    try:
        wavelet = model.blocks[block_idx].mlp.wavelet
    except (AttributeError, IndexError) as e:
        raise ValueError(
            f"Cannot find wavelet layer at blocks[{block_idx}].mlp.wavelet: {e}"
        ) from e

    energy_params = wavelet.extract_energy_params()

    specs = []
    for params in energy_params:
        weight = params["weight"]
        specs.append(WaveletLevelSpec(
            level=params["level"],
            band=params["band"],
            dim=weight.shape[0],
            weight_shape=tuple(weight.shape),
            max_connections=min(weight.shape[0], 5),  # wavelet ≤5 neighbors
            energy_params=params,
        ))

    # Sort coarsest-first (highest level number = coarsest)
    specs.sort(key=lambda s: (-s.level, s.band))
    return specs


def wavelet_to_resolution_ladder(
    specs: list[WaveletLevelSpec],
) -> list[ResolutionLevel]:
    """Convert wavelet level specs into a curriculum resolution ladder.

    Maps wavelet levels to ResolutionLevel objects where K = dim
    (the feature dimension at each level serves as the resolution).

    Returns ladder in coarse-to-fine order (ascending K).
    """
    # Deduplicate by level (detail + approx at same level → use detail dim)
    level_dims = {}
    for spec in specs:
        if spec.level not in level_dims or spec.band == "detail":
            level_dims[spec.level] = spec.dim

    # Sort by dim ascending (coarse → fine)
    items = sorted(level_dims.items(), key=lambda x: x[1])
    return [
        ResolutionLevel(k=dim, label=f"wavelet-L{level}")
        for level, dim in items
    ]


def estimate_level_cost(spec: WaveletLevelSpec) -> float:
    """Estimate computational cost for a wavelet level.

    Cost is proportional to the weight matrix size (number of multiply-adds).
    """
    if spec.weight_shape:
        return float(np.prod(spec.weight_shape))
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  THRML cascade solver
# ═══════════════════════════════════════════════════════════════════════════

def make_quantimork_cascade_solver_thrml(
    wavelet_specs: list[WaveletLevelSpec],
    coupling_precision: float | None = None,
) -> "Callable":
    """Build unified THRML cascade graph from wavelet level specifications.

    Maps QuantiMORK's wavelet hierarchy directly to curriculum_thrml's
    unified factor graph:
    - Each wavelet level → one CategoricalNode (K = dim at that level)
    - Wavelet energy_params → per-level CategoricalEBMFactor prior weights
    - Inter-level structure → non-square coupling factors

    Instead of generic Beta priors, uses the actual wavelet weight matrices
    from WaveletLevelSpec.energy_params to set per-level bias.

    Args:
        wavelet_specs: from collect_wavelet_level_specs()
        coupling_precision: precision for inter-level coupling (default: auto)

    Returns:
        A callable(k, prior_weights_or_None) → (posterior, s, c)

    References:
        - quantimork_thrml.thrml_verify: build_single_level_graph() (per-level pattern)
        - curriculum_thrml: build_unified_cascade_graph() (unified cascade)
    """
    import jax.numpy as jnp
    from geodesic_thrml.curriculum_thrml import (
        build_unified_cascade_graph,
        _run_unified_sampling,
        _extract_level_posterior,
        _posterior_to_stv,
    )

    ladder = wavelet_to_resolution_ladder(wavelet_specs)

    # Extract per-level prior weights from wavelet energy params
    level_priors = {}
    for spec in wavelet_specs:
        if spec.energy_params is not None and "weight" in spec.energy_params:
            weight = spec.energy_params["weight"]
            # Diagonal of weight matrix → per-bin energy (self-energy)
            if hasattr(weight, 'numpy'):
                weight = weight.numpy()
            w = np.asarray(weight, dtype=np.float64)
            if w.ndim == 2:
                diag = np.diag(w) if w.shape[0] == w.shape[1] else w.mean(axis=0)
            else:
                diag = w
            # Center for stability
            diag = diag - diag.mean()
            level_priors[spec.dim] = diag

    def solve(k: int, prior_weights: np.ndarray | None):
        """Solve at resolution k using the unified factor graph."""
        # Find the matching level in the ladder
        target_levels = [l for l in ladder if l.k == k]
        if not target_levels:
            # Fallback: single-level solve
            single_ladder = [ResolutionLevel(k=k)]
            graph = build_unified_cascade_graph(single_ladder)
        else:
            graph = build_unified_cascade_graph(
                ladder, coupling_precision=coupling_precision)

        # Inject wavelet-derived priors if available
        if k in level_priors:
            from thrml.models.discrete_ebm import CategoricalEBMFactor
            from thrml.block_management import Block
            wp = jnp.array(level_priors[k][:k])  # truncate/pad to k
            if len(wp) < k:
                wp = jnp.pad(wp, (0, k - len(wp)))
            wp = wp - jnp.mean(wp)
            # Find the node for this level
            for i, level in enumerate(graph["ladder"]):
                if level.k == k:
                    factor = CategoricalEBMFactor(
                        [Block([graph["nodes"][i]])], wp[None, :])
                    graph["factors"].append(factor)
                    break

        # Also inject external prior_weights if provided
        if prior_weights is not None:
            from thrml.models.discrete_ebm import CategoricalEBMFactor
            from thrml.block_management import Block
            pw = jnp.array(prior_weights)
            for i, level in enumerate(graph["ladder"]):
                if level.k == k:
                    factor = CategoricalEBMFactor(
                        [Block([graph["nodes"][i]])], pw[None, :])
                    graph["factors"].append(factor)
                    break

        samples = _run_unified_sampling(graph, seed=42)

        # Extract posterior from the target level
        for i, level in enumerate(graph["ladder"]):
            if level.k == k:
                posterior = _extract_level_posterior(samples, i, k)
                s, c = _posterior_to_stv(posterior, k)
                return np.array(posterior), float(s), float(c)

        # Fallback: extract from last level
        posterior = _extract_level_posterior(samples, len(ladder) - 1, k)
        s, c = _posterior_to_stv(posterior, k)
        return np.array(posterior), float(s), float(c)

    return solve
