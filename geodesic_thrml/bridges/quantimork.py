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

from geodesic_thrml.curriculum import ResolutionLevel


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
