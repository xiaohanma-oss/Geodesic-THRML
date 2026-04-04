"""
geodesic_thrml.bridges.moses — MOSES-THRML bridge (stub)
==========================================================

MOSES-THRML is not yet started.  This bridge defines the interface
that will be implemented when the program evolution engine is built.

MOSES's program fitness landscape is the highest multimodal risk
among all four sub-projects — DTM will be needed from day one.
The geodesic controller's role here is to:
    1. Evaluate program skeleton fitness (coarse, fast)
    2. Select promising skeletons via f⊗g scoring
    3. Cascade to full parameter search only on selected skeletons

This maps to GEO-EVO (Hyperon whitepaper): bidirectional guidance
via optimal transport in program space.

References:
    - Hyperon whitepaper: GEO-EVO (MOSES + geodesic control)
    - genenergy-logic §6: cross-module scheduling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class ProgramSkeletonSpec:
    """Specification for a program skeleton (coarse representation).

    Attributes:
        skeleton_id: unique identifier
        structure: symbolic representation of program structure
        fitness_estimate: coarse fitness from skeleton evaluation
        n_parameters: number of parameters to optimize (for cost estimation)
    """
    skeleton_id: str
    structure: Any
    fitness_estimate: float
    n_parameters: int


def collect_program_skeleton_specs(
    population: Sequence[Any],
    fitness_fn: Callable[[Any], float],
    structure_fn: Callable[[Any], Any] | None = None,
) -> list[ProgramSkeletonSpec]:
    """Extract skeleton specs from a MOSES population.

    Args:
        population: list of program individuals
        fitness_fn: callable(program) → fitness score
        structure_fn: optional callable(program) → skeleton structure

    Returns:
        List of ProgramSkeletonSpec.

    Raises:
        NotImplementedError: MOSES-THRML not yet available.
    """
    raise NotImplementedError(
        "MOSES-THRML is not yet implemented. "
        "This bridge will be activated when the program evolution "
        "engine is built. See Hyperon whitepaper GEO-EVO for design."
    )
