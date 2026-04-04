"""
geodesic_thrml.bridges.moses — MOSES-THRML bridge (GEO-EVO)
==============================================================

Adapts MOSES-THRML's deme/metapopulation state for geodesic scheduling,
implementing the GEO-EVO pattern from Hyperon whitepaper §6.2.1:

    "GEO-EVO's two-ended guidance maintains forward reachability factors
    (can we get there from here?) and backward compatibility factors
    (is it useful for our goals?). The system expands search where the
    product of these factors increases most per unit effort."

GEO-EVO mapping to geodesic controller:

    f(deme) = forward reachability
            = normalized fitness of best program found so far
            → "can we get there from here?"

    g(deme) = backward compatibility
            = proximity of deme's best behavior to target behavior
            → "is it useful for our goals?"

    cost(deme) = knob space size × chain count
              → computational budget for one deme expansion

    ρ = f · g  → expand the deme where ρ/cost increases most

This is the highest multimodal risk among all four sub-projects.
The geodesic controller is critical here: MOSES's program fitness
landscape has exponentially many local optima, and blind search
wastes most of its budget in dead-end valleys.  GEO-EVO's bidirectional
guidance reduces effective search space from O(n) to O(√n).

References:
    - Hyperon whitepaper §6.2.1: GEO-EVO
    - moses_thrml.deme: Deme, Metapopulation, run_deme()
    - moses_thrml.search: run_thermodynamic_search(), SamplingResult
    - moses_thrml.knobs: KnobSpace
    - genenergy-logic §6: cross-module scheduling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from geodesic_thrml.scores import RuleSpec


@dataclass
class DemeSpec:
    """One MOSES deme translated for GEO-EVO geodesic scoring.

    Maps MOSES concepts to geodesic concepts:
        best_fitness    → forward factor f (can we get there from here?)
        goal_proximity  → backward factor g (is it useful for our goals?)
        n_knobs         → cost (larger knob space = more expensive to search)
        acceptance_rate → health indicator (too low = stuck, too high = random walk)

    Attributes:
        deme_id: unique identifier (usually exemplar name)
        n_knobs: size of the knob space (binary variables)
        best_fitness: best fitness found so far (-best_energy)
        acceptance_rate: MH acceptance rate (healthy: 20%-60%)
        is_done: whether this deme has completed its search
        best_bits: knob vector of best program found (for behavior eval)
    """
    deme_id: str
    n_knobs: int
    best_fitness: float
    acceptance_rate: float
    is_done: bool
    best_bits: np.ndarray | None = None


def collect_deme_specs(metapop: Any) -> list[DemeSpec]:
    """Extract deme specs from a MOSES Metapopulation.

    Args:
        metapop: a moses_thrml.deme.Metapopulation instance

    Returns:
        List of DemeSpec, one per deme.
    """
    specs = []
    for deme in metapop.demes:
        acceptance = 0.0
        if deme.result is not None and hasattr(deme.result, 'acceptance_rates'):
            rates = deme.result.acceptance_rates
            acceptance = float(np.mean(rates)) if len(rates) > 0 else 0.0

        specs.append(DemeSpec(
            deme_id=deme.knob_space.exemplar or f"deme-{len(specs)}",
            n_knobs=deme.knob_space.n_knobs,
            best_fitness=deme.score,
            acceptance_rate=acceptance,
            is_done=deme.is_done,
            best_bits=deme.best_program,
        ))
    return specs


def compute_forward_reachability(
    deme_specs: list[DemeSpec],
) -> np.ndarray:
    """GEO-EVO forward factor: "can we get there from here?"

    f(deme) = normalized fitness of best program in the deme.
    Higher fitness → more promising starting point for further search.

    Returns:
        Normalized scores in [0, 1], shape [n_demes].
    """
    if not deme_specs:
        return np.array([])
    fitnesses = np.array([d.best_fitness for d in deme_specs])
    f_min, f_max = fitnesses.min(), fitnesses.max()
    if f_max > f_min:
        scores = (fitnesses - f_min) / (f_max - f_min)
    else:
        scores = np.ones(len(deme_specs)) * 0.5
    # Normalize to probability distribution
    total = scores.sum()
    return scores / total if total > 0 else np.ones(len(scores)) / len(scores)


def compute_backward_compatibility(
    deme_specs: list[DemeSpec],
    target_behavior_fn: Callable[[np.ndarray], float] | None = None,
) -> np.ndarray:
    """GEO-EVO backward factor: "is it useful for our goals?"

    g(deme) = proximity of deme's best program behavior to target behavior.
    Without a target behavior function, uses acceptance rate as proxy
    (healthy acceptance = deme is in a productive region of program space).

    Args:
        deme_specs: list of DemeSpec
        target_behavior_fn: callable(knob_bits) → similarity score in [0,1]
            If provided, evaluates each deme's best program against target.
            If None, uses acceptance_rate as a health-based proxy.

    Returns:
        Normalized scores in [0, 1], shape [n_demes].
    """
    if not deme_specs:
        return np.array([])

    if target_behavior_fn is not None:
        scores = np.array([
            target_behavior_fn(d.best_bits) if d.best_bits is not None else 0.0
            for d in deme_specs
        ])
    else:
        # Proxy: acceptance rate in healthy range [0.2, 0.6] → higher score
        scores = np.array([
            1.0 - abs(d.acceptance_rate - 0.4) / 0.4
            for d in deme_specs
        ])
        scores = np.clip(scores, 0.01, 1.0)

    total = scores.sum()
    return scores / total if total > 0 else np.ones(len(scores)) / len(scores)


def deme_specs_to_rule_specs(
    deme_specs: list[DemeSpec],
    target_behavior_fn: Callable[[np.ndarray], float] | None = None,
    n_chains: int = 50,
) -> list[RuleSpec]:
    """Convert deme specs to RuleSpec for the geodesic controller.

    Implements the full GEO-EVO mapping:
        f = forward_reachability (fitness-based)
        g = backward_compatibility (target-behavior-based)
        cost = n_knobs × n_chains
        touched_nodes = {deme_id} (different demes are independent → parallel-safe)

    Args:
        deme_specs: list of DemeSpec from collect_deme_specs
        target_behavior_fn: optional target behavior evaluation
        n_chains: number of parallel MH chains per deme (for cost estimation)

    Returns:
        List of RuleSpec for controller.select_step().
    """
    if not deme_specs:
        return []

    f_scores = compute_forward_reachability(deme_specs)
    g_scores = compute_backward_compatibility(deme_specs, target_behavior_fn)

    specs = []
    for i, d in enumerate(deme_specs):
        specs.append(RuleSpec(
            name=d.deme_id,
            posterior=np.ones(16) / 16,  # placeholder — MOSES uses bits, not histograms
            conclusion_stv=(float(f_scores[i]), float(g_scores[i])),
            premise_confidences=[float(f_scores[i])],
            cost=float(d.n_knobs * n_chains),
            touched_nodes=frozenset([d.deme_id]),  # demes are independent → weakness ≈ 0
        ))
    return specs


def estimate_deme_cost(n_knobs: int, n_chains: int = 50) -> float:
    """Estimate computational cost for searching a deme.

    Cost ∝ knob space size × chain count.  On TSU hardware, this maps
    to the number of pbits × number of parallel thermal relaxation runs.
    """
    return float(n_knobs * n_chains)
