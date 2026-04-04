"""
geodesic_thrml.controller — Geodesic step selection and annealing
==================================================================

Finite-temperature implementation of the geodesic controller
(genenergy-logic §5).  Selects inference steps by combining forward
reachability (f), backward utility (g), and cost into a Boltzmann
distribution over candidates.

    T → 0:  exact Pareto selector (argmax ρ/cost)
    T > 0:  Boltzmann exploration with natural temperature annealing

The controller operates in CPU-space, pre-selecting which steps to
send to the TSU.  When thrml is available, it can also build a factor
graph for joint (rule, conclusion) sampling on hardware.

References:
    - genenergy-logic §5: log-free geodesic controller
    - evidence-conservation-theorems §2.5: weakness gauge
    - noncommutative-evidence §10.2: Noether anomaly as design parameter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geodesic_thrml.scores import (
    RuleSpec, compute_forward_scores, compute_backward_scores,
    compute_geodesic_energy, energy_to_probs, compute_rho,
    partition_parallel_groups,
)


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


def select_step(
    specs: Sequence[RuleSpec],
    goal_stv: tuple[float, float] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    seed: int | None = None,
) -> SelectionResult:
    """Select the next inference step via geodesic energy.

    At T→0, deterministically picks argmin(energy) = argmax(f·g/cost).
    At T>0, samples from Boltzmann distribution for exploration.

    Args:
        specs: candidate rule specifications
        goal_stv: target (strength, confidence) for backward scoring
        temperature: T > 0; lower = more greedy
        cost_weight: λ, penalty for expensive steps
        seed: random seed for reproducibility

    Returns:
        SelectionResult with selected step and diagnostics.
    """
    if not specs:
        raise ValueError("No candidate specs provided")

    f = compute_forward_scores(specs)
    g = compute_backward_scores(specs, goal_stv)
    costs = np.array([s.cost for s in specs])

    energy = compute_geodesic_energy(f, g, costs, temperature, cost_weight)
    probs = energy_to_probs(energy, temperature)
    rho = compute_rho(f, g)

    # Select: deterministic at very low T, stochastic otherwise
    if temperature < 1e-4:
        idx = int(np.argmin(energy))
    else:
        rng = np.random.default_rng(seed)
        idx = int(rng.choice(len(specs), p=probs))

    return SelectionResult(
        selected_idx=idx,
        selected_name=specs[idx].name,
        rule_probs=probs,
        rho=rho,
        energy=energy,
        f_scores=f,
        g_scores=g,
    )


def select_batch(
    specs: Sequence[RuleSpec],
    goal_stv: tuple[float, float] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    weakness_threshold: float = 0.05,
    seed: int | None = None,
) -> BatchResult:
    """Select a batch of steps that can safely run in parallel.

    First selects the primary step via geodesic energy, then finds
    all steps in the same parallel group (weakness < threshold).

    Thm 3 (weakness-bounded leakage) guarantees that parallel
    execution within a group introduces bounded evidence leakage.

    Args:
        specs: candidate rule specifications
        goal_stv: target (strength, confidence)
        temperature: T > 0
        cost_weight: λ
        weakness_threshold: max weakness for parallel grouping
        seed: random seed

    Returns:
        BatchResult with primary selection and parallel group.
    """
    primary = select_step(specs, goal_stv, temperature, cost_weight, seed)
    groups = partition_parallel_groups(list(specs), weakness_threshold)

    # Find the group containing the selected step
    selected_group = [primary.selected_idx]
    for group in groups:
        if primary.selected_idx in group:
            selected_group = group
            break

    return BatchResult(
        primary=primary,
        parallel_group=selected_group,
        parallel_names=[specs[i].name for i in selected_group],
    )


def annealing_schedule(
    t_start: float,
    t_end: float,
    n_steps: int,
    strategy: str = "exponential",
) -> list[float]:
    """Generate a temperature annealing schedule.

    High temperature → exploration of multiple modes.
    Low temperature → exploitation of the best mode.

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
        import math
        return [t_end + 0.5 * (t_start - t_end) *
                (1 + math.cos(math.pi * i / (n_steps - 1)))
                for i in range(n_steps)]

    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Use 'exponential', 'linear', or 'cosine'.")


def multi_step_select(
    specs: Sequence[RuleSpec],
    goal_stv: tuple[float, float] | None = None,
    n_steps: int = 5,
    t_start: float = 2.0,
    t_end: float = 0.01,
    strategy: str = "exponential",
    cost_weight: float = DEFAULT_COST_WEIGHT,
    seed: int = 42,
) -> list[SelectionResult]:
    """Run annealed multi-step selection.

    Applies temperature annealing across multiple selection rounds.
    Early rounds explore (high T); later rounds exploit (low T).

    Returns:
        List of SelectionResult, one per step.
    """
    temps = annealing_schedule(t_start, t_end, n_steps, strategy)
    results = []
    for i, t in enumerate(temps):
        result = select_step(specs, goal_stv, t, cost_weight, seed=seed + i)
        results.append(result)
    return results
