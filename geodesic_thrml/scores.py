"""
geodesic_thrml.scores — Forward/backward factors and weakness gauge
====================================================================

Computes the components of the geodesic energy function:

    E(step_i) = -(log f(i) + log g(i)) / T  +  λ·cost(i)

where f = forward reachability, g = backward utility, T = temperature,
λ = cost weight.

Also implements the weakness gauge (evidence-conservation-theorems §2.5,
noncommutative-evidence §2.2) for determining which inference steps
can safely execute in parallel.

References:
    - genenergy-logic §5: geodesic controller energy
    - evidence-conservation-theorems §2.5: weakness gauge
    - noncommutative-evidence §2.2: non-commutative weakness
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

EPS = 1e-30  # log-safety clamp


@dataclass
class RuleSpec:
    """Specification for a candidate inference step.

    Attributes:
        name: human-readable identifier
        posterior: K-bin posterior histogram from executing this rule
        conclusion_stv: (strength, confidence) of derived conclusion
        premise_confidences: confidences of input premises
        cost: computational cost (default 1.0)
        touched_nodes: set of node IDs this rule reads/writes
    """
    name: str
    posterior: np.ndarray
    conclusion_stv: tuple[float, float]
    premise_confidences: list[float]
    cost: float = 1.0
    touched_nodes: frozenset[str] = frozenset()


def compute_forward_scores(specs: Sequence[RuleSpec]) -> np.ndarray:
    """Forward factor f(r): how strongly the premises support each rule.

    f(r) = Σ c2w(premise_confidence_i), normalized to [0, 1].
    Higher confidence premises → higher forward score.

    Args:
        specs: list of RuleSpec

    Returns:
        Array of shape [n_rules] with normalized forward scores.
    """
    scores = np.array([
        sum(_c2w(c) for c in spec.premise_confidences)
        for spec in specs
    ], dtype=np.float64)
    total = scores.sum()
    if total > 0:
        scores = scores / total
    else:
        scores = np.ones(len(specs)) / len(specs)
    return scores


def compute_backward_scores(
    specs: Sequence[RuleSpec],
    goal_stv: tuple[float, float] | None = None,
) -> np.ndarray:
    """Backward factor g(r): how useful each rule's conclusion is for the goal.

    g(r) = exp(-‖conclusion_stv_r - goal_stv‖²)
    If no goal is given, returns uniform scores.

    Args:
        specs: list of RuleSpec
        goal_stv: target (strength, confidence), or None for uniform

    Returns:
        Array of shape [n_rules] with normalized backward scores.
    """
    if goal_stv is None:
        return np.ones(len(specs)) / len(specs)

    gs, gc = goal_stv
    scores = np.array([
        math.exp(-((spec.conclusion_stv[0] - gs) ** 2 +
                    (spec.conclusion_stv[1] - gc) ** 2))
        for spec in specs
    ], dtype=np.float64)
    total = scores.sum()
    if total > 0:
        scores = scores / total
    return scores


def compute_geodesic_energy(
    f_scores: np.ndarray,
    g_scores: np.ndarray,
    costs: np.ndarray,
    temperature: float = 1.0,
    cost_weight: float = 0.1,
) -> np.ndarray:
    """Geodesic energy for each candidate step.

    E(r) = -(log f(r) + log g(r)) / T + λ·cost(r)

    Lower energy = better step.  At T→0, the minimum-energy step
    is the Pareto-optimal argmax(f·g / cost).

    Args:
        f_scores: forward factors [n_rules]
        g_scores: backward factors [n_rules]
        costs: computational costs [n_rules]
        temperature: T > 0; T→0 recovers exact Pareto
        cost_weight: λ, penalty for expensive steps

    Returns:
        Energy array [n_rules].  Lower is better.
    """
    log_f = np.log(np.clip(f_scores, EPS, None))
    log_g = np.log(np.clip(g_scores, EPS, None))

    # Clip for T→0 stability
    t = max(temperature, 1e-6)
    energy = -(log_f + log_g) / t + cost_weight * costs
    return energy


def energy_to_probs(energy: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert energy to Boltzmann probabilities.

    P(r) ∝ exp(-E(r) / T)

    Args:
        energy: energy array [n_rules]
        temperature: T > 0

    Returns:
        Probability array [n_rules], sums to 1.
    """
    t = max(temperature, 1e-6)
    logits = -energy / t
    logits = logits - logits.max()  # numerical stability
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return probs


def compute_rho(f_scores: np.ndarray, g_scores: np.ndarray) -> np.ndarray:
    """Reinforcement ρ = f ⊗ g (element-wise product in multiplicative reals).

    Along geodesic paths, ρ is constant (Thm 3.1).

    Returns:
        Array of shape [n_rules].
    """
    return f_scores * g_scores


# ═══════════════════════════════════════════════════════════════════════════
#  Weakness gauge (v2.1)
# ═══════════════════════════════════════════════════════════════════════════

def compute_weakness(step_a: RuleSpec, step_b: RuleSpec) -> float:
    """Weakness gauge: quantifies non-commutativity of two inference steps.

    Two steps that touch disjoint nodes commute exactly (weakness = 0).
    Overlapping nodes → positive weakness → reordering has a cost.

    Implementation: Jaccard overlap of touched_nodes as a proxy for
    the quantale weakness gauge δ(a,b).

    References:
        evidence-conservation-theorems §2.5
        noncommutative-evidence §2.2

    Args:
        step_a: first rule spec
        step_b: second rule spec

    Returns:
        Weakness ∈ [0, 1].  0 = commutative (safe to parallelize).
    """
    a_nodes = step_a.touched_nodes
    b_nodes = step_b.touched_nodes
    if not a_nodes and not b_nodes:
        return 0.0
    intersection = a_nodes & b_nodes
    union = a_nodes | b_nodes
    if not union:
        return 0.0
    return len(intersection) / len(union)


def partition_parallel_groups(
    specs: Sequence[RuleSpec],
    weakness_threshold: float = 0.05,
) -> list[list[int]]:
    """Partition steps into groups that can safely execute in parallel.

    Two steps can be in the same parallel group if their weakness
    is below the threshold (near-commutative).

    Algorithm: greedy graph coloring where steps with weakness ≥ threshold
    are connected by conflict edges.  Each color = one parallel group.

    Args:
        specs: list of RuleSpec
        weakness_threshold: maximum weakness for parallel execution

    Returns:
        List of groups, each group is a list of spec indices.
    """
    n = len(specs)
    if n == 0:
        return []

    # Build conflict graph
    conflicts = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if compute_weakness(specs[i], specs[j]) >= weakness_threshold:
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Greedy coloring
    color_of = {}
    for i in range(n):
        neighbor_colors = {color_of[j] for j in conflicts[i] if j in color_of}
        c = 0
        while c in neighbor_colors:
            c += 1
        color_of[i] = c

    n_colors = max(color_of.values(), default=-1) + 1
    groups = [[] for _ in range(n_colors)]
    for i in range(n):
        groups[color_of[i]].append(i)
    return groups


def _c2w(c: float) -> float:
    """Confidence → evidence weight.  Mirrors pln_thrml.beta.c2w."""
    if c >= 1.0:
        return 1e6
    if c <= 0.0:
        return 0.0
    return c / (1.0 - c)
