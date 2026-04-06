"""
geodesic_thrml.controller_thrml — Step selection compiled to THRML
===================================================================

Compiles the geodesic controller's Boltzmann step selection to a
single-node THRML factor graph.  Instead of computing probabilities
on CPU and sampling with numpy, the geodesic energy vector becomes
the bias of a pdit CategoricalNode — TSU samples in one clock cycle.

Mapping:
    energy_to_probs() + rng.choice()  →  single pdit with bias = -E/T
    select_batch() parallel grouping  →  unchanged (CPU-side scheduling)

The CPU fallback (controller.py) is available for environments
without TSU hardware.

References:
    - genenergy-logic §5: log-free geodesic controller
    - tsu-architecture §III: Gibbs update P(X_i) = σ(2β(Σ J_ij x_j + h_i))
    - hyperon-whitepaper §6.1.1: Q_logic (CPU) / Q_tv (TSU) split
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from geodesic_thrml.scores import (
    RuleSpec,
    compute_forward_scores,
    compute_backward_scores,
    compute_geodesic_energy,
    energy_to_probs,
    compute_rho,
    partition_parallel_groups,
)
from geodesic_thrml.types import (
    SelectionResult,
    BatchResult,
    DEFAULT_TEMPERATURE,
    DEFAULT_COST_WEIGHT,
    annealing_schedule,
)

import jax.numpy as jnp
from thrml.pgm import CategoricalNode
from thrml.block_management import Block
from thrml.models.discrete_ebm import CategoricalEBMFactor

from geodesic_thrml.sampling import (
    LIGHT_CONFIG,
    assemble_sampling_program,
    run_gibbs_sampling,
    extract_posterior,
)


def select_step_thrml(
    specs: Sequence[RuleSpec],
    goal_stv: tuple[float, float] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    seed: int = 42,
) -> SelectionResult:
    """Select next inference step via THRML single-node Boltzmann sampling.

    The geodesic energy E(i) = -(log f(i) + log g(i))/T + λ*cost(i)
    becomes the bias vector of a single CategoricalNode with K=n_rules
    categories.  TSU samples from the exact Boltzmann distribution in
    hardware.

    For small n_rules (< ~50), this is functionally equivalent to the
    CPU path but validates the THRML compilation pipeline.  For large
    candidate pools or joint (rule, conclusion) sampling, the hardware
    path avoids explicit probability computation.

    Args:
        specs: candidate rule specifications
        goal_stv: target (strength, confidence) for backward scoring
        temperature: T > 0; lower = more greedy
        cost_weight: λ, penalty for expensive steps
        seed: random seed

    Returns:
        SelectionResult with selected step and diagnostics.

    References:
        - genenergy-logic §5: geodesic controller
        - tsu-architecture §III: single-node Gibbs sampling
    """
    if not specs:
        raise ValueError("No candidate specs provided")

    n_rules = len(specs)

    # Compute scores (CPU — these are lightweight O(n) operations)
    f = compute_forward_scores(specs)
    g = compute_backward_scores(specs, goal_stv)
    costs = np.array([s.cost for s in specs])
    energy = compute_geodesic_energy(f, g, costs, temperature, cost_weight)
    rho = compute_rho(f, g)

    # Encode energy as pdit bias: h_i = -E(i)/T
    t = max(temperature, 1e-6)
    bias = jnp.array(-energy / t)
    bias = bias - jnp.mean(bias)  # center for stability

    # Build single-node factor graph
    node = CategoricalNode()
    block = Block([node])
    factor = CategoricalEBMFactor([block], bias[None, :])
    program = assemble_sampling_program([block], [], [factor], k=n_rules)

    # Sample via shared facility
    samples = run_gibbs_sampling(
        program, [block], config=LIGHT_CONFIG, k=n_rules, seed=seed)

    # Mode of samples = selected step
    posterior = extract_posterior(samples, 0, 0, n_rules)
    idx = int(jnp.argmax(jnp.array(posterior)))

    # CPU probabilities for diagnostics (cheap to compute anyway)
    probs = energy_to_probs(energy, temperature)

    return SelectionResult(
        selected_idx=idx,
        selected_name=specs[idx].name,
        rule_probs=np.array(probs),
        rho=np.array(rho),
        energy=np.array(energy),
        f_scores=np.array(f),
        g_scores=np.array(g),
    )


def select_batch_thrml(
    specs: Sequence[RuleSpec],
    goal_stv: tuple[float, float] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    weakness_threshold: float = 0.05,
    seed: int = 42,
) -> BatchResult:
    """Select a batch of parallel-safe steps via THRML.

    Primary selection uses THRML Boltzmann sampling; parallel grouping
    remains CPU-side (it's a graph-coloring decision, not a sampling
    problem).

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
    primary = select_step_thrml(
        specs, goal_stv, temperature, cost_weight, seed)
    groups = partition_parallel_groups(list(specs), weakness_threshold)

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


def multi_step_select_thrml(
    specs: Sequence[RuleSpec],
    goal_stv: tuple[float, float] | None = None,
    n_steps: int = 5,
    t_start: float = 2.0,
    t_end: float = 0.01,
    strategy: str = "exponential",
    cost_weight: float = DEFAULT_COST_WEIGHT,
    seed: int = 42,
) -> list[SelectionResult]:
    """Run annealed multi-step selection using THRML.

    Each step uses THRML Boltzmann sampling at the current temperature.

    Returns:
        List of SelectionResult, one per step.
    """
    temps = annealing_schedule(t_start, t_end, n_steps, strategy)
    results = []
    for i, t in enumerate(temps):
        result = select_step_thrml(
            specs, goal_stv, t, cost_weight, seed=seed + i)
        results.append(result)
    return results
