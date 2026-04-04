"""
geodesic_thrml.bridges.pln — PLN-THRML bridge
===============================================

Adapts PLN-THRML's beta engine for use with the geodesic curriculum
(coarse-to-fine K cascade) and controller (rule posterior precomputation).

References:
    - pln_thrml.beta: Beta-discretized factor graphs for PLN inference
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def make_pln_cascade_solver(
    build_fn: Callable,
    sample_fn: Callable,
    marginal_fn: Callable,
    target_node_fn: Callable,
    posterior_to_stv_fn: Callable,
    beta_prior_weights_fn: Callable,
    **fixed_params,
) -> Callable:
    """Create a solve_fn compatible with curriculum.cascade_solve.

    This wraps PLN-THRML's build → sample → measure pipeline into
    the (k, prior_weights) → (posterior, s, c) interface.

    Args:
        build_fn: graph builder (e.g., build_beta_chain) — must accept k= kwarg
        sample_fn: run_beta_sampling
        marginal_fn: estimate_beta_marginal
        target_node_fn: callable(graph) -> target node for measurement
        posterior_to_stv_fn: posterior_to_stv from pln_thrml.beta
        beta_prior_weights_fn: beta_prior_weights from pln_thrml.beta
        **fixed_params: fixed parameters passed to build_fn

    Returns:
        A callable(k, prior_weights_or_None) -> (posterior, s, c)
    """
    def solve(k: int, prior_weights: np.ndarray | None):
        graph = build_fn(**fixed_params, k=k)
        samples = sample_fn(graph)
        target = target_node_fn(graph)
        posterior, s, c = marginal_fn(samples, graph, target, k=k)
        return np.array(posterior), float(s), float(c)

    return solve


def collect_pln_rule_specs(
    premises_stv: list[tuple[float, float]],
    rules: list[dict],
    k: int = 16,
    sample_fn: Callable | None = None,
) -> list[dict]:
    """Precompute posteriors for each candidate PLN rule.

    Each rule spec contains:
        - name: rule identifier
        - posterior: K-bin posterior histogram
        - conclusion_stv: (strength, confidence)
        - premise_confidences: list of input confidences
        - cost: computational cost estimate

    Args:
        premises_stv: list of (strength, confidence) for available premises
        rules: list of rule dicts, each with 'name', 'build_fn', 'target_fn'
        k: bin count
        sample_fn: sampling function (default: pln_thrml.beta.run_beta_sampling)

    Returns:
        List of rule spec dicts, one per applicable rule.
    """
    specs = []
    for rule in rules:
        try:
            graph = rule["build_fn"](k=k)
            if sample_fn is None:
                from pln_thrml.beta import run_beta_sampling
                sample_fn = run_beta_sampling
            samples = sample_fn(graph)
            target = rule["target_fn"](graph)

            from pln_thrml.beta import estimate_beta_marginal
            posterior, s, c = estimate_beta_marginal(samples, graph, target, k=k)

            specs.append({
                "name": rule["name"],
                "posterior": np.array(posterior),
                "conclusion_stv": (float(s), float(c)),
                "premise_confidences": [p[1] for p in premises_stv],
                "cost": rule.get("cost", 1.0),
            })
        except Exception:
            # Rule not applicable with current premises — skip
            continue

    return specs
