"""
geodesic_thrml.bridges.pln — PLN-THRML bridge
===============================================

Translates PLN-THRML's inference results into RuleSpec for the
geodesic controller.  Does NOT re-do PLN inference — PLN-THRML
already builds factor graphs, runs THRML Gibbs sampling, and
returns (posterior, strength, confidence).  This bridge simply
wraps those results as RuleSpec.

References:
    - pln_thrml.beta: sample_and_measure() → (posterior, s, c)
    - Hyperon whitepaper §6.1: PLN + geodesic control
"""

from __future__ import annotations

import numpy as np

from geodesic_thrml.scores import RuleSpec


def pln_result_to_rule_spec(
    name: str,
    posterior: np.ndarray,
    strength: float,
    confidence: float,
    premise_confidences: list[float],
    cost: float = 1.0,
    touched_nodes: frozenset[str] = frozenset(),
) -> RuleSpec:
    """Wrap a PLN-THRML sampling result as RuleSpec.

    PLN-THRML's sample_and_measure() (or estimate_beta_marginal())
    already returns a proper posterior from THRML Gibbs sampling.
    This function packages it for the geodesic controller — no
    re-sampling needed.

    Args:
        name: rule identifier (e.g., "modus_ponens")
        posterior: K-bin posterior histogram from PLN-THRML sampling
        strength: posterior mean (from PLN-THRML moment matching)
        confidence: posterior concentration (from PLN-THRML)
        premise_confidences: confidences of input premises
        cost: computational cost of applying this rule
        touched_nodes: graph nodes this rule reads/writes

    Returns:
        RuleSpec ready for controller.select_step_thrml().
    """
    return RuleSpec(
        name=name,
        posterior=np.asarray(posterior),
        conclusion_stv=(strength, confidence),
        premise_confidences=list(premise_confidences),
        cost=cost,
        touched_nodes=touched_nodes,
    )


def pln_results_to_rule_specs(
    results: list[dict],
) -> list[RuleSpec]:
    """Batch-convert PLN-THRML results to RuleSpec list.

    Convenience wrapper for multiple rules.  Each dict should have:
        name, posterior, strength, confidence, premise_confidences,
        and optionally cost, touched_nodes.

    Args:
        results: list of dicts from PLN-THRML inference

    Returns:
        List of RuleSpec for controller.select_step_thrml().
    """
    return [
        pln_result_to_rule_spec(
            name=r["name"],
            posterior=r["posterior"],
            strength=r["strength"],
            confidence=r["confidence"],
            premise_confidences=r["premise_confidences"],
            cost=r.get("cost", 1.0),
            touched_nodes=r.get("touched_nodes", frozenset()),
        )
        for r in results
    ]
