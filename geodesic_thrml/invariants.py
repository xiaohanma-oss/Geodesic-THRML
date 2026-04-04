"""
geodesic_thrml.invariants — Five-theorem runtime safety checks
================================================================

Implements runtime-checkable invariants from the evidence conservation
theorems paper.  Each check is O(1) or O(n) — lightweight enough to
run on every inference step without affecting sampling performance.

Theorem source: "Five Theorems on Evidence Conservation in
Quantale-Based Inference Control" (Goertzel 2026)

Usage:
    - Thm 2 (hallucination bound): check every step (cheapest)
    - Thm 3 (leakage bound): check when steps run in parallel
    - Thm 4 (evidence monotonicity): check at end of inference chain
    - Thm 1 (ρ conservation): check along multi-step paths
    - Thm 5 (entropy non-decrease): diagnostic logging only
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

from geodesic_thrml.capsules import EvidenceCapsule


@dataclass
class InvariantViolation:
    """Record of a violated invariant."""
    theorem: str
    message: str
    observed: float
    bound: float


def check_hallucination_bound(
    conclusion_strength: float,
    capsule: EvidenceCapsule,
    tolerance: float = 0.05,
) -> InvariantViolation | None:
    """Thm 4.2: conclusion strength ≤ capsule evidence mass.

    The cheapest safety check — just compares two numbers.
    A violation means the inference chain is generating confidence
    beyond what the input evidence supports (hallucination).

    Args:
        conclusion_strength: strength of the derived conclusion
        capsule: evidence capsule tracking provenance
        tolerance: slack for numerical noise

    Returns:
        InvariantViolation if bound is exceeded, None otherwise.
    """
    bound = capsule.mass
    if conclusion_strength > bound + tolerance:
        return InvariantViolation(
            theorem="Thm 2 (hallucination bound)",
            message=f"conclusion strength {conclusion_strength:.4f} > "
                    f"capsule mass {bound:.4f} + tolerance {tolerance}",
            observed=conclusion_strength,
            bound=bound,
        )
    return None


def check_evidence_monotonicity(
    evidence_pre: float,
    evidence_post: float,
    tolerance: float = 0.01,
) -> InvariantViolation | None:
    """Thm 6.1: total evidence is non-increasing under capsule-respecting inference.

    Check at the end of an inference chain.  A violation indicates
    evidence was injected without proper tracking, or double-counting
    occurred.

    Args:
        evidence_pre: total evidence mass before inference step(s)
        evidence_post: total evidence mass after inference step(s)
        tolerance: slack for numerical noise

    Returns:
        InvariantViolation if evidence increased, None otherwise.
    """
    if evidence_post > evidence_pre + tolerance:
        return InvariantViolation(
            theorem="Thm 4 (evidence monotonicity / DPI)",
            message=f"evidence increased: {evidence_pre:.4f} → {evidence_post:.4f}",
            observed=evidence_post,
            bound=evidence_pre,
        )
    return None


def check_leakage_bound(
    observed_leakage: float,
    pairwise_weaknesses: Sequence[float],
    tolerance: float = 0.01,
) -> InvariantViolation | None:
    """Thm 5.3: reordering leakage ≤ sum of pairwise weaknesses.

    Check when multiple inference steps are executed in parallel
    (reordered from the sequential geodesic path).  The weakness
    gauge δ(a,b) measures non-commutativity cost; total leakage
    is bounded by Σ δ for transposed pairs.

    Args:
        observed_leakage: measured evidence discrepancy from reordering
        pairwise_weaknesses: list of δ(step_i, step_j) for transposed pairs
        tolerance: slack for numerical noise

    Returns:
        InvariantViolation if bound is exceeded, None otherwise.
    """
    bound = sum(pairwise_weaknesses)
    if observed_leakage > bound + tolerance:
        return InvariantViolation(
            theorem="Thm 3 (weakness-bounded leakage)",
            message=f"leakage {observed_leakage:.4f} > "
                    f"weakness sum {bound:.4f} + tolerance {tolerance}",
            observed=observed_leakage,
            bound=bound,
        )
    return None


def check_rho_conservation(
    rho_sequence: Sequence[float],
    tolerance: float = 0.1,
) -> InvariantViolation | None:
    """Thm 3.1: ρ = f ⊗ g is constant along geodesic paths.

    At T=0 this is exact (Noether invariant); at finite temperature
    it is a soft constraint with deviations suppressed by e^{-ΔE/T}.

    Args:
        rho_sequence: ρ values along inference path
        tolerance: maximum allowed deviation from mean

    Returns:
        InvariantViolation if ρ varies too much, None otherwise.
    """
    if len(rho_sequence) < 2:
        return None
    rho_mean = sum(rho_sequence) / len(rho_sequence)
    max_dev = max(abs(r - rho_mean) for r in rho_sequence)
    if max_dev > tolerance:
        return InvariantViolation(
            theorem="Thm 1 (ρ conservation / Noether)",
            message=f"ρ deviation {max_dev:.4f} > tolerance {tolerance} "
                    f"(mean ρ = {rho_mean:.4f})",
            observed=max_dev,
            bound=tolerance,
        )
    return None


def check_entropy_non_decrease(
    entropy_pre: float,
    entropy_post: float,
    tolerance: float = 0.01,
) -> InvariantViolation | None:
    """Thm 7.10: join-collision entropy is non-decreasing.

    This is a diagnostic-level check — violations are logged as
    warnings rather than errors, since the quantale entropy is an
    approximation in discrete settings.

    Args:
        entropy_pre: entropy before inference step
        entropy_post: entropy after inference step
        tolerance: slack for discretization noise

    Returns:
        InvariantViolation if entropy decreased significantly, None otherwise.
    """
    if entropy_post < entropy_pre - tolerance:
        violation = InvariantViolation(
            theorem="Thm 5 (entropy non-decrease)",
            message=f"entropy decreased: {entropy_pre:.4f} → {entropy_post:.4f}",
            observed=entropy_post,
            bound=entropy_pre,
        )
        warnings.warn(f"[diagnostic] {violation.message}", stacklevel=2)
        return violation
    return None


def run_all_checks(
    conclusion_strength: float | None = None,
    capsule: EvidenceCapsule | None = None,
    evidence_pre: float | None = None,
    evidence_post: float | None = None,
    rho_sequence: Sequence[float] | None = None,
) -> list[InvariantViolation]:
    """Run all applicable invariant checks and return violations.

    Only checks with sufficient arguments are run.
    """
    violations = []

    if conclusion_strength is not None and capsule is not None:
        v = check_hallucination_bound(conclusion_strength, capsule)
        if v:
            violations.append(v)

    if evidence_pre is not None and evidence_post is not None:
        v = check_evidence_monotonicity(evidence_pre, evidence_post)
        if v:
            violations.append(v)

    if rho_sequence is not None:
        v = check_rho_conservation(rho_sequence)
        if v:
            violations.append(v)

    return violations
