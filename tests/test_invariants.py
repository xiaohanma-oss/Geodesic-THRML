"""Tests for geodesic_thrml.invariants — Five-theorem runtime safety checks."""

import warnings

from geodesic_thrml.capsules import EvidenceCapsule, make_capsule
from geodesic_thrml.invariants import (
    check_hallucination_bound,
    check_evidence_monotonicity,
    check_leakage_bound,
    check_rho_conservation,
    check_entropy_non_decrease,
    run_all_checks,
)


class TestHallucinationBound:
    """Thm 4.2: conclusion strength ≤ capsule evidence mass."""

    def test_within_bound(self):
        capsule = make_capsule("obs1", weight=0.8)
        assert check_hallucination_bound(0.7, capsule) is None

    def test_at_bound(self):
        capsule = make_capsule("obs1", weight=0.8)
        assert check_hallucination_bound(0.8, capsule) is None

    def test_violation(self):
        capsule = make_capsule("obs1", weight=0.3)
        v = check_hallucination_bound(0.9, capsule)
        assert v is not None
        assert "hallucination" in v.theorem.lower() or "Thm 2" in v.theorem

    def test_tolerance(self):
        capsule = make_capsule("obs1", weight=0.8)
        # 0.84 is within tolerance of 0.05
        assert check_hallucination_bound(0.84, capsule, tolerance=0.05) is None
        # 0.86 exceeds
        assert check_hallucination_bound(0.86, capsule, tolerance=0.05) is not None

    def test_multi_source_capsule(self):
        capsule = EvidenceCapsule(
            frozenset(["a", "b", "c"]),
            {"a": 0.3, "b": 0.3, "c": 0.3},
        )
        # mass = 0.9, so strength 0.8 is ok
        assert check_hallucination_bound(0.8, capsule) is None


class TestEvidenceMonotonicity:
    """Thm 6.1: total evidence non-increasing."""

    def test_decreasing_ok(self):
        assert check_evidence_monotonicity(1.0, 0.8) is None

    def test_constant_ok(self):
        assert check_evidence_monotonicity(1.0, 1.0) is None

    def test_increasing_violation(self):
        v = check_evidence_monotonicity(0.5, 0.8)
        assert v is not None
        assert "monotonicity" in v.theorem.lower() or "DPI" in v.theorem


class TestLeakageBound:
    """Thm 5.3: reordering leakage ≤ sum of pairwise weaknesses."""

    def test_within_bound(self):
        assert check_leakage_bound(0.05, [0.03, 0.04]) is None

    def test_violation(self):
        v = check_leakage_bound(0.5, [0.01, 0.02])
        assert v is not None
        assert "leakage" in v.theorem.lower() or "weakness" in v.theorem.lower()

    def test_zero_weakness_zero_leakage(self):
        """Commutative steps: zero weakness, zero leakage."""
        assert check_leakage_bound(0.0, [0.0, 0.0]) is None


class TestRhoConservation:
    """Thm 3.1: ρ = f ⊗ g constant along geodesic."""

    def test_constant_rho(self):
        assert check_rho_conservation([1.0, 1.0, 1.0]) is None

    def test_small_variation(self):
        assert check_rho_conservation([1.0, 1.05, 0.98], tolerance=0.1) is None

    def test_large_variation(self):
        v = check_rho_conservation([1.0, 0.5, 1.5], tolerance=0.1)
        assert v is not None

    def test_single_element(self):
        """Single ρ value: nothing to check."""
        assert check_rho_conservation([1.0]) is None


class TestEntropyNonDecrease:
    """Thm 7.10: join-collision entropy non-decreasing."""

    def test_increasing_ok(self):
        assert check_entropy_non_decrease(1.0, 1.5) is None

    def test_constant_ok(self):
        assert check_entropy_non_decrease(1.0, 1.0) is None

    def test_decrease_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = check_entropy_non_decrease(1.0, 0.5)
            assert v is not None
            assert len(w) == 1
            assert "diagnostic" in str(w[0].message).lower()


class TestRunAllChecks:
    def test_no_violations(self):
        capsule = make_capsule("obs1", weight=1.0)
        violations = run_all_checks(
            conclusion_strength=0.7,
            capsule=capsule,
            evidence_pre=1.0,
            evidence_post=0.9,
            rho_sequence=[1.0, 1.01, 0.99],
        )
        assert len(violations) == 0

    def test_multiple_violations(self):
        capsule = make_capsule("obs1", weight=0.3)
        violations = run_all_checks(
            conclusion_strength=0.9,  # violates hallucination bound
            capsule=capsule,
            evidence_pre=0.5,
            evidence_post=0.8,  # violates monotonicity
            rho_sequence=[1.0, 0.1, 2.0],  # violates ρ conservation
        )
        assert len(violations) == 3

    def test_partial_args(self):
        """Only runs checks for which args are provided."""
        violations = run_all_checks(rho_sequence=[1.0, 1.0, 1.0])
        assert len(violations) == 0
