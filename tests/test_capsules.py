"""Tests for geodesic_thrml.capsules — Evidence capsule provenance tracking."""

from geodesic_thrml.capsules import (
    EvidenceCapsule, make_capsule, merge_capsules,
    overlap_ratio, double_counting_penalty,
)


class TestEvidenceCapsule:
    def test_mass_single(self):
        c = make_capsule("obs1", weight=2.5)
        assert c.mass == 2.5

    def test_mass_default_weight(self):
        c = make_capsule("obs1")
        assert c.mass == 1.0

    def test_mass_empty(self):
        c = EvidenceCapsule()
        assert c.mass == 0.0

    def test_size(self):
        c = EvidenceCapsule(sources=frozenset(["a", "b", "c"]))
        assert c.size == 3

    def test_frozen(self):
        """Capsules should be immutable (frozen dataclass)."""
        c = make_capsule("obs1")
        try:
            c.sources = frozenset(["hacked"])
            assert False, "Should not allow mutation"
        except AttributeError:
            pass


class TestMergeCapsules:
    def test_idempotent(self):
        """merge(A, A) = A."""
        a = make_capsule("obs1", 2.0)
        merged = merge_capsules(a, a)
        assert merged.sources == a.sources
        assert merged.mass == a.mass

    def test_commutative(self):
        """merge(A, B) = merge(B, A)."""
        a = make_capsule("obs1", 1.0)
        b = make_capsule("obs2", 2.0)
        ab = merge_capsules(a, b)
        ba = merge_capsules(b, a)
        assert ab.sources == ba.sources
        assert ab.mass == ba.mass

    def test_overlap_counted_once(self):
        """Shared sources in merge(A∪B, B∪C) — B counted once."""
        ab = EvidenceCapsule(frozenset(["a", "b"]), {"a": 1.0, "b": 1.0})
        bc = EvidenceCapsule(frozenset(["b", "c"]), {"b": 1.0, "c": 1.0})
        merged = merge_capsules(ab, bc)
        assert merged.sources == frozenset(["a", "b", "c"])
        assert merged.mass == 3.0  # a + b + c, not a + b + b + c

    def test_disjoint_merge(self):
        a = make_capsule("obs1", 1.0)
        b = make_capsule("obs2", 2.0)
        merged = merge_capsules(a, b)
        assert merged.mass == 3.0
        assert merged.size == 2

    def test_weight_max_on_overlap(self):
        """Overlapping sources take the max weight."""
        a = EvidenceCapsule(frozenset(["x"]), {"x": 1.0})
        b = EvidenceCapsule(frozenset(["x"]), {"x": 3.0})
        merged = merge_capsules(a, b)
        assert merged.weights["x"] == 3.0


class TestOverlapRatio:
    def test_identical(self):
        a = EvidenceCapsule(frozenset(["x", "y"]))
        assert overlap_ratio(a, a) == 1.0

    def test_disjoint(self):
        a = EvidenceCapsule(frozenset(["x"]))
        b = EvidenceCapsule(frozenset(["y"]))
        assert overlap_ratio(a, b) == 0.0

    def test_partial(self):
        a = EvidenceCapsule(frozenset(["x", "y"]))
        b = EvidenceCapsule(frozenset(["y", "z"]))
        assert abs(overlap_ratio(a, b) - 1 / 3) < 1e-10

    def test_empty(self):
        a = EvidenceCapsule()
        b = make_capsule("x")
        assert overlap_ratio(a, b) == 0.0


class TestDoubleCountingPenalty:
    def test_no_overlap(self):
        caps = [make_capsule("a"), make_capsule("b"), make_capsule("c")]
        assert double_counting_penalty(caps) == 0.0

    def test_full_overlap(self):
        a = make_capsule("x", 2.0)
        penalty = double_counting_penalty([a, a, a])
        # naive sum = 6.0, merged mass = 2.0, penalty = 4.0
        assert abs(penalty - 4.0) < 1e-10

    def test_partial_overlap(self):
        ab = EvidenceCapsule(frozenset(["a", "b"]), {"a": 1.0, "b": 1.0})
        bc = EvidenceCapsule(frozenset(["b", "c"]), {"b": 1.0, "c": 1.0})
        # naive sum = 4.0, merged mass = 3.0, penalty = 1.0
        assert abs(double_counting_penalty([ab, bc]) - 1.0) < 1e-10

    def test_empty(self):
        assert double_counting_penalty([]) == 0.0
