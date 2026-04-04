"""Tests for geodesic_thrml.scores — forward/backward factors and weakness gauge."""

import numpy as np
import pytest

from geodesic_thrml.scores import (
    RuleSpec, compute_forward_scores, compute_backward_scores,
    compute_geodesic_energy, energy_to_probs, compute_rho,
    compute_weakness, partition_parallel_groups,
)


def _make_spec(name, s=0.7, c=0.8, cost=1.0, nodes=frozenset()):
    return RuleSpec(
        name=name,
        posterior=np.ones(16) / 16,
        conclusion_stv=(s, c),
        premise_confidences=[c],
        cost=cost,
        touched_nodes=nodes,
    )


class TestForwardScores:
    def test_higher_confidence_higher_score(self):
        specs = [_make_spec("low", c=0.3), _make_spec("high", c=0.9)]
        f = compute_forward_scores(specs)
        assert f[1] > f[0]

    def test_normalized(self):
        specs = [_make_spec("a", c=0.5), _make_spec("b", c=0.8)]
        f = compute_forward_scores(specs)
        assert abs(f.sum() - 1.0) < 1e-10

    def test_single_rule(self):
        specs = [_make_spec("only", c=0.9)]
        f = compute_forward_scores(specs)
        assert abs(f[0] - 1.0) < 1e-10


class TestBackwardScores:
    def test_goal_aligned_higher(self):
        specs = [_make_spec("far", s=0.2, c=0.3), _make_spec("near", s=0.8, c=0.9)]
        g = compute_backward_scores(specs, goal_stv=(0.8, 0.9))
        assert g[1] > g[0]

    def test_no_goal_uniform(self):
        specs = [_make_spec("a"), _make_spec("b")]
        g = compute_backward_scores(specs, goal_stv=None)
        np.testing.assert_allclose(g[0], g[1])

    def test_normalized(self):
        specs = [_make_spec("a", s=0.3), _make_spec("b", s=0.8)]
        g = compute_backward_scores(specs, goal_stv=(0.7, 0.8))
        assert abs(g.sum() - 1.0) < 1e-10


class TestGeodesicEnergy:
    def test_lower_energy_for_better_step(self):
        """High f, high g, low cost → low energy."""
        f = np.array([0.8, 0.2])
        g = np.array([0.7, 0.3])
        costs = np.array([1.0, 1.0])
        e = compute_geodesic_energy(f, g, costs)
        assert e[0] < e[1]

    def test_cost_penalty(self):
        """Higher cost → higher energy."""
        f = np.array([0.5, 0.5])
        g = np.array([0.5, 0.5])
        costs = np.array([1.0, 10.0])
        e = compute_geodesic_energy(f, g, costs, cost_weight=1.0)
        assert e[1] > e[0]

    def test_low_temperature_sharpens(self):
        """Low T makes energy differences more extreme."""
        f = np.array([0.8, 0.2])
        g = np.array([0.5, 0.5])
        costs = np.array([1.0, 1.0])
        e_high = compute_geodesic_energy(f, g, costs, temperature=10.0)
        e_low = compute_geodesic_energy(f, g, costs, temperature=0.01)
        # Difference should be larger at low T
        assert abs(e_low[0] - e_low[1]) > abs(e_high[0] - e_high[1])


class TestEnergyToProbs:
    def test_sums_to_one(self):
        energy = np.array([1.0, 2.0, 3.0])
        p = energy_to_probs(energy)
        assert abs(p.sum() - 1.0) < 1e-10

    def test_lower_energy_higher_prob(self):
        energy = np.array([0.5, 2.0])
        p = energy_to_probs(energy)
        assert p[0] > p[1]

    def test_low_t_concentrates(self):
        """Very low T → almost all probability on lowest energy."""
        energy = np.array([0.1, 1.0, 5.0])
        p = energy_to_probs(energy, temperature=0.001)
        assert p[0] > 0.99


class TestRho:
    def test_product(self):
        f = np.array([0.5, 0.3])
        g = np.array([0.4, 0.6])
        rho = compute_rho(f, g)
        np.testing.assert_allclose(rho, [0.2, 0.18])


class TestWeakness:
    def test_disjoint_zero(self):
        a = _make_spec("a", nodes=frozenset(["X", "Y"]))
        b = _make_spec("b", nodes=frozenset(["Z", "W"]))
        assert compute_weakness(a, b) == 0.0

    def test_identical_one(self):
        a = _make_spec("a", nodes=frozenset(["X", "Y"]))
        b = _make_spec("b", nodes=frozenset(["X", "Y"]))
        assert compute_weakness(a, b) == 1.0

    def test_partial_overlap(self):
        a = _make_spec("a", nodes=frozenset(["X", "Y"]))
        b = _make_spec("b", nodes=frozenset(["Y", "Z"]))
        # Jaccard: 1/3
        assert abs(compute_weakness(a, b) - 1 / 3) < 1e-10

    def test_empty_nodes_zero(self):
        a = _make_spec("a", nodes=frozenset())
        b = _make_spec("b", nodes=frozenset())
        assert compute_weakness(a, b) == 0.0


class TestPartitionParallelGroups:
    def test_all_disjoint(self):
        specs = [
            _make_spec("a", nodes=frozenset(["X"])),
            _make_spec("b", nodes=frozenset(["Y"])),
            _make_spec("c", nodes=frozenset(["Z"])),
        ]
        groups = partition_parallel_groups(specs)
        # All disjoint → all in one group
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_all_conflicting(self):
        specs = [
            _make_spec("a", nodes=frozenset(["X"])),
            _make_spec("b", nodes=frozenset(["X"])),
            _make_spec("c", nodes=frozenset(["X"])),
        ]
        groups = partition_parallel_groups(specs)
        # All conflict → each in own group
        assert len(groups) == 3

    def test_mixed(self):
        specs = [
            _make_spec("a", nodes=frozenset(["X"])),
            _make_spec("b", nodes=frozenset(["Y"])),
            _make_spec("c", nodes=frozenset(["X", "Y"])),  # conflicts with a and b
        ]
        groups = partition_parallel_groups(specs)
        # a and b can be parallel, c cannot be with either
        assert len(groups) == 2

    def test_empty(self):
        assert partition_parallel_groups([]) == []
