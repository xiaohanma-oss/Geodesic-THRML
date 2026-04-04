"""Tests for geodesic_thrml.controller — step selection and annealing."""

import numpy as np
import pytest

from geodesic_thrml.scores import RuleSpec
from geodesic_thrml.controller import (
    select_step, select_batch, annealing_schedule, multi_step_select,
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


# ═══════════════════════════════════════════════════════════════════════════
#  select_step
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectStep:
    def test_single_rule(self):
        specs = [_make_spec("only")]
        result = select_step(specs)
        assert result.selected_idx == 0
        assert result.selected_name == "only"
        assert abs(result.rule_probs.sum() - 1.0) < 1e-10

    def test_high_confidence_preferred(self):
        """High-confidence rule should have higher forward score."""
        specs = [_make_spec("weak", c=0.2), _make_spec("strong", c=0.9)]
        result = select_step(specs, temperature=0.001)
        assert result.selected_name == "strong"

    def test_goal_aligned_preferred(self):
        """Goal-aligned rule should be selected with backward scoring."""
        specs = [_make_spec("far", s=0.2), _make_spec("near", s=0.8)]
        result = select_step(specs, goal_stv=(0.8, 0.8), temperature=0.001)
        assert result.selected_name == "near"

    def test_low_cost_preferred(self):
        """Lower cost should be preferred, all else equal."""
        specs = [
            _make_spec("cheap", s=0.7, c=0.8, cost=0.1),
            _make_spec("expensive", s=0.7, c=0.8, cost=10.0),
        ]
        result = select_step(specs, temperature=0.001, cost_weight=1.0)
        assert result.selected_name == "cheap"

    def test_rho_computed(self):
        specs = [_make_spec("a"), _make_spec("b")]
        result = select_step(specs)
        assert result.rho.shape == (2,)
        assert np.all(result.rho >= 0)

    def test_t_zero_deterministic(self):
        """At T→0, selection should be deterministic."""
        specs = [_make_spec("bad", c=0.1), _make_spec("good", c=0.9)]
        results = [select_step(specs, temperature=1e-6, seed=i)
                   for i in range(5)]
        names = [r.selected_name for r in results]
        assert all(n == names[0] for n in names)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_step([])

    def test_nonzero_probs(self):
        """All rules should have nonzero probability at finite T."""
        specs = [_make_spec("a", c=0.1), _make_spec("b", c=0.9)]
        result = select_step(specs, temperature=1.0)
        assert np.all(result.rule_probs > 0)


# ═══════════════════════════════════════════════════════════════════════════
#  select_batch
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectBatch:
    def test_disjoint_all_parallel(self):
        """Disjoint rules → all in one parallel group."""
        specs = [
            _make_spec("a", nodes=frozenset(["X"])),
            _make_spec("b", nodes=frozenset(["Y"])),
            _make_spec("c", nodes=frozenset(["Z"])),
        ]
        result = select_batch(specs, temperature=0.001)
        assert len(result.parallel_group) == 3

    def test_conflicting_single(self):
        """All-conflicting rules → primary alone."""
        specs = [
            _make_spec("a", c=0.9, nodes=frozenset(["X"])),
            _make_spec("b", c=0.1, nodes=frozenset(["X"])),
        ]
        result = select_batch(specs, temperature=0.001)
        assert len(result.parallel_group) == 1

    def test_primary_in_group(self):
        """Primary selection should always be in its parallel group."""
        specs = [_make_spec("a"), _make_spec("b")]
        result = select_batch(specs, temperature=1.0, seed=42)
        assert result.primary.selected_idx in result.parallel_group


# ═══════════════════════════════════════════════════════════════════════════
#  annealing_schedule
# ═══════════════════════════════════════════════════════════════════════════

class TestAnnealingSchedule:
    def test_exponential_endpoints(self):
        schedule = annealing_schedule(10.0, 0.01, 5, "exponential")
        assert len(schedule) == 5
        assert abs(schedule[0] - 10.0) < 1e-6
        assert abs(schedule[-1] - 0.01) < 1e-3

    def test_linear_endpoints(self):
        schedule = annealing_schedule(10.0, 0.0, 5, "linear")
        assert abs(schedule[0] - 10.0) < 1e-6
        assert abs(schedule[-1] - 0.0) < 1e-6

    def test_cosine_endpoints(self):
        schedule = annealing_schedule(10.0, 0.01, 5, "cosine")
        assert abs(schedule[0] - 10.0) < 1e-3
        assert abs(schedule[-1] - 0.01) < 1e-3

    def test_monotonically_decreasing(self):
        for strat in ["exponential", "linear", "cosine"]:
            schedule = annealing_schedule(10.0, 0.01, 10, strat)
            for i in range(len(schedule) - 1):
                assert schedule[i] >= schedule[i + 1] - 1e-10, \
                    f"{strat}: not monotonically decreasing at step {i}"

    def test_single_step(self):
        schedule = annealing_schedule(5.0, 0.1, 1)
        assert schedule == [5.0]

    def test_empty(self):
        assert annealing_schedule(5.0, 0.1, 0) == []

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            annealing_schedule(1.0, 0.1, 5, "invalid")


# ═══════════════════════════════════════════════════════════════════════════
#  multi_step_select
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiStepSelect:
    def test_returns_correct_count(self):
        specs = [_make_spec("a", c=0.3), _make_spec("b", c=0.9)]
        results = multi_step_select(specs, n_steps=5)
        assert len(results) == 5

    def test_converges_to_best(self):
        """At low final T, last step should select the best rule."""
        specs = [_make_spec("bad", c=0.1), _make_spec("good", c=0.9)]
        results = multi_step_select(
            specs, n_steps=10, t_start=5.0, t_end=0.001,
        )
        # Last result should strongly prefer "good"
        assert results[-1].rule_probs[1] > 0.9

    def test_early_steps_explore(self):
        """At high initial T, early steps should have more uniform probs."""
        specs = [_make_spec("a", c=0.3), _make_spec("b", c=0.9)]
        results = multi_step_select(
            specs, n_steps=10, t_start=100.0, t_end=0.001,
        )
        # Early step should be more uniform than late step
        early_entropy = -np.sum(results[0].rule_probs *
                                np.log(results[0].rule_probs + 1e-30))
        late_entropy = -np.sum(results[-1].rule_probs *
                               np.log(results[-1].rule_probs + 1e-30))
        assert early_entropy > late_entropy


# ═══════════════════════════════════════════════════════════════════════════
#  Pareto dominance (T→0 convergence)
# ═══════════════════════════════════════════════════════════════════════════

class TestParetoDominance:
    def test_pareto_optimal_selected(self):
        """At T→0, the Pareto-optimal step (max f·g/cost) should win."""
        specs = [
            _make_spec("dominated", s=0.3, c=0.3, cost=5.0),
            _make_spec("pareto", s=0.8, c=0.9, cost=1.0),
        ]
        result = select_step(specs, goal_stv=(0.8, 0.9), temperature=1e-6)
        assert result.selected_name == "pareto"

    def test_rho_conservation_approximate(self):
        """ρ values across multiple annealing steps should be consistent."""
        specs = [_make_spec("a", c=0.5), _make_spec("b", c=0.8)]
        results = multi_step_select(specs, n_steps=5, t_start=1.0, t_end=0.1)
        # ρ is computed from f and g which are fixed → should be identical
        for r in results:
            np.testing.assert_allclose(r.rho, results[0].rho)
