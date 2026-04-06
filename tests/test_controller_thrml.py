"""Tests for geodesic_thrml.controller_thrml — THRML step selection."""

import numpy as np
import pytest

from geodesic_thrml.scores import RuleSpec
from geodesic_thrml.controller_thrml import (
    select_step_thrml,
    select_batch_thrml,
    multi_step_select_thrml,
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
#  select_step_thrml
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectStepThrml:
    def test_single_rule(self):
        specs = [_make_spec("only")]
        result = select_step_thrml(specs)
        assert result.selected_idx == 0
        assert result.selected_name == "only"

    def test_high_confidence_preferred(self):
        specs = [_make_spec("weak", c=0.1), _make_spec("strong", c=0.95)]
        result = select_step_thrml(specs, temperature=0.01, seed=42)
        assert result.selected_name == "strong"

    def test_goal_aligned_preferred(self):
        specs = [_make_spec("far", s=0.1), _make_spec("near", s=0.85)]
        result = select_step_thrml(
            specs, goal_stv=(0.85, 0.85), temperature=0.01, seed=42)
        assert result.selected_name == "near"

    def test_returns_valid_diagnostics(self):
        specs = [_make_spec("a"), _make_spec("b")]
        result = select_step_thrml(specs)
        assert result.f_scores.shape == (2,)
        assert result.g_scores.shape == (2,)
        assert result.energy.shape == (2,)
        assert result.rho.shape == (2,)
        assert abs(result.rule_probs.sum() - 1.0) < 1e-10

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_step_thrml([])


# ═══════════════════════════════════════════════════════════════════════════
#  select_batch_thrml
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectBatchThrml:
    def test_disjoint_all_parallel(self):
        specs = [
            _make_spec("a", nodes=frozenset(["X"])),
            _make_spec("b", nodes=frozenset(["Y"])),
            _make_spec("c", nodes=frozenset(["Z"])),
        ]
        result = select_batch_thrml(specs, temperature=0.01)
        assert len(result.parallel_group) == 3

    def test_conflicting_single(self):
        specs = [
            _make_spec("a", c=0.9, nodes=frozenset(["X"])),
            _make_spec("b", c=0.1, nodes=frozenset(["X"])),
        ]
        result = select_batch_thrml(specs, temperature=0.01)
        assert len(result.parallel_group) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  multi_step_select_thrml
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiStepSelectThrml:
    def test_returns_correct_count(self):
        specs = [_make_spec("a", c=0.3), _make_spec("b", c=0.9)]
        results = multi_step_select_thrml(specs, n_steps=3)
        assert len(results) == 3

    def test_converges_to_best(self):
        specs = [_make_spec("bad", c=0.1), _make_spec("good", c=0.95)]
        results = multi_step_select_thrml(
            specs, n_steps=5, t_start=5.0, t_end=0.001)
        assert results[-1].rule_probs[1] > 0.9


class TestDiagnosticsValidity:
    def test_energy_and_probs_consistent(self):
        """Energy and rule_probs should be consistent Boltzmann."""
        specs = [_make_spec("a", c=0.3), _make_spec("b", c=0.9)]
        result = select_step_thrml(specs, goal_stv=(0.8, 0.8), temperature=1.0, seed=42)
        # Lower energy → higher probability
        if result.energy[0] < result.energy[1]:
            assert result.rule_probs[0] > result.rule_probs[1]
        else:
            assert result.rule_probs[1] >= result.rule_probs[0]
