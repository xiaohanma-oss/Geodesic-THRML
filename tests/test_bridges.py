"""Tests for geodesic_thrml.bridges — sub-project bridge adapters."""

import numpy as np
import pytest

from geodesic_thrml.bridges.quantimork import (
    WaveletLevelSpec, wavelet_to_resolution_ladder, estimate_level_cost,
)
from geodesic_thrml.bridges.ecan import (
    extract_ecan_snapshot, sti_to_forward_scores,
    extract_hjb_factors,
)
from geodesic_thrml.bridges.moses import (
    DemeSpec, compute_forward_reachability, compute_backward_compatibility,
    estimate_deme_cost,
)


# ═══════════════════════════════════════════════════════════════════════════
#  QuantiMORK bridge
# ═══════════════════════════════════════════════════════════════════════════

class TestWaveletToResolutionLadder:
    def test_three_levels(self):
        specs = [
            WaveletLevelSpec(level=3, band="detail", dim=64, weight_shape=(64, 64), max_connections=5),
            WaveletLevelSpec(level=3, band="approx", dim=64, weight_shape=(64, 64), max_connections=5),
            WaveletLevelSpec(level=2, band="detail", dim=128, weight_shape=(128, 128), max_connections=5),
            WaveletLevelSpec(level=1, band="detail", dim=256, weight_shape=(256, 256), max_connections=5),
        ]
        ladder = wavelet_to_resolution_ladder(specs)
        # Should be coarse → fine
        ks = [l.k for l in ladder]
        assert ks == [64, 128, 256]

    def test_labels_contain_level(self):
        specs = [
            WaveletLevelSpec(level=2, band="detail", dim=128, weight_shape=(128, 128), max_connections=5),
            WaveletLevelSpec(level=1, band="detail", dim=256, weight_shape=(256, 256), max_connections=5),
        ]
        ladder = wavelet_to_resolution_ladder(specs)
        assert "L2" in ladder[0].label
        assert "L1" in ladder[1].label

    def test_empty_specs(self):
        assert wavelet_to_resolution_ladder([]) == []


class TestEstimateLevelCost:
    def test_proportional_to_dim(self):
        small = WaveletLevelSpec(level=3, band="detail", dim=64, weight_shape=(64, 64), max_connections=5)
        large = WaveletLevelSpec(level=1, band="detail", dim=256, weight_shape=(256, 256), max_connections=5)
        assert estimate_level_cost(large) > estimate_level_cost(small)


# ═══════════════════════════════════════════════════════════════════════════
#  ECAN bridge
# ═══════════════════════════════════════════════════════════════════════════

class TestECANSnapshot:
    def test_extract(self):
        snap = extract_ecan_snapshot(
            sti_values=[0.3, 0.5, 0.2],
            atom_ids=["A", "B", "C"],
        )
        assert snap.total_sti == pytest.approx(1.0)
        assert snap.lti_values is None
        assert len(snap.atom_ids) == 3

    def test_with_lti(self):
        snap = extract_ecan_snapshot(
            sti_values=[0.5, 0.5],
            atom_ids=["X", "Y"],
            lti_values=[0.8, 0.2],
        )
        assert snap.lti_values is not None
        np.testing.assert_allclose(snap.lti_values, [0.8, 0.2])


class TestSTIToForwardScores:
    def test_normalized(self):
        snap = extract_ecan_snapshot([0.3, 0.5, 0.2], ["A", "B", "C"])
        scores = sti_to_forward_scores(snap)
        assert abs(scores.sum() - 1.0) < 1e-10

    def test_higher_sti_higher_score(self):
        snap = extract_ecan_snapshot([0.1, 0.9], ["low", "high"])
        scores = sti_to_forward_scores(snap)
        assert scores[1] > scores[0]

    def test_zero_sti_uniform(self):
        snap = extract_ecan_snapshot([0.0, 0.0, 0.0], ["A", "B", "C"])
        scores = sti_to_forward_scores(snap)
        np.testing.assert_allclose(scores, [1/3, 1/3, 1/3])


# ═══════════════════════════════════════════════════════════════════════════
#  ECAN bridge — HJB mode (whitepaper §5.3 aligned)
# ═══════════════════════════════════════════════════════════════════════════

class TestHJBFactors:
    """HJB value function → geodesic forward/backward factors."""

    def test_goal_has_highest_g(self):
        """V=0 at goal → g highest at goal position."""
        # 5 atoms in a ring, goal at position 2 (V=0, others V=high)
        V = np.array([5.0, 3.0, 0.0, 3.0, 5.0])
        ids = ["a0", "a1", "goal", "a3", "a4"]
        factors = extract_hjb_factors(V, ids)
        assert np.argmax(factors.g_scores) == 2  # goal position

    def test_gradient_highest_near_goal(self):
        """|∇V| largest near goal (steep descent toward V=0)."""
        V = np.array([5.0, 3.0, 0.0, 3.0, 5.0])
        ids = ["a0", "a1", "goal", "a3", "a4"]
        factors = extract_hjb_factors(V, ids)
        # Neighbors of goal (positions 1 and 3) should have high f
        assert factors.f_scores[1] > factors.f_scores[0]
        assert factors.f_scores[3] > factors.f_scores[4]

    def test_flat_v_uniform_f(self):
        """Flat V → zero gradient → approximately uniform f."""
        V = np.array([1.0, 1.0, 1.0, 1.0])
        ids = ["a", "b", "c", "d"]
        factors = extract_hjb_factors(V, ids)
        np.testing.assert_allclose(factors.f_scores, [0.25] * 4, atol=0.01)

    def test_normalized(self):
        V = np.array([5.0, 2.0, 0.5, 2.0, 5.0])
        ids = [f"a{i}" for i in range(5)]
        factors = extract_hjb_factors(V, ids)
        assert abs(factors.f_scores.sum() - 1.0) < 1e-10
        assert abs(factors.g_scores.sum() - 1.0) < 1e-10

    def test_2d_grid_flattened(self):
        """2D V field [1, N] should be handled correctly."""
        V = np.array([[5.0, 2.0, 0.0, 2.0, 5.0]])  # shape [1, 5]
        ids = [f"a{i}" for i in range(5)]
        factors = extract_hjb_factors(V, ids)
        assert np.argmax(factors.g_scores) == 2

    def test_mismatched_length_raises(self):
        V = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="atom_ids"):
            extract_hjb_factors(V, ["a", "b"])



# ═══════════════════════════════════════════════════════════════════════════
#  MOSES bridge (GEO-EVO)
# ═══════════════════════════════════════════════════════════════════════════

def _make_deme_spec(deme_id, n_knobs=10, fitness=0.8, acceptance=0.4, done=True):
    return DemeSpec(
        deme_id=deme_id, n_knobs=n_knobs, best_fitness=fitness,
        acceptance_rate=acceptance, is_done=done,
        best_bits=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8),
    )


class TestForwardReachability:
    """GEO-EVO forward factor: 'can we get there from here?'"""

    def test_higher_fitness_higher_score(self):
        specs = [_make_deme_spec("bad", fitness=0.2), _make_deme_spec("good", fitness=0.9)]
        f = compute_forward_reachability(specs)
        assert f[1] > f[0]

    def test_normalized(self):
        specs = [_make_deme_spec("a", fitness=0.5), _make_deme_spec("b", fitness=0.8)]
        f = compute_forward_reachability(specs)
        assert abs(f.sum() - 1.0) < 1e-10

    def test_equal_fitness_uniform(self):
        specs = [_make_deme_spec("a", fitness=0.5), _make_deme_spec("b", fitness=0.5)]
        f = compute_forward_reachability(specs)
        np.testing.assert_allclose(f[0], f[1])


class TestBackwardCompatibility:
    """GEO-EVO backward factor: 'is it useful for our goals?'"""

    def test_with_target_fn(self):
        specs = [_make_deme_spec("a"), _make_deme_spec("b")]
        # Target function: prefer demes where first knob is 1
        g = compute_backward_compatibility(specs, lambda bits: float(bits[0]))
        assert g.shape == (2,)
        assert abs(g.sum() - 1.0) < 1e-10

    def test_proxy_healthy_acceptance(self):
        """Without target fn, healthy acceptance rate (0.4) → higher score."""
        specs = [
            _make_deme_spec("stuck", acceptance=0.01),
            _make_deme_spec("healthy", acceptance=0.4),
            _make_deme_spec("random", acceptance=0.95),
        ]
        g = compute_backward_compatibility(specs)
        assert g[1] > g[0]  # healthy > stuck
        assert g[1] > g[2]  # healthy > random walk



class TestEstimateDemeCost:
    def test_proportional(self):
        assert estimate_deme_cost(20, 50) > estimate_deme_cost(10, 50)


# ═══════════════════════════════════════════════════════════════════════════
#  THRML bridge tests — proper posteriors (not uniform placeholders)
# ═══════════════════════════════════════════════════════════════════════════

class TestHJBToRuleSpecsThrml:
    """ECAN bridge: HJB V → THRML-sampled posteriors."""

    def test_posterior_not_uniform(self):
        """THRML posterior should NOT be uniform (unlike placeholder)."""
        from geodesic_thrml.bridges.ecan import hjb_to_rule_specs_thrml
        V = np.array([5.0, 2.0, 0.0, 2.0, 5.0])
        ids = ["a0", "a1", "goal", "a3", "a4"]
        specs = hjb_to_rule_specs_thrml(V, ids, k=16, n_batches=10, seed=42)
        # Goal atom (V=0, highest g) should have non-uniform posterior
        goal_spec = specs[2]
        assert goal_spec.posterior.shape == (16,)
        assert abs(goal_spec.posterior.sum() - 1.0) < 1e-5
        # Not uniform: std should be significantly above uniform std
        uniform_std = np.std(np.ones(16) / 16)
        assert np.std(goal_spec.posterior) > uniform_std * 2

    def test_goal_atom_posterior_peaked_higher_than_far(self):
        """Goal atom (highest g) should have posterior peaked at higher bin than far atom."""
        from geodesic_thrml.bridges.ecan import hjb_to_rule_specs_thrml
        V = np.array([5.0, 2.0, 0.0, 2.0, 5.0])
        ids = ["a0", "a1", "goal", "a3", "a4"]
        specs = hjb_to_rule_specs_thrml(V, ids, k=16, n_batches=10, seed=42)
        goal_peak = np.argmax(specs[2].posterior)   # V=0, highest g
        far_peak = np.argmax(specs[0].posterior)     # V=5, lowest g
        assert goal_peak > far_peak

    def test_produces_correct_count(self):
        from geodesic_thrml.bridges.ecan import hjb_to_rule_specs_thrml
        V = np.array([3.0, 0.0, 3.0])
        ids = ["a", "goal", "b"]
        specs = hjb_to_rule_specs_thrml(V, ids, k=8)
        assert len(specs) == 3
        assert specs[1].name == "goal"

    def test_consumable_by_controller(self):
        """RuleSpecs should work with select_step_thrml."""
        from geodesic_thrml.bridges.ecan import hjb_to_rule_specs_thrml
        from geodesic_thrml.controller_thrml import select_step_thrml
        from geodesic_thrml.scores import RuleSpec

        V = np.array([5.0, 2.0, 0.0, 2.0, 5.0])
        ids = ["a0", "a1", "goal", "a3", "a4"]
        specs_raw = hjb_to_rule_specs_thrml(V, ids, k=16, n_batches=5)
        # Convert to RuleSpec (they already are, but verify interface)
        rule_specs = [
            RuleSpec(
                name=s.name, posterior=s.posterior,
                conclusion_stv=s.conclusion_stv,
                premise_confidences=s.premise_confidences,
                cost=s.cost, touched_nodes=s.touched_nodes,
            ) for s in specs_raw
        ]
        result = select_step_thrml(rule_specs, temperature=0.01, seed=42)
        assert result.selected_name in ids


class TestDemeSpecsToRuleSpecsThrml:
    """MOSES bridge: deme fitness → THRML-sampled posteriors."""

    def test_posterior_not_uniform(self):
        """THRML posterior should NOT be uniform."""
        from geodesic_thrml.bridges.moses import deme_specs_to_rule_specs_thrml
        demes = [_make_deme_spec("good", fitness=0.9, acceptance=0.4),
                 _make_deme_spec("bad", fitness=0.1, acceptance=0.01)]
        specs = deme_specs_to_rule_specs_thrml(demes, k=16, n_batches=10, seed=42)
        for spec in specs:
            assert spec.posterior.shape == (16,)
            assert abs(spec.posterior.sum() - 1.0) < 1e-5
            uniform_std = np.std(np.ones(16) / 16)
            assert np.std(spec.posterior) > uniform_std * 2

    def test_good_deme_peaked_higher(self):
        """Better deme (higher fitness + healthy acceptance) → posterior peaked higher."""
        from geodesic_thrml.bridges.moses import deme_specs_to_rule_specs_thrml
        demes = [_make_deme_spec("good", fitness=0.9, acceptance=0.4),
                 _make_deme_spec("bad", fitness=0.1, acceptance=0.01)]
        specs = deme_specs_to_rule_specs_thrml(demes, k=16, n_batches=10, seed=42)
        good_peak = np.argmax(specs[0].posterior)
        bad_peak = np.argmax(specs[1].posterior)
        assert good_peak > bad_peak

    def test_empty(self):
        from geodesic_thrml.bridges.moses import deme_specs_to_rule_specs_thrml
        assert deme_specs_to_rule_specs_thrml([]) == []

    def test_consumable_by_controller(self):
        """RuleSpecs should work with select_step_thrml."""
        from geodesic_thrml.bridges.moses import deme_specs_to_rule_specs_thrml
        from geodesic_thrml.controller_thrml import select_step_thrml
        demes = [_make_deme_spec("a", fitness=0.5), _make_deme_spec("b", fitness=0.8)]
        specs = deme_specs_to_rule_specs_thrml(demes, k=16, n_batches=5)
        result = select_step_thrml(specs, temperature=0.01, seed=42)
        assert result.selected_name in ["a", "b"]


class TestPLNBridge:
    """PLN bridge: wrap PLN-THRML results as RuleSpec (no re-sampling)."""

    def test_single_result(self):
        from geodesic_thrml.bridges.pln import pln_result_to_rule_spec
        posterior = np.array([0.05, 0.1, 0.3, 0.35, 0.15, 0.05])
        spec = pln_result_to_rule_spec(
            name="modus_ponens",
            posterior=posterior,
            strength=0.7,
            confidence=0.8,
            premise_confidences=[0.9, 0.8],
        )
        assert spec.name == "modus_ponens"
        np.testing.assert_array_equal(spec.posterior, posterior)
        assert spec.conclusion_stv == (0.7, 0.8)
        assert spec.premise_confidences == [0.9, 0.8]
        assert spec.cost == 1.0

    def test_batch_results(self):
        from geodesic_thrml.bridges.pln import pln_results_to_rule_specs
        results = [
            {"name": "mp", "posterior": np.ones(16)/16,
             "strength": 0.7, "confidence": 0.8,
             "premise_confidences": [0.9]},
            {"name": "ded", "posterior": np.ones(16)/16,
             "strength": 0.6, "confidence": 0.7,
             "premise_confidences": [0.8], "cost": 2.0},
        ]
        specs = pln_results_to_rule_specs(results)
        assert len(specs) == 2
        assert specs[0].name == "mp"
        assert specs[1].cost == 2.0

    def test_consumable_by_controller(self):
        from geodesic_thrml.bridges.pln import pln_result_to_rule_spec
        from geodesic_thrml.controller_thrml import select_step_thrml
        specs = [
            pln_result_to_rule_spec("a", np.ones(16)/16, 0.7, 0.8, [0.9]),
            pln_result_to_rule_spec("b", np.ones(16)/16, 0.3, 0.4, [0.5]),
        ]
        result = select_step_thrml(specs, temperature=0.01, seed=42)
        assert result.selected_name in ["a", "b"]


class TestQuantiMORKCascadeSolverThrml:
    """QuantiMORK bridge: wavelet levels → THRML unified cascade."""

    def test_produces_solver(self):
        from geodesic_thrml.bridges.quantimork import (
            make_quantimork_cascade_solver_thrml, WaveletLevelSpec,
        )
        specs = [
            WaveletLevelSpec(level=2, band="detail", dim=8, weight_shape=(8, 8),
                             max_connections=5, energy_params={"weight": np.eye(8) * 0.5}),
            WaveletLevelSpec(level=1, band="detail", dim=16, weight_shape=(16, 16),
                             max_connections=5, energy_params={"weight": np.eye(16) * 0.3}),
        ]
        solver = make_quantimork_cascade_solver_thrml(specs)
        assert callable(solver)

    def test_solver_returns_valid_posterior(self):
        from geodesic_thrml.bridges.quantimork import (
            make_quantimork_cascade_solver_thrml, WaveletLevelSpec,
        )
        specs = [
            WaveletLevelSpec(level=2, band="detail", dim=8, weight_shape=(8, 8),
                             max_connections=5, energy_params={"weight": np.eye(8) * 0.5}),
            WaveletLevelSpec(level=1, band="detail", dim=16, weight_shape=(16, 16),
                             max_connections=5, energy_params={"weight": np.eye(16) * 0.3}),
        ]
        solver = make_quantimork_cascade_solver_thrml(specs)
        posterior, s, c = solver(8, None)
        assert posterior.shape == (8,)
        assert abs(posterior.sum() - 1.0) < 1e-5
        assert 0.0 < s < 1.0
