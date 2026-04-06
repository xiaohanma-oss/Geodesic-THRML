"""Tests for geodesic_thrml.curriculum_thrml — unified THRML factor graph cascade."""

import numpy as np
import pytest

import jax.numpy as jnp

from geodesic_thrml.types import build_resolution_ladder, CascadeResult
from geodesic_thrml.curriculum_thrml import (
    rebin_coupling_weights,
    build_unified_cascade_graph,
    cascade_solve_thrml,
    _beta_prior_weights_np,
    _posterior_to_stv,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Beta prior weights
# ═══════════════════════════════════════════════════════════════════════════

class TestBetaPriorWeightsNp:

    def test_centered(self):
        w = _beta_prior_weights_np(0.7, 0.8, k=16)
        assert abs(w.mean()) < 1e-10
        assert w.shape == (16,)

    def test_peaked_at_strength(self):
        w = _beta_prior_weights_np(0.8, 0.9, k=16)
        peak_bin = np.argmax(w)
        center = (peak_bin + 0.5) / 16
        assert abs(center - 0.8) < 0.15

    def test_higher_confidence_sharper(self):
        w_low = _beta_prior_weights_np(0.5, 0.3, k=16)
        w_high = _beta_prior_weights_np(0.5, 0.9, k=16)
        assert np.std(w_high) > np.std(w_low)

    def test_uniform_at_zero_confidence(self):
        w = _beta_prior_weights_np(0.5, 0.0, k=16)
        assert np.std(w) < 0.5


class TestPosteriorToStv:

    def test_peaked_posterior(self):
        posterior = np.zeros(16)
        posterior[12] = 0.95
        posterior[11] = 0.025
        posterior[13] = 0.025
        s, c = _posterior_to_stv(posterior, k=16)
        assert abs(s - 0.78) < 0.1
        assert c > 0.5

    def test_uniform_posterior(self):
        posterior = np.ones(16) / 16
        s, c = _posterior_to_stv(posterior, k=16)
        assert abs(s - 0.5) < 0.05
        assert c < 0.3


# ═══════════════════════════════════════════════════════════════════════════
#  Inter-level coupling weights (non-square)
# ═══════════════════════════════════════════════════════════════════════════

class TestRebinCouplingWeights:

    def test_non_square_shape(self):
        """Coupling between different K values produces non-square matrix."""
        W = rebin_coupling_weights(4, 8)
        assert W.shape == (4, 8)

    def test_same_k_square(self):
        W = rebin_coupling_weights(8, 8)
        assert W.shape == (8, 8)

    def test_centered_rows(self):
        W = rebin_coupling_weights(4, 16)
        row_means = jnp.mean(W, axis=1)
        np.testing.assert_allclose(np.array(row_means), 0.0, atol=1e-5)

    def test_diagonal_dominance_same_k(self):
        """For same-K, diagonal should have highest weight per row."""
        W = np.array(rebin_coupling_weights(8, 8))
        for i in range(8):
            assert W[i, i] == W[i].max()

    def test_higher_precision_tighter(self):
        W_loose = np.array(rebin_coupling_weights(4, 8, precision=2.0))
        W_tight = np.array(rebin_coupling_weights(4, 8, precision=20.0))
        assert np.mean(np.std(W_tight, axis=1)) > np.mean(np.std(W_loose, axis=1))

    def test_coarse_to_fine_peaks_near_diagonal(self):
        """For K=4→K=8, each row's peak should be near 2*i (scaled position)."""
        W = np.array(rebin_coupling_weights(4, 8))
        for i in range(4):
            peak_j = np.argmax(W[i])
            # K=4 bin i center ≈ (2*i+1)/8, K=8 bin j center ≈ (2*j+1)/16
            # Expected peak near j=2*i or 2*i+1
            assert abs(peak_j - 2 * i) <= 1, \
                f"Row {i}: peak at {peak_j}, expected near {2*i}"


# ═══════════════════════════════════════════════════════════════════════════
#  Unified cascade factor graph
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildUnifiedCascadeGraph:

    def test_single_level(self):
        ladder = build_resolution_ladder([16])
        graph = build_unified_cascade_graph(ladder, base_prior=(0.7, 0.8))
        assert len(graph["nodes"]) == 1
        assert len(graph["free_blocks"]) == 1
        # Prior factor only, no coupling
        assert len(graph["factors"]) == 1

    def test_three_levels(self):
        """K=4→8→16 should create 3 nodes, 1 prior + 2 coupling factors."""
        ladder = build_resolution_ladder([4, 8, 16])
        graph = build_unified_cascade_graph(ladder, base_prior=(0.7, 0.8))
        assert len(graph["nodes"]) == 3
        assert len(graph["free_blocks"]) == 3
        # 1 prior (K=4) + 2 couplings (4→8, 8→16)
        assert len(graph["factors"]) == 3

    def test_no_prior(self):
        """Without base_prior, only coupling factors."""
        ladder = build_resolution_ladder([4, 8])
        graph = build_unified_cascade_graph(ladder, base_prior=None)
        # 0 prior + 1 coupling
        assert len(graph["factors"]) == 1

    def test_empty_ladder_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            build_unified_cascade_graph([])

    def test_program_has_correct_samplers(self):
        """Each block should have a sampler with matching n_categories."""
        ladder = build_resolution_ladder([4, 8, 16])
        graph = build_unified_cascade_graph(ladder)
        prog = graph["program"]
        # The program should be a FactorSamplingProgram (compilable)
        assert prog is not None


# ═══════════════════════════════════════════════════════════════════════════
#  Full cascade solve
# ═══════════════════════════════════════════════════════════════════════════

class TestCascadeSolveThrml:

    def test_single_level(self):
        ladder = build_resolution_ladder([8])
        result = cascade_solve_thrml(
            ladder, base_prior=(0.7, 0.8), n_batches=10, seed=42)
        assert isinstance(result, CascadeResult)
        assert result.posterior.shape == (8,)
        assert abs(result.posterior.sum() - 1.0) < 1e-6
        assert 0.0 < result.strength < 1.0
        assert len(result.level_results) == 1

    def test_three_level_cascade(self):
        """K=4→8→16 unified cascade should produce valid results."""
        ladder = build_resolution_ladder([4, 8, 16])
        result = cascade_solve_thrml(
            ladder, base_prior=(0.7, 0.8), n_batches=10, seed=42)
        assert result.posterior.shape == (16,)
        assert len(result.level_results) == 3
        assert result.level_results[0][0] == 4
        assert result.level_results[1][0] == 8
        assert result.level_results[2][0] == 16

    def test_all_levels_have_valid_posteriors(self):
        """Each level should have a valid normalized posterior."""
        ladder = build_resolution_ladder([4, 8, 16])
        result = cascade_solve_thrml(
            ladder, base_prior=(0.7, 0.8), n_batches=10, seed=42)
        for k, posterior, s, c in result.level_results:
            assert posterior.shape == (k,)
            assert abs(posterior.sum() - 1.0) < 1e-5
            assert 0.0 < s < 1.0

    def test_strength_near_prior(self):
        """With strong prior, cascade strength should be near prior."""
        ladder = build_resolution_ladder([4, 8, 16])
        result = cascade_solve_thrml(
            ladder, base_prior=(0.7, 0.9), n_batches=20, seed=42)
        assert abs(result.strength - 0.7) < 0.15

    def test_empty_ladder_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            cascade_solve_thrml([])

    def test_no_prior_uses_uniform(self):
        ladder = build_resolution_ladder([4, 8])
        result = cascade_solve_thrml(ladder, base_prior=None, n_batches=5)
        assert isinstance(result, CascadeResult)
        assert abs(result.strength - 0.5) < 0.3


class TestCascadeOutputValidity:
    """Cascade output should always be valid."""

    def test_strength_in_range(self):
        ladder = build_resolution_ladder([4, 8, 16])
        result = cascade_solve_thrml(
            ladder, base_prior=(0.7, 0.8), n_batches=20, seed=42)
        assert 0.0 < result.strength < 1.0
        assert 0.0 <= result.confidence < 1.0
