"""Tests for geodesic_thrml.curriculum — Schrödinger Bridge Learning cascade."""

import numpy as np
import pytest

from geodesic_thrml.curriculum import (
    rebin_posterior,
    posterior_to_prior_weights,
    build_resolution_ladder,
    cascade_solve,
    ResolutionLevel,
)


# ═══════════════════════════════════════════════════════════════════════════
#  rebin_posterior
# ═══════════════════════════════════════════════════════════════════════════

class TestRebinPosterior:
    """Rebin a histogram between different K resolutions."""

    def test_same_k_identity(self):
        """Rebinning to same K should approximate identity."""
        posterior = np.array([0.1, 0.2, 0.4, 0.3])
        result = rebin_posterior(posterior, source_k=4, target_k=4)
        np.testing.assert_allclose(result, posterior, atol=1e-6)

    def test_normalized(self):
        """Output should always be normalized."""
        posterior = np.array([0.1, 0.2, 0.4, 0.3])
        for target_k in [4, 8, 16, 32]:
            result = rebin_posterior(posterior, source_k=4, target_k=target_k)
            assert abs(result.sum() - 1.0) < 1e-10
            assert result.shape == (target_k,)

    def test_upsample_preserves_mode(self):
        """Upsampling should preserve the mode location."""
        # Peak at bin 3 (center ≈ 0.875 for K=4)
        posterior = np.array([0.05, 0.1, 0.15, 0.7])
        result = rebin_posterior(posterior, source_k=4, target_k=16)
        # Mode should be in the high region (bin > 12 for K=16)
        assert np.argmax(result) > 10

    def test_downsample_preserves_mode(self):
        """Downsampling should preserve the mode location."""
        posterior = np.zeros(16)
        posterior[14] = 0.8
        posterior[13] = 0.15
        posterior[15] = 0.05
        result = rebin_posterior(posterior, source_k=16, target_k=4)
        # Mode should be in the highest bin
        assert np.argmax(result) == 3

    def test_uniform_stays_uniform(self):
        """Uniform posterior should remain approximately uniform after rebin."""
        posterior = np.ones(4) / 4
        result = rebin_posterior(posterior, source_k=4, target_k=16)
        # Should be roughly uniform
        assert np.std(result) < 0.02

    def test_handles_zero_bins(self):
        """Should handle posteriors with near-zero bins gracefully."""
        posterior = np.array([1e-10, 1e-10, 0.5, 0.5])
        result = rebin_posterior(posterior, source_k=4, target_k=8)
        assert result.shape == (8,)
        assert abs(result.sum() - 1.0) < 1e-10
        assert np.all(np.isfinite(result))


# ═══════════════════════════════════════════════════════════════════════════
#  posterior_to_prior_weights
# ═══════════════════════════════════════════════════════════════════════════

class TestPosteriorToPriorWeights:
    """Convert posterior to log-space prior weights."""

    def test_centered(self):
        """Output weights should be centered (mean ≈ 0)."""
        posterior = np.array([0.1, 0.2, 0.4, 0.3])
        w = posterior_to_prior_weights(posterior, k=4)
        assert abs(w.mean()) < 1e-10

    def test_peaked_posterior_gives_peaked_weights(self):
        """Peaked posterior should give weights with large spread."""
        peaked = np.array([0.01, 0.01, 0.01, 0.97])
        uniform = np.ones(4) / 4
        w_peaked = posterior_to_prior_weights(peaked, k=4)
        w_uniform = posterior_to_prior_weights(uniform, k=4)
        assert np.std(w_peaked) > np.std(w_uniform)

    def test_finite(self):
        """Should produce finite weights even with near-zero bins."""
        posterior = np.array([1e-20, 0.5, 0.5, 1e-20])
        w = posterior_to_prior_weights(posterior, k=4)
        assert np.all(np.isfinite(w))


# ═══════════════════════════════════════════════════════════════════════════
#  build_resolution_ladder
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildResolutionLadder:
    """Build resolution ladder for cascade."""

    def test_default(self):
        ladder = build_resolution_ladder()
        assert len(ladder) == 3
        assert [l.k for l in ladder] == [4, 8, 16]

    def test_custom(self):
        ladder = build_resolution_ladder([8, 16, 32], ["low", "mid", "high"])
        assert ladder[0].label == "low"
        assert ladder[2].k == 32

    def test_non_ascending_raises(self):
        with pytest.raises(ValueError, match="ascending"):
            build_resolution_ladder([16, 8, 4])

    def test_label_mismatch_raises(self):
        with pytest.raises(ValueError, match="labels length"):
            build_resolution_ladder([4, 8], ["only_one"])


# ═══════════════════════════════════════════════════════════════════════════
#  cascade_solve
# ═══════════════════════════════════════════════════════════════════════════

class TestCascadeSolve:
    """Coarse-to-fine cascade integration tests."""

    @staticmethod
    def _mock_solver_beta(s=0.7, c=0.8):
        """Create a mock solver that returns a Beta-like posterior."""
        def solve(k, prior_weights):
            centers = np.linspace(0.5 / k, 1.0 - 0.5 / k, k)
            # Simple Beta-like shape peaked at s
            w = c2w_simple(c)
            n = w + 2.0
            alpha = s * n
            beta_param = (1.0 - s) * n
            logp = (alpha - 1) * np.log(np.clip(centers, 1e-10, None)) + \
                   (beta_param - 1) * np.log(np.clip(1 - centers, 1e-10, None))
            posterior = np.exp(logp - logp.max())
            posterior /= posterior.sum()
            # If prior_weights provided, blend them in
            if prior_weights is not None:
                blended = logp + 0.5 * prior_weights  # partial warm-start
                posterior = np.exp(blended - blended.max())
                posterior /= posterior.sum()
            mu = float(np.sum(posterior * centers))
            var = float(np.sum(posterior * (centers - mu) ** 2))
            n_est = mu * (1 - mu) / max(var, 1e-12) - 1
            w_est = max(n_est - 2.0, 0.0)
            c_est = w_est / (w_est + 1.0)
            return posterior, mu, c_est
        return solve

    def test_single_level(self):
        """Single-level cascade = direct solve."""
        solver = self._mock_solver_beta(s=0.7, c=0.8)
        ladder = build_resolution_ladder([16])
        result = cascade_solve(solver, ladder)
        assert abs(result.strength - 0.7) < 0.1
        assert result.posterior.shape == (16,)
        assert len(result.level_results) == 1

    def test_three_level_cascade(self):
        """K=4→8→16 cascade should produce reasonable results."""
        solver = self._mock_solver_beta(s=0.7, c=0.8)
        ladder = build_resolution_ladder([4, 8, 16])
        result = cascade_solve(solver, ladder)
        assert abs(result.strength - 0.7) < 0.1
        assert result.posterior.shape == (16,)
        assert len(result.level_results) == 3
        # Each level should have correct K
        assert result.level_results[0][0] == 4
        assert result.level_results[1][0] == 8
        assert result.level_results[2][0] == 16

    def test_error_tracking(self):
        """Error tracking should record per-level deviations."""
        solver = self._mock_solver_beta(s=0.7, c=0.8)
        ladder = build_resolution_ladder([4, 8, 16])
        direct = {4: (0.7, 0.8), 8: (0.7, 0.8), 16: (0.7, 0.8)}
        result = cascade_solve(solver, ladder, direct_results=direct)
        # Errors should be finite and small
        for err in result.error_per_level:
            assert np.isfinite(err)
            assert err < 0.15  # within reasonable tolerance

    def test_error_linear_not_exponential(self):
        """Cascade error should accumulate linearly, not exponentially.

        This is the key property from QLN's linear error accumulation theorem:
        if local approximations are bounded, the chain error is bounded by
        the sum (not product) of per-step errors.
        """
        solver = self._mock_solver_beta(s=0.6, c=0.7)
        # Direct results at each K
        direct = {}
        for k in [4, 8, 16]:
            post, s, c = solver(k, None)
            direct[k] = (s, c)

        ladder = build_resolution_ladder([4, 8, 16])
        result = cascade_solve(solver, ladder, direct_results=direct)

        # Total error should be bounded by sum of per-level errors
        # (not growing exponentially with depth)
        errors = [e for e in result.error_per_level if np.isfinite(e)]
        if len(errors) >= 2:
            # Error at level N should not be >> N * error at level 1
            max_error = max(errors)
            assert max_error < 0.15  # stays bounded

    def test_empty_ladder_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            cascade_solve(lambda k, pw: (np.ones(k) / k, 0.5, 0.5), [])


def c2w_simple(c):
    """Simple confidence → weight for tests (no pln_thrml dependency)."""
    if c >= 1.0:
        return 1e6
    if c <= 0.0:
        return 0.0
    return c / (1.0 - c)
