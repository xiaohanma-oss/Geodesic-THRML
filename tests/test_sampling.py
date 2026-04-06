"""Tests for geodesic_thrml.sampling — shared THRML sampling facility."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from thrml.pgm import CategoricalNode
from thrml.block_management import Block
from thrml.models.discrete_ebm import CategoricalEBMFactor

from geodesic_thrml.sampling import (
    SamplingConfig,
    PLN_DEFAULT_CONFIG,
    QM_DEFAULT_CONFIG,
    LIGHT_CONFIG,
    greedy_color,
    assemble_sampling_program,
    init_block_states,
    run_gibbs_sampling,
    extract_posterior,
    extract_node_posterior,
    bin_centers,
    posterior_to_stv,
    diagnose_convergence,
)


# ═══════════════════════════════════════════════════════════════════════════
#  SamplingConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestSamplingConfig:

    def test_to_schedule(self):
        config = SamplingConfig(n_warmup=100, n_samples=500, steps_per_sample=2)
        sched = config.to_schedule()
        assert sched.n_warmup == 100
        assert sched.n_samples == 500
        assert sched.steps_per_sample == 2

    def test_predefined_configs(self):
        assert PLN_DEFAULT_CONFIG.n_batches == 50
        assert PLN_DEFAULT_CONFIG.n_warmup == 500
        assert QM_DEFAULT_CONFIG.n_batches == 30
        assert QM_DEFAULT_CONFIG.n_warmup == 300
        assert LIGHT_CONFIG.n_batches == 10
        assert LIGHT_CONFIG.n_warmup == 50


# ═══════════════════════════════════════════════════════════════════════════
#  Graph coloring
# ═══════════════════════════════════════════════════════════════════════════

class TestGreedyColor:

    def test_chain_graph_two_colors(self):
        """A-B-C chain should use 2 colors."""
        names = ["A", "B", "C"]
        adj = {"A": {"B"}, "B": {"A", "C"}, "C": {"B"}}
        groups = greedy_color(names, adj)
        assert len(groups) == 2
        # No two adjacent nodes in same group
        for grp in groups:
            for n in grp:
                for nb in adj.get(n, set()):
                    assert nb not in grp

    def test_star_graph(self):
        """Star with center + 3 leaves needs 2 colors."""
        names = ["center", "a", "b", "c"]
        adj = {
            "center": {"a", "b", "c"},
            "a": {"center"}, "b": {"center"}, "c": {"center"},
        }
        groups = greedy_color(names, adj)
        assert len(groups) == 2

    def test_disconnected_graph(self):
        """Disconnected nodes need only 1 color."""
        names = ["a", "b", "c"]
        adj = {}
        groups = greedy_color(names, adj)
        assert len(groups) == 1
        assert set(groups[0]) == {"a", "b", "c"}

    def test_complete_graph_k3(self):
        """K3 complete graph needs 3 colors."""
        names = ["a", "b", "c"]
        adj = {"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"}}
        groups = greedy_color(names, adj)
        assert len(groups) == 3

    def test_empty_graph(self):
        groups = greedy_color([], {})
        assert groups == []


# ═══════════════════════════════════════════════════════════════════════════
#  Graph assembly
# ═══════════════════════════════════════════════════════════════════════════

class TestAssembleSamplingProgram:

    def test_single_block_uniform_k(self):
        node = CategoricalNode()
        block = Block([node])
        bias = jnp.zeros((1, 8))
        factor = CategoricalEBMFactor([block], bias)
        prog = assemble_sampling_program([block], [], [factor], k=8)
        assert prog is not None

    def test_multi_block_uniform_k(self):
        n1, n2 = CategoricalNode(), CategoricalNode()
        b1, b2 = Block([n1]), Block([n2])
        bias1 = jnp.zeros((1, 16))
        bias2 = jnp.zeros((1, 16))
        f1 = CategoricalEBMFactor([b1], bias1)
        f2 = CategoricalEBMFactor([b2], bias2)
        prog = assemble_sampling_program([b1, b2], [], [f1, f2], k=16)
        assert prog is not None

    def test_multi_k_list(self):
        """Different K per block (curriculum cascade pattern)."""
        n1, n2 = CategoricalNode(), CategoricalNode()
        b1, b2 = Block([n1]), Block([n2])
        bias1 = jnp.zeros((1, 4))
        f1 = CategoricalEBMFactor([b1], bias1)
        prog = assemble_sampling_program([b1, b2], [], [f1], k=[4, 8])
        assert prog is not None

    def test_k_list_length_mismatch_raises(self):
        n1 = CategoricalNode()
        b1 = Block([n1])
        with pytest.raises(ValueError, match="k list length"):
            assemble_sampling_program([b1], [], [], k=[4, 8])


# ═══════════════════════════════════════════════════════════════════════════
#  State initialization
# ═══════════════════════════════════════════════════════════════════════════

class TestInitBlockStates:

    def test_shapes_and_dtype(self):
        n1, n2 = CategoricalNode(), CategoricalNode()
        b1 = Block([n1, n2])
        key = jax.random.PRNGKey(0)
        states, _ = init_block_states([b1], n_batches=5, k=16, key=key)
        assert len(states) == 1
        assert states[0].shape == (5, 2)
        assert states[0].dtype == jnp.uint8

    def test_value_range(self):
        n1 = CategoricalNode()
        b1 = Block([n1])
        key = jax.random.PRNGKey(0)
        states, _ = init_block_states([b1], n_batches=100, k=8, key=key)
        vals = np.array(states[0]).flatten()
        assert vals.min() >= 0
        assert vals.max() < 8

    def test_multi_k(self):
        n1, n2 = CategoricalNode(), CategoricalNode()
        b1, b2 = Block([n1]), Block([n2])
        key = jax.random.PRNGKey(0)
        states, _ = init_block_states([b1, b2], n_batches=5, k=[4, 16], key=key)
        assert len(states) == 2
        assert np.array(states[0]).max() < 4
        assert np.array(states[1]).max() < 16


# ═══════════════════════════════════════════════════════════════════════════
#  Gibbs sampling
# ═══════════════════════════════════════════════════════════════════════════

class TestRunGibbsSampling:

    def _make_biased_program(self, k=8, peak_bin=6):
        """Single-node graph strongly biased toward peak_bin."""
        node = CategoricalNode()
        block = Block([node])
        bias = jnp.zeros(k)
        bias = bias.at[peak_bin].set(10.0)
        bias = bias - jnp.mean(bias)
        factor = CategoricalEBMFactor([block], bias[None, :])
        prog = assemble_sampling_program([block], [], [factor], k=k)
        return prog, [block]

    def test_biased_single_node(self):
        """Strongly biased node should produce peaked posterior."""
        prog, blocks = self._make_biased_program(k=8, peak_bin=6)
        samples = run_gibbs_sampling(
            prog, blocks, config=LIGHT_CONFIG, k=8, seed=42)
        posterior = extract_posterior(samples, 0, 0, k=8)
        assert posterior.shape == (8,)
        assert abs(posterior.sum() - 1.0) < 1e-6
        assert np.argmax(posterior) == 6

    def test_output_shape(self):
        prog, blocks = self._make_biased_program()
        samples = run_gibbs_sampling(
            prog, blocks, config=LIGHT_CONFIG, k=8, seed=0)
        # samples should be indexable by block
        assert samples is not None


# ═══════════════════════════════════════════════════════════════════════════
#  Posterior extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractPosterior:

    def test_sums_to_one(self):
        node = CategoricalNode()
        block = Block([node])
        bias = jnp.array([0.0, 0.0, 0.0, 5.0])[None, :]
        factor = CategoricalEBMFactor([block], bias)
        prog = assemble_sampling_program([block], [], [factor], k=4)
        samples = run_gibbs_sampling(
            prog, [block], config=LIGHT_CONFIG, k=4, seed=0)
        posterior = extract_posterior(samples, 0, 0, k=4)
        assert abs(posterior.sum() - 1.0) < 1e-6
        assert posterior.shape == (4,)

    def test_peaked_at_biased_bin(self):
        node = CategoricalNode()
        block = Block([node])
        bias = jnp.array([0.0, 0.0, 10.0, 0.0])[None, :]
        factor = CategoricalEBMFactor([block], bias)
        prog = assemble_sampling_program([block], [], [factor], k=4)
        samples = run_gibbs_sampling(
            prog, [block], config=LIGHT_CONFIG, k=4, seed=42)
        posterior = extract_posterior(samples, 0, 0, k=4)
        assert np.argmax(posterior) == 2


class TestExtractNodePosterior:

    def test_finds_node_by_identity(self):
        n1, n2 = CategoricalNode(), CategoricalNode()
        b1, b2 = Block([n1]), Block([n2])
        bias1 = jnp.array([10.0, 0.0, 0.0, 0.0])[None, :]
        bias2 = jnp.array([0.0, 0.0, 0.0, 10.0])[None, :]
        f1 = CategoricalEBMFactor([b1], bias1)
        f2 = CategoricalEBMFactor([b2], bias2)
        prog = assemble_sampling_program([b1, b2], [], [f1, f2], k=4)
        samples = run_gibbs_sampling(
            prog, [b1, b2], config=LIGHT_CONFIG, k=4, seed=42)

        p1 = extract_node_posterior(samples, [b1, b2], n1, k=4)
        p2 = extract_node_posterior(samples, [b1, b2], n2, k=4)
        assert np.argmax(p1) == 0
        assert np.argmax(p2) == 3

    def test_missing_node_raises(self):
        n1 = CategoricalNode()
        b1 = Block([n1])
        missing = CategoricalNode()
        bias = jnp.zeros((1, 4))
        factor = CategoricalEBMFactor([b1], bias)
        prog = assemble_sampling_program([b1], [], [factor], k=4)
        samples = run_gibbs_sampling(
            prog, [b1], config=LIGHT_CONFIG, k=4, seed=0)
        with pytest.raises(ValueError, match="not found"):
            extract_node_posterior(samples, [b1], missing, k=4)


# ═══════════════════════════════════════════════════════════════════════════
#  Moment matching
# ═══════════════════════════════════════════════════════════════════════════

class TestBinCenters:

    def test_k4(self):
        c = bin_centers(4)
        expected = np.array([0.125, 0.375, 0.625, 0.875])
        np.testing.assert_allclose(c, expected)

    def test_k16_range(self):
        c = bin_centers(16)
        assert c[0] > 0.0
        assert c[-1] < 1.0
        assert len(c) == 16


class TestPosteriorToStv:

    def test_peaked_high_confidence(self):
        posterior = np.zeros(16)
        posterior[12] = 0.95
        posterior[11] = 0.025
        posterior[13] = 0.025
        s, c = posterior_to_stv(posterior, k=16)
        assert abs(s - 0.78) < 0.1
        assert c > 0.5

    def test_uniform_low_confidence(self):
        posterior = np.ones(16) / 16
        s, c = posterior_to_stv(posterior, k=16)
        assert abs(s - 0.5) < 0.05
        assert c < 0.3

    def test_custom_centers(self):
        """QuantiMORK-style centers in [-1, 1]."""
        k = 8
        centers = np.linspace(-0.875, 0.875, k)
        posterior = np.zeros(k)
        posterior[6] = 1.0  # center = 0.625
        s, c = posterior_to_stv(posterior, k=k, centers=centers)
        assert abs(s - 0.625) < 0.01

    def test_zero_posterior(self):
        posterior = np.zeros(8)
        s, c = posterior_to_stv(posterior, k=8)
        assert s == 0.5
        assert c == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Convergence diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class TestDiagnoseConvergence:

    def test_converged_chain(self):
        """Well-mixed chain should converge."""
        node = CategoricalNode()
        block = Block([node])
        # Mild bias — should mix well
        bias = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.1])
        bias = bias - jnp.mean(bias)
        factor = CategoricalEBMFactor([block], bias[None, :])
        prog = assemble_sampling_program([block], [], [factor], k=8)

        config = SamplingConfig(
            n_warmup=200, n_samples=1000, steps_per_sample=2, n_batches=20)
        samples = run_gibbs_sampling(
            prog, [block], config=config, k=8, seed=42)

        diag = diagnose_convergence(samples, 0, 0, k=8)
        assert "r_hat" in diag
        assert "ess" in diag
        assert "converged" in diag
        # Should have reasonable R-hat for a simple single-node model
        assert diag["r_hat"] < 2.0  # loose bound
        assert diag["ess"] > 0
