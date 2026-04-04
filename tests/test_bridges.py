"""Tests for geodesic_thrml.bridges — sub-project bridge adapters."""

import numpy as np
import pytest

from geodesic_thrml.bridges.quantimork import (
    WaveletLevelSpec, wavelet_to_resolution_ladder, estimate_level_cost,
)
from geodesic_thrml.bridges.ecan import (
    extract_ecan_snapshot, sti_to_forward_scores,
)
from geodesic_thrml.bridges.moses import collect_program_skeleton_specs


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
#  MOSES bridge (stub)
# ═══════════════════════════════════════════════════════════════════════════

class TestMOSESBridge:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError, match="MOSES-THRML"):
            collect_program_skeleton_specs([], lambda x: 0.0)
