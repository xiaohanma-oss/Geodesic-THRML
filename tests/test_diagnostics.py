"""Tests for geodesic_thrml.diagnostics — H¹ cohomology detection."""

from geodesic_thrml.diagnostics import (
    InferenceGraph, compute_betti_1, classify_topology,
    locate_cycles, suggest_strategy,
)


class TestBetti1:
    def test_chain_is_tree(self):
        """A→B→C→D is a tree: β₁ = 0."""
        g = InferenceGraph(
            nodes=["A", "B", "C", "D"],
            edges=[("A", "B"), ("B", "C"), ("C", "D")],
        )
        assert compute_betti_1(g) == 0

    def test_single_cycle(self):
        """A→B→C→A has one cycle: β₁ = 1."""
        g = InferenceGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],
        )
        assert compute_betti_1(g) == 1

    def test_two_cycles(self):
        """Two independent cycles sharing an edge: β₁ = 2."""
        # A→B→C→A and A→B→D→A
        g = InferenceGraph(
            nodes=["A", "B", "C", "D"],
            edges=[("A", "B"), ("B", "C"), ("C", "A"), ("B", "D"), ("D", "A")],
        )
        assert compute_betti_1(g) == 2

    def test_single_node(self):
        g = InferenceGraph(nodes=["A"], edges=[])
        assert compute_betti_1(g) == 0

    def test_disconnected_trees(self):
        """Two disconnected chains: β₁ = 0."""
        g = InferenceGraph(
            nodes=["A", "B", "C", "D"],
            edges=[("A", "B"), ("C", "D")],
        )
        assert compute_betti_1(g) == 0

    def test_v_shape(self):
        """A→B←C (inverted V): β₁ = 0."""
        g = InferenceGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("C", "B")],
        )
        assert compute_betti_1(g) == 0

    def test_diamond(self):
        """A→B, A→C, B→D, C→D (diamond): β₁ = 1."""
        g = InferenceGraph(
            nodes=["A", "B", "C", "D"],
            edges=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )
        assert compute_betti_1(g) == 1


class TestClassifyTopology:
    def test_tree_classification(self):
        g = InferenceGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C")],
        )
        report = classify_topology(g)
        assert report.classification == "tree"
        assert report.betti_1 == 0
        assert report.cycle_edges == []

    def test_cyclic_classification(self):
        g = InferenceGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],
        )
        report = classify_topology(g)
        assert report.classification == "cyclic"
        assert report.betti_1 == 1
        assert len(report.cycle_edges) >= 1


class TestLocateCycles:
    def test_tree_no_cycles(self):
        g = InferenceGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C")],
        )
        assert locate_cycles(g) == []

    def test_cycle_finds_back_edge(self):
        g = InferenceGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],
        )
        back = locate_cycles(g)
        assert len(back) >= 1
        # The back edge should be one of the cycle edges
        for u, v in back:
            assert (u, v) in set(g.edges)


class TestSuggestStrategy:
    def test_tree_local(self):
        g = InferenceGraph(nodes=["A", "B"], edges=[("A", "B")])
        report = classify_topology(g)
        assert suggest_strategy(report) == "local"

    def test_single_cycle_annealing(self):
        g = InferenceGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],
        )
        report = classify_topology(g)
        assert suggest_strategy(report) == "annealing"

    def test_many_cycles_dtm(self):
        # 3+ independent cycles
        g = InferenceGraph(
            nodes=["A", "B", "C", "D", "E"],
            edges=[("A", "B"), ("B", "A"),   # cycle 1
                   ("B", "C"), ("C", "B"),   # cycle 2
                   ("C", "D"), ("D", "C"),   # cycle 3
                   ("D", "E")],
        )
        report = classify_topology(g)
        assert suggest_strategy(report) == "dtm"
