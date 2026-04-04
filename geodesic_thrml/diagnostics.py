"""
geodesic_thrml.diagnostics — H¹ cohomology detection for inference graphs
==========================================================================

Analyzes the topology of inference graphs to determine whether local
message passing is safe (tree-like, H¹=0) or whether cycles introduce
dependencies that require heavier techniques (annealing, DTM).

For a directed inference graph:
    H¹ = 0  →  tree-like: local merges are exact, no double-counting risk
    H¹ ≠ 0  →  cycles exist: need annealing/DTM for cyclic subgraphs

Implementation uses the standard relation: for a connected graph,
    β₁ = |E| - |V| + 1  (first Betti number = number of independent cycles)
which equals dim(H¹) for a simplicial 1-complex.

References:
    - cohomology-quantum-evidence: topological analysis of evidence dependencies
    - evidence-conservation-theorems: tree-like graphs have exact merges
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass
class InferenceGraph:
    """Directed inference graph.

    Attributes:
        nodes: list of node identifiers
        edges: list of (source, target) pairs
    """
    nodes: list[str]
    edges: list[tuple[str, str]]


@dataclass
class TopologyReport:
    """Result of topological analysis.

    Attributes:
        classification: 'tree' or 'cyclic'
        betti_1: first Betti number (0 = tree, >0 = cycles)
        n_components: number of connected components
        cycle_edges: edges participating in cycles (empty if tree)
    """
    classification: str
    betti_1: int
    n_components: int
    cycle_edges: list[tuple[str, str]]


def compute_betti_1(graph: InferenceGraph) -> int:
    """Compute the first Betti number β₁ = |E| - |V| + n_components.

    For a connected graph, β₁ = |E| - |V| + 1 = number of independent cycles.
    For disconnected graphs, we sum across components.
    """
    n_components = _count_components(graph)
    return len(graph.edges) - len(graph.nodes) + n_components


def classify_topology(graph: InferenceGraph) -> TopologyReport:
    """Classify inference graph topology.

    Returns a TopologyReport with:
        - 'tree' if H¹ = 0 (no cycles, local merges safe)
        - 'cyclic' if H¹ ≠ 0 (cycles present, need heavier techniques)
    """
    b1 = compute_betti_1(graph)
    n_comp = _count_components(graph)
    cycles = locate_cycles(graph) if b1 > 0 else []
    return TopologyReport(
        classification="tree" if b1 == 0 else "cyclic",
        betti_1=b1,
        n_components=n_comp,
        cycle_edges=cycles,
    )


def locate_cycles(graph: InferenceGraph) -> list[tuple[str, str]]:
    """Find edges that participate in cycles.

    Uses DFS on the undirected version of the graph.  Returns the
    "back edges" — removing any one of them would break its cycle.
    """
    adj = defaultdict(set)
    for u, v in graph.edges:
        adj[u].add(v)
        adj[v].add(u)

    visited = set()
    parent = {}
    back_edges = []

    def dfs(node, par):
        visited.add(node)
        parent[node] = par
        for nb in adj[node]:
            if nb not in visited:
                dfs(nb, node)
            elif nb != par:
                # Back edge found — part of a cycle
                # Record as directed edge from the original graph if it exists
                if (node, nb) in edge_set:
                    back_edges.append((node, nb))
                elif (nb, node) in edge_set:
                    back_edges.append((nb, node))

    edge_set = set(graph.edges)
    for node in graph.nodes:
        if node not in visited:
            dfs(node, None)

    return back_edges


def suggest_strategy(report: TopologyReport) -> str:
    """Suggest a sampling strategy based on topology.

    - tree: lightweight local message passing
    - cyclic with few cycles: annealing on cycle subgraph
    - cyclic with many cycles: full DTM
    """
    if report.classification == "tree":
        return "local"
    if report.betti_1 <= 2:
        return "annealing"
    return "dtm"


def _count_components(graph: InferenceGraph) -> int:
    """Count connected components (treating edges as undirected)."""
    adj = defaultdict(set)
    for u, v in graph.edges:
        adj[u].add(v)
        adj[v].add(u)

    visited = set()
    count = 0
    for node in graph.nodes:
        if node not in visited:
            count += 1
            _bfs(node, adj, visited)
    return count


def _bfs(start, adj, visited):
    """BFS from start, marking visited."""
    queue = [start]
    visited.add(start)
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
