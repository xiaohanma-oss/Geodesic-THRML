"""
geodesic_thrml.capsules — Evidence capsule provenance tracking
==============================================================

Implements the evidence capsule system from genenergy-logic §4.4 and
evidence-conservation-theorems §2.3.

An evidence capsule tracks which primitive evidence items support a
derived conclusion.  Overlap-aware merging ensures shared evidence is
counted once, not twice — eliminating "phantom multimodality" caused
by double-counting.

Key properties:
    - merge is idempotent: merge(A, A) = A
    - merge is commutative: merge(A, B) = merge(B, A)
    - mass is subadditive: mass(merge(A, B)) ≤ mass(A) + mass(B)
    - shared evidence counted once: mass(merge(A∪B, B∪C)) uses B once

References:
    - genenergy-logic §4.4-4.5: evidence capsules, double-counting penalty
    - evidence-conservation-theorems §2.3: formal capsule definition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class EvidenceCapsule:
    """Provenance record for a derived proposition.

    Attributes:
        sources: frozenset of primitive evidence item IDs
        weights: mapping from source ID to quantale-valued weight (default 1.0)
    """
    sources: frozenset[str] = field(default_factory=frozenset)
    weights: Mapping[str, float] = field(default_factory=dict)

    @property
    def mass(self) -> float:
        """Total evidence mass: sum of weights for all sources.

        This is the computable upper bound used in the hallucination
        bound (Thm 4.2): conclusion strength ≤ mass(capsule).
        """
        if not self.sources:
            return 0.0
        return sum(self.weights.get(s, 1.0) for s in self.sources)

    @property
    def size(self) -> int:
        """Number of distinct primitive evidence items."""
        return len(self.sources)


def merge_capsules(a: EvidenceCapsule, b: EvidenceCapsule) -> EvidenceCapsule:
    """Overlap-aware merge: shared sources counted once.

    This is simply set union on sources (Definition 2.9 in
    evidence-conservation-theorems).  Idempotent: merge(A, A) = A.

    Weights: for shared sources, take the max weight.
    """
    merged_sources = a.sources | b.sources
    merged_weights = dict(a.weights)
    for s, w in b.weights.items():
        if s in merged_weights:
            merged_weights[s] = max(merged_weights[s], w)
        else:
            merged_weights[s] = w
    return EvidenceCapsule(sources=merged_sources, weights=merged_weights)


def overlap_ratio(a: EvidenceCapsule, b: EvidenceCapsule) -> float:
    """Fraction of shared evidence between two capsules.

    Returns 0.0 if no overlap, 1.0 if identical sources.
    """
    if not a.sources or not b.sources:
        return 0.0
    shared = a.sources & b.sources
    total = a.sources | b.sources
    return len(shared) / len(total)


def double_counting_penalty(capsules: list[EvidenceCapsule]) -> float:
    """Quantify the degree of evidence double-counting across capsules.

    Measures how much the naive sum of individual masses exceeds the
    mass of their overlap-aware merge.  Zero means no double-counting.

    From genenergy-logic §4.5:
        penalty = Σ mass(c_i) - mass(merge_all(c_i))
    """
    if not capsules:
        return 0.0
    naive_sum = sum(c.mass for c in capsules)
    merged = capsules[0]
    for c in capsules[1:]:
        merged = merge_capsules(merged, c)
    return max(naive_sum - merged.mass, 0.0)


def make_capsule(source_id: str, weight: float = 1.0) -> EvidenceCapsule:
    """Create a capsule from a single primitive evidence item."""
    return EvidenceCapsule(
        sources=frozenset([source_id]),
        weights={source_id: weight},
    )
