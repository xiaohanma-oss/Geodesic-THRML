"""
geodesic_thrml.bridges.ecan — ECAN-THRML bridge (stub)
========================================================

ECAN-THRML currently uses Lattice Boltzmann (D2Q9) for attention flow,
without direct thrml factor graph integration.  This bridge provides
the interface for when ECAN gains TSU support.

Available now:
    - V_history (HJB value function) extraction
    - STI/LTI distribution snapshots for geodesic scoring

Future (when ECAN adds thrml):
    - Factor graph extraction for joint geodesic sampling
    - Attention-weighted node selection for parallel groups

References:
    - ecan_thrml.hjb: solve_hjb() → V_history
    - ecan_thrml.bridge: ring_to_lbm(), lbm_to_sti()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class ECANSnapshot:
    """Snapshot of ECAN attention state for geodesic scoring.

    Attributes:
        sti_values: Short-Term Importance per atom
        lti_values: Long-Term Importance per atom (optional)
        atom_ids: identifiers for each atom
        total_sti: total STI budget (conservation constraint)
    """
    sti_values: np.ndarray
    lti_values: np.ndarray | None
    atom_ids: list[str]
    total_sti: float


def extract_ecan_snapshot(
    sti_values: Sequence[float],
    atom_ids: Sequence[str],
    lti_values: Sequence[float] | None = None,
) -> ECANSnapshot:
    """Create an ECAN snapshot from raw STI/LTI values.

    Args:
        sti_values: STI per atom
        atom_ids: atom identifiers
        lti_values: optional LTI per atom

    Returns:
        ECANSnapshot for geodesic scoring.
    """
    sti = np.asarray(sti_values, dtype=np.float64)
    lti = np.asarray(lti_values, dtype=np.float64) if lti_values is not None else None
    return ECANSnapshot(
        sti_values=sti,
        lti_values=lti,
        atom_ids=list(atom_ids),
        total_sti=float(sti.sum()),
    )


def sti_to_forward_scores(snapshot: ECANSnapshot) -> np.ndarray:
    """Convert STI distribution to forward scores for geodesic controller.

    Higher STI → higher forward reachability (the atom is "active"
    and its inference pathways are primed).

    Returns:
        Normalized scores, shape [n_atoms].
    """
    sti = snapshot.sti_values
    total = sti.sum()
    if total > 0:
        return sti / total
    return np.ones(len(sti)) / len(sti)
