"""
geodesic_thrml.bridges.ecan — ECAN-THRML bridge
==================================================

Maps ECAN's attention state to geodesic factors.  Two modes:

1. **HJB mode** (preferred, aligns with whitepaper §5.3-5.4):
   Uses the solved HJB value function V(x) from ecan_thrml.hjb.
   V encodes goal-directed attention routing cost — it already contains
   both forward and backward information:
       - Low V(x) → close to goal → high backward factor g
       - Large |∇V(x)| → strong attention flow → high forward factor f
   The optimal attention flow follows −∇V.

2. **STI snapshot mode** (simplified fallback):
   Uses raw STI values as forward scores.  This is a simplification
   that discards the goal-directed information encoded in V.
   Use only when HJB solve is unavailable.

Whitepaper §4.4: "ECAN allocates attention — short-term and long-term
importance — over the metagraph, which PRIMUS uses as the scheduler
for both loops."  ECAN and the geodesic controller are co-schedulers,
not a provider-consumer pair.

References:
    - Hyperon whitepaper §5.3: ECAN meets optimal transport
    - Hyperon whitepaper §5.4: incompressible-fluid networks
    - ecan_thrml.hjb: solve_hjb() → V_final, V_history
    - ecan_thrml.bridge: ring_to_lbm(), lbm_to_sti()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from geodesic_thrml.scores import RuleSpec


# ═══════════════════════════════════════════════════════════════════════════
#  Mode 1: HJB value function (preferred)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HJBFactors:
    """Forward and backward factors extracted from HJB value function.

    Attributes:
        f_scores: attention flow intensity per atom (|∇V| based)
        g_scores: goal proximity per atom (exp(-V) based)
        atom_ids: identifiers for each atom
        v_values: raw V values at each atom position
    """
    f_scores: np.ndarray
    g_scores: np.ndarray
    atom_ids: list[str]
    v_values: np.ndarray


def extract_hjb_factors(
    V_final: np.ndarray,
    atom_ids: list[str],
) -> HJBFactors:
    """Extract geodesic forward/backward factors from HJB value function.

    V(x) is the goal-directed attention routing cost (whitepaper §5.3):
        - Low V → close to goal → high backward score (goal proximity)
        - Large |∇V| → strong gradient → high forward score (attention flow)

    The optimal attention flow follows −∇V, routing attention toward goals.

    Args:
        V_final: solved value function, shape [Ny, Nx] or [N] (1D ring)
        atom_ids: identifiers for each atom position

    Returns:
        HJBFactors with normalized f and g scores.
    """
    V = np.asarray(V_final, dtype=np.float64)

    # Handle both 1D ring [N] and 2D grid [Ny, Nx] → flatten to 1D
    if V.ndim == 2 and V.shape[0] == 1:
        V = V.squeeze(0)  # [1, N] → [N]
    elif V.ndim == 2:
        V = V.flatten()  # [Ny, Nx] → [N]

    n = len(atom_ids)
    if len(V) != n:
        raise ValueError(f"V has {len(V)} values but {n} atom_ids")

    # Backward factor g: goal proximity
    # Low V = close to goal = high g
    # g(x) = exp(-V(x) / scale) where scale normalizes the range
    v_range = V.max() - V.min()
    scale = max(v_range, 1e-10)
    g_raw = np.exp(-(V - V.min()) / scale)
    g_total = g_raw.sum()
    g_scores = g_raw / g_total if g_total > 0 else np.ones(n) / n

    # Forward factor f: attention flow intensity
    # |∇V| = gradient magnitude = where attention is actively flowing
    # For 1D: |∇V|_i ≈ |V[i+1] - V[i-1]| / 2  (central difference, periodic)
    grad = np.abs(np.roll(V, -1) - np.roll(V, 1)) / 2.0
    f_raw = grad + 1e-10  # avoid zero
    f_total = f_raw.sum()
    f_scores = f_raw / f_total if f_total > 0 else np.ones(n) / n

    return HJBFactors(
        f_scores=f_scores,
        g_scores=g_scores,
        atom_ids=list(atom_ids),
        v_values=V,
    )


def hjb_to_rule_specs(
    V_final: np.ndarray,
    atom_ids: list[str],
    atom_costs: Sequence[float] | None = None,
) -> list[RuleSpec]:
    """Convert HJB value function to RuleSpec list for geodesic controller.

    Each atom becomes a candidate "step" — the controller decides which
    atoms' inference neighborhoods to expand next.

    Args:
        V_final: solved HJB value function
        atom_ids: identifiers for each atom
        atom_costs: per-atom computational cost (default 1.0 each)

    Returns:
        List of RuleSpec for controller.select_step().
    """
    factors = extract_hjb_factors(V_final, atom_ids)
    costs = atom_costs if atom_costs is not None else [1.0] * len(atom_ids)

    specs = []
    for i, aid in enumerate(atom_ids):
        specs.append(RuleSpec(
            name=aid,
            posterior=np.ones(16) / 16,  # placeholder — ECAN uses V, not histograms
            conclusion_stv=(float(factors.g_scores[i]),
                            float(factors.f_scores[i])),
            premise_confidences=[float(factors.f_scores[i])],
            cost=float(costs[i]),
            touched_nodes=frozenset([aid]),
        ))
    return specs


# ═══════════════════════════════════════════════════════════════════════════
#  Mode 2: STI snapshot (simplified fallback)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ECANSnapshot:
    """Snapshot of ECAN attention state for simplified geodesic scoring.

    NOTE: This is a simplified fallback.  For full whitepaper alignment,
    use extract_hjb_factors() with the solved V_final from solve_hjb().
    Raw STI values lack the goal-directed routing information that V encodes.

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
        ECANSnapshot for simplified geodesic scoring.
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

    NOTE: This is a simplified fallback that only provides forward scores.
    It discards the goal-directed information that the HJB value function
    encodes.  For full whitepaper alignment, use extract_hjb_factors().

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
