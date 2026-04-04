"""
geodesic-thrml — Geodesic inference control on thermodynamic hardware
=====================================================================

Shared scheduling layer for PLN-THRML, QuantiMORK-THRML, ECAN-THRML,
and MOSES-THRML.  Implements the geodesic controller (genenergy-logic §5)
as a finite-temperature Gibbs selector on TSU.

    T → 0:  exact Pareto selector (argmax ρ/cost)
    T > 0:  Boltzmann exploration with annealing

Modules:
    curriculum   — Schrödinger Bridge Learning (coarse-to-fine cascade)
    capsules     — Evidence capsule provenance tracking
    diagnostics  — H¹ cohomology detection for inference graphs
    invariants   — Five-theorem runtime safety checks
    scores       — Forward/backward factor computation + weakness gauge
    controller   — Mixture-of-Energies step selection + annealing
    bridges/     — Sub-project adapters (PLN, QuantiMORK, ECAN, MOSES)
"""

__version__ = "0.1.0"
