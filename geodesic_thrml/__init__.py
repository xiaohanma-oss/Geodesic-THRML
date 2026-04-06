"""
geodesic-thrml — Geodesic inference control on thermodynamic hardware
=====================================================================

Shared scheduling layer for PLN-THRML, QuantiMORK-THRML, ECAN-THRML,
and MOSES-THRML.  Implements the geodesic controller (genenergy-logic §5)
as a finite-temperature Gibbs selector on TSU.

    T → 0:  exact Pareto selector (argmax ρ/cost)
    T > 0:  Boltzmann exploration with annealing

Modules:
    types             — Shared data classes (CascadeResult, SelectionResult, etc.)
    curriculum_thrml  — SB cascade → unified THRML factor graph
    controller_thrml  — Step selection → THRML Boltzmann sampling
    capsules          — Evidence capsule provenance tracking
    diagnostics       — H¹ cohomology detection for inference graphs
    invariants        — Five-theorem runtime safety checks
    scores            — Forward/backward factor computation + weakness gauge
    bridges/          — Sub-project adapters (PLN, QuantiMORK, ECAN, MOSES)
"""

__version__ = "0.1.0"

# Types and utilities
from geodesic_thrml.types import (  # noqa: F401
    CascadeResult,
    ResolutionLevel,
    SelectionResult,
    BatchResult,
    build_resolution_ladder,
    annealing_schedule,
    DEFAULT_TEMPERATURE,
    DEFAULT_COST_WEIGHT,
)

# Shared THRML sampling facility
from geodesic_thrml.sampling import (  # noqa: F401
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
    posterior_to_stv,
    diagnose_convergence,
)

# Primary API (THRML)
from geodesic_thrml.curriculum_thrml import (  # noqa: F401
    cascade_solve_thrml,
    rebin_coupling_weights,
    build_unified_cascade_graph,
)
from geodesic_thrml.controller_thrml import (  # noqa: F401
    select_step_thrml,
    select_batch_thrml,
    multi_step_select_thrml,
)
