"""
geodesic_thrml.controller — Re-exports from types.py
=====================================================

The controller implementation is in controller_thrml.py (THRML
Boltzmann sampling).  This module re-exports shared types for
backwards compatibility.
"""

from geodesic_thrml.types import (  # noqa: F401
    SelectionResult,
    BatchResult,
    DEFAULT_TEMPERATURE,
    DEFAULT_COST_WEIGHT,
    annealing_schedule,
)
