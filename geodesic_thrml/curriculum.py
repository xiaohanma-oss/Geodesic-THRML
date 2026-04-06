"""
geodesic_thrml.curriculum — Re-exports from types.py
=====================================================

The curriculum implementation is in curriculum_thrml.py (unified THRML
factor graph).  This module re-exports shared types for backwards
compatibility.
"""

from geodesic_thrml.types import (  # noqa: F401
    CascadeResult,
    ResolutionLevel,
    build_resolution_ladder,
)
