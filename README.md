# Geodesic-THRML

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)

> Geodesic inference control compiled to thermodynamic factor graphs.
> Shared scheduling layer for
> [PLN-THRML](https://github.com/xiaohanma-oss/PLN-THRML),
> [QuantiMORK-THRML](https://github.com/xiaohanma-oss/QuantiMORK-THRML),
> [ECAN-THRML](https://github.com/xiaohanma-oss/ECAN-THRML), and
> MOSES-THRML.

## Overview

Geodesic-THRML implements the geodesic controller from the Hyperon whitepaper
(§5.2, genenergy-logic §5) as a finite-temperature Gibbs selector.  It sits
**above** all four sub-projects, deciding where each TSU should spend its
sampling budget.

```
        Geodesic Control (global scheduler)
        ┌───────┼───────┼───────┐
        ↓       ↓       ↓       ↓
    PLN-THRML  QM-THRML ECAN   MOSES
    (reasoning) (perception) (attention) (evolution)
```

### Core idea

At each inference step the controller evaluates:

```
E(step_i) = -(log f(i) + log g(i)) / T  +  λ·cost(i)
```

where **f** = forward reachability (from premises), **g** = backward utility
(toward goals), **T** = temperature, **λ** = cost weight.

- **T → 0**: exact Pareto selector — picks the step maximizing ρ = f·g per unit cost
- **T > 0**: Boltzmann exploration — naturally explores alternatives, annealing recovers the optimum

## Modules

| Module | Purpose |
|--------|---------|
| `curriculum.py` | Schrödinger Bridge Learning — coarse-to-fine cascade (K=4→8→16 for PLN, wavelet levels for QuantiMORK) |
| `capsules.py` | Evidence capsule provenance tracking — prevents double-counting |
| `diagnostics.py` | H¹ cohomology detection — identifies cycles in inference graphs |
| `invariants.py` | Five-theorem runtime checks — hallucination bound, evidence monotonicity, ρ conservation |
| `scores.py` | Forward/backward factors, geodesic energy, weakness gauge, parallel grouping |
| `controller.py` | Mixture-of-Energies step selection, annealing schedule, batch parallel selection |
| `bridges/pln.py` | PLN bridge — precomputes rule posteriors via pln_thrml.beta |
| `bridges/quantimork.py` | QuantiMORK bridge — wavelet level specs extraction |
| `bridges/ecan.py` | ECAN bridge — STI/LTI snapshot to forward scores |
| `bridges/moses.py` | MOSES bridge — program skeleton specs (stub, not yet started) |

## Installation

```bash
git clone https://github.com/xiaohanma-oss/Geodesic-THRML.git
cd Geodesic-THRML
pip install -e ".[dev]"    # core + pytest + PLN bridge
pytest                     # run all tests
```

## Quick start

```python
from geodesic_thrml.curriculum import cascade_solve, build_resolution_ladder

# Coarse-to-fine PLN inference: K=4 → K=8 → K=16
ladder = build_resolution_ladder(k_levels=[4, 8, 16])
result = cascade_solve(rule_fn, premises_stv=(0.8, 0.9), ladder=ladder)
# result.strength, result.confidence — within ±0.05/±0.15 of direct K=16
```

## Theoretical foundation

- **Geodesic control**: genenergy-logic §5 (Pareto dominance on ρ = f⊗g / cost)
- **ρ conservation**: Thm 3.1 from evidence-conservation-theorems (Noether invariant)
- **Hallucination bound**: Thm 4.2 (conclusion strength ≤ initial evidence mass)
- **Weakness gauge**: noncommutative-evidence §2.2 (quantifies step non-commutativity)
- **H¹ diagnostics**: cohomology-quantum-evidence (tree vs cyclic topology)
- **SB Learning**: Hyperon whitepaper "bridging" (geodesics from simple to accurate models)

## Project structure

```
geodesic_thrml/
├── __init__.py
├── curriculum.py         # SB Learning coarse-to-fine cascade
├── capsules.py           # Evidence capsule provenance
├── diagnostics.py        # H¹ cohomology detection
├── invariants.py         # Five-theorem runtime checks
├── scores.py             # Forward/backward factors + weakness gauge
├── controller.py         # Step selection + annealing
└── bridges/
    ├── pln.py            # PLN-THRML bridge
    ├── quantimork.py     # QuantiMORK-THRML bridge
    ├── ecan.py           # ECAN-THRML bridge (future)
    └── moses.py          # MOSES-THRML bridge (future)
tests/
├── test_curriculum.py    # K cascade verification
├── test_capsules.py      # Merge idempotency
├── test_diagnostics.py   # Tree/cycle detection
├── test_invariants.py    # Five-theorem assertions
├── test_scores.py        # Forward/backward factors + weakness gauge
├── test_controller.py    # Mixture-of-Energies + annealing
└── test_bridges.py       # Bridge-specific regression tests
```

## Known limitations

- **thrml Mixture-of-Energies**: `SquareCategoricalEBMFactor` requires square
  weight tensors (same K for both blocks). The controller's joint (rule, conclusion)
  factor graph must use K=16 with zero-padding for unused rule slots when n_rules < 16.
- **MOSES bridge**: stub only — MOSES-THRML not yet started.
- **ECAN bridge**: no thrml factor graph integration yet (ECAN uses Lattice Boltzmann).

## Contributing

See [PLN-THRML CONTRIBUTING.md](https://github.com/xiaohanma-oss/PLN-THRML/blob/main/CONTRIBUTING.md)
for shared contribution guidelines across the -THRML project family.
