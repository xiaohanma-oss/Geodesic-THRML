# Geodesic-THRML

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)

> Geodesic inference control compiled to thermodynamic factor graphs.
> Shared scheduling layer for
> [PLN-THRML](https://github.com/xiaohanma-oss/PLN-THRML),
> [QuantiMORK-THRML](https://github.com/xiaohanma-oss/QuantiMORK-THRML),
> [ECAN-THRML](https://github.com/xiaohanma-oss/ECAN-THRML), and
> [MOSES-THRML](https://github.com/xiaohanma-oss/MOSES-THRML).

## Table of Contents

- [Overview](#overview)
- [Why this matters](#why-this-matters)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Results](#results)
- [Modules](#modules)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Theoretical foundation](#theoretical-foundation)
- [Hyperon integration outlook](#hyperon-integration-outlook)
- [Cross-project comparison](#cross-project-comparison)
- [Sister Projects](#sister-projects)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

Geodesic-THRML implements the geodesic controller from the Hyperon whitepaper
(§5.2, genenergy-logic §5) — a cross-paradigm Occam bias plus compositional
scheduler.  It sits **above** all four sub-projects, deciding where each TSU
should spend its sampling budget.

```
        Geodesic Control (global scheduler)
        ┌───────┼───────┼───────┐
        ↓       ↓       ↓       ↓
    PLN-THRML  QM-THRML ECAN   MOSES
    (reasoning) (perception) (attention) (evolution)
```

As an implementation convenience, `sampling.py` also provides shared THRML
utilities (factor graph assembly, Gibbs sampling, posterior extraction) that
sibling projects can import to avoid duplicating boilerplate.

<details>
<summary><strong>New to geodesic control? (30-second primer)</strong></summary>

Imagine four inference engines competing for limited hardware time.  Each
candidate step has two scores:

| Score | Meaning | Analogy |
|-------|---------|---------|
| **f** (forward) | How strongly the premises support this step | "Can we get there from here?" |
| **g** (backward) | How useful the result is for our goal | "Does it bring us closer to the answer?" |

The controller combines them into a single energy:

    E(step) = -(log f + log g) / T  +  λ · cost

Lower energy = better step.  At high temperature **T** the controller
explores broadly; as **T** → 0 it greedily picks the best.  The product
**ρ = f · g** is a Noether invariant — it stays constant along the optimal
path (Thm 3.1), which means the controller never drifts off the geodesic.

</details>

<details>
<summary><strong>New to Hyperon's cognitive stack? (30-second primer)</strong></summary>

Hyperon is an AGI framework with four major inference subsystems:

| Module | Role | Example |
|--------|------|---------|
| **PLN** | Probabilistic logic — combines uncertain premises into conclusions | "If A→B (0.8) and B→C (0.7), how strong is A→C?" |
| **ECAN** | Attention allocation — routes resources to important atoms | "Which 100 of 10,000 concepts should be active right now?" |
| **QuantiMORK** | Perception — predictive coding with wavelet-sparse neural layers | "What does this image look like at different resolutions?" |
| **MOSES** | Program evolution — searches for programs that fit target behavior | "Find a short program that predicts this dataset." |

Each module produces candidate inference steps.  **Geodesic Control** is the
scheduler that decides which step to execute next, balancing exploration
(trying new modules) against exploitation (deepening the best current path).
Without it, each module greedily consumes all the budget.

</details>

**Technical summary**: The geodesic controller compiles the Hyperon whitepaper's
cross-paradigm Occam bias (genenergy-logic §6; energy formula from whitepaper §5.2) to a single-node THRML factor
graph: candidate steps from PLN, QuantiMORK, ECAN, and MOSES are scored by
forward reachability **f** and backward utility **g**, combined into geodesic
energy E = −(log f + log g)/T + λ·cost, and encoded as a CategoricalNode bias
for TSU Gibbs sampling. The product ρ = f·g is a Noether invariant (Thm 3.1),
and five runtime checks guard evidence conservation. `sampling.py` also
provides shared THRML utilities (graph assembly, Gibbs sampling, posterior
extraction) that sibling projects import directly.

### Core idea

At each inference step the controller evaluates:

```
E(step_i) = -(log f(i) + log g(i)) / T  +  λ·cost(i)
```

where **f** = forward reachability (from premises), **g** = backward utility
(toward goals), **T** = temperature, **λ** = cost weight.

- **T → 0**: exact Pareto selector — picks the step maximizing ρ = f·g per unit cost
- **T > 0**: Boltzmann exploration — naturally explores alternatives, annealing recovers the optimum

## Why this matters

### Cross-module scheduling

The core problem is allocating finite TSU sampling budget across four
inference subsystems that each want all of it.  Each scheduling approach
hits different bottlenecks:

|               | CPU (hand-tuned) | GPU (batched) | Geodesic on TSU |
|---------------|------------------|---------------|-----------------|
| Parallelism   | Sequential rule selection | Batched tensor ops | All candidates scored via factor graph |
| Bottleneck    | Search space explosion | Communication overhead | Mixing time (energy barriers) |
| Cross-module  | Hand-written priority heuristics | Not supported | ρ = f·g automatic Pareto across all modules |
| Exploration   | ε-greedy or random | N/A | Temperature annealing (Boltzmann) |

### Energy efficiency

The TSU architecture paper ([arXiv:2510.23972](https://arxiv.org/abs/2510.23972))
reports ~10,000× lower energy per sample vs GPU baselines (E_cell ≈ 2
femtojoules).  Geodesic step selection compiles to a single-node Boltzmann
sample — the controller decision itself costs one TSU clock cycle.

## Installation

```bash
git clone https://github.com/xiaohanma-oss/Geodesic-THRML.git
cd Geodesic-THRML
pip install -e ".[dev]"    # core (jax + numpy + thrml) + pytest
pytest                     # run all tests
```

## Quick start

### THRML factor graph path (primary)

```python
from geodesic_thrml.curriculum_thrml import cascade_solve_thrml
from geodesic_thrml.controller_thrml import select_step_thrml
from geodesic_thrml.types import build_resolution_ladder

# SB cascade as unified THRML factor graph — K=4, K=8, K=16 nodes
# coupled by non-square CategoricalEBMFactor, joint Gibbs sampling
ladder = build_resolution_ladder([4, 8, 16])
result = cascade_solve_thrml(ladder, base_prior=(0.8, 0.9))

# Step selection via single-node Boltzmann sampling on TSU
from geodesic_thrml.scores import RuleSpec
specs = [...]  # candidate inference steps
result = select_step_thrml(specs, goal_stv=(0.9, 0.95), temperature=1.0)
```

### Shared THRML utilities

```python
from geodesic_thrml.sampling import (
    assemble_sampling_program, run_gibbs_sampling,
    extract_posterior, posterior_to_stv,
    PLN_DEFAULT_CONFIG,
)
from thrml.pgm import CategoricalNode
from thrml.block_management import Block
from thrml.models.discrete_ebm import CategoricalEBMFactor

# Build your domain-specific factor graph
node = CategoricalNode()
block = Block([node])
factor = CategoricalEBMFactor([block], my_weights[None, :])
program = assemble_sampling_program([block], [], [factor], k=16)

# Sample + extract posterior — no boilerplate
samples = run_gibbs_sampling(program, [block], config=PLN_DEFAULT_CONFIG, k=16)
posterior = extract_posterior(samples, 0, 0, k=16)
s, c = posterior_to_stv(posterior, k=16)
```

## How it works

### Step 1 — Bridges collect candidates

Each sub-project reports its candidate steps as `RuleSpec` objects via a
bridge module.  A `RuleSpec` carries the posterior histogram, conclusion
truth value, premise confidences, computational cost, and the set of
graph nodes it touches.

### Step 2 — Score and rank

`scores.py` computes forward factor **f** (premise support), backward
factor **g** (goal proximity), and geodesic energy **E**.  The weakness
gauge measures non-commutativity between pairs of steps — disjoint steps
(weakness ≈ 0) can execute in parallel.

### Step 3 — Select and anneal

`controller_thrml.py` encodes the energy vector as a single
`CategoricalNode` bias and runs THRML Gibbs sampling.  An annealing
schedule (exponential / linear / cosine) lowers **T** from exploration
to exploitation over multiple rounds.

### Step 4 — Verify safety

After each step, `invariants.py` runs five O(1) checks and
`capsules.py` tracks evidence provenance:

| Check | Theorem | Cost |
|-------|---------|------|
| ρ conservation | Thm 3.1 — ρ = f·g constant along geodesics | O(n) |
| Hallucination bound | Thm 4.2 — conclusion strength ≤ evidence mass | O(1) |
| Leakage bound | Thm 5.3 — parallel leakage ≤ Σ weakness | O(n²) |
| Evidence monotonicity | Thm 6.1 — evidence non-increasing (DPI) | O(1) |
| Entropy non-decrease | Thm 7.10 — diagnostic only | O(1) |

### Hardware mapping

| Geodesic concept | thrml construct | TSU hardware |
|------------------|-----------------|--------------|
| Energy E(i) | `CategoricalNode` bias h_i | pdit bias voltage |
| Boltzmann selection | Block Gibbs sampling | Thermal relaxation |
| SB cascade K=4→8→16 | Non-square `CategoricalEBMFactor` | DTM chain (tsu-arch §II) |
| Parallel grouping | CPU-side graph coloring | Independent chip regions |

## Results

All results are from the existing test suite (169 tests, all passing).

### Cascade precision

With a Beta prior of (s=0.7, c=0.9), the unified THRML cascade
(K=4→8→16) recovers strength within ±0.15 of the prior:

| Cascade | K levels | Final strength | Deviation from prior |
|---------|----------|---------------|---------------------|
| Single level | K=8 | 0.0 < s < 1.0 | Valid range |
| Three levels | K=4→8→16 | s ≈ 0.7 | |s - 0.7| < 0.15 |

### Controller convergence

Multi-step annealing (T=2.0→0.01, exponential, 5 steps) consistently
selects the lowest-energy candidate by the final step.  At T < 1e-4 the
controller is deterministic (argmin E).

### Five-theorem safety

All invariant checks pass across the test suite:
- Hallucination bound: 0 violations (with tolerance 0.01)
- ρ conservation: 0 violations (tolerance 0.1)
- Leakage bound: 0 violations
- Evidence monotonicity: 0 violations
- Capsule merge idempotency: verified (merge(A, A) = A)

### Bridge coverage

| Bridge | Status | Posterior quality |
|--------|--------|-------------------|
| PLN | Wraps PLN-THRML sampling results → RuleSpec | Proper THRML posteriors |
| QuantiMORK | Wavelet levels → unified THRML cascade | Proper THRML posteriors |
| ECAN | HJB value function → f=|∇V|, g=exp(-V) | Proper THRML posteriors |
| MOSES | GEO-EVO f·g two-node factor graph | Proper THRML posteriors |

## Modules

| Module | Purpose |
|--------|---------|
| `sampling.py` | Shared THRML utilities — graph assembly, Gibbs sampling, posterior extraction, moment matching, convergence diagnostics |
| `curriculum_thrml.py` | SB cascade → unified THRML factor graph with non-square inter-level coupling (primary) |
| `controller_thrml.py` | Step selection → single-node THRML Boltzmann sampling (primary) |
| `scores.py` | Forward/backward factors, geodesic energy, weakness gauge, parallel grouping |
| `capsules.py` | Evidence capsule provenance tracking — prevents double-counting |
| `diagnostics.py` | H¹ cohomology detection — identifies cycles in inference graphs |
| `invariants.py` | Five-theorem runtime checks — hallucination bound, evidence monotonicity, ρ conservation |
| `controller.py` | CPU fallback — step selection, annealing schedule, batch parallel selection |
| `curriculum.py` | Re-exports shared types for backwards compatibility |
| `bridges/pln.py` | PLN bridge — wraps PLN-THRML results as RuleSpec |
| `bridges/quantimork.py` | QuantiMORK bridge — wavelet hierarchy → THRML unified cascade |
| `bridges/ecan.py` | ECAN bridge — HJB value function → THRML-sampled posteriors |
| `bridges/moses.py` | MOSES bridge — GEO-EVO deme fitness → THRML two-node f·g factor graph |

## API reference

### Shared sampling facility (`geodesic_thrml.sampling`)

**Configuration**:

| Class / Constant | Description |
|------------------|-------------|
| `SamplingConfig(n_warmup, n_samples, steps_per_sample, n_batches, seed)` | Sampling parameters for THRML Gibbs sampling |
| `PLN_DEFAULT_CONFIG` | 500 warmup, 2000 samples, 50 batches (PLN-THRML default) |
| `QM_DEFAULT_CONFIG` | 300 warmup, 1000 samples, 30 batches (QuantiMORK default) |
| `LIGHT_CONFIG` | 50 warmup, 200 samples, 10 batches (controller/lightweight) |

**Graph assembly & sampling**:

| Function | Description |
|----------|-------------|
| `greedy_color(names, adjacency)` | Partition nodes into independent groups via greedy graph coloring |
| `assemble_sampling_program(free_blocks, clamped_blocks, factors, k)` | Build `FactorSamplingProgram` from blocks and factors; `k` may be int or list[int] for non-square cascades |
| `init_block_states(blocks, n_batches, k, key)` | Initialize random per-block states for Gibbs sampling |
| `run_gibbs_sampling(program, free_blocks, *, config, k, ...)` | Run vmapped block Gibbs sampling; returns per-block sample arrays |

**Posterior extraction & moment matching**:

| Function | Description |
|----------|-------------|
| `extract_posterior(samples, block_idx, node_idx, k)` | Extract K-bin normalized posterior for a single node |
| `extract_node_posterior(samples, free_blocks, node, k)` | Extract posterior by finding node identity in free_blocks |
| `posterior_to_stv(posterior, k, centers=None)` | Convert K-bin posterior to (strength, confidence) via moment matching |
| `diagnose_convergence(samples, block_idx, node_idx, k)` | Compute split-R-hat and ESS for convergence assessment |

### Curriculum (`geodesic_thrml.curriculum_thrml`)

| Function | Description |
|----------|-------------|
| `cascade_solve_thrml(ladder, base_prior, schedule, ...)` | Run coarse-to-fine SB cascade as a unified THRML factor graph |
| `build_unified_cascade_graph(ladder, base_prior, coupling_precision)` | Build ONE factor graph for the entire K cascade (returns dict with `program`, `nodes`, `free_blocks`, `factors`, `ladder`) |
| `rebin_coupling_weights(source_k, target_k, precision)` | Build K_src × K_tgt inter-level coupling weight matrix |

### Controller (`geodesic_thrml.controller_thrml`)

| Function | Description |
|----------|-------------|
| `select_step_thrml(specs, goal_stv, temperature, cost_weight, seed)` | Select next step via single-node THRML Boltzmann sampling → `SelectionResult` |
| `select_batch_thrml(specs, goal_stv, temperature, ..., weakness_threshold)` | Select a batch of parallel-safe steps → `BatchResult` |
| `multi_step_select_thrml(specs, goal_stv, n_steps, t_start, t_end, strategy)` | Annealed multi-step selection → list of `SelectionResult` |

### Scores (`geodesic_thrml.scores`)

| Function / Class | Description |
|------------------|-------------|
| `RuleSpec(name, posterior, conclusion_stv, premise_confidences, cost, touched_nodes)` | Specification for a candidate inference step |
| `compute_forward_scores(specs)` | Forward factor f(r): premise support (normalized evidence weights) |
| `compute_backward_scores(specs, goal_stv)` | Backward factor g(r): goal proximity (exp(−‖Δstv‖²)) |
| `compute_geodesic_energy(f, g, costs, temperature, cost_weight)` | Geodesic energy E = −(log f + log g)/T + λ·cost |
| `energy_to_probs(energy, temperature)` | Convert energy to Boltzmann probabilities P ∝ exp(−E/T) |
| `compute_rho(f, g)` | Reinforcement ρ = f · g (Noether invariant along geodesics) |
| `compute_weakness(step_a, step_b)` | Weakness gauge: Jaccard overlap of touched_nodes (0 = commutative) |
| `partition_parallel_groups(specs, weakness_threshold)` | Partition steps into parallel-safe groups via greedy coloring |

### Capsules (`geodesic_thrml.capsules`)

| Function / Class | Description |
|------------------|-------------|
| `EvidenceCapsule(sources, weights)` | Provenance record; `.mass` = total evidence, `.size` = distinct items |
| `merge_capsules(a, b)` | Overlap-aware merge: shared sources counted once (idempotent) |
| `overlap_ratio(a, b)` | Fraction of shared evidence (Jaccard) |
| `double_counting_penalty(capsules)` | Σ mass(c_i) − mass(merge_all) |
| `make_capsule(source_id, weight)` | Create capsule from a single evidence item |

### Diagnostics (`geodesic_thrml.diagnostics`)

| Function / Class | Description |
|------------------|-------------|
| `InferenceGraph(nodes, edges)` | Directed inference graph |
| `TopologyReport(classification, betti_1, n_components, cycle_edges)` | Result of topological analysis ('tree' or 'cyclic') |
| `compute_betti_1(graph)` | First Betti number β₁ = \|E\| − \|V\| + n_components |
| `classify_topology(graph)` | Classify graph → `TopologyReport` |
| `locate_cycles(graph)` | Find back-edges participating in cycles |
| `suggest_strategy(report)` | Suggest sampling strategy: 'local', 'annealing', or 'dtm' |

### Invariants (`geodesic_thrml.invariants`)

| Function / Class | Description |
|------------------|-------------|
| `InvariantViolation(theorem, message, observed, bound)` | Record of a violated invariant |
| `check_hallucination_bound(strength, capsule, tolerance)` | Thm 4.2: conclusion strength ≤ capsule mass |
| `check_evidence_monotonicity(pre, post, tolerance)` | Thm 6.1: total evidence non-increasing |
| `check_leakage_bound(leakage, weaknesses, tolerance)` | Thm 5.3: reordering leakage ≤ Σ weakness |
| `check_rho_conservation(rho_sequence, tolerance)` | Thm 3.1: ρ constant along geodesic paths |
| `check_entropy_non_decrease(pre, post, tolerance)` | Thm 7.10: diagnostic-level entropy check |
| `run_all_checks(...)` | Run all applicable checks; returns list of violations |

### Bridges (`geodesic_thrml.bridges`)

**PLN** (`bridges.pln`):

| Function | Description |
|----------|-------------|
| `pln_result_to_rule_spec(name, posterior, strength, confidence, ...)` | Wrap a PLN-THRML sampling result as `RuleSpec` |
| `pln_results_to_rule_specs(results)` | Batch-convert list of dicts → list of `RuleSpec` |

**QuantiMORK** (`bridges.quantimork`):

| Function / Class | Description |
|------------------|-------------|
| `WaveletLevelSpec(level, band, dim, weight_shape, max_connections, ...)` | Specification for one wavelet level |
| `collect_wavelet_level_specs(model, block_idx)` | Extract wavelet level specs from a `WaveletPCTransformer` |
| `wavelet_to_resolution_ladder(specs)` | Convert wavelet specs → curriculum `ResolutionLevel` ladder |
| `make_quantimork_cascade_solver_thrml(specs, coupling_precision)` | Build unified THRML cascade solver from wavelet hierarchy |

**ECAN** (`bridges.ecan`):

| Function / Class | Description |
|------------------|-------------|
| `HJBFactors(f_scores, g_scores, atom_ids, v_values)` | Forward/backward factors from HJB value function |
| `extract_hjb_factors(V_final, atom_ids)` | Extract geodesic factors from solved HJB V(x): f = \|∇V\|, g = exp(−V) |
| `hjb_to_rule_specs_thrml(V_final, atom_ids, ...)` | HJB value function → `RuleSpec` with THRML-sampled posteriors |
| `extract_ecan_snapshot(sti_values, atom_ids, lti_values)` | Create STI/LTI snapshot (simplified fallback) |
| `sti_to_forward_scores(snapshot)` | STI → normalized forward scores (fallback, no goal info) |

**MOSES** (`bridges.moses`):

| Function / Class | Description |
|------------------|-------------|
| `DemeSpec(deme_id, n_knobs, best_fitness, acceptance_rate, is_done, ...)` | One MOSES deme translated for GEO-EVO scoring |
| `collect_deme_specs(metapop)` | Extract deme specs from a `Metapopulation` |
| `compute_forward_reachability(deme_specs)` | GEO-EVO f: normalized best fitness per deme |
| `compute_backward_compatibility(deme_specs, target_behavior_fn)` | GEO-EVO g: target proximity or acceptance rate proxy |
| `deme_specs_to_rule_specs_thrml(deme_specs, ...)` | Deme specs → `RuleSpec` with THRML two-node f·g factor graph |

## Project structure

```
geodesic_thrml/
├── __init__.py
├── types.py                  # Shared data classes (CascadeResult, SelectionResult, etc.)
├── sampling.py               # Shared THRML sampling facility (graph assembly, Gibbs, posterior)
├── curriculum_thrml.py       # SB cascade → unified THRML factor graph
├── controller_thrml.py       # Step selection → THRML Boltzmann sampling
├── scores.py                 # Forward/backward factors + weakness gauge
├── capsules.py               # Evidence capsule provenance
├── diagnostics.py            # H¹ cohomology detection
├── invariants.py             # Five-theorem runtime checks
└── bridges/
    ├── pln.py                # PLN bridge — THRML cascade solver
    ├── quantimork.py         # QuantiMORK bridge — wavelet → THRML cascade
    ├── ecan.py               # ECAN bridge — HJB V → THRML posteriors
    └── moses.py              # MOSES bridge — deme f·g → THRML posteriors
tests/
├── test_sampling.py          # Shared sampling facility tests
├── test_curriculum_thrml.py  # Unified cascade tests
├── test_controller_thrml.py  # THRML selection tests
├── test_bridges.py           # Bridge tests (all THRML)
├── test_capsules.py          # Merge idempotency
├── test_diagnostics.py       # Tree/cycle detection
├── test_invariants.py        # Five-theorem assertions
└── test_scores.py            # Forward/backward factors + weakness gauge
```

## Theoretical foundation

- **Geodesic control**: genenergy-logic §5 (Pareto dominance on ρ = f⊗g / cost)
- **ρ conservation**: Thm 3.1 from evidence-conservation-theorems (Noether invariant)
- **Hallucination bound**: Thm 4.2 (conclusion strength ≤ initial evidence mass)
- **Weakness gauge**: noncommutative-evidence §2.2 (quantifies step non-commutativity)
- **H¹ diagnostics**: cohomology-quantum-evidence (tree vs cyclic topology)
- **SB Learning**: Hyperon whitepaper "bridging" (geodesics from simple to accurate models)

## Hyperon integration outlook

See [PLN-THRML README](https://github.com/xiaohanma-oss/PLN-THRML#hyperon-integration-outlook)
for the full heterogeneous pipeline design (Control → Compile → Sample).

This project contributes the **Control tier**: geodesic scheduling that
allocates finite TSU sampling budget across the four sub-projects. In the
three-tier pipeline:

| Tier | Hardware | What Geodesic-THRML does |
|------|----------|--------------------------|
| Control | CPU | Collect `RuleSpec` from bridges, run five-theorem safety checks, parallel grouping |
| Compile | CPU | Encode geodesic energy E(i) as CategoricalNode bias |
| Sample | TSU | Single-node Boltzmann selection (one clock cycle per decision) |

The controller decision itself is lightweight (one TSU sample), but it
coordinates budget across workloads that each consume thousands of TSU
cycles (PLN factor graph inference, QuantiMORK wavelet PC, ECAN LBM
diffusion, MOSES evolutionary search). Whether the gain from principled
scheduling outweighs the overhead of bridge construction and safety checks
is an open question — the current test suite validates correctness, not
end-to-end throughput.

## Cross-project comparison

| | PLN-THRML | ECAN-THRML | QuantiMORK-THRML | MOSES-THRML | Geodesic-THRML |
|--|-----------|------------|------------------|-------------|----------------|
| Compiles | PLN inference rules | ECAN attention diffusion | Predictive coding layers | MOSES program search | Cross-module scheduling |
| TSU primitive | Factor graph Gibbs | LBM collision + streaming | Factor graph Gibbs (wavelet) | EBM Gibbs (SA equiv.) | Single-node Boltzmann |
| Variables | TruthValue (s, c) | STI density field | Wavelet coefficients | Knob vector (binary) | Geodesic energy E(i) |
| CPU handles | Rule structure (Q_logic) | Goal/boundary conditions | PC iterations, attention | Representation building | Bridge construction, safety checks |
| TSU handles | Posterior sampling | Diffusion/advection | Per-level weight sampling | Deme-local knob search | Step selection (1 sample) |
| Formal basis | Boltzmann factor graph | Navier-Stokes ↔ LBM | PC free energy ↔ EBM | MH ↔ Gibbs | ρ = f·g Noether invariant |
| Upstream baseline | [trueagi-io/PLN](https://github.com/trueagi-io/PLN) | [metta-attention](https://github.com/iCog-Labs-Dev/metta-attention) | [PC-Transformers](https://github.com/iCog-Labs-Dev/PC-Transformers) | [metta-moses](https://github.com/iCog-Labs-Dev/metta-moses) | Hyperon whitepaper §5.2 |

## Sister Projects

Five projects compiling Hyperon's cognitive architecture to thermodynamic hardware:

| Project | What it compiles |
|---------|-----------------|
| [PLN-THRML](https://github.com/xiaohanma-oss/PLN-THRML) | Probabilistic inference → Boltzmann energy tables |
| [ECAN-THRML](https://github.com/xiaohanma-oss/ECAN-THRML) | Attention diffusion → Lattice Boltzmann simulation |
| [MOSES-THRML](https://github.com/xiaohanma-oss/MOSES-THRML) | Program evolution → Boltzmann sampling |
| [QuantiMORK-THRML](https://github.com/xiaohanma-oss/QuantiMORK-THRML) | Predictive coding → wavelet-sparse factor graphs |
| **[Geodesic-THRML](https://github.com/xiaohanma-oss/Geodesic-THRML)** | **Unified geodesic scheduler for all above** |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Acknowledgements

- [Hyperon](https://github.com/trueagi-io/hyperon-experimental) — TrueAGI
- [thrml](https://github.com/extropic-ai/thrml) — Extropic AI factor graph library

## License

[MIT](LICENSE) — Copyright (c) 2026 Xiaohan Ma
