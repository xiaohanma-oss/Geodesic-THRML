"""
geodesic_thrml.sampling — Shared THRML factor graph sampling facility
======================================================================

Thin wrappers around thrml's block Gibbs sampler that eliminate
boilerplate in PLN-THRML, QuantiMORK-THRML, and Geodesic-THRML's
own internal consumers (curriculum_thrml, controller_thrml, bridges).

Layers (bottom-up):
    1. Graph coloring   — greedy_color()
    2. Graph assembly   — assemble_sampling_program()
    3. State init       — init_block_states()
    4. Vmapped sampling — run_gibbs_sampling()
    5. Posterior extract — extract_posterior(), extract_node_posterior()
    6. Moment matching  — posterior_to_stv()
    7. Diagnostics      — diagnose_convergence()

NOT used by ECAN-THRML (Lattice Boltzmann, no factor graphs).

References:
    - tsu-architecture §III: Gibbs update on TSU
    - genenergy-logic §3.3: posterior moment matching
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import jax
import jax.numpy as jnp
from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec, SamplingSchedule, sample_states,
)
from thrml.models.discrete_ebm import CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SamplingConfig:
    """Sampling parameters for THRML Gibbs sampling.

    Each consumer project creates its own config with project-appropriate
    defaults.

    Attributes:
        n_warmup: burn-in samples to discard
        n_samples: post-warmup samples to collect
        steps_per_sample: Gibbs sweeps between recorded samples
        n_batches: independent parallel chains
        seed: random seed
    """
    n_warmup: int = 500
    n_samples: int = 2000
    steps_per_sample: int = 3
    n_batches: int = 50
    seed: int = 42

    def to_schedule(self) -> SamplingSchedule:
        """Convert to thrml SamplingSchedule."""
        return SamplingSchedule(
            n_warmup=self.n_warmup,
            n_samples=self.n_samples,
            steps_per_sample=self.steps_per_sample,
        )


# Predefined configs
PLN_DEFAULT_CONFIG = SamplingConfig(
    n_warmup=500, n_samples=2000, steps_per_sample=3, n_batches=50)
QM_DEFAULT_CONFIG = SamplingConfig(
    n_warmup=300, n_samples=1000, steps_per_sample=3, n_batches=30)
LIGHT_CONFIG = SamplingConfig(
    n_warmup=50, n_samples=200, steps_per_sample=1, n_batches=10)


# ═══════════════════════════════════════════════════════════════════════════
#  Graph coloring
# ═══════════════════════════════════════════════════════════════════════════

def greedy_color(
    names: Sequence[str],
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    """Greedy graph coloring: partition nodes into independent groups.

    Returns list of groups where no two nodes within a group are adjacent.
    Used for block partitioning in factor graphs with arbitrary topology.

    Args:
        names: node identifiers in visitation order
        adjacency: mapping from node name to set of neighbor names

    Returns:
        List of groups (lists of names).  Group count = chromatic number
        under greedy ordering.
    """
    color_of: dict[str, int] = {}
    for name in names:
        neighbor_colors = {
            color_of[nb] for nb in adjacency.get(name, set())
            if nb in color_of
        }
        c = 0
        while c in neighbor_colors:
            c += 1
        color_of[name] = c

    n_colors = max(color_of.values(), default=-1) + 1
    groups: list[list[str]] = [[] for _ in range(n_colors)]
    for name in names:
        groups[color_of[name]].append(name)
    return groups


# ═══════════════════════════════════════════════════════════════════════════
#  Graph assembly
# ═══════════════════════════════════════════════════════════════════════════

def assemble_sampling_program(
    free_blocks: list[Block],
    clamped_blocks: list[Block],
    factors: list,
    k: int | list[int] = 16,
) -> FactorSamplingProgram:
    """Build a FactorSamplingProgram from blocks and factors.

    Args:
        free_blocks: blocks to be sampled
        clamped_blocks: blocks held fixed (may be empty list)
        factors: list of CategoricalEBMFactor / SquareCategoricalEBMFactor
        k: bin count.  If int, all free blocks use the same k.
           If list[int], one k per free block (for non-square cascades).

    Returns:
        Ready-to-sample FactorSamplingProgram.
    """
    if isinstance(k, int):
        samplers = [
            CategoricalGibbsConditional(n_categories=k)
            for _ in free_blocks
        ]
    else:
        if len(k) != len(free_blocks):
            raise ValueError(
                f"k list length {len(k)} != free_blocks length {len(free_blocks)}")
        samplers = [
            CategoricalGibbsConditional(n_categories=ki)
            for ki in k
        ]

    spec = BlockGibbsSpec(free_blocks, clamped_blocks)
    return FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=samplers,
        factors=factors,
        other_interaction_groups=[],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  State initialization
# ═══════════════════════════════════════════════════════════════════════════

def init_block_states(
    blocks: list[Block],
    n_batches: int,
    k: int | list[int],
    key: jax.Array,
) -> tuple[list[jnp.ndarray], jax.Array]:
    """Initialize per-block random states for Gibbs sampling.

    Args:
        blocks: list of Block objects (free blocks)
        n_batches: number of parallel chains
        k: bin count(s) — int for uniform, list[int] per block
        key: JAX PRNG key

    Returns:
        (init_state_list, remaining_key) where each element has
        shape [n_batches, n_nodes_in_block] as uint8 in [0, k).
    """
    if isinstance(k, int):
        k_list = [k] * len(blocks)
    else:
        k_list = list(k)

    init_state = []
    for block, ki in zip(blocks, k_list):
        key, subkey = jax.random.split(key)
        init_state.append(
            jax.random.randint(
                subkey, (n_batches, len(block.nodes)),
                minval=0, maxval=ki, dtype=jnp.uint8,
            )
        )
    return init_state, key


# ═══════════════════════════════════════════════════════════════════════════
#  Vmapped Gibbs sampling
# ═══════════════════════════════════════════════════════════════════════════

def run_gibbs_sampling(
    program: FactorSamplingProgram,
    free_blocks: list[Block],
    *,
    clamped_blocks: list[Block] | None = None,
    config: SamplingConfig | None = None,
    k: int | list[int] = 16,
    init_state: list[jnp.ndarray] | None = None,
    clamped_state: list[jnp.ndarray] | None = None,
    seed: int = 42,
):
    """Run vmapped block Gibbs sampling.

    Wraps the jax.vmap + sample_states pattern used by every THRML
    consumer project.

    Args:
        program: assembled FactorSamplingProgram
        free_blocks: blocks to sample (also used as observe_blocks)
        clamped_blocks: fixed blocks (None or empty for fully free graphs)
        config: SamplingConfig (default: PLN_DEFAULT_CONFIG)
        k: bin count(s) for random initialization (ignored if init_state given)
        init_state: pre-built initial state (if None, random init)
        clamped_state: pre-built clamped state (if None, empty)
        seed: random seed

    Returns:
        Raw samples — list of per-observed-block arrays.
        Each array has shape [n_batches, n_samples, n_nodes_in_block].
    """
    if config is None:
        config = PLN_DEFAULT_CONFIG

    schedule = config.to_schedule()
    n_batches = config.n_batches
    key = jax.random.PRNGKey(seed)

    observe_blocks = list(free_blocks)

    # Initialize free-block states
    if init_state is None:
        init_state, key = init_block_states(free_blocks, n_batches, k, key)

    keys = jax.random.split(key, n_batches)

    # Dispatch on clamped vs unclamped
    has_clamped = clamped_state is not None and len(clamped_state) > 0

    if has_clamped:
        samples = jax.jit(jax.vmap(
            lambda s, c, k_: sample_states(
                k_, program, schedule, s, c, observe_blocks)
        ))(init_state, clamped_state, keys)
    else:
        samples = jax.jit(jax.vmap(
            lambda s, k_: sample_states(
                k_, program, schedule, s, [], observe_blocks)
        ))(init_state, keys)

    return samples


# ═══════════════════════════════════════════════════════════════════════════
#  Posterior extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_posterior(
    samples,
    block_idx: int,
    node_idx: int,
    k: int,
) -> np.ndarray:
    """Extract K-bin normalized posterior for a single node.

    Args:
        samples: raw output from run_gibbs_sampling
        block_idx: index into the observed blocks list
        node_idx: index of the node within the block
        k: number of bins

    Returns:
        Normalized posterior array of shape [K].
    """
    level_samples = jax.tree.leaves(samples[block_idx])
    # Each leaf: [n_samples, n_nodes_in_block] or similar
    # Flatten all leaves, then pick the right node
    if node_idx == 0 and all(
        leaf.ndim <= 1 or leaf.shape[-1] == 1 for leaf in level_samples
    ):
        # Single-node block (curriculum, controller) — flatten everything
        flat = jnp.concatenate([jnp.ravel(leaf) for leaf in level_samples])
    else:
        # Multi-node block — select the specific node
        flat = jnp.concatenate([
            jnp.ravel(leaf[..., node_idx]) for leaf in level_samples
        ])

    flat = flat.astype(jnp.int32)
    counts = jnp.bincount(flat, length=k).astype(jnp.float32)
    total = jnp.sum(counts)
    if total == 0:
        return np.ones(k) / k
    return np.array(counts / total)


def extract_node_posterior(
    samples,
    free_blocks: list[Block],
    node,
    k: int,
) -> np.ndarray:
    """Extract posterior for a node, finding it by identity in free_blocks.

    Args:
        samples: raw output from run_gibbs_sampling
        free_blocks: the free_blocks used during sampling
        node: the CategoricalNode to extract
        k: number of bins

    Returns:
        Normalized posterior array of shape [K].

    Raises:
        ValueError: if node is not found in any free block.
    """
    for bi, block in enumerate(free_blocks):
        for ni, n in enumerate(block.nodes):
            if n is node:
                return extract_posterior(samples, bi, ni, k)
    raise ValueError("Node not found in any free block")


# ═══════════════════════════════════════════════════════════════════════════
#  Moment matching
# ═══════════════════════════════════════════════════════════════════════════

def bin_centers(k: int) -> np.ndarray:
    """Evenly-spaced bin centers in (0, 1)."""
    return np.linspace(0.5 / k, 1.0 - 0.5 / k, k)


def posterior_to_stv(
    posterior: np.ndarray | jnp.ndarray,
    k: int,
    centers: np.ndarray | None = None,
) -> tuple[float, float]:
    """Convert K-bin posterior to (strength, confidence) via moment matching.

    By default, bin centers are evenly spaced in (0, 1) — suitable for
    PLN-THRML's Beta-discretized nodes.

    For QuantiMORK (arbitrary [lo, hi] ranges), pass custom centers.

    Args:
        posterior: K-bin histogram (will be normalized)
        k: number of bins
        centers: bin center values.  If None, uses standard (0, 1) centers.

    Returns:
        (strength, confidence) tuple.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    total = posterior.sum()
    if total < 1e-30:
        return 0.5, 0.0
    posterior = posterior / total

    if centers is None:
        centers = bin_centers(k)
    else:
        centers = np.asarray(centers, dtype=np.float64)

    mu = float(np.sum(posterior * centers))
    var = float(np.sum(posterior * (centers - mu) ** 2))

    if var < 1e-12:
        return mu, min(1e6 / (1e6 + 1.0), 0.9999)

    # Moment-matching: Var(Beta) = mu*(1-mu)/(n+1)
    n = mu * (1.0 - mu) / var - 1.0
    n = max(n, 0.01)
    w_eff = max(n - 2.0, 0.0)
    c = w_eff / (w_eff + 1.0)
    return mu, c


# ═══════════════════════════════════════════════════════════════════════════
#  Convergence diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_convergence(
    samples,
    block_idx: int,
    node_idx: int,
    k: int,
) -> dict:
    """Compute split-R-hat and ESS for convergence assessment.

    Uses batches as independent chains.  Each chain is split in half
    for R-hat computation.

    Args:
        samples: raw output from run_gibbs_sampling
        block_idx: index into observed blocks
        node_idx: index of node within block
        k: number of bins (used for bin-center mapping)

    Returns:
        Dict with keys:
            r_hat:     float — split-R-hat statistic (< 1.05 is good)
            ess:       int   — effective sample size
            converged: bool  — True if r_hat < 1.05 and ess > 400
    """
    centers = bin_centers(k)

    # Extract per-batch samples: [n_batches, n_samples]
    level_leaves = jax.tree.leaves(samples[block_idx])
    # Reconstruct [n_batches, n_samples] for the target node
    per_batch = []
    for leaf in level_leaves:
        arr = np.array(leaf)
        if arr.ndim == 3:
            # [n_batches, n_samples, n_nodes] → select node
            per_batch.append(arr[:, :, node_idx])
        elif arr.ndim == 2:
            per_batch.append(arr)
        else:
            continue
    if not per_batch:
        return {"r_hat": float("inf"), "ess": 0, "converged": False}

    # [n_batches, total_samples]
    chain_data = np.concatenate(per_batch, axis=1).astype(np.float64)
    # Map to continuous via bin centers
    chain_data = centers[np.clip(chain_data.astype(int), 0, k - 1)]

    n_chains, n_samples = chain_data.shape
    if n_chains < 2 or n_samples < 4:
        return {"r_hat": float("inf"), "ess": 0, "converged": False}

    # Split-R-hat: split each chain in half
    half = n_samples // 2
    splits = np.concatenate([chain_data[:, :half], chain_data[:, half:2*half]], axis=0)
    m = splits.shape[0]  # 2 * n_chains
    n = half

    chain_means = splits.mean(axis=1)
    chain_vars = splits.var(axis=1, ddof=1)
    grand_mean = chain_means.mean()

    B = n * np.var(chain_means, ddof=1)  # between-chain variance
    W = np.mean(chain_vars)  # within-chain variance

    if W < 1e-30:
        r_hat = 1.0 if B < 1e-30 else float("inf")
    else:
        var_hat = (1.0 - 1.0 / n) * W + B / n
        r_hat = float(np.sqrt(var_hat / W))

    # ESS via autocorrelation
    pooled = chain_data.mean(axis=0)  # pool chains
    pooled = pooled - pooled.mean()
    n_pool = len(pooled)
    max_lag = min(n_pool // 2, 200)

    ess = n_pool  # start with full count
    for lag in range(1, max_lag):
        autocorr = float(np.mean(pooled[:n_pool - lag] * pooled[lag:]))
        var0 = float(np.mean(pooled ** 2))
        if var0 < 1e-30:
            break
        rho = autocorr / var0
        if rho < 0.05:
            break
        ess -= 2 * rho * n_pool / max_lag

    ess = max(int(ess), 1)

    return {
        "r_hat": r_hat,
        "ess": ess,
        "converged": r_hat < 1.05 and ess > 400,
    }
