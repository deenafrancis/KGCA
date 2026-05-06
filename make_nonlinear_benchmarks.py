#!/usr/bin/env python3
"""
Generate higher-dimensional synthetic benchmarks for KGCA/GCA.

This script creates two benchmark families designed to answer reviewer
concerns about (i) limited low-dimensional VAR simulations and
(ii) absence of non-VAR nonlinear dynamical systems.

Families
--------
1) hd_nlds   : higher-dimensional sparse nonlinear latent state-space model
2) l96       : coupled Lorenz-96 system with a known scalar driver->driven pair

Saved format
------------
Each replicate is written as a compressed NPZ with keys:
    X    : observed multivariate time series, shape [T, D]
    S    : latent/signals array, shape [T, K_latent]
           For l96, columns 0 and 1 are the scalar ground-truth driver/driven
           signals, followed by the full latent state.
    meta : python dict (saved as object array)
A sidecar JSON file is also written for easy inspection.

The format is intentionally aligned with the dataset pattern used in the
current project (load_dataset(...)->X,S,meta).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


Array = np.ndarray


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def standardize_cols(x: Array, eps: float = 1e-8) -> Array:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def save_dataset_npz(out_path: Path, X: Array, S: Array, meta: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X.astype(np.float32), S=S.astype(np.float32), meta=np.array(meta, dtype=object))
    json_path = out_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)


# -----------------------------------------------------------------------------
# Observation models
# -----------------------------------------------------------------------------

def make_observations(
    latent_state: Array,
    D_obs: int,
    rng: np.random.Generator,
    mixing: str = 'nonlinear',
    noise_std: float = 0.08,
) -> Array:
    """
    Map latent state [T, K] to observed signal [T, D].

    mixing='linear'    : X = S A + noise
    mixing='nonlinear' : random linear mix + smooth nonlinear transforms + cross-terms
    """
    T, K = latent_state.shape
    A = rng.normal(scale=1.0 / math.sqrt(K), size=(K, D_obs))
    X_lin = latent_state @ A

    if mixing == 'linear':
        X = X_lin
    elif mixing == 'nonlinear':
        B = rng.normal(scale=1.0 / math.sqrt(K), size=(K, D_obs))
        C = rng.normal(scale=1.0 / math.sqrt(K), size=(K, D_obs))
        X_nl1 = np.tanh(latent_state @ B)
        X_nl2 = 0.35 * np.sin(latent_state @ C)

        # weak pairwise interaction channels built from random latent pairs
        X_cross = np.zeros((T, D_obs), dtype=float)
        pair_count = max(2, D_obs // 8)
        for _ in range(pair_count):
            i, j = rng.choice(K, size=2, replace=False)
            col = rng.integers(0, D_obs)
            X_cross[:, col] += 0.25 * latent_state[:, i] * latent_state[:, j]

        X = 0.65 * X_lin + 0.25 * X_nl1 + 0.10 * X_nl2 + X_cross
    else:
        raise ValueError(f'Unknown mixing={mixing!r}')

    X = standardize_cols(X)
    X += noise_std * rng.normal(size=X.shape)
    return X


# -----------------------------------------------------------------------------
# Benchmark 1: higher-dimensional nonlinear latent state-space model
# -----------------------------------------------------------------------------

def simulate_hd_nlds(
    T: int,
    K_state: int,
    D_obs: int,
    rng: np.random.Generator,
    burn: int = 500,
    n_pairs: int = 3,
    mixing: str = 'nonlinear',
    noise_std: float = 0.05,
    null: bool = False,
) -> Tuple[Array, Array, Dict]:
    """
    Higher-dimensional sparse nonlinear state-space model.

    This is intentionally non-VAR in the sense that the latent updates include
    nonlinear self-dynamics and nonlinear directed couplings, rather than only
    linear lag matrices.

    Ground-truth directed pairs are (0->1), (2->3), ..., up to n_pairs.
    Remaining states are nuisance dynamics with no directed coupling.
    """
    if 2 * n_pairs > K_state:
        raise ValueError('Need K_state >= 2 * n_pairs')

    edges = [(2 * p, 2 * p + 1) for p in range(n_pairs)]

    total = T + burn
    s = np.zeros((total, K_state), dtype=float)
    s_prev2 = np.zeros(K_state, dtype=float)

    # small random initial condition
    s[0] = 0.2 * rng.normal(size=K_state)
    s_prev2[:] = 0.2 * rng.normal(size=K_state)

    # fixed nuisance local coupling pattern
    local = rng.normal(scale=0.12, size=(K_state, K_state))
    local *= (rng.random((K_state, K_state)) < 0.08)
    np.fill_diagonal(local, 0.0)

    for t in range(1, total):
        prev = s[t - 1]

        # stable nonlinear self dynamics + second-order memory
        base = (
            0.62 * prev
            + 0.12 * np.sin(prev)
            - 0.05 * prev**3
            + 0.10 * s_prev2
            + local @ np.tanh(prev)
        )

        if not null:
            for i, j in edges:
                # directed nonlinear influence i -> j
                base[j] += 0.55 * np.tanh(prev[i]) + 0.18 * prev[i] * np.tanh(prev[j])

        s_next = base + noise_std * rng.normal(size=K_state)
        s_next = np.clip(s_next, -3.5, 3.5)

        s[t] = s_next
        s_prev2 = prev.copy()

    s = s[burn:]
    s = standardize_cols(s)
    X = make_observations(s, D_obs=D_obs, rng=rng, mixing=mixing, noise_std=0.08)

    meta = {
        'family': 'hd_nlds_null' if null else 'hd_nlds',
        'T': int(T),
        'K_state': int(K_state),
        'D_obs': int(D_obs),
        'burn': int(burn),
        'mixing': mixing,
        'noise_std': float(noise_std),
        'true_pairs': edges if not null else [],
        'driver_idx': 0,
        'driven_idx': 1,
        'null': bool(null),
        'notes': 'Sparse nonlinear discrete-time latent dynamics with known scalar directed pairs embedded in higher dimension.',
    }
    return X, s, meta


# -----------------------------------------------------------------------------
# Benchmark 2: coupled Lorenz-96 with scalar driver/driven readouts
# -----------------------------------------------------------------------------

def l96_rhs(x: Array, forcing: float) -> Array:
    # Standard Lorenz-96: dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def rk4_step(rhs, x: Array, dt: float, *args) -> Array:
    k1 = rhs(x, *args)
    k2 = rhs(x + 0.5 * dt * k1, *args)
    k3 = rhs(x + 0.5 * dt * k2, *args)
    k4 = rhs(x + dt * k3, *args)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_coupled_l96(
    T: int,
    K_u: int,
    K_v: int,
    D_obs: int,
    rng: np.random.Generator,
    burn: int = 1000,
    dt: float = 0.01,
    steps_per_sample: int = 5,
    F_u: float = 8.0,
    F_v: float = 8.0,
    gamma: float = 0.55,
    mixing: str = 'nonlinear',
    noise_std: float = 0.03,
    null: bool = False,
) -> Tuple[Array, Array, Dict]:
    """
    Coupled Lorenz-96 benchmark.

    U evolves autonomously under Lorenz-96 dynamics.
    V evolves under Lorenz-96 plus a unidirectional forcing pattern driven by a
    scalar readout y_t = w_u^T U_t. The corresponding scalar readout on V is
    z_t = w_v^T V_t, where w_v matches the coupling pattern.

    Saved latent array S stacks [y, z, U..., V...], so columns 0 and 1 are the
    ground-truth scalar driver/driven signals compatible with the current plots.
    """
    total_samples = T + burn
    n_steps = total_samples * steps_per_sample

    # readout / coupling patterns
    idx_u = np.arange(K_u)
    idx_v = np.arange(K_v)
    w_u = np.cos(2 * np.pi * idx_u / K_u)
    w_u /= np.linalg.norm(w_u)
    w_v = np.sin(2 * np.pi * idx_v / K_v + np.pi / 4.0)
    w_v /= np.linalg.norm(w_v)

    u = F_u + 0.5 * rng.normal(size=K_u)
    v = F_v + 0.5 * rng.normal(size=K_v)

    U_series = np.zeros((total_samples, K_u), dtype=float)
    V_series = np.zeros((total_samples, K_v), dtype=float)

    def full_rhs(state: Array) -> Array:
        uu = state[:K_u]
        vv = state[K_u:]
        y = float(w_u @ uu)
        du = l96_rhs(uu, F_u)
        coupling = 0.0 if null else gamma * y * w_v
        dv = l96_rhs(vv, F_v) + coupling
        return np.concatenate([du, dv])

    state = np.concatenate([u, v])
    sample_idx = 0
    for step in range(n_steps):
        state = rk4_step(lambda s: full_rhs(s), state, dt)
        state += noise_std * math.sqrt(dt) * rng.normal(size=state.shape)
        if (step + 1) % steps_per_sample == 0:
            U_series[sample_idx] = state[:K_u]
            V_series[sample_idx] = state[K_u:]
            sample_idx += 1

    U_series = U_series[burn:]
    V_series = V_series[burn:]

    # center state anomalies
    U_series = U_series - U_series.mean(axis=0, keepdims=True)
    V_series = V_series - V_series.mean(axis=0, keepdims=True)

    y = standardize_cols((U_series @ w_u[:, None]))[:, 0]
    z = standardize_cols((V_series @ w_v[:, None]))[:, 0]

    full_state = np.concatenate([U_series, V_series], axis=1)
    full_state = standardize_cols(full_state)
    X = make_observations(full_state, D_obs=D_obs, rng=rng, mixing=mixing, noise_std=0.08)

    S = np.column_stack([y, z, full_state])
    meta = {
        'family': 'l96_null' if null else 'coupled_l96',
        'T': int(T),
        'K_u': int(K_u),
        'K_v': int(K_v),
        'K_state': int(K_u + K_v),
        'D_obs': int(D_obs),
        'burn': int(burn),
        'dt': float(dt),
        'steps_per_sample': int(steps_per_sample),
        'F_u': float(F_u),
        'F_v': float(F_v),
        'gamma': float(0.0 if null else gamma),
        'mixing': mixing,
        'driver_idx': 0,
        'driven_idx': 1,
        'driver_name': 'scalar readout of U-ring',
        'driven_name': 'scalar readout of V-ring',
        'true_pairs': [(0, 1)] if not null else [],
        'null': bool(null),
        'notes': 'Non-VAR nonlinear continuous-time latent dynamics; observations are random linear/nonlinear mixtures of full state.',
    }
    return X, S, meta


# -----------------------------------------------------------------------------
# CLI / experiment grid
# -----------------------------------------------------------------------------

@dataclass
class JobConfig:
    family: str
    tag: str
    run_idx: int
    seed: int
    T: int
    D_obs: int
    mixing: str
    null: bool
    K_state: int | None = None
    n_pairs: int | None = None
    K_u: int | None = None
    K_v: int | None = None


def run_job(job: JobConfig, out_dir: Path) -> Path:
    rng = np.random.default_rng(job.seed)

    if job.family == 'hd_nlds':
        X, S, meta = simulate_hd_nlds(
            T=job.T,
            K_state=int(job.K_state),
            D_obs=job.D_obs,
            rng=rng,
            n_pairs=int(job.n_pairs),
            mixing=job.mixing,
            null=job.null,
        )
    elif job.family == 'l96':
        X, S, meta = simulate_coupled_l96(
            T=job.T,
            K_u=int(job.K_u),
            K_v=int(job.K_v),
            D_obs=job.D_obs,
            rng=rng,
            mixing=job.mixing,
            null=job.null,
        )
    else:
        raise ValueError(f'Unknown family={job.family!r}')

    name = f'{job.tag}_run{job.run_idx}'
    out_path = out_dir / f'{name}.npz'
    meta['name'] = name
    meta['seed'] = int(job.seed)
    save_dataset_npz(out_path, X, S, meta)
    return out_path


def build_default_jobs(args: argparse.Namespace) -> List[JobConfig]:
    jobs: List[JobConfig] = []
    seeds = [args.seed + i for i in range(args.n_runs)]

    if args.family in ('hd_nlds', 'all'):
        for mixing in args.mixings:
            for is_null in ([False, True] if args.include_null else [False]):
                tag = f"hd_nlds_{mixing}{'_null' if is_null else ''}"
                for run_idx, seed in enumerate(seeds):
                    jobs.append(JobConfig(
                        family='hd_nlds',
                        tag=tag,
                        run_idx=run_idx,
                        seed=seed,
                        T=args.T_hd,
                        D_obs=args.D_hd,
                        mixing=mixing,
                        null=is_null,
                        K_state=args.K_hd,
                        n_pairs=args.P_hd,
                    ))

    if args.family in ('l96', 'all'):
        for mixing in args.mixings:
            for is_null in ([False, True] if args.include_null else [False]):
                tag = f"l96_{mixing}{'_null' if is_null else ''}"
                for run_idx, seed in enumerate(seeds):
                    jobs.append(JobConfig(
                        family='l96',
                        tag=tag,
                        run_idx=run_idx,
                        seed=seed,
                        T=args.T_l96,
                        D_obs=args.D_l96,
                        mixing=mixing,
                        null=is_null,
                        K_u=args.K_u,
                        K_v=args.K_v,
                    ))

    return jobs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Generate higher-dimensional nonlinear synthetic benchmarks for KGCA.')
    ap.add_argument('--out_dir', default='datasets', help='Where to write dataset NPZ/JSON files.')
    ap.add_argument('--family', choices=['hd_nlds', 'l96', 'all'], default='all')
    ap.add_argument('--n_runs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mixings', nargs='+', default=['linear', 'nonlinear'], choices=['linear', 'nonlinear'])
    ap.add_argument('--include_null', action='store_true', help='Also generate null-control datasets with no directed coupling.')

    # HD-NLDS defaults
    ap.add_argument('--T_hd', type=int, default=3000)
    ap.add_argument('--K_hd', type=int, default=20)
    ap.add_argument('--D_hd', type=int, default=50)
    ap.add_argument('--P_hd', type=int, default=3, help='Number of true directed pairs embedded in hd_nlds.')

    # L96 defaults
    ap.add_argument('--T_l96', type=int, default=2500)
    ap.add_argument('--K_u', type=int, default=10)
    ap.add_argument('--K_v', type=int, default=10)
    ap.add_argument('--D_l96', type=int, default=50)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_default_jobs(args)
    manifest = []
    for job in jobs:
        path = run_job(job, out_dir)
        manifest.append({**asdict(job), 'path': str(path)})
        print(f'Wrote {path}')

    manifest_path = out_dir / 'benchmark_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote {manifest_path}')


if __name__ == '__main__':
    main()
