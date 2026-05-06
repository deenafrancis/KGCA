#!/usr/bin/env python3
"""
Evaluation of PCA, ICA, linear GCA, exact KGCA, and RFF-KGCA on the
synthetic benchmarks produced by make_nonlinear_benchmarks.py.

Features
--------
- Evaluates grouped datasets such as:
    hd_nlds_linear_run0.npz
    hd_nlds_nonlinear_run3.npz
    hd_nlds_linear_null_run7.npz
    l96_linear_run0.npz
    l96_nonlinear_null_run4.npz
- Saves one JSON per completed run in per_run/ so interruption is safe.
- Rebuilds runs.csv and runs_restricted.csv from per-run JSON files.
- Skips completed runs automatically on restart.
- Optionally saves rep*_cache.npz for later plotting.
- Handles both exact KGCA and RFF-KGCA.
"""

from __future__ import annotations

import argparse
import csv
import glob
import inspect
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from methods.pca import pca_fit, pca_transform
from methods.ica import ica_fit, ica_transform
from methods.gca import run_gca_original_multi
from methods.kgca import KGCAConfig, run_kernel_gca
from kgca_rff import KGCARFFConfig, run_kernel_gca_rff
from metrics import granger_delta, granger_gap, circshift_pvalue


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def atomic_json_dump(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_dataset_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    S = data["S"]
    meta_obj = data.get("meta", None)
    if meta_obj is None:
        meta: Dict[str, Any] = {}
    elif isinstance(meta_obj, np.ndarray) and meta_obj.dtype == object and meta_obj.size == 1:
        meta = meta_obj.item()
    elif isinstance(meta_obj, np.ndarray) and meta_obj.dtype == object:
        try:
            meta = meta_obj.item()
        except Exception:
            meta = {"meta_raw": repr(meta_obj)}
    elif isinstance(meta_obj, dict):
        meta = meta_obj
    else:
        meta = {"meta_raw": repr(meta_obj)}
    return X, S, meta


def discover_dataset_groups(dataset_dir: str | Path, pattern: str = "*.npz") -> Dict[str, List[str]]:
    paths = sorted(glob.glob(os.path.join(str(dataset_dir), pattern)))
    groups: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        name = Path(p).stem
        m = re.match(r"^(.*)_run(\d+)$", name)
        if not m:
            continue
        tag = m.group(1)
        groups[tag].append(p)
    return {k: sorted(v) for k, v in groups.items()}


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def zscore_cols(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)


def topk_pairs_metrics(C_te: np.ndarray, L: int, lam: float, k: int = 2):
    M = C_te.shape[1]
    pairs = []
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            y = C_te[:, i]
            z = C_te[:, j]
            d = granger_delta(y, z, L, lam)
            g = d - granger_delta(z, y, L, lam)
            pairs.append((i, j, d, g))
    pairs.sort(key=lambda t: t[2], reverse=True)
    top = pairs[:k]
    return float(np.mean([t[2] for t in top])), float(np.mean([t[3] for t in top])), top


def fixed_pairs_metrics(
    C_te: np.ndarray,
    L: int,
    lam: float,
    pairs,
    seed_base: int = 0,
    n_shifts: int = 200,
    min_shift: int = 50,
):
    deltas, gaps, pvals = [], [], []
    for idx, (i, j) in enumerate(pairs):
        y = C_te[:, i]
        z = C_te[:, j]
        d = granger_delta(y, z, L, lam)
        g = granger_gap(y, z, L, lam)
        _, pv = circshift_pvalue(
            y, z, L, lam,
            n_shifts=n_shifts,
            min_shift=min_shift,
            seed=seed_base + idx,
        )
        deltas.append(d)
        gaps.append(g)
        pvals.append(pv)
    return float(np.mean(deltas)), float(np.mean(gaps)), float(np.mean(np.array(pvals) < 0.05))


def learned_pair_metrics(
    Y_te: np.ndarray,
    Z_te: np.ndarray,
    L: int,
    lam: float,
    seed_base: int = 0,
    n_shifts: int = 200,
    min_shift: int = 50,
    topk: int = 2,
):
    P = Y_te.shape[1]
    deltas = [granger_delta(Y_te[:, p], Z_te[:, p], L, lam) for p in range(P)]
    gaps = [granger_gap(Y_te[:, p], Z_te[:, p], L, lam) for p in range(P)]
    topk = min(topk, P)
    topk_delta = float(np.mean(sorted(deltas, reverse=True)[:topk]))
    topk_gap = float(np.mean(sorted(gaps, reverse=True)[:topk]))
    pairmean_delta = float(np.mean(deltas))
    pairmean_gap = float(np.mean(gaps))
    pvals = []
    for p in range(P):
        _, pv = circshift_pvalue(
            Y_te[:, p], Z_te[:, p], L, lam,
            n_shifts=n_shifts,
            min_shift=min_shift,
            seed=seed_base + p,
        )
        pvals.append(pv)
    sig_rate = float(np.mean(np.array(pvals) < 0.05))
    return {
        "topk_delta": topk_delta,
        "topk_gap": topk_gap,
        "pairmean_delta": pairmean_delta,
        "pairmean_gap": pairmean_gap,
        "sig_rate": sig_rate,
        "deltas": [float(x) for x in deltas],
        "gaps": [float(x) for x in gaps],
        "pvals": [float(x) for x in pvals],
    }


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

def instantiate_config(cls, **kwargs):
    sig = inspect.signature(cls)
    supported = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in kwargs:
            supported[name] = kwargs[name]
    return cls(**supported)


def default_pairs_for_tag(tag: str, meta0: Dict[str, Any], args: argparse.Namespace) -> int:
    if args.override_P is not None:
        return args.override_P
    if "l96" in tag:
        return args.P_l96
    if "hd_nlds" in tag:
        if "true_pairs" in meta0 and isinstance(meta0["true_pairs"], (list, tuple)):
            return max(1, len(meta0["true_pairs"]))
        return args.P_hd
    if "true_pairs" in meta0 and isinstance(meta0["true_pairs"], (list, tuple)) and len(meta0["true_pairs"]) > 0:
        return len(meta0["true_pairs"])
    return args.P_hd


def default_dims_for_unsupervised(P: int, min_components: int = 6) -> int:
    return max(2 * P, min_components)


# -----------------------------------------------------------------------------
# Resume / CSV rebuild helpers
# -----------------------------------------------------------------------------

def per_run_json_path(outdir: Path, run_idx: int) -> Path:
    return outdir / "per_run" / f"rep{run_idx}_metrics.json"


def load_completed_runs(outdir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    per_run_dir = outdir / "per_run"
    if not per_run_dir.exists():
        return out
    for p in sorted(per_run_dir.glob("rep*_metrics.json")):
        m = re.match(r"rep(\d+)_metrics\.json$", p.name)
        if not m:
            continue
        run_idx = int(m.group(1))
        with open(p, "r") as f:
            out[run_idx] = json.load(f)
    return out


def rebuild_csvs_from_json(outdir: Path):
    completed = load_completed_runs(outdir)
    runs_csv = outdir / "runs.csv"
    runsR_csv = outdir / "runs_restricted.csv"

    with open(runs_csv, "w", newline="") as f1, open(runsR_csv, "w", newline="") as f2:
        w1 = csv.writer(f1)
        w2 = csv.writer(f2)

        w1.writerow([
            "run", "file",
            "kgca_top2_delta_test", "kgca_top2_gap_test", "kgca_sig_rate",
            "rff_top2_delta_test", "rff_top2_gap_test", "rff_sig_rate",
            "gca_top2_delta_test", "gca_top2_gap_test", "gca_sig_rate",
            "pca_top2_delta_test", "pca_top2_gap_test", "pca_sig_rate",
            "ica_top2_delta_test", "ica_top2_gap_test", "ica_sig_rate",
        ])

        w2.writerow([
            "run", "file",
            "pca_fixed_delta_test", "pca_fixed_gap_test", "pca_fixed_sig_rate",
            "ica_fixed_delta_test", "ica_fixed_gap_test", "ica_fixed_sig_rate",
            "gca_pairmean_delta_test", "gca_pairmean_gap_test", "gca_sig_rate",
            "kgca_pairmean_delta_test", "kgca_pairmean_gap_test", "kgca_sig_rate",
            "rff_pairmean_delta_test", "rff_pairmean_gap_test", "rff_sig_rate",
        ])

        for run_idx in sorted(completed):
            r = completed[run_idx]
            w1.writerow([
                r["run"], r["file"],
                r["kgca"]["topk_delta"], r["kgca"]["topk_gap"], r["kgca"]["sig_rate"],
                r["rff"]["topk_delta"], r["rff"]["topk_gap"], r["rff"]["sig_rate"],
                r["gca"]["topk_delta"], r["gca"]["topk_gap"], r["gca"]["sig_rate"],
                r["pca"]["top2_delta"], r["pca"]["top2_gap"], r["pca"]["sig_rate"],
                r["ica"]["top2_delta"], r["ica"]["top2_gap"], r["ica"]["sig_rate"],
            ])

            w2.writerow([
                r["run"], r["file"],
                r["pca"]["fixed_delta"], r["pca"]["fixed_gap"], r["pca"]["fixed_sig"],
                r["ica"]["fixed_delta"], r["ica"]["fixed_gap"], r["ica"]["fixed_sig"],
                r["gca"]["pairmean_delta"], r["gca"]["pairmean_gap"], r["gca"]["sig_rate"],
                r["kgca"]["pairmean_delta"], r["kgca"]["pairmean_gap"], r["kgca"]["sig_rate"],
                r["rff"]["pairmean_delta"], r["rff"]["pairmean_gap"], r["rff"]["sig_rate"],
            ])

        f1.flush()
        os.fsync(f1.fileno())
        f2.flush()
        os.fsync(f2.fileno())


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------

def evaluate_group(tag: str, files: Sequence[str], out_root: Path, args: argparse.Namespace) -> Path:
    X0, S0, meta0 = load_dataset_npz(files[0])
    P = default_pairs_for_tag(tag, meta0, args)
    M = default_dims_for_unsupervised(P, min_components=args.min_unsup_components)
    fixed_pairs = [(2 * i, 2 * i + 1) for i in range(P)]

    if args.fresh:
        outdir = out_root / f"{tag}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        outdir = out_root / f"{tag}_eval"

    outdir = out_root / f"{tag}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "per_run").mkdir(parents=True, exist_ok=True)

    config = {
        "dataset_tag": tag,
        "n_files": len(files),
        "P": P,
        "M_unsup": M,
        "L": args.L,
        "lam": args.lam,
        "train_frac": args.train_frac,
        "iters": args.iters,
        "lr": args.lr,
        "gca_restarts": args.gca_restarts,
        "rff_dim": args.rff_dim,
        "rff_gamma": args.rff_gamma,
        "n_shifts": args.n_shifts,
        "min_shift": args.min_shift,
        "save_caches": bool(args.save_caches),
        "source_meta_example": meta0,
    }
    atomic_json_dump(config, outdir / "config.json")

    kg_cfg_base = {
        "L": args.L,
        "lam": args.lam,
        "lr": args.lr,
        "iters": args.iters,
        "seed": 0,
        "n_pairs": P,
    }
    rff_cfg_base = {
        "L": args.L,
        "lam": args.lam,
        "lr": args.lr,
        "iters": args.iters,
        "seed": 0,
        "n_pairs": P,
        "rff_dim": args.rff_dim,
        "D_rff": args.rff_dim,
        "m": args.rff_dim,
        "gamma": args.rff_gamma,
    }

    completed = load_completed_runs(outdir)
    if completed:
        print(f"[resume] Found {len(completed)} completed runs in {outdir}")
        rebuild_csvs_from_json(outdir)

    for run_idx, path in enumerate(files):
        rep_json = per_run_json_path(outdir, run_idx)
        if rep_json.exists():
            print(f"[skip] {tag} run {run_idx} already completed")
            continue

        print(f"[run] {tag} run {run_idx+1}/{len(files)} : {os.path.basename(path)}")
        X, S, meta = load_dataset_npz(path)
        T = X.shape[0]
        Ttr = int(T * args.train_frac)
        X_tr, X_te = X[:Ttr], X[Ttr:]

        # PCA
        pcam = pca_fit(X_tr)
        C_pca_te = pca_transform(pcam, X_te, n_components=M)
        C_pca_te = zscore_cols(C_pca_te)
        pca_top2_d, pca_top2_g, pca_top_pairs = topk_pairs_metrics(C_pca_te, args.L, args.lam, k=min(2, P))
        pca_pvals = []
        for q, (i, j, _, _) in enumerate(pca_top_pairs):
            _, pv = circshift_pvalue(
                C_pca_te[:, i], C_pca_te[:, j], args.L, args.lam,
                n_shifts=args.n_shifts, min_shift=args.min_shift, seed=1000 + run_idx * 100 + q
            )
            pca_pvals.append(pv)
        pca_sig = float(np.mean(np.array(pca_pvals) < 0.05)) if pca_pvals else 0.0
        pca_fd, pca_fg, pca_fs = fixed_pairs_metrics(
            C_pca_te, args.L, args.lam, fixed_pairs,
            seed_base=5000 + run_idx * 100,
            n_shifts=args.n_shifts,
            min_shift=args.min_shift,
        )
        print("  PCA done")

        # ICA
        icam = ica_fit(X_tr, n_components=M, seed=run_idx)
        C_ica_te = ica_transform(icam, X_te)
        C_ica_te = zscore_cols(C_ica_te)
        ica_top2_d, ica_top2_g, ica_top_pairs = topk_pairs_metrics(C_ica_te, args.L, args.lam, k=min(2, P))
        ica_pvals = []
        for q, (i, j, _, _) in enumerate(ica_top_pairs):
            _, pv = circshift_pvalue(
                C_ica_te[:, i], C_ica_te[:, j], args.L, args.lam,
                n_shifts=args.n_shifts, min_shift=args.min_shift, seed=2000 + run_idx * 100 + q
            )
            ica_pvals.append(pv)
        ica_sig = float(np.mean(np.array(ica_pvals) < 0.05)) if ica_pvals else 0.0
        ica_fd, ica_fg, ica_fs = fixed_pairs_metrics(
            C_ica_te, args.L, args.lam, fixed_pairs,
            seed_base=6000 + run_idx * 100,
            n_shifts=args.n_shifts,
            min_shift=args.min_shift,
        )
        print("  ICA done")

        # Linear GCA
        Yg_te, Zg_te = run_gca_original_multi(
            X_tr, X_te, L=args.L, n_pairs=P, seed=run_idx, n_restarts=args.gca_restarts
        )
        Yg_te = zscore_cols(Yg_te)
        Zg_te = zscore_cols(Zg_te)
        gca_res = learned_pair_metrics(
            Yg_te, Zg_te, args.L, args.lam,
            seed_base=3000 + run_idx * 100,
            n_shifts=args.n_shifts,
            min_shift=args.min_shift,
            topk=min(2, P),
        )
        print("  Linear GCA done")

        # Exact KGCA
        kg_cfg = instantiate_config(KGCAConfig, **{**kg_cfg_base, "seed": run_idx})
        kg_out = run_kernel_gca(X_tr, X_te, kg_cfg, seed=run_idx)
        if isinstance(kg_out, tuple) and len(kg_out) == 5:
            Yk_te, Zk_te, _, _, kg_meta = kg_out
        elif isinstance(kg_out, tuple) and len(kg_out) == 3:
            Yk_te, Zk_te, kg_meta = kg_out
        else:
            raise RuntimeError(f"Unexpected return signature from run_kernel_gca: {type(kg_out)}")
        Yk_te = zscore_cols(Yk_te)
        Zk_te = zscore_cols(Zk_te)
        kg_res = learned_pair_metrics(
            Yk_te, Zk_te, args.L, args.lam,
            seed_base=4000 + run_idx * 100,
            n_shifts=args.n_shifts,
            min_shift=args.min_shift,
            topk=min(2, P),
        )
        print("  Exact KGCA done")

        # RFF KGCA
        rff_cfg = instantiate_config(KGCARFFConfig, **{**rff_cfg_base, "seed": run_idx})
        rff_out = run_kernel_gca_rff(X_tr, X_te, rff_cfg, seed=run_idx)
        if isinstance(rff_out, tuple) and len(rff_out) == 5:
            Yr_te, Zr_te, _, _, rff_meta = rff_out
        elif isinstance(rff_out, tuple) and len(rff_out) == 3:
            Yr_te, Zr_te, rff_meta = rff_out
        else:
            raise RuntimeError(f"Unexpected return signature from run_kernel_gca_rff: {type(rff_out)}")
        Yr_te = zscore_cols(Yr_te)
        Zr_te = zscore_cols(Zr_te)
        rff_res = learned_pair_metrics(
            Yr_te, Zr_te, args.L, args.lam,
            seed_base=7000 + run_idx * 100,
            n_shifts=args.n_shifts,
            min_shift=args.min_shift,
            topk=min(2, P),
        )
        print("  RFF KGCA done")

        # Optional cache
        if args.save_caches:
            final_cache = outdir / f"rep{run_idx}_cache.npz"
            tmp_cache = outdir / f"rep{run_idx}_cache.npz.tmp"
            with open(tmp_cache, "wb") as f:
                np.savez_compressed(
                    f,
                    C_pca_te=C_pca_te,
                    C_ica_te=C_ica_te,
                    Yg_te=Yg_te, Zg_te=Zg_te,
                    Yk_te=Yk_te, Zk_te=Zk_te,
                    Yr_te=Yr_te, Zr_te=Zr_te,
                    X_te=X_te, S_te=S[Ttr:],
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_cache, final_cache)

        rep_record = {
            "run": run_idx,
            "file": os.path.basename(path),
            "dataset_path": str(path),
            "tag": tag,
            "P": P,
            "L": args.L,
            "lam": args.lam,
            "train_frac": args.train_frac,
            "pca": {
                "top2_delta": pca_top2_d,
                "top2_gap": pca_top2_g,
                "sig_rate": pca_sig,
                "fixed_delta": pca_fd,
                "fixed_gap": pca_fg,
                "fixed_sig": pca_fs,
            },
            "ica": {
                "top2_delta": ica_top2_d,
                "top2_gap": ica_top2_g,
                "sig_rate": ica_sig,
                "fixed_delta": ica_fd,
                "fixed_gap": ica_fg,
                "fixed_sig": ica_fs,
            },
            "gca": gca_res,
            "kgca": kg_res,
            "rff": rff_res,
            "meta": meta,
        }

        atomic_json_dump(rep_record, rep_json)
        rebuild_csvs_from_json(outdir)
        print(f"[saved] {rep_json}")

    rebuild_csvs_from_json(outdir)
    print(f"Wrote {outdir / 'runs.csv'}")
    print(f"Wrote {outdir / 'runs_restricted.csv'}")
    return outdir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate all methods on new benchmark datasets.")
    ap.add_argument("--dataset_dir", default="datasets", type=str)
    ap.add_argument(
        "--tags", nargs="*", default=None,
        help="Specific dataset tags to evaluate, e.g. hd_nlds_linear l96_nonlinear. Defaults to all grouped *_run*.npz files."
    )
    ap.add_argument("--pattern", default="*.npz", type=str,
                    help="Glob pattern used when discovering dataset files.")
    ap.add_argument("--outdir_root", default="results", type=str)

    ap.add_argument("--train_frac", default=0.7, type=float)
    ap.add_argument("--L", default=5, type=int, help="Lag order for evaluation/predictors.")
    ap.add_argument("--lam", default=1e-2, type=float)
    ap.add_argument("--override_P", default=None, type=int,
                    help="Override number of learned/fixed pairs for all tags.")
    ap.add_argument("--P_hd", default=3, type=int)
    ap.add_argument("--P_l96", default=1, type=int)
    ap.add_argument("--min_unsup_components", default=6, type=int)

    ap.add_argument("--iters", default=200, type=int)
    ap.add_argument("--lr", default=5e-2, type=float)
    ap.add_argument("--gca_restarts", default=6, type=int)

    ap.add_argument("--rff_dim", default=500, type=int, help="RFF feature dimension for RFF-KGCA.")
    ap.add_argument("--rff_gamma", default=None, type=float,
                    help="Optional RFF gamma override. Leave unset to let the implementation choose its heuristic if supported.")

    ap.add_argument("--n_shifts", default=200, type=int)
    ap.add_argument("--min_shift", default=50, type=int)
    ap.add_argument("--save_caches", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="Create a new timestamped output directory instead of resuming in a stable one.")
    return ap.parse_args()


def main():
    args = parse_args()
    groups = discover_dataset_groups(args.dataset_dir, pattern=args.pattern)
    if args.tags:
        wanted = set(args.tags)
        groups = {k: v for k, v in groups.items() if k in wanted}
    if not groups:
        raise RuntimeError(f"No dataset groups found in {args.dataset_dir} matching tags={args.tags}")

    out_root = Path(args.outdir_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Evaluating tags:")
    for tag, files in groups.items():
        print(f"  {tag}: {len(files)} files")

    for tag, files in groups.items():
        evaluate_group(tag, files, out_root, args)

    print("Done.")


if __name__ == "__main__":
    main()

'''
def evaluate_group(tag: str, files: Sequence[str], out_root: Path, args: argparse.Namespace) -> Path:
    X0, S0, meta0 = load_dataset_npz(files[0])
    P = default_pairs_for_tag(tag, meta0, args)
    M = default_dims_for_unsupervised(P, min_components=args.min_unsup_components)
    fixed_pairs = [(2 * i, 2 * i + 1) for i in range(P)]

    outdir = out_root / f"{tag}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset_tag": tag,
        "n_files": len(files),
        "P": P,
        "M_unsup": M,
        "L": args.L,
        "lam": args.lam,
        "train_frac": args.train_frac,
        "iters": args.iters,
        "lr": args.lr,
        "gca_restarts": args.gca_restarts,
        "rff_dim": args.rff_dim,
        "rff_gamma": args.rff_gamma,
        "n_shifts": args.n_shifts,
        "min_shift": args.min_shift,
        "source_meta_example": meta0,
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    runs_csv = outdir / "runs.csv"
    runsR_csv = outdir / "runs_restricted.csv"

    f1 = open(runs_csv, "w", newline="")
    w1 = csv.writer(f1)
    w1.writerow([
        "run", "file",
        "kgca_top2_delta_test", "kgca_top2_gap_test", "kgca_sig_rate",
        "rff_top2_delta_test", "rff_top2_gap_test", "rff_sig_rate",
        "gca_top2_delta_test", "gca_top2_gap_test", "gca_sig_rate",
        "pca_top2_delta_test", "pca_top2_gap_test", "pca_sig_rate",
        "ica_top2_delta_test", "ica_top2_gap_test", "ica_sig_rate",
    ])

    f2 = open(runsR_csv, "w", newline="")
    w2 = csv.writer(f2)
    w2.writerow([
        "run", "file",
        "pca_fixed_delta_test", "pca_fixed_gap_test", "pca_fixed_sig_rate",
        "ica_fixed_delta_test", "ica_fixed_gap_test", "ica_fixed_sig_rate",
        "gca_pairmean_delta_test", "gca_pairmean_gap_test", "gca_sig_rate",
        "kgca_pairmean_delta_test", "kgca_pairmean_gap_test", "kgca_sig_rate",
        "rff_pairmean_delta_test", "rff_pairmean_gap_test", "rff_sig_rate",
    ])

    kg_cfg_base = {
        "L": args.L,
        "lam": args.lam,
        "lr": args.lr,
        "iters": args.iters,
        "seed": 0,
        "n_pairs": P,
    }
    rff_cfg_base = {
        "L": args.L,
        "lam": args.lam,
        "lr": args.lr,
        "iters": args.iters,
        "seed": 0,
        "n_pairs": P,
        "rff_dim": args.rff_dim,
        "D_rff": args.rff_dim,
        "m": args.rff_dim,
        "gamma": args.rff_gamma,
    }

    for run_idx, path in enumerate(files):
        X, S, meta = load_dataset_npz(path)
        T = X.shape[0]
        Ttr = int(T * args.train_frac)
        X_tr, X_te = X[:Ttr], X[Ttr:]

        # PCA
        pcam = pca_fit(X_tr)
        C_pca_te = pca_transform(pcam, X_te, n_components=M)
        C_pca_te = zscore_cols(C_pca_te)
        pca_top2_d, pca_top2_g, pca_top_pairs = topk_pairs_metrics(C_pca_te, args.L, args.lam, k=min(2, P))
        pca_pvals = []
        for q, (i, j, _, _) in enumerate(pca_top_pairs):
            _, pv = circshift_pvalue(C_pca_te[:, i], C_pca_te[:, j], args.L, args.lam,
                                     n_shifts=args.n_shifts, min_shift=args.min_shift, seed=1000 + run_idx * 100 + q)
            pca_pvals.append(pv)
        pca_sig = float(np.mean(np.array(pca_pvals) < 0.05)) if pca_pvals else 0.0
        pca_fd, pca_fg, pca_fs = fixed_pairs_metrics(C_pca_te, args.L, args.lam, fixed_pairs,
                                                     seed_base=5000 + run_idx * 100,
                                                     n_shifts=args.n_shifts,
                                                     min_shift=args.min_shift)

        print("PCA done")
        # ICA
        icam = ica_fit(X_tr, n_components=M, seed=run_idx)
        C_ica_te = ica_transform(icam, X_te)
        C_ica_te = zscore_cols(C_ica_te)
        ica_top2_d, ica_top2_g, ica_top_pairs = topk_pairs_metrics(C_ica_te, args.L, args.lam, k=min(2, P))
        ica_pvals = []
        for q, (i, j, _, _) in enumerate(ica_top_pairs):
            _, pv = circshift_pvalue(C_ica_te[:, i], C_ica_te[:, j], args.L, args.lam,
                                     n_shifts=args.n_shifts, min_shift=args.min_shift, seed=2000 + run_idx * 100 + q)
            ica_pvals.append(pv)
        ica_sig = float(np.mean(np.array(ica_pvals) < 0.05)) if ica_pvals else 0.0
        ica_fd, ica_fg, ica_fs = fixed_pairs_metrics(C_ica_te, args.L, args.lam, fixed_pairs,
                                                     seed_base=6000 + run_idx * 100,
                                                     n_shifts=args.n_shifts,
                                                     min_shift=args.min_shift)

        print("ICA done")
        # Linear GCA
        Yg_te, Zg_te = run_gca_original_multi(
            X_tr, X_te, L=args.L, n_pairs=P, seed=run_idx, n_restarts=args.gca_restarts
        )
        Yg_te = zscore_cols(Yg_te)
        Zg_te = zscore_cols(Zg_te)
        gca_res = learned_pair_metrics(Yg_te, Zg_te, args.L, args.lam,
                                       seed_base=3000 + run_idx * 100,
                                       n_shifts=args.n_shifts,
                                       min_shift=args.min_shift,
                                       topk=min(2, P))

        print("Linear GCA done")
        # Exact KGCA
        kg_cfg = instantiate_config(KGCAConfig, **{**kg_cfg_base, "seed": run_idx})
        Yk_te, Zk_te, _, _, kg_meta = run_kernel_gca(X_tr, X_te, kg_cfg, seed=run_idx)
        Yk_te = zscore_cols(Yk_te)
        Zk_te = zscore_cols(Zk_te)
        kg_res = learned_pair_metrics(Yk_te, Zk_te, args.L, args.lam,
                                      seed_base=4000 + run_idx * 100,
                                      n_shifts=args.n_shifts,
                                      min_shift=args.min_shift,
                                      topk=min(2, P))

        # RFF KGCA
        rff_cfg = instantiate_config(KGCARFFConfig, **{**rff_cfg_base, "seed": run_idx})
        Yr_te, Zr_te, _, _, rff_meta = run_kernel_gca_rff(X_tr, X_te, rff_cfg, seed=run_idx)
        Yr_te = zscore_cols(Yr_te)
        Zr_te = zscore_cols(Zr_te)
        rff_res = learned_pair_metrics(Yr_te, Zr_te, args.L, args.lam,
                                       seed_base=7000 + run_idx * 100,
                                       n_shifts=args.n_shifts,
                                       min_shift=args.min_shift,
                                       topk=min(2, P))

        print("Run done = ", run_idx)
        # Optional cache for later plotting/debugging
        if args.save_caches:
            np.savez_compressed(
                outdir / f"rep{run_idx}_cache.npz",
                C_pca_te=C_pca_te,
                C_ica_te=C_ica_te,
                Yg_te=Yg_te, Zg_te=Zg_te,
                Yk_te=Yk_te, Zk_te=Zk_te,
                Yr_te=Yr_te, Zr_te=Zr_te,
                X_te=X_te, S_te=S[Ttr:],
            )

        w1.writerow([
            run_idx, os.path.basename(path),
            kg_res["topk_delta"], kg_res["topk_gap"], kg_res["sig_rate"],
            rff_res["topk_delta"], rff_res["topk_gap"], rff_res["sig_rate"],
            gca_res["topk_delta"], gca_res["topk_gap"], gca_res["sig_rate"],
            pca_top2_d, pca_top2_g, pca_sig,
            ica_top2_d, ica_top2_g, ica_sig,
        ])

        f1.flush()
        f2.flush()

        w2.writerow([
            run_idx, os.path.basename(path),
            pca_fd, pca_fg, pca_fs,
            ica_fd, ica_fg, ica_fs,
            gca_res["pairmean_delta"], gca_res["pairmean_gap"], gca_res["sig_rate"],
            kg_res["pairmean_delta"], kg_res["pairmean_gap"], kg_res["sig_rate"],
            rff_res["pairmean_delta"], rff_res["pairmean_gap"], rff_res["sig_rate"],
        ])
        f1.flush()
        f2.flush()

    f1.close()
    f2.close()

    print(f"Wrote {runs_csv}")
    print(f"Wrote {runsR_csv}")
    return outdir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate all methods on new benchmark datasets.")
    ap.add_argument("--dataset_dir", default="datasets", type=str)
    ap.add_argument("--tags", nargs="*", default=None,
                    help="Specific dataset tags to evaluate, e.g. hd_nlds_linear l96_nonlinear. Defaults to all *_run*.npz groups.")
    ap.add_argument("--pattern", default="*.npz", type=str,
                    help="Glob pattern used when discovering dataset files.")
    ap.add_argument("--outdir_root", default="results", type=str)

    ap.add_argument("--train_frac", default=0.7, type=float)
    ap.add_argument("--L", default=5, type=int, help="Lag order for evaluation/predictors.")
    ap.add_argument("--lam", default=1e-2, type=float)
    ap.add_argument("--override_P", default=None, type=int, help="Override number of learned/fixed pairs for all tags.")
    ap.add_argument("--P_hd", default=3, type=int)
    ap.add_argument("--P_l96", default=1, type=int)
    ap.add_argument("--min_unsup_components", default=6, type=int)

    ap.add_argument("--iters", default=200, type=int)
    ap.add_argument("--lr", default=5e-2, type=float)
    ap.add_argument("--gca_restarts", default=6, type=int)

    ap.add_argument("--rff_dim", default=500, type=int, help="RFF feature dimension for RFF-KGCA.")
    ap.add_argument("--rff_gamma", default=None, type=float,
                    help="Optional RFF gamma override. Leave unset to let the implementation choose its default/heuristic if supported.")

    ap.add_argument("--n_shifts", default=200, type=int)
    ap.add_argument("--min_shift", default=50, type=int)
    ap.add_argument("--save_caches", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    groups = discover_dataset_groups(args.dataset_dir, pattern=args.pattern)
    if args.tags:
        groups = {k: v for k, v in groups.items() if k in set(args.tags)}
    if not groups:
        raise RuntimeError(f"No dataset groups found in {args.dataset_dir} matching tags={args.tags}")

    out_root = Path(args.outdir_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Evaluating tags:")
    for tag, files in groups.items():
        print(f"  {tag}: {len(files)} files")

    for tag, files in groups.items():
        evaluate_group(tag, files, out_root, args)

    print("Done.")


if __name__ == "__main__":
    main()
'''
