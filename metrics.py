# metrics.py
import numpy as np

def ridge_value(X, u, lam):
    N, d = X.shape
    A = X.T @ X + lam*np.eye(d)
    w = np.linalg.solve(A, X.T @ u)
    e = u - X@w
    return float((e@e + lam*(w@w))/N)

def granger_delta(y, z, L, lam):
    T = len(y)
    z_plus = z[L:]
    Zp = np.stack([z[L-l:T-l] for l in range(1, L+1)], axis=1)
    Yp = np.stack([y[L-l:T-l] for l in range(1, L+1)], axis=1)
    Psi_r = ridge_value(Zp, z_plus, lam)
    Psi_f = ridge_value(np.concatenate([Zp, Yp], axis=1), z_plus, lam)
    return float(np.log(Psi_r + 1e-12) - np.log(Psi_f + 1e-12))

def granger_gap(y, z, L, lam):
    return granger_delta(y, z, L, lam) - granger_delta(z, y, L, lam)

def circshift_pvalue(y, z, L, lam, n_shifts=200, min_shift=50, seed=0):
    rng = np.random.default_rng(seed)
    T = len(y)
    obs = granger_delta(y, z, L, lam)
    null = []
    for _ in range(n_shifts):
        s = int(rng.integers(min_shift, T - min_shift))
        null.append(granger_delta(np.roll(y, s), z, L, lam))
    null = np.asarray(null)
    p = float((np.sum(null >= obs) + 1.0) / (len(null) + 1.0))
    return obs, p

import numpy as np

def granger_scores(y, z, L=3, lam=1e-2):
    """
    Compute ridge-regularized Granger predictive gain scores.

    Parameters
    ----------
    y : array-like, shape (T,)
        Driver time series
    z : array-like, shape (T,)
        Target time series
    L : int
        Lag order
    lam : float
        Ridge regularization parameter

    Returns
    -------
    dict with keys:
        "Psi_r"    : reduced model loss
        "Psi_f"    : full model loss
        "Delta"    : Psi_r - Psi_f
        "Delta_log": log(Psi_r) - log(Psi_f)
    """
    y = np.asarray(y)
    z = np.asarray(z)

    T = len(z)
    if T <= L:
        raise ValueError("Time series too short for given lag L")

    # Targets
    z_t = z[L:]

    # Lag matrices
    Zp = np.stack([z[L - l:T - l] for l in range(1, L + 1)], axis=1)
    Yp = np.stack([y[L - l:T - l] for l in range(1, L + 1)], axis=1)

    N = z_t.shape[0]

    # Ridge regression helper
    def ridge_value(X, y):
        d = X.shape[1]
        A = X.T @ X + lam * np.eye(d)
        w = np.linalg.solve(A, X.T @ y)
        e = y - X @ w
        return (e @ e) / N

    # Reduced and full losses
    Psi_r = ridge_value(Zp, z_t)
    Psi_f = ridge_value(np.concatenate([Zp, Yp], axis=1), z_t)

    return {
        "Psi_r": float(Psi_r),
        "Psi_f": float(Psi_f),
        "Delta": float(Psi_r - Psi_f),
        "Delta_log": float(np.log(Psi_r + 1e-12) - np.log(Psi_f + 1e-12)),
    }
