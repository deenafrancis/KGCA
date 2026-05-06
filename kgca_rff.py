import numpy as np
from dataclasses import dataclass


# ---------- shared utilities ----------

def build_selectors(T, L):
    N = T - L
    S_plus = np.zeros((N, T))
    for i in range(N):
        S_plus[i, L + i] = 1.0

    S_lags = np.zeros((L, N, T))
    for l in range(1, L + 1):
        Sl = np.zeros((N, T))
        for i in range(N):
            Sl[i, L + i - l] = 1.0
        S_lags[l - 1] = Sl
    return S_plus, S_lags


def ridge_value_and_grads(X, u, lam):
    """
    Psi = min_w (1/N)||u - X w||^2 + (lam/N)||w||^2
    returns Psi and gradients wrt u and X
    """
    N, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    w = np.linalg.solve(A, X.T @ u)
    e = u - X @ w
    Psi = (e @ e + lam * (w @ w)) / N

    grad_u = (2.0 / N) * e
    grad_X = -(2.0 / N) * np.outer(e, w)
    return float(Psi), grad_u, grad_X


def pack_lag(D_lags, c):
    """
    D_lags: shape (L, N, M)
    c: shape (M,)
    returns Xlag: shape (N, L)
    """
    L, N, _ = D_lags.shape
    X = np.zeros((N, L))
    for li in range(L):
        X[:, li] = D_lags[li] @ c
    return X


def normalize_primal(c, eps):
    nrm = np.linalg.norm(c)
    return c / max(nrm, eps)


def primal_gram_schmidt(c, prev, eps):
    for pj in prev:
        c = c - float(c @ pj) * pj
    return normalize_primal(c, eps)


# ---------- RFF utilities ----------

def zscore_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def median_gamma(X, max_points=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_points, len(X)), replace=False)
    Xs = X[idx]
    d2 = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
    vals = d2[d2 > 0]
    if len(vals) == 0:
        return 1.0
    med = np.median(vals)
    return 1.0 / max(med, 1e-12)


class RFFMap:
    def __init__(self, gamma, n_features, seed=0):
        self.gamma = gamma
        self.n_features = n_features
        self.seed = seed
        self.W = None
        self.b = None

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        D = X.shape[1]
        # For exp(-gamma ||x-x'||^2), omega ~ N(0, 2 gamma I)
        self.W = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 * self.gamma),
            size=(D, self.n_features),
        )
        self.b = rng.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=(self.n_features,),
        )
        return self

    def transform(self, X):
        if self.W is None or self.b is None:
            raise RuntimeError("Call fit() before transform().")
        Z = X @ self.W + self.b[None, :]
        return np.sqrt(2.0 / self.n_features) * np.cos(Z)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------- RFF-KGCA ----------

@dataclass
class KGCARFFConfig:
    L: int = 3
    lam: float = 1e-2
    lr: float = 5e-2
    iters: int = 200
    seed: int = 0
    eps: float = 1e-12
    grad_clip: float = 10.0
    n_pairs: int = 2

    n_features: int = 256
    gamma: float = None
    standardize: bool = True
    max_points_gamma: int = 800


def kgca_primal_objective_and_grads(a, b, Phi, cfg):
    """
    Same KGCA objective as dual version, but with explicit features:
      y = Phi @ a
      z = Phi @ b
    """
    T = Phi.shape[0]
    L = cfg.L

    S_plus, S_lags = build_selectors(T, L)

    D_plus = S_plus @ Phi                     # (N, M)
    D_lags = np.array([S_lags[i] @ Phi for i in range(L)])  # (L, N, M)

    z_plus = D_plus @ b
    Zp = pack_lag(D_lags, b)
    Yp = pack_lag(D_lags, a)
    Xf = np.concatenate([Zp, Yp], axis=1)

    Psi_r, gu_r, gX_r = ridge_value_and_grads(Zp, z_plus, cfg.lam)
    Psi_f, gu_f, gX_f = ridge_value_and_grads(Xf, z_plus, cfg.lam)

    gXf_Z = gX_f[:, :L]
    gXf_Y = gX_f[:, L:]

    grad_b_r = D_plus.T @ gu_r + sum(D_lags[li].T @ gX_r[:, li] for li in range(L))
    grad_b_f = D_plus.T @ gu_f + sum(D_lags[li].T @ gXf_Z[:, li] for li in range(L))
    grad_a_f = sum(D_lags[li].T @ gXf_Y[:, li] for li in range(L))

    # time reversal
    Phi_t = Phi[::-1].copy()
    D_plus_t = S_plus @ Phi_t
    D_lags_t = np.array([S_lags[i] @ Phi_t for i in range(L)])

    y_plus_t = D_plus_t @ a
    Yp_t = pack_lag(D_lags_t, a)
    Zp_t = pack_lag(D_lags_t, b)
    Xf_t = np.concatenate([Yp_t, Zp_t], axis=1)

    tPsi_r, tgu_r, tgX_r = ridge_value_and_grads(Yp_t, y_plus_t, cfg.lam)
    tPsi_f, tgu_f, tgX_f = ridge_value_and_grads(Xf_t, y_plus_t, cfg.lam)

    tgXf_Y = tgX_f[:, :L]
    tgXf_Z = tgX_f[:, L:]

    grad_a_tr = D_plus_t.T @ tgu_r + sum(D_lags_t[li].T @ tgX_r[:, li] for li in range(L))
    grad_a_tf = D_plus_t.T @ tgu_f + sum(D_lags_t[li].T @ tgXf_Y[:, li] for li in range(L))
    grad_b_tf = sum(D_lags_t[li].T @ tgXf_Z[:, li] for li in range(L))

    eps = cfg.eps
    J = float(np.log((Psi_f * tPsi_f + eps) / (Psi_r * tPsi_r + eps)))

    grad_a = (grad_a_f / (Psi_f + eps)) + (grad_a_tf / (tPsi_f + eps)) - (grad_a_tr / (tPsi_r + eps))
    grad_b = (grad_b_f / (Psi_f + eps)) - (grad_b_r / (Psi_r + eps)) + (grad_b_tf / (tPsi_f + eps))

    return J, grad_a, grad_b


def fit_kgca_rff_train(Phi_tr, cfg):
    rng = np.random.default_rng(cfg.seed)
    M = Phi_tr.shape[1]

    As = []
    Bs = []
    histories = []

    for p in range(cfg.n_pairs):
        a = primal_gram_schmidt(rng.normal(size=M), As, cfg.eps)
        b = primal_gram_schmidt(rng.normal(size=M), Bs, cfg.eps)

        hist = []
        for _ in range(cfg.iters):
            J, ga, gb = kgca_primal_objective_and_grads(a, b, Phi_tr, cfg)

            gnorm = max(np.linalg.norm(np.concatenate([ga, gb])), cfg.eps)
            if gnorm > cfg.grad_clip:
                s = cfg.grad_clip / gnorm
                ga *= s
                gb *= s

            a = primal_gram_schmidt(a - cfg.lr * ga, As, cfg.eps)
            b = primal_gram_schmidt(b - cfg.lr * gb, Bs, cfg.eps)

            hist.append(J)

        As.append(a)
        Bs.append(b)
        histories.append(hist)

    return {
        "A": np.stack(As, axis=1),   # (M, P)
        "B": np.stack(Bs, axis=1),   # (M, P)
        "histories": histories,
    }


def run_kernel_gca_rff(X_tr, X_te, cfg_rff, seed=0):
    """
    RFF approximation to KGCA itself:
      kernel map ≈ explicit features Phi
      optimize KGCA objective in primal feature space

    Returns:
      Y_te: (Tte, P)
      Z_te: (Tte, P)
      A:    (M, P)
      B:    (M, P)
      cache: dict
    """
    seed = cfg_rff.seed if seed is None else seed

    if cfg_rff.standardize:
        mu_x, sd_x = zscore_fit(X_tr)
        Xtr_s = zscore_apply(X_tr, mu_x, sd_x)
        Xte_s = zscore_apply(X_te, mu_x, sd_x)
    else:
        Xtr_s = X_tr.copy()
        Xte_s = X_te.copy()

    gamma = cfg_rff.gamma
    if gamma is None:
        gamma = median_gamma(Xtr_s, max_points=cfg_rff.max_points_gamma, seed=seed)

    rff = RFFMap(gamma=gamma, n_features=cfg_rff.n_features, seed=seed)
    Phi_tr = rff.fit_transform(Xtr_s)
    Phi_te = rff.transform(Xte_s)

    # center explicit features using training mean
    mu_phi = Phi_tr.mean(axis=0, keepdims=True)
    Phi_tr = Phi_tr - mu_phi
    Phi_te = Phi_te - mu_phi

    out = fit_kgca_rff_train(Phi_tr, KGCARFFConfig(**{**cfg_rff.__dict__, "seed": seed}))
    A = out["A"]
    B = out["B"]

    Y_te = Phi_te @ A
    Z_te = Phi_te @ B

    cache = {
        "gamma": float(gamma),
        "n_features": int(cfg_rff.n_features),
        "kgca_J_pair1": out["histories"][0] if len(out["histories"]) > 0 else None,
        "approximation": "RFF primal KGCA",
    }
    return Y_te, Z_te, A, B, cache