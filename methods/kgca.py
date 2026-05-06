from dataclasses import dataclass
import numpy as np
from metrics import *

def fit_multiple_pairs(K_tr, cfg):
    """
    Multi-pair KGCA training on a centered Gram matrix K_tr.

    Returns a dict with:
      alphas: (Ttr, P)
      betas:  (Ttr, P)
      histories: list[dict] or list[list] depending on your internal logging
    """
    # If you already have fit_kgca_train that returns alphas/betas/histories for n_pairs,
    # this wrapper simply calls it.
    out = fit_kgca_train(K_tr, cfg)

    # Normalize output keys to match what your scripts expect
    return {
        "alphas": out["alphas"],
        "betas": out["betas"],
        "histories": out.get("histories", []),
    }

def center_gram(K):
    T = K.shape[0]
    H = np.eye(T) - np.ones((T, T)) / T
    return H @ K @ H

def rbf_kernel(X, gamma):
    X2 = np.sum(X**2, axis=1, keepdims=True)
    D2 = X2 + X2.T - 2 * (X @ X.T)
    return np.exp(-gamma * D2)

def center_cross_kernel(K_te_tr, K_tr_centered):
    m_col = K_tr_centered.mean(axis=0)
    m_all = float(K_tr_centered.mean())
    m_te  = K_te_tr.mean(axis=1)
    return K_te_tr - m_te[:, None] - m_col[None, :] + m_all

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
    N, d = X.shape
    A = X.T @ X + lam*np.eye(d)
    w = np.linalg.solve(A, X.T @ u)
    e = u - X@w
    Psi = (e@e + lam*(w@w))/N
    grad_u = (2.0/N) * e
    grad_X = -(2.0/N) * np.outer(e, w)
    return float(Psi), grad_u, grad_X

def pack_lag(D_lags, c):
    L, N, _ = D_lags.shape
    X = np.zeros((N, L))
    for li in range(L):
        X[:, li] = D_lags[li] @ c
    return X

def normalize_dual(c, K, eps):
    n2 = float(c.T @ (K @ c))
    return c / np.sqrt(max(n2, eps))

def rkhs_gram_schmidt(c, prev, K, eps):
    for pj in prev:
        c = c - float(c.T @ (K @ pj)) * pj
    return normalize_dual(c, K, eps)

@dataclass
class KGCAConfig:
    L: int = 3
    lam: float = 1e-2
    lr: float = 5e-2
    iters: int = 200
    seed: int = 0
    eps: float = 1e-12
    grad_clip: float = 10.0
    n_pairs: int = 2

def kgca_objective_and_grads(alpha, beta, K, cfg):
    T = K.shape[0]; L = cfg.L
    S_plus, S_lags = build_selectors(T, L)
    D_plus = S_plus @ K
    D_lags = np.array([S_lags[i] @ K for i in range(L)])

    z_plus = D_plus @ beta
    Zp = pack_lag(D_lags, beta)
    Yp = pack_lag(D_lags, alpha)
    Xf = np.concatenate([Zp, Yp], axis=1)

    Psi_r, gu_r, gX_r = ridge_value_and_grads(Zp, z_plus, cfg.lam)
    Psi_f, gu_f, gX_f = ridge_value_and_grads(Xf, z_plus, cfg.lam)

    gXf_Z = gX_f[:, :L]
    gXf_Y = gX_f[:, L:]

    grad_beta_r = D_plus.T @ gu_r + sum(D_lags[li].T @ gX_r[:, li] for li in range(L))
    grad_beta_f = D_plus.T @ gu_f + sum(D_lags[li].T @ gXf_Z[:, li] for li in range(L))
    grad_alpha_f = sum(D_lags[li].T @ gXf_Y[:, li] for li in range(L))

    # time reversal
    R = np.flipud(np.eye(T))
    Kt = R @ K
    D_plus_t = S_plus @ Kt
    D_lags_t = np.array([S_lags[i] @ Kt for i in range(L)])

    y_plus_t = D_plus_t @ alpha
    Yp_t = pack_lag(D_lags_t, alpha)
    Zp_t = pack_lag(D_lags_t, beta)
    Xf_t = np.concatenate([Yp_t, Zp_t], axis=1)

    tPsi_r, tgu_r, tgX_r = ridge_value_and_grads(Yp_t, y_plus_t, cfg.lam)
    tPsi_f, tgu_f, tgX_f = ridge_value_and_grads(Xf_t, y_plus_t, cfg.lam)

    tgXf_Y = tgX_f[:, :L]
    tgXf_Z = tgX_f[:, L:]

    grad_alpha_tr = D_plus_t.T @ tgu_r + sum(D_lags_t[li].T @ tgX_r[:, li] for li in range(L))
    grad_alpha_tf = D_plus_t.T @ tgu_f + sum(D_lags_t[li].T @ tgXf_Y[:, li] for li in range(L))
    grad_beta_tf  = sum(D_lags_t[li].T @ tgXf_Z[:, li] for li in range(L))

    eps = cfg.eps
    J = float(np.log((Psi_f*tPsi_f + eps) / (Psi_r*tPsi_r + eps)))
    grad_alpha = (grad_alpha_f/(Psi_f+eps)) + (grad_alpha_tf/(tPsi_f+eps)) - (grad_alpha_tr/(tPsi_r+eps))
    grad_beta  = (grad_beta_f /(Psi_f+eps)) - (grad_beta_r /(Psi_r+eps)) + (grad_beta_tf /(tPsi_f+eps))
    return J, grad_alpha, grad_beta

def fit_kgca_train(K_tr, cfg):
    rng = np.random.default_rng(cfg.seed)
    Ttr = K_tr.shape[0]
    alphas=[]; betas=[]; histories=[]
    for p in range(cfg.n_pairs):
        alpha = rkhs_gram_schmidt(rng.normal(size=Ttr), alphas, K_tr, cfg.eps)
        beta  = rkhs_gram_schmidt(rng.normal(size=Ttr), betas,  K_tr, cfg.eps)
        hist=[]
        for _ in range(cfg.iters):
            J, ga, gb = kgca_objective_and_grads(alpha, beta, K_tr, cfg)
            gnorm = max(np.linalg.norm(np.concatenate([ga, gb])), cfg.eps)
            if gnorm > cfg.grad_clip:
                s = cfg.grad_clip / gnorm
                ga *= s; gb *= s
            alpha = rkhs_gram_schmidt(alpha - cfg.lr*ga, alphas, K_tr, cfg.eps)
            beta  = rkhs_gram_schmidt(beta  - cfg.lr*gb, betas,  K_tr, cfg.eps)
            hist.append(J)
        alphas.append(alpha); betas.append(beta); histories.append(hist)
    return {"alphas": np.stack(alphas, axis=1), "betas": np.stack(betas, axis=1), "histories": histories}

def run_kernel_gca(X_tr, X_te, cfg_k, seed):
    """
    Median heuristic gamma on subset -> centered K_tr -> fit -> centered cross-kernel projection to test.

    Returns:
      Y_te: (Tte, P)
      Z_te: (Tte, P)
      alphas: (Ttr, P)
      betas:  (Ttr, P)
      cache: dict (includes gamma, objective history)
    """
    import numpy as np

    Ttr = X_tr.shape[0]
    rng = np.random.default_rng(seed)
    subs = rng.choice(Ttr, size=min(800, Ttr), replace=False)

    d2 = np.sum((X_tr[subs][:, None, :] - X_tr[subs][None, :, :])**2, axis=2)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / max(med, 1e-12)

    K_tr = center_gram(rbf_kernel(X_tr, gamma=gamma))
    out = fit_kgca_train(K_tr, KGCAConfig(**{**cfg_k.__dict__, "seed": seed}))
    alphas = out["alphas"]
    betas  = out["betas"]

    # cross-kernel projection
    X2_te = np.sum(X_te**2, axis=1, keepdims=True)
    X2_tr = np.sum(X_tr**2, axis=1, keepdims=True).T
    D2 = X2_te + X2_tr - 2*(X_te @ X_tr.T)
    K_te_tr = np.exp(-gamma * D2)
    K_te_tr_c = center_cross_kernel(K_te_tr, K_tr)

    Y_te = K_te_tr_c @ alphas
    Z_te = K_te_tr_c @ betas

    cache = {
        "gamma": float(gamma),
        "kgca_J_pair1": out["histories"][0] if len(out.get("histories", [])) > 0 else None,
    }
    return Y_te, Z_te, alphas, betas, cache
