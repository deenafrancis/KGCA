# methods/ica.py
import numpy as np

def _sym_decorrelation(W):
    s, u = np.linalg.eigh(W @ W.T)
    return (u @ np.diag(1.0/np.sqrt(s + 1e-12)) @ u.T) @ W

def ica_fit(X_tr, n_components, seed=0, iters=300, tol=1e-6):
    rng = np.random.default_rng(seed)
    mu = X_tr.mean(axis=0, keepdims=True)
    Xc = X_tr - mu

    cov = np.cov(Xc, rowvar=False)
    d, E = np.linalg.eigh(cov)
    idx = np.argsort(d)[::-1]
    d = d[idx]; E = E[:, idx]
    Kwhite = E @ np.diag(1.0/np.sqrt(d + 1e-12)) @ E.T
    Xw = Xc @ Kwhite.T

    W = _sym_decorrelation(rng.normal(size=(n_components, Xw.shape[1])))

    def g(u): return np.tanh(u)
    def gprime(u):
        t = np.tanh(u); return 1.0 - t*t

    for _ in range(iters):
        W_old = W.copy()
        WX = Xw @ W.T
        W = (g(WX).T @ Xw)/Xw.shape[0] - np.diag(gprime(WX).mean(axis=0)) @ W
        W = _sym_decorrelation(W)
        if np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1.0)) < tol:
            break

    return {"mu": mu, "Kwhite": Kwhite, "W": W}

def ica_transform(model, X):
    Xc = X - model["mu"]
    Xw = Xc @ model["Kwhite"].T
    return Xw @ model["W"].T
