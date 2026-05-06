# methods/pca.py
import numpy as np

def pca_fit(X_tr):
    mu = X_tr.mean(axis=0, keepdims=True)
    Xc = X_tr - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return {"mu": mu, "Vt": Vt, "S": S}

def pca_transform(model, X, n_components):
    Xc = X - model["mu"]
    V = model["Vt"][:n_components, :].T
    return Xc @ V
