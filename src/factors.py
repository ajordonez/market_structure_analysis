"""PCA via SVD and sector-clustering helpers for the sector factor analysis."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def pca_svd(excess_returns):
    # Standardize each stock (mean 0, std 1) so no single volatile stock dominates, then SVD.
    # Returns (Z, U, s, Vt) with X = Z.T = U diag(s) Vt.
    Z = (excess_returns - excess_returns.mean()) / excess_returns.std()
    X = Z.T.values                                   # n_stocks x n_days
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    signs = np.where(U.sum(axis=0) >= 0, 1.0, -1.0)  # resolve sign ambiguity
    U, Vt = U * signs, Vt * signs[:, None]
    return Z, U, s, Vt


def knn_sector_purity(coords, sectors, k=5):
    # Fraction of each point's k nearest neighbours (excluding itself) that share its sector.
    sectors = np.asarray(sectors)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
    return float((sectors[idx[:, 1:]] == sectors[:, None]).mean())


def chance_sector_purity(sectors):
    # Expected k-NN sector purity if neighbours were picked at random.
    counts = pd.Series(sectors).value_counts().values
    n_stocks = len(sectors)
    return float((counts * (counts - 1)).sum() / (n_stocks * (n_stocks - 1)))
