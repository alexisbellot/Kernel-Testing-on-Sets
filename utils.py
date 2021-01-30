import numpy as np
import numpy.ma as ma
import scipy.linalg
import scipy.optimize
import scipy.stats
import typing
import warnings
from numpy import diag, exp, sqrt
from numpy.matlib import repmat
from sklearn.metrics import euclidean_distances
from typing import Union, List


def columnwise_normalizes(*Xs) -> typing.List[Union[None, np.ndarray]]:
    """normalize per column for multiple data"""
    return [columnwise_normalize(X) for X in Xs]


def columnwise_normalize(X: np.ndarray) -> Union[None, np.ndarray]:
    """normalize per column"""
    if X is None:
        return None
    return (X - np.mean(X, 0)) / np.std(X, 0)  # broadcast


def ensure_symmetric(x: np.ndarray) -> np.ndarray:
    return (x + x.T) / 2


def truncated_eigen(eig_vals, eig_vecs=None, relative_threshold=1e-5):
    """Retain eigenvalues and corresponding eigenvectors where an eigenvalue > max(eigenvalues)*relative_threshold"""
    indices = np.where(eig_vals > max(eig_vals) * relative_threshold)[0]
    if eig_vecs is not None:
        return eig_vals[indices], eig_vecs[:, indices]
    else:
        return eig_vals[indices]


def eigdec(X: np.ndarray, top_N: int = None):
    """Eigendecomposition with top N descending ordered eigenvalues and corresponding eigenvectors"""
    if top_N is None:
        top_N = len(X)

    X = ensure_symmetric(X)
    M = len(X)

    # ascending M-1-N <= <= M-1
    w, v = scipy.linalg.eigh(X, eigvals=(M - 1 - top_N + 1, M - 1))

    # descending
    return w[::-1], v[:, ::-1]


def centering(M: np.ndarray) -> Union[None, np.ndarray]:
    """Matrix Centering"""
    if M is None:
        return None
    n = len(M)
    H = np.eye(n) - 1 / n
    return H @ M @ H


def pdinv(x: np.ndarray) -> np.ndarray:
    """Inverse of a positive definite matrix"""
    U = scipy.linalg.cholesky(x)
    Uinv = scipy.linalg.inv(U)
    return Uinv @ Uinv.T



def rbf_kernel_median(data: np.ndarray, *args, without_two=False):
    """A list of RBF kernel matrices for data sets in arguments based on median heuristic"""
    if args is None:
        args = []

    outs = []
    for x in [data, *args]:
        D_squared = euclidean_distances(x, squared=True)
        # masking upper triangle and the diagonal.
        mask = np.triu(np.ones(D_squared.shape), 0)
        median_squared_distance = ma.median(ma.array(D_squared, mask=mask))
        if without_two:
            kx = exp(-D_squared / median_squared_distance)
        else:
            kx = exp(-0.5 * D_squared / median_squared_distance)
        outs.append(kx)

    if len(outs) == 1:
        return outs[0]
    else:
        return outs















