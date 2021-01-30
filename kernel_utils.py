"""
This script contains some utility functions that we use in our implementation' 

We acknowledge the use of existing code from the repositories in https://github.com/wittawatj/ that served as a skeleton for the implementation of some of our tests, especially kernel-based tests. Thank you!
"""


from __future__ import print_function
from __future__ import division
from builtins import map
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod
import math
import matplotlib.pyplot as plt
import autograd.numpy as np
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


class Kernel(with_metaclass(ABCMeta, object)):
    """Abstract class for kernels"""

    @abstractmethod
    def eval(self, X1, X2):
        """Evalute the kernel on data X1 and X2 """
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ..."""
        pass

class KGauss(Kernel):

    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0'
        self.sigma2 = sigma2

    def eval(self, X1, X2):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.
        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array
        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        D2 = np.sum(X1**2, 1)[:, np.newaxis] - 2*np.dot(X1, X2.T) + np.sum(X2**2, 1)
        K = np.exp(old_div(-D2,self.sigma2))
        
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...
        Parameters
        ----------
        X, Y : n x d numpy array
        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = np.sum( (X-Y)**2, 1)
        Kvec = np.exp(old_div(-D2,self.sigma2))
        return Kvec

    def __str__(self):
        return "KGauss(w2=%.3f)"%self.sigma2