import numpy as np
from scipy.stats import binom
from scipy.spatial.distance import cdist


## Spatial Max and rand_k
# compute MSE Spatial Max
# utilities for spatial family encoder
def _compute_binom_pmf(K, p):
    pmf_vec = binom.pmf(np.arange(K + 1), n=K, p=p)
    return pmf_vec


def _compute_conditional_expectation(n, d, k, fn=lambda x: x, t=1):
    # compute E_{M | M >= t}[fn(X)] for X = {t, t+1, ..., n}
    if t < 0:
        raise Exception('t >= 0!')
    val = fn(np.arange(t, n + 1))
    prob = _compute_binom_pmf(n - t, k / d)
    expectation = np.dot(val, prob)
    return expectation

def compute_c1_c2_beta_bar(n, dim, k, T_fn=lambda x: x):
    fn_beta = lambda x: 1 / T_fn(x)
    beta_bar_inv = k / dim * _compute_conditional_expectation(n=n, d=dim, k=k, fn=fn_beta, t=1)
    beta_bar = 1 / beta_bar_inv
    fn_c1 = lambda x: 1 / (T_fn(x) ** 2)
    c1 = (beta_bar ** 2) * k / dim * _compute_conditional_expectation(n, dim, k, fn_c1, 1) - dim / k
    fn_c2 = lambda x: 1 / (T_fn(x) ** 2)
    c2 = 1 - (beta_bar ** 2) * (k ** 2) / (dim ** 2) * _compute_conditional_expectation(n, dim, k, fn_c2, 2)
    return beta_bar, c1, c2


def compute_MSE_Rand_k_Spatial(X, k, ratio=None):
    n, d = X.shape
    if ratio is None:  # by default use Spatial Average
        T_fn = lambda x: 1 + n / 2 * (x - 1)
    else:
        T_fn = lambda x: 1 + ratio * (x - 1) / (n - 1)
    beta_bar, c1, c2 = compute_c1_c2_beta_bar(n, d, k, T_fn)
    R1 = np.sum(X ** 2)
    D = cdist(X, X, metric=lambda x, y: np.dot(x, y))
    D[np.arange(n), np.arange(n)] = 0
    R2 = np.sum(D)
    MSE = (d / k - 1) * np.sum(X ** 2) / (n ** 2) + (c1 * R1 - c2 * R2) / (n ** 2)
    return MSE

def compute_MSE_Rand_k(X, k):
    n, d = X.shape
    R1 = np.sum(X ** 2)
    return (d / k - 1) / (n ** 2) * R1
