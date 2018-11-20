import numpy as np
from scipy.stats import multivariate_normal

# TODO: This file is useless, put everything in utils and in em


def multi_gaussian(MU, SIGMA):
    '''
    MU IS a K by d matrix
    SIGMA IS a K by d x d matrix
    return a function that given x (T by d) returns a T by K vector of the K gaussian pdfs
    '''
    K, d = MU.shape

    def f(x):
        if x.ndim != 2:
            x = x[np.newaxis, :]
        assert x.ndim == 2
        densities = np.zeros((x.shape[0], K))
        for k in range(K):
            densities[:, k] = multivariate_normal.pdf(x, MU[k], SIGMA[k])
        return densities

    return f


def alpha_rec(t, pi, X, A, emission_density):
    '''pi has to be a K by 1 matrix. Returns a K by 1 matrix. X is a T by d matrix'''
    if t == 0:
        return pi * emission_density(X[0])
    return emission_density(X[t]) * (A @ alpha_rec(t - 1, pi, X, A, emission_density))


def beta_rec(t, X, A, emission_density):
    T = X.shape[0]
    K = A.shape[0]
    if t == T - 1:
        return np.ones((K, 1))
    return A @ np.squeeze((emission_density(X[t + 1]) * beta_rec(t + 1, X, A, emission_density)).T)


def alpha_dyna_prog(pi, X, A, emission_density):
    T = X.shape[0]
    K = A.shape[0]
    alpha = np.zeros((T, K))
    alpha[0] = pi * emission_density(X[0])
    for t in range(1, T):
        alpha[t] = emission_density(X[t]) * (A @ alpha[t - 1])
    return alpha


def beta_dyna_prog(X, A, emission_density):
    T = X.shape[0]
    K = A.shape[0]
    beta = np.ones((T, K))
    for t in range(T - 1)[::-1]:
        beta[t] = np.squeeze(A @ (emission_density(X[t + 1]) * beta[t + 1]).T)
    return beta


def alpha_tilde_dyna_prog(pi, X, A, emission_density):
    T = X.shape[0]
    K = A.shape[0]
    alpha = np.zeros((T, K))
    c = np.zeros(T)
    alpha[0] = pi * emission_density(X[0])
    c[0] = np.sum(alpha[0])
    alpha[0] /= c[0]
    # TODO: this is vectorizable
    for t in range(1, T):
        alpha[t] = emission_density(X[t]) * (A @ alpha[t - 1])
        c[t] = np.sum(alpha[t])
        alpha[t] /= c[t]
    return alpha, c


def beta_tilde_dyna_prog(X, A, emission_density, c):
    T = X.shape[0]
    K = A.shape[0]
    beta = np.ones((T, K))
    # TODO: this is vectorizable
    for t in range(T - 1)[::-1]:
        beta[t] = np.squeeze(A.T @ (emission_density(X[t + 1]) * beta[t + 1]).T)
        beta[t] /= c[t + 1]
    return beta


def smoothing_conditional(pi, X, A, emission_density):
    """p is the smoothing node conditional T x K, p2 is the smoothing edge conditional T x K x K"""
    alpha_tilde, c = alpha_tilde_dyna_prog(pi, X, A, emission_density)  # T x K
    beta_tilde = beta_tilde_dyna_prog(X, A, emission_density, c)  # T x K

    p = alpha_tilde * beta_tilde

    emissions = emission_density(X)  # T x K

    p2 = A * emissions[1:, :, np.newaxis] * alpha_tilde[:-1, np.newaxis, :] * beta_tilde[1:, :, np.newaxis]
    p2 = np.einsum('klt -> tkl', np.einsum('tkl -> klt', p2) / c[1:])

    return p, p2, c


def viterbi(pi, X, A, c, emission_density):
    # TODO: this is vectorizable
    T = X.shape[0]
    K = A.shape[0]
    alpha = np.zeros((T, K))
    argmaxes = np.zeros((T - 1, K))
    true_argmaxes = np.zeros(T, dtype=int)
    alpha[0] = pi * emission_density(X[0]) / c[0]
    for t in range(1, T):
        alpha[t] = emission_density(X[t]) / c[t] * np.max(A * alpha[t - 1], axis=1)
        argmaxes[t - 1] = np.argmax(A * alpha[t - 1], axis=1)

    true_argmaxes[T - 1] = np.argmax(alpha[T - 1])
    for t in range(0, T - 1)[::-1]:
        # print(t, true_argmaxes[t + 1])
        true_argmaxes[t] = argmaxes[t, true_argmaxes[t + 1]]

    return true_argmaxes

