import numpy as np
from kmeans import kmeans


class EMGMM:
    def __init__(self, K=4, d=2, var=1, spherical=True):
        self.K = K
        self.d = d
        self.spherical = spherical

        # Initialize parameters
        self.pi = np.ones(K) / K
        self.variances = np.ones(K) * var
        self.sigma = np.zeros((K, d, d))
        for k in range(K):
            self.sigma[k, :, :] = self.variances[k] * np.eye(d)

        self.mu = None

    def _init_means(self, X):
        model = kmeans(K=self.K, d=self.d)
        _ = model.fit(X)
        self.mu = model.centroids

    def E_step_old(self, X):
        # E STEP WITHOUT EINSTEIN MAGIC
        n = X.shape[0]
        tau = np.zeros((n, self.K))

        for i in range(n):
            for j in range(self.K):
                tau[i, j] = (self.pi[j] * (np.linalg.det(self.sigma[j, :, :]) ** (- 1. / 2)) *
                             np.exp(-.5 * np.linalg.multi_dot([(X[i] - self.mu[j]).T,
                                                               np.linalg.inv(self.sigma[j, :, :]),
                                                               X[i] - self.mu[j]])))

        tau = (tau.T / tau.sum(axis=1)).T
        return tau

    def E_step(self, X):
        # TODO: find a way to vectorize this to avoid looping through the classes
        n = X.shape[0]

        tau = np.zeros((n, self.K))
        for j in range(self.K):
            to_exp = np.einsum('nd, df, nf -> n',
                               X - self.mu[j],
                               np.linalg.inv(self.sigma[j]),
                               X - self.mu[j])
            tau[:, j] = self.pi[j] * np.linalg.det(self.sigma[j, :, :]) ** (- 1. / 2) * np.exp(- .5 * to_exp)

        tau = (tau.T / tau.sum(axis=1)).T
        return tau

    def M_step(self, X, tau):
        n = X.shape[0]

        new_pi = tau.sum(axis=0) / n

        new_mu = (np.einsum('nk, nd -> kd', tau, X).T / tau.sum(axis=0)).T

        if self.spherical:
            new_variances = np.einsum('nk, nkd, nkd -> k',
                                      tau,
                                      X[:, np.newaxis, :] - new_mu,
                                      X[:, np.newaxis, :] - new_mu) / (self.d * np.einsum('nk->k', tau))
            new_sigma = np.einsum('dfk->kdf',
                                  np.repeat(np.eye(self.d)[:, :, np.newaxis], self.K, axis=2) * new_variances)

        else:
            new_sigma = np.einsum('dfk -> kdf', np.einsum('nk, nkd, nkf -> dfk',
                                                          tau,
                                                          X[:, np.newaxis, :] - new_mu,
                                                          X[:, np.newaxis, :] - new_mu) / np.einsum('nk->k', tau))

        return new_pi, new_mu, new_sigma

    def M_step_old(self, X, tau):
        # M STEP WITHOUT EINSTEIN MAGIC
        n = X.shape[0]

        new_pi = tau.sum(axis=0) / n
        new_mu = np.zeros((self.K, self.d))

        for j in range(self.K):
            new_mu[j] = (tau[:, j] * X.T).T.sum(axis=0) / tau[:, j].sum()

        new_variances = np.zeros(self.K)
        new_sigma = np.zeros((self.K, self.d, self.d))
        if self.spherical:
            for k in range(self.K):
                for i in range(n):
                    new_variances[k] += tau[i, k] * np.dot(X[i] - new_mu[k], X[i] - new_mu[k])
                new_variances[k] /= (self.d * tau[:, k].sum())
                new_sigma[k] = new_variances[k] * np.eye(self.d)
        else:
            for k in range(self.K):
                for i in range(n):
                    new_sigma[k] += tau[i, k] * np.outer(X[i] - new_mu[k], X[i] - new_mu[k])
                new_sigma[k] /= tau[:, k].sum()

        return new_pi, new_mu, new_sigma

    def fit(self, X, epsilon=1e-4, old=False):
        # When old is True, we use ugly loops instead of Einstein summation
        self._init_means(X)
        while True:
            # E STEP
            tau = self.E_step(X) if not old else self.E_step_old(X)
            # M STEP
            new_pi, new_mu, new_sigma = self.M_step(X, tau) if not old else self.M_step_old(X, tau)

            change = 0
            change = max(change, np.max(np.abs(self.pi - new_pi)))
            change = max(change, np.max(np.abs(self.mu - new_mu)))
            change = max(change, np.max(np.abs(self.sigma - new_sigma)))

            self.pi = new_pi
            self.mu = new_mu
            self.sigma = new_sigma
            if change < epsilon:
                return tau

    def log_likelihood(self, X):
        # TODO: vectorize this
        n = X.shape[0]
        L = 0

        tau = self.E_step(X)

        for i in range(n):
            for j in range(self.K):
                L += tau[i, j] * (np.log(self.pi[j]) - .5 * np.log((2 * np.pi) ** self.d *
                                                                   np.linalg.det(self.sigma[j])) -
                                  .5 * np.linalg.multi_dot([(X[i] - self.mu[j]).T, np.linalg.inv(self.sigma[j]),
                                                            X[i] - self.mu[j]])
                                  )
        L /= n

        return L
