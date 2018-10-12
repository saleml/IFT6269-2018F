import numpy as np
from base_predictor import Base
import utils


class QDA(Base):
    def fit(self, X, Y):
        super().fit(X, Y)
        assert len(np.unique(Y)) == 2

        # Number of points belonging to each class
        n0 = X[Y == 0].shape[0]
        n1 = X[Y == 1].shape[0]

        n = X.shape[0]
        assert n == n0 + n1

        # Define the parameters
        pi = np.mean(Y)

        mu_0 = np.mean(X[Y == 0], axis=0).reshape((self.d, 1))
        mu_1 = np.mean(X[Y == 1], axis=0).reshape((self.d, 1))

        sigma_0 = np.cov(X[Y == 0], rowvar=False)
        sigma_1 = np.cov(X[Y == 1], rowvar=False)

        self.params = {'pi': pi, 'mu_0': mu_0, 'mu_1': mu_1, 'sigma_0': sigma_0, 'sigma_1': sigma_1}

    def predict(self, X, return_prob=False):
        super().predict(X)

        n = X.shape[0]
        d = X.shape[1]

        pi = self.params['pi']
        mu_0, mu_1 = self.params['mu_0'], self.params['mu_1']
        sigma_0, sigma_1 = self.params['sigma_0'], self.params['sigma_1']

        sigma_0_inv = np.linalg.inv(sigma_0)
        sigma_1_inv = np.linalg.inv(sigma_1)

        bias = (.5 * (np.dot(np.dot(mu_0.T, sigma_0_inv), mu_0) -
                      np.dot(np.dot(mu_1.T, sigma_1_inv), mu_1)) +
                np.log(pi / (1 - pi) * np.sqrt(np.linalg.det(sigma_0) / np.linalg.det(sigma_1))))

        linear_part = np.dot(X, np.dot(sigma_1_inv, mu_1) - np.dot(sigma_0_inv, mu_0))

        # This can be drastically improved if I find a way of vectorizing it !
        quadratic_part = np.zeros((n, 1))
        for i in range(n):
            quadratic_part[i][0] = .5 * np.dot(np.dot(X[i, :].reshape(1, d), (sigma_0_inv - sigma_1_inv)),
                                               X[i, :].reshape(d, 1)).squeeze().item()
        prob = quadratic_part + linear_part + bias
        if return_prob:
            return prob
        else:
            return prob > .5

    def decision_boundary(self, X):
        super().decision_boundary(X)

        pi = self.params['pi']
        mu_0, mu_1 = self.params['mu_0'], self.params['mu_1']
        sigma_0 = self.params['sigma_0']
        sigma_1 = self.params['sigma_1']

        sigma_0_inv = np.linalg.inv(sigma_0)
        sigma_1_inv = np.linalg.inv(sigma_1)


        x0_MIN, x0_MAX = np.min(X[:, 0]), np.max(X[:, 0])
        x1_MIN, x1_MAX = np.min(X[:, 1]), np.max(X[:, 1])

        # Define the parameters that define the separator line a x_0**2 + b x_0 * x_1 + c x_1**2 + d x_0 + e x_1 + f = 0
        tmp = sigma_0_inv - sigma_1_inv
        a = .5 * tmp[0, 0]
        b = .5 * (tmp[0, 1] + tmp[1, 0])
        c = .5 * tmp[1, 1]

        d = (np.dot(sigma_1_inv[0, :], mu_1) - np.dot(sigma_0_inv[0, :], mu_0)).squeeze().item()
        e = (np.dot(sigma_1_inv[1, :], mu_1) - np.dot(sigma_0_inv[1, :], mu_0)).squeeze().item()
        f = (.5 * (np.dot(np.dot(mu_0.T, sigma_0_inv), mu_0) -
                      np.dot(np.dot(mu_1.T, sigma_1_inv), mu_1)) +
             np.log(pi / (1 - pi) * np.sqrt(np.linalg.det(sigma_0) / np.linalg.det(sigma_1)))).squeeze().item()

        x0range = np.arange(x0_MIN - 5, x0_MAX + 5, 0.01)
        x1range = np.arange(x1_MIN - 5, x1_MAX + 5, 0.01)

        return (x0range, x1range), (a, b, c, d, e, f)
