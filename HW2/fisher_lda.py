import numpy as np
from base_predictor import Base
import utils


class FDA(Base):
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

        sigma = n0/float(n) * np.cov(X[Y == 0], rowvar=False) + n1/float(n) * np.cov(X[Y == 1], rowvar=False)

        self.params = {'pi': pi, 'mu_0': mu_0, 'mu_1': mu_1, 'sigma': sigma}

    def predict(self, X, return_prob=False):
        super().predict(X)

        pi = self.params['pi']
        mu_0, mu_1 = self.params['mu_0'], self.params['mu_1']
        sigma = self.params['sigma']

        sigma_inv = np.linalg.inv(sigma)

        bias = .5 * np.dot(np.dot((mu_0 + mu_1).T, sigma_inv), mu_0 - mu_1) + np.log(pi / (1 - pi))

        prob = utils.sigmoid(np.dot(X, np.dot(sigma_inv, mu_1 - mu_0)) + bias)
        if return_prob:
            return prob
        else:
            return prob > .5

    def decision_boundary(self, X):
        super().decision_boundary(X)

        pi = self.params['pi']
        mu_0, mu_1 = self.params['mu_0'], self.params['mu_1']
        sigma = self.params['sigma']

        sigma_inv = np.linalg.inv(sigma)

        x0_MIN, x0_MAX = np.min(X[:, 0]), np.max(X[:, 0])
        x1_MIN, x1_MAX = np.min(X[:, 1]), np.max(X[:, 1])

        # Define the parameters that define the separator line a x_0 + b x_1 + c = 0
        a = np.dot(sigma_inv[0, :], mu_1 - mu_0).squeeze().item()
        b = np.dot(sigma_inv[1, :], mu_1 - mu_0).squeeze().item()
        c = (.5 * np.dot(np.dot((mu_0 + mu_1).T, sigma_inv), mu_0 - mu_1) + np.log(pi / (1 - pi))).squeeze().item()

        if b != 0:  # define x1 as a function of x0
            x0range = np.arange(x0_MIN, x0_MAX, 0.01)
            x1range = -a / b * x0range - c / b
        else:  # define x0 as function of x1
            x1range = np.arange(x1_MIN, x1_MAX, 0.01)
            x0range = -b / a * x1range - c / a

        return (x0range, x1range), (a, b, c)


