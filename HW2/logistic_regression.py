import numpy as np
from base_predictor import Base
import utils


class LogReg(Base):
    def fit(self, X, Y, epsilon=1e-5):
        super().fit(X, Y)
        assert len(np.unique(Y)) == 2

        n = X.shape[0]
        d = X.shape[1]

        X_with_bias = np.ones((n, d + 1))
        X_with_bias[:, :d] = X

        w = np.zeros((d + 1, 1))
        while True:
            MU = utils.sigmoid(np.dot(X_with_bias, w))  # should be n x 1
            D = np.diag(np.squeeze(MU * (1 - MU)))  # should be n x n
            H = - np.dot(np.dot(X_with_bias.T, D), X_with_bias)
            gradient = np.dot(X_with_bias.T, Y.reshape(n, 1) - MU)
            # print(np.max(np.abs(gradient)))
            if np.max(np.abs(gradient)) < epsilon:
                break
            to_substract, _, _, _ = np.linalg.lstsq(H, gradient, rcond=None)
            w -= to_substract

        self.params = {'w': w}

    def predict(self, X, return_prob=False):
        n = X.shape[0]
        d = X.shape[1]

        X_with_bias = np.ones((n, d + 1))
        X_with_bias[:, :d] = X

        w = self.params['w']

        prob = utils.sigmoid(np.dot(X_with_bias, w))
        if return_prob:
            return prob
        else:
            return prob > .5

    def decision_boundary(self, X):
        super().decision_boundary(X)

        w = self.params['w']

        x0_MIN, x0_MAX = np.min(X[:, 0]), np.max(X[:, 0])
        x1_MIN, x1_MAX = np.min(X[:, 1]), np.max(X[:, 1])

        # Define the parameters that define the separator line a x_0 + b x_1 + c = 0
        a = w[0].squeeze().item()
        b = w[1].squeeze().item()
        c = w[2].squeeze().item()

        if b != 0:  # define x1 as a function of x0
            x0range = np.arange(x0_MIN, x0_MAX, 0.01)
            x1range = -a / b * x0range - c / b
        else:  # define x0 as function of x1
            x1range = np.arange(x1_MIN, x1_MAX, 0.01)
            x0range = -b / a * x1range - c / a

        return (x0range, x1range), (a, b, c)


