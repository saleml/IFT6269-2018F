import numpy as np
from base_predictor import Base
import utils


class LinReg(Base):
    def fit(self, X, Y):
        super().fit(X, Y)
        assert len(np.unique(Y)) == 2

        n = X.shape[0]
        d = X.shape[1]

        X_with_bias = np.ones((n, d + 1))
        X_with_bias[:, :d] = X

        w, _, _, _ = np.linalg.lstsq(np.dot(X_with_bias.T, X_with_bias),
                                     np.dot(X_with_bias.T, Y.reshape(n, 1)))

        self.params = {'w': w}

    def predict(self, X, return_prob=False):
        # Note that these are not actual probabilities, but just a number that can be anywhere in the real line.
        n = X.shape[0]
        d = X.shape[1]

        X_with_bias = np.ones((n, d + 1))
        X_with_bias[:, :d] = X

        w = self.params['w']

        prob = np.dot(X_with_bias, w)
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
        c = w[2].squeeze().item() - 0.5

        if b != 0:  # define x1 as a function of x0
            x0range = np.arange(x0_MIN, x0_MAX, 0.01)
            x1range = -a / b * x0range - c / b
        else:  # define x0 as function of x1
            x1range = np.arange(x1_MIN, x1_MAX, 0.01)
            x0range = -b / a * x1range - c / a

        return (x0range, x1range), (a, b, c)
