import numpy as np
from inference import smoothing_conditional, multi_gaussian, viterbi


class EMHMM:
    """performing EM for ONE long chain modeled with HMM"""
    def __init__(self, mu=None, sigma=None, A=None, pi = None, K=4, d=2, var=1):
        self.K = K
        self.d = d

        # Initialize parameters
        if pi is None:
            self.pi = np.ones(K) / K
        else:
            self.pi = pi

        if A is None:
            self.A = np.zeros((K, K))
        else:
            self.A = A

        if mu is None:
            self.mu = np.zeros((K, d))
        else:
            self.mu = mu

        if sigma is None:
            self.sigma = np.zeros((K, d, d))
            for k in range(K):
                self.sigma[k, :, :] = var * np.eye(d)
        else:
            self.sigma = sigma

        self.emission_density_generator = multi_gaussian

    def E_step(self, X):
        return smoothing_conditional(self.pi, X, self.A, self.emission_density_generator(self.mu, self.sigma))

    def M_step(self, X, tau1, tau2):
        """tau1 is a T x K matrix, tau2 is a (T-1) x K x K tensor. X is a T x d matrix"""
        assert np.abs(self.pi.sum() - 1) < 1e-4
        assert np.max(np.abs(np.sum(self.A, axis=0) - np.ones(self.K))) < 1e-4

        T = X.shape[0]
        d = X.shape[1]

        new_pi = tau1[0]

        new_A = np.einsum('tkl -> kl', tau2) / np.einsum('tkl -> l', tau2)

        new_mu = (np.einsum('tk, td -> kd', tau1, X).T / np.einsum('tk -> k', tau1)).T

        new_mu = np.zeros((tau1.shape[1], d))
        for k in range(tau1.shape[1]):
            for t in range(T):
                new_mu[k] += tau1[t, k] * X[t]
            new_mu[k] /= np.sum(tau1[:, k])

        new_sigma = np.einsum('dfk -> kdf', np.einsum('tk, tkd, tkf -> dfk',
                                                      tau1,
                                                      X[:, np.newaxis, :] - new_mu,
                                                      X[:, np.newaxis, :] - new_mu) / np.einsum('tk -> k', tau1))

        # new_sigma = np.zeros((tau1.shape[1], d, d))
        # for k in range(tau1.shape[1]):
        #     for t in range(T):
        #         new_sigma[k] += tau1[t, k] * np.outer(X[t] - new_mu[k], X[t] - new_mu[k])
        #     new_sigma[k] /= np.sum(tau1[:, k])

        return new_pi, new_A, new_mu, new_sigma

    def fit(self, X, X_test=None, epsilon=1e-3):
        """X_test is only for likelihood evaluation at each training step"""
        train_log_likelihoods = []
        test_log_likelihoods = []
        step = 0
        while True:
            step += 1
            # E STEP
            tau1, tau2, self.c = self.E_step(X)
            # Log-likelihoods
            train_log_likelihood = self.log_likelihood(X, tau1, tau2)

            train_log_likelihoods.append(train_log_likelihood)
            if X_test is not None:
                test_log_likelihoods.append(self.log_likelihood(X_test))
            # M STEP
            new_pi, new_A, new_mu, new_sigma = self.M_step(X, tau1, tau2)

            change = 0
            change = max(change, np.max(np.abs(self.A - new_A)))
            change = max(change, np.max(np.abs(self.pi - new_pi)))
            change = max(change, np.max(np.abs(self.mu - new_mu)))
            change = max(change, np.max(np.abs(self.sigma - new_sigma)))

            self.pi = new_pi
            self.A = new_A
            self.mu = new_mu
            self.sigma = new_sigma

            if change < epsilon:
                # Log-likelihoods
                train_log_likelihoods.append(self.log_likelihood(X, tau1, tau2))
                if X_test is not None:
                    test_log_likelihoods.append(self.log_likelihood(X_test))
                return tau1, tau2, train_log_likelihoods, test_log_likelihoods

    def log_likelihood(self, X, tau1=None, tau2=None):
        # TODO: vectorize this
        T = X.shape[0]
        L = 0

        if tau1 is None or tau2 is None:
            tau1, tau2, c = self.E_step(X)

        densities = self.emission_density_generator(self.mu, self.sigma)(X[0:T])  # T x K

        for k in range(self.K):
            L += tau1[0, k] * np.log(self.pi[k])
            for t in range(T):
                L += tau1[t, k] * np.log(densities[t, k])
                if t < T - 1:
                    for m in range(self.K):
                        L += tau2[t, k, m] * np.log(self.A[k, m])

        L /= T

        return L

    def viterbi_decoding(self, X):
        return viterbi(self.pi, X, self.A, self.c, self.emission_density_generator(self.mu, self.sigma))
