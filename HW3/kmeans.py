import numpy as np
from matplotlib import pyplot as plt


class kmeans:
    def __init__(self, K=4, std=1, d=None):
        # Number of clusers
        self.K = K

        # Standard deviation of the initialization of the centroids
        self.std = std

        # Dimension of the data
        self.d = d

        # Centroids
        self.centroids = None
        if self.d is not None:
            self._init_centroids()

    def _init_centroids(self):
        self.centroids = self.std * np.random.randn(self.K, self.d)

    def cluster_assignments(self, X, one_hot=True):
        '''
        if one_hot is True, it returns a (n x k) one-hot matrix
        '''
        distance_matrix = ((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2)
        assignments = np.argmin(distance_matrix, axis=0)
        if not one_hot:
            return assignments
        assignments_matrix = np.zeros((X.shape[0], self.K))
        assignments_matrix[np.arange(X.shape[0]), assignments] = 1
        return assignments_matrix

    def kmeans_objective(self, X):
        distance_matrix = ((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2).T
        assignments_matrix = self.cluster_assignments(X)
        return np.sum(distance_matrix * assignments_matrix)

    def fit(self, X, epsilon=1e-4):
        if self.d is None:
            self.d = X.shape[1]
            self._init_centroids()
        old_objective = self.kmeans_objective(X)
        kmeans_objective_values = [old_objective]
        while True:
            z = self.cluster_assignments(X)
            self.centroids = ((z.T[:, :, np.newaxis] * X).sum(axis=1).T / (epsilon + z.sum(axis=0))).T
            new_objective = self.kmeans_objective(X)
            improvement = - new_objective + old_objective
            old_objective = new_objective
            kmeans_objective_values.append(old_objective)
            if improvement < epsilon:
                return kmeans_objective_values








