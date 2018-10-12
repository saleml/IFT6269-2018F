class Base:
    def __init__(self, d=None):
        # Dictionary of parameters
        self.params = None

        # Dimension of the data
        self.d = d

    def fit(self, X, Y):
        if self.d is None:
            self.d = X.shape[1]
        assert X.ndim == 2 and X.shape[1] == self.d
        assert Y.ndim == 1 and len(Y) == X.shape[0]
        pass

    def predict(self, X):
        assert X.ndim == 2 and X.shape[1] == self.d
        return None

    def decision_boundary(self, X):
        assert X.ndim == 2 and X.shape[1] == self.d
        assert self.d == 2
        return None

    def classification_error(self, X, Y):
        predictions = self.predict(X).squeeze()
        assert Y.ndim == 1 == predictions.ndim
        return 1 - sum(predictions == Y)/float(len(Y))
