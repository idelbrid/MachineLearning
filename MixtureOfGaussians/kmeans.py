import numpy as np

class KMeans:
    def __init__(self, K, max_iter=200, init_method='random'):
        self.K = K
        self.init_method = init_method
        self.max_iter = max_iter

    def fit(self, X):
        self.X = X
        self.ndim = len(X[0])
        mean = np.zeros((self.K, self.ndim))
        if self.init_method == 'random':
            mean = X[np.random.choice(range(0, len(X)), self.K), :]
        dists = np.zeros((len(X), self.K))
        for iter in range(self.max_iter):
            for k in range(self.K):
                dists[:, k] = np.sqrt((np.square(X - mean[k])).sum(axis=1))

            labels = np.argmin(dists, axis=1)
            for k in range(self.K):
                mean[k] = X[labels == k].mean(axis=0)
        self.mean = mean

    def pred(self, test_data):
        dists = np.zeros((len(test_data), self.K))
        for k in range(self.K):
                dists[:, k] = np.sqrt((np.square(test_data - self.mean[k])).sum(axis=1))
        labels = np.argmin(dists, axis=1)
        return labels