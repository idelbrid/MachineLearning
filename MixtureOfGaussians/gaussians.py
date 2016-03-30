import sys
import numpy as np
from scipy.stats import multivariate_normal as normal
from kmeans import KMeans

class GaussianMixture:
    def __init__(self, max_iter, num_gaussians, init_means=None, init_cov=None, init_weights=None, init_method='random',
                 init_seed=123456):  # todo finish?
        if init_method not in ['random', 'kmeans']:
            sys.exit("undefined initialization method")

        self.init_means = init_means
        self.init_cov = init_cov
        self.init_weights = init_weights
        self.init_method = init_method
        self.X = None
        self.ndim = None
        self.max_iter = max_iter
        self.num_gaussians = num_gaussians
        self.random_seed = init_seed

    def fit(self, X):
        self.X = X
        self.N = X.shape[0]
        self.ndim = X.shape[1]
        np.random.seed(self.random_seed)
        matX = np.asmatrix(X)
        if self.init_method == 'random':
            if self.init_means is not None:
                mu = self.init_means
            else:
                mu = X[np.random.choice(range(0, len(X)), self.num_gaussians), :]
            if self.init_cov is not None:
                sigma = self.init_cov
            else:
                sigma = list()
                for k in range(self.num_gaussians):
                    sigma.append(np.identity(self.ndim, dtype=np.float64))
                    sigma[k] += np.random.rand(self.ndim, self.ndim)
                    sigma[k] = np.dot(sigma[k], sigma[k].T)
                    sigma[k] /= sigma[k].sum()

                    # lowerbound = k * self.N / self.num_gaussians
                    # upperbound = lowerbound + 20
                    # sigma[k] = np.cov(X[lowerbound:upperbound, :].T)
            if self.init_weights is not None:
                lmbda = self.init_weights
            else:
                lmbda = np.random.rand(self.num_gaussians)
                lmbda /= lmbda.sum()
        elif self.init_method == 'kmeans':
            model = KMeans(K=self.num_gaussians, max_iter=20)
            model.fit(X)
            labels = model.pred(X)
            mu = np.zeros((self.num_gaussians, self.ndim))
            sigma = [np.zeros((self.ndim, self.ndim))] * self.num_gaussians
            for k in range(self.num_gaussians):
                cluster = X[labels == k]
                mu[k] = cluster.mean(axis=0)
                sigma[k] = np.cov(cluster.T)
            if self.init_weights is not None:
                lmbda = self.init_weights
            else:
                lmbda = np.random.rand(self.num_gaussians)
                lmbda /= lmbda.sum()

        for iter in range(self.max_iter):
            phat = np.zeros((self.N, self.num_gaussians))
            N = np.zeros(self.num_gaussians)

            # E step
            for k in range(0, self.num_gaussians):
                normal_var = normal(mean=mu[k], cov=sigma[k])
                phat[:, k] = lmbda[k] * normal_var.pdf(X)
            phat /= phat.sum(axis=1)[:, None]

            # for n in range(0, self.N):  # for each training point...
            #     for k in range(0, self.num_gaussians):
            #         normalx = normal(mean=mu[k], cov=sigma[k]).pdf(X[n, :])
            #         phat[n, k] = lmbda[k] * normalx
            #     phat[n, :] /= phat[n, :].sum()

            # M step
            for k in range(self.num_gaussians):
                N[k] = phat[:, k].sum()
                mu[k] = np.dot(phat[:, k], X) / N[k]
                intermed = np.multiply(phat[:, k], (matX - mu[k]).T).T
                sigma[k] = np.dot(intermed.T, (matX - mu[k])) / N[k]
                lmbda[k] = N[k] / self.N

            pass  # end of this iteration
        self.mu = mu
        self.sigma = sigma
        self.lmbda = lmbda

    def pred(self, test_data):  # use learned gaussians to predict most likely distribution and total likelihood of the pt
        num_pts = len(test_data)
        p = np.zeros((num_pts, self.num_gaussians), dtype=np.float64)

        for k in range(self.num_gaussians):
            normal_var = normal(mean=self.mu[k], cov=self.sigma[k])
            p[:, k] = self.lmbda[k] * normal_var.pdf(test_data)
        pred_labels = p.argmax(axis=1)
        likelihood = p.sum(axis=1)
        return pred_labels, likelihood


def read_data(file_str, num_feats):
    """ INPUT: string denoting the file containing the dataset, number of features
        OUTPUT: matrix of the data """
    with open(file_str, 'r') as data_file:  # Reading number of observations
        for i, l in enumerate(data_file):  # lines from stackoverflow article
            pass                           #
        size = i + 1                       #
        data = np.zeros((size, num_feats), dtype=np.float64)

    with open(file_str, 'r') as data_file:  # Reading data to matrix
        for i, l in enumerate(data_file):
            for j, word in enumerate(l.split()):
                if word == '\n':
                    pass  # if the word is the end of the line, skip it
                else:
                    data[i, j] = float(word)
    return data


def batch_test():  # subroutine for repeated tests with different number of gaussians
    log_likelihoods = np.zeros(60)
    log_likelihoods_train = np.zeros(60)
    for iterations in range(1, 61):
        model = GaussianMixture(iterations, num_gaussians=20, init_method='random', init_seed=456789)
        model.fit(train_data)  # fitting
        train_labels, train_likelihoods = model.pred(train_data)
        test_labels, test_likelihoods = model.pred(test_data)

        log_likelihoods_train[iterations-1] = np.log(train_likelihoods).sum()
        log_likelihoods[iterations-1] = np.log(test_likelihoods).sum()

        print log_likelihoods[iterations-1], 'log likelihood ', iterations
    with open('num_iterations_likelihoods_many_extra_gaussians.csv', 'w+') as f:
        f.write("number of iterations,log likelihood test,log likelihood train\n")
        for i, row in enumerate(np.vstack((log_likelihoods, log_likelihoods_train)).T):
            f.write("{},{},{}\n".format(i+1, row[0], row[1]))

if __name__ == "__main__":

    num_feats = 2  # known ahead of time

    data = read_data('../points.dat', num_feats)
    train_data = data[:(len(data) * 9 / 10), :]  # first 90%
    test_data = data[(len(data)) * 9 / 10:, :]  # last 10%

    means = np.array([[1.3, 2],    # eye-balled means for manual initialization
                      [2, -1.3],
                      [-1.5, 1],
                      [-1, -1.5]])
    cov = [np.array([[1, 0],
                     [0, 1]])]*4

    model = GaussianMixture(30, num_gaussians=4, init_method='random', init_seed=456789)
    model.fit(train_data)  # fitting
    train_labels, train_likelihoods = model.pred(train_data)
    test_labels, test_likelihoods = model.pred(test_data)

    log_likelihood = np.log(test_likelihoods).sum()  # likelihoods returned are for each point

    print np.log(train_likelihoods).sum(), "log likelihood on first 10% of data"
    print log_likelihood, "log likelihood on last 10% of data"

    # with open('label_logs_rand.csv', 'w+') as f:
    #     f.write("dim1,dim2,label,likelihood\n")
    #     for row in np.vstack((test_data[:, 0], test_data[:, 1], test_labels, test_likelihoods)).T:
    #         f.write("{},{},{},{}\n".format(row[0], row[1], row[2], row[3]))
    #
    # batch_test()