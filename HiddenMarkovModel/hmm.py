import sys
import numpy as np
from scipy.stats import multivariate_normal as normal


class HMM:
    def __init__(self, max_iter, init_method='random', num_gaussians=3,
                 init_seed=123456):
        if init_method not in ['random']:
            sys.exit("undefined initialization method")

        self.num_gaussians = num_gaussians
        self.init_method = init_method
        self.X = None
        self.max_iter = max_iter
        self.random_seed = init_seed
        self.T = None
        self.A = None
        self.B = None
        self.pi = None
        self.mu = None
        self.sigma = None

    def forward_backward(self):  # todo: actually test, speed up, alter for initial states
        T = self.T
        alpha = np.zeros((T, self.num_gaussians))
        beta = np.zeros((T, self.num_gaussians))
        A = self.A
        B = self.B
        # alpha[-1, self.num_gaussians] = 1
        # alpha[-1, :self.num_gaussians] = 0
        for t in range(0, T):  # Standard calculation
            if t == 0:
                for j in range(self.num_gaussians):
                    alpha[0, j] = self.pi[j] * self.B[0, j]
            else:
                for j in range(self.num_gaussians):
                    for i in range(self.num_gaussians):
                        alpha[t, j] += alpha[t-1, i] * A[j, i] * B[t+1, j]

        for t in range(T-1, -1, -1):  # off by one?
            if t == T-1:
                for i in range(self.num_gaussians):
                    beta[T-1, i] = 1
            else:
                for i in range(self.num_gaussians):
                    for j in range(self.num_gaussians):
                        beta[t, i] += beta[t+1, j] * A[j, i] * B[t+1, j]

        a = np.zeros(T, self.num_gaussians)
        b = np.zeros(T, self.num_gaussians)
        for t in range(0, T):  # Faster? Works? Todo verify if this is an alternative
            a[t, :] = np.dot(A, a[t-1, :]) * B[t, :]

        for t in range(T-1, 0, -1):
            b[t, :] = np.dot(A, b[t+1, :]) * B[t+1, :]

        # a == alpha?
        # b == beta?
        return alpha[:-1, :-1], beta[:-1, :-1]

    def fit(self, X):
        self.X = X
        matX = np.asmatrix(self.X)
        self.T = X.shape[0]
        self.ndim = X.shape[1]
        np.random.seed(self.random_seed)

        # initialization schemes
        if self.init_method == 'random':
            self.A = np.zeros((self.num_gaussians, self.num_gaussians))
            self.A += 1.0 / self.num_gaussians
            self.A += np.random.rand(self.num_gaussians, self.num_gaussians) * 0.1
            self.A /= self.A.sum(axis=0)

            self.pi = np.zeros(self.num_gaussians)
            self.pi += 1.0
            self.pi += np.random.rand(self.num_gaussians) * 0.01
            self.pi /= self.pi.sum()

            self.mu = X[np.random.choice(range(0, len(X)), self.num_gaussians), :]  # sample from the data
            self.sigma = list()
            self.B = np.zeros((self.T, self.num_gaussians))
            for k in range(self.num_gaussians):
                    self.sigma.append(np.identity(self.ndim, dtype=np.float64))
                    self.sigma[k] += np.random.rand(self.ndim, self.ndim)  # purely synthetic
                    self.sigma[k] = np.dot(self.sigma[k], self.sigma[k].T)  # making it positive semi-definite
                    self.sigma[k] /= self.sigma[k].sum()
                    self.B[:, k] = normal(self.mu[k], self.sigma[k]).pdf(X)

        ######## BEGIN ACTUAL ALGORITHM ###################
        gamma = np.zeros((self.T, self.num_gaussians))
        ksi = np.zeros((self.num_gaussians, self.num_gaussians, self.T))
        for iter in range(self.max_iter):

            # E step
            phat = np.zeros((self.T, self.num_gaussians))
            alpha, beta = self.forward_backward()
            ect = np.zeros((self.num_gaussians, self.num_gaussians))

            for t in range(self.T):
                for i in range(self.num_gaussians):
                    gamma[t, i] = alpha[t, i] * beta[t, i] / NORMALIZATION  # TODO
                    for j in range(self.num_gaussians):
                        ksi[i, j, t] = alpha[j, t-1] * beta[i, t] * self.A[i, j] * self.B[t, i] / NORMAL

            # M step
            self.A = ect / ect.sum(axis=1)  # check this is the right axis
            self.pi = gamma[1, :] / gamma[1, :].sum()
            for k in range(self.num_gaussians):
                norm = phat[:, k].sum()
                self.mu[k] = np.dot(phat[:, k], X) / norm
                intermed = np.multiply(phat[:, k], (matX - mu[k]).T).T
                self.sigma[k] = np.dot(intermed.T, (matX - mu[k])) / norm
                self.B[:, k] = normal(self.mu[k], self.sigma[k]).pdf(X)



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


# def batch_test():  # subroutine for repeated tests with different number of gaussians
#     log_likelihoods = np.zeros(60)
#     log_likelihoods_train = np.zeros(60)
#     for iterations in range(1, 61):
#         model = GaussianMixture(iterations, num_gaussians=20, init_method='random', init_seed=456789)
#         model.fit(train_data)  # fitting
#         train_labels, train_likelihoods = model.pred(train_data)
#         test_labels, test_likelihoods = model.pred(test_data)
#
#         log_likelihoods_train[iterations-1] = np.log(train_likelihoods).sum()
#         log_likelihoods[iterations-1] = np.log(test_likelihoods).sum()
#
#         print log_likelihoods[iterations-1], 'log likelihood ', iterations
#     with open('num_iterations_likelihoods_many_extra_gaussians.csv', 'w+') as f:
#         f.write("number of iterations,log likelihood test,log likelihood train\n")
#         for i, row in enumerate(np.vstack((log_likelihoods, log_likelihoods_train)).T):
#             f.write("{},{},{}\n".format(i+1, row[0], row[1]))

if __name__ == "__main__":

    num_feats = 2  # known ahead of time

    data = read_data('../points.dat', num_feats)
    train_data = data[:(len(data) * 9 / 10), :]  # first 90%
    test_data = data[(len(data)) * 9 / 10:, :]  # last 10%

    model = HMM(30)
    model.fit(train_data)  # fitting
    train_labels, train_likelihoods = model.pred(train_data)
    test_labels, test_likelihoods = model.pred(test_data)

    log_likelihood = np.log(test_likelihoods).sum()  # likelihoods returned are for each point

    print np.log(train_likelihoods).sum(), "log likelihood on first 90% of data"
    print log_likelihood, "log likelihood on last 10% of data"

    # with open('label_logs_rand.csv', 'w+') as f:
    #     f.write("dim1,dim2,label,likelihood\n")
    #     for row in np.vstack((test_data[:, 0], test_data[:, 1], test_labels, test_likelihoods)).T:
    #         f.write("{},{},{},{}\n".format(row[0], row[1], row[2], row[3]))
    #
    # batch_test()