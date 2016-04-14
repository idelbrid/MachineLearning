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
        np.random.seed(self.random_seed)
        self.T = None
        self.A = None
        self.B = None
        self.pi = None
        self.mu = None
        self.sigma = None

    def forward_backward(self, X=None):

        T = self.T
        A = self.A
        if X is None:
            B = self.B  # training
        else:
            B = np.zeros((T, self.num_gaussians))  # testing
            for k in range(self.num_gaussians):
                B[:, k] = normal(self.mu[k], self.sigma[k]).pdf(X)
        # alpha[-1, self.num_gaussians] = 1
        # alpha[-1, :self.num_gaussians] = 0

        c = np.zeros(T)
        # alpha = np.zeros((T, self.num_gaussians))
        # beta = np.zeros((T, self.num_gaussians))
        # for t in range(0, T):  # Standard calculation
        #     if t == 0:
        #         for j in range(self.num_gaussians):
        #             alpha[0, j] = self.pi[j] * self.B[0, j]
        #
        #     else:
        #         for j in range(self.num_gaussians):
        #             for i in range(self.num_gaussians):
        #                 alpha[t, j] += alpha[t-1, i] * A[i, j] * B[t, j]
        #     c[t] = alpha[t, :].sum()
        #     alpha[t, :] /= c[t]  # Normalize to alpha hat as in Bishop
        #
        # for t in range(T-1, -1, -1):
        #     if t == T-1:
        #         for i in range(self.num_gaussians):
        #             beta[T-1, i] = 1
        #     else:
        #         for i in range(self.num_gaussians):
        #             for j in range(self.num_gaussians):
        #                 beta[t, i] += beta[t+1, j] * A[i, j] * B[t+1, j]
        #         beta[t, :] /= c[t+1]

        # return alpha, beta
        # a == alpha
        # b == beta

        a = np.zeros((T, self.num_gaussians))
        b = np.zeros((T, self.num_gaussians))
        for j in range(self.num_gaussians):
            a[0, j] = self.pi[j] * self.B[0, j]
        c[0] = a[0, :].sum()
        a[0, :] /= c[0]

        for t in range(1, T):  # The same as above, but vectorized
            a[t, :] = np.dot(A.T, a[t-1, :]) * B[t, :]
            c[t] = a[t, :].sum()
            a[t, :] /= c[t]

        for j in range(self.num_gaussians):
            b[T-1, j] = 1
        for t in range(T-2, -1, -1):
            b[t, :] = np.dot(A, (b[t+1, :]) * B[t+1, :] )
            b[t, :] /= c[t+1]

        return a, b, c

    def fit(self, X):
        self.X = X
        matX = np.asmatrix(self.X)
        self.T = X.shape[0]
        self.ndim = X.shape[1]

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
            alpha, beta, c = self.forward_backward()  # c is scaling factors
            ect = np.zeros((self.num_gaussians, self.num_gaussians))

            for i in range(self.num_gaussians):
                gamma[0, i] = alpha[0, i] * beta[0, i]
            for t in range(1, self.T):  # skip 0 - covered by pi
                for i in range(self.num_gaussians):
                    gamma[t, i] = alpha[t, i] * beta[t, i]  # normalize below
                    for j in range(self.num_gaussians):
                        ksi[j, i, t] = alpha[t-1, j] * beta[t, i] * self.A[j, i] * self.B[t, i] / c[t] # may still have i, j backwards
                        ect[j, i] += ksi[j, i, t]  # expected count of zn =j and zn-1 =i is sum over t of p(z_{n-1}=j, z_n=i)

            # M step
            self.A = ect / ect.sum(axis=1)  # appears to be right axis... sum over the posterior, i.e. axis 1 (rows)
            self.pi = gamma[0, :] / gamma[0, :].sum()  # Checks out with maths
            for k in range(self.num_gaussians):
                norm = gamma[:, k].sum()
                self.mu[k] = np.dot(gamma[:, k], X) / norm  # looks right
                intermed = np.multiply(gamma[:, k], (matX - self.mu[k]).T).T
                self.sigma[k] = np.dot(intermed.T, (matX - self.mu[k])) / norm  # since taken from GAUSS MIX, correct
                self.B[:, k] = normal(self.mu[k], self.sigma[k]).pdf(X)  # recalculating the table of densities

    def pred(self, test_data):  # todo
        if len(test_data) != self.T:
            raise ValueError("Testing data does not match the length of the training data.")
        alpha, beta, c = self.forward_backward(test_data)

        gamma = alpha * beta
        pred_labels = gamma.argmax(axis=1)  # predicted labels, i.e. choices of latent variable z
        testB = np.zeros((self.T, self.num_gaussians))
        for k in range(self.num_gaussians):
            testB[:, k] = normal(self.mu[k], self.sigma[k]).pdf(test_data)

        ksi = np.zeros((self.num_gaussians, self.num_gaussians, self.T))
        for t in range(1, self.T):  # skip 0 - covered by pi
            for i in range(self.num_gaussians):
                for j in range(self.num_gaussians):
                    ksi[j, i, t] = alpha[t-1, j] * beta[t, i] * self.A[j, i] * testB[t, i] / c[t] # may still have i, j backwards

        log_likelihood = (gamma[0, :] * self.pi).sum() + (ksi.sum(axis=2) * np.log(self.A)).sum() + (gamma * testB).sum()
        return pred_labels, log_likelihood


def best_of_10(X, max_iter):
    models = list()
    for i in range(10):
        m = HMM(max_iter=max_iter)

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
    train_labels, log_likelihood = model.pred(train_data)
    # test_labels, test_likelihoods = model.pred(test_data)


    print log_likelihood, "log likelihood on first 90% of data"

    # with open('label_logs_rand.csv', 'w+') as f:
    #     f.write("dim1,dim2,label,likelihood\n")
    #     for row in np.vstack((test_data[:, 0], test_data[:, 1], test_labels, test_likelihoods)).T:
    #         f.write("{},{},{},{}\n".format(row[0], row[1], row[2], row[3]))
    #
    # batch_test()