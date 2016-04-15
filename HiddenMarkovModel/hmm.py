import sys
import numpy as np
from scipy.stats import multivariate_normal as normal

np.random.seed(123456)
class HMM:
    def __init__(self, max_iter, init_method='random', num_gaussians=3,
                 init_seed=None):
        if init_method not in ['random']:
            sys.exit("undefined initialization method")

        self.num_gaussians = num_gaussians
        self.init_method = init_method
        self.X = None
        self.max_iter = max_iter
        self.random_seed = init_seed
        if init_seed is not None:
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
        gamma = np.zeros((self.T, self.num_gaussians))  # gamma(n, i) = p(z_n=i | X, theta)
        ksi = np.zeros((self.num_gaussians, self.num_gaussians, self.T))  # ksi[i, j, t] = p(z_n = i, z_n-1 = j | X, theta)
        for iter in range(self.max_iter):

            # E step
            alpha, beta, c = self.forward_backward()  # c is scaling factors
            ect = np.zeros((self.num_gaussians, self.num_gaussians))

            for i in range(self.num_gaussians):
                gamma[0, i] = alpha[0, i] * beta[0, i]
            for t in range(1, self.T):  # skip 0 - covered by pi
                for i in range(self.num_gaussians):
                    gamma[t, i] = alpha[t, i] * beta[t, i]
                    for j in range(self.num_gaussians):
                        ksi[j, i, t] = alpha[t-1, j] * beta[t, i] * self.A[j, i] * self.B[t, i] / c[t]
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

    def pred(self, test_data):  # calculate best labels (settings for z) and the log likelihood P(X | A, mu, pi, sigma)
        if len(test_data) != self.T:
            raise ValueError("Testing data does not match the length of the training data.")
        alpha, beta, c = self.forward_backward(test_data)  # c is scaling constants

        gamma = alpha * beta  # gamma(n, i) = p(z_n=i | X, theta)
        pred_labels = gamma.argmax(axis=1)  # predicted labels, i.e. choices of latent variable z
        testB = np.zeros((self.T, self.num_gaussians))  # B[t, i] = p(x_t | z_t, theta) (emission probability)
        for k in range(self.num_gaussians):
            testB[:, k] = normal(self.mu[k], self.sigma[k]).pdf(test_data)

        ksi = np.zeros((self.num_gaussians, self.num_gaussians, self.T))  # ksi[i, j, t] = p(z_n = i, z_n-1 = j | X, theta)
        for t in range(1, self.T):  # skip 0 - covered by pi
            for i in range(self.num_gaussians):
                for j in range(self.num_gaussians):
                    ksi[j, i, t] = alpha[t-1, j] * beta[t, i] * self.A[j, i] * testB[t, i] / c[t]

        log_likelihood = np.log(c).sum()
        return pred_labels, log_likelihood

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


def best_of_10(X, **kwargs):  # tries 10 times with different random initializations.
    models = list()
    best_pair = (None, -np.inf)
    best_index = -1
    for i in range(10):
        models.append(HMM(init_seed=None, **kwargs))
        try:
            models[i].fit(X)  # different due to random number generator state advancing
            prediction, log_likelihood = models[i].pred(X)
        except Exception:  # bad, but it will work... if the random matrix gave bad result, move to new random matrix
            models[i].fit(X)
            prediction, log_likelihood = model.pred(X)
        # with open('HMMTests/HMMiter-{}gauss-{}try-{}.csv'.format(kwargs['max_iter'],kwargs['num_gaussians'],i), 'w+') as f:
        #     f.write("x,y,z,loglikelihood\n")
        #     for j, row in enumerate(np.hstack((X, prediction[:,np.newaxis]))):
        #         f.write("{},{},{},{}\n".format(row[0], row[1], row[2], log_likelihood))
        if log_likelihood > best_pair[1]:
            best_index = i
            best_pair = (prediction, log_likelihood)

    return models[best_index], best_pair[0], best_pair[1]


def batch_test(data):  # scripting subroutine for repeated tests with different number of gaussians, etc.
    log_likelihoods = np.zeros((51, 7))
    with open('HMM-likelihood-vs-iterations-vs-gaussians.csv', 'w+') as f:
        f.write("number of iterations,number of gaussians,log likelihood\n")

    for iterations in range(10, 60):
        with open('HMM-likelihood-vs-iterations-vs-gaussians.csv', 'a') as f:
            for gaussians in range(2, 9):
                model, labels, loglikelihood = best_of_10(data, num_gaussians=gaussians, max_iter=iterations)
                log_likelihoods[iterations - 10][gaussians - 2] = loglikelihood
                f.write("{},{},{}\n".format(iterations,gaussians, loglikelihood))
                print loglikelihood, 'log likelihood', iterations, 'iterations,', gaussians, 'gaussians.'

if __name__ == "__main__":

    num_feats = 2  # known ahead of time

    data = read_data('../points.dat', num_feats)
    train_data = data[:(len(data) * 9 / 10), :]  # first 90%
    test_data = data[(len(data)) * 9 / 10:, :]  # last 10%

    model = HMM(30, num_gaussians=16)
    model.fit(train_data)  # fitting
    train_labels, log_likelihood = model.pred(train_data)

    print log_likelihood, "log likelihood on first 90% of data"

    bettermodel, newlabels, newloglikelihood = best_of_10(train_data, num_gaussians=16, max_iter=30)
    print newloglikelihood, " log likelihood on first 90% of data (best model of 10)"

    # for g in range(9, 14):
    #     best_of_10(train_data, num_gaussians=g, max_iter=30)

    # batch_test(train_data)