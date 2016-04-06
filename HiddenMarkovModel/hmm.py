import sys
import numpy as np

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
        self.transition = None
        self.emission = None

    def forward_backward(self):  # todo: actually test, speed up
        T = self.T
        alpha = np.zeros(T, len(self.zval_range))
        beta = np.zeros(T, len(self.zval_range))
        A = self.transition
        B = self.emission

        for t in range(0, T):  # Standard calculation
            for j in self.zval_range:
                for i in self.zval_range:
                    alpha[t, j] += alpha[t-1, i] * A[j, i] * B[self.X[t], j]

        for t in range(T-1, 0, -1):  # off by one?
            for i in self.zval_range:
                for j in self.zval_range:
                    beta[t, i] += beta[t+1, j] * A[j,i] * B[self.X[t+1], j]

        a = np.zeros(T, self.zval_range)
        b = np.zeros(T, self.zval_range)
        for t in range(0, T):  # Faster?
            a[t, :] = np.dot(A, a[t-1, :]) * B[self.X[t], :]

        for t in range(T-1, 0, -1):
            b[t, :] = np.dot(A, b[t+1, :]) * B[self.X[t+1], :]

        # a == alpha?
        # b == beta?
        return alpha, beta

    def fit(self, X):
        self.X = X
        self.N = X.shape[0]
        self.ndim = X.shape[1]
        np.random.seed(self.random_seed)

        # initialization schemes
        if self.init_method == 'random':
            pass

        ######## BEGIN ACTUAL ALGORITHM ###################
        gamma = np.zeros(self.T, len(self.zval_range))
        for iter in range(self.max_iter):
            alpha, beta = self.forward_backward()
            for t in range(self.T):
                for i in self.zval_range:
                    gamma[t, i] = alpha[t, i] * beta[t, i] / NORMALIZATION  # TODO



            # E step


            # M step

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