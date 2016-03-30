import sys
import numpy as np
from scipy.stats import multivariate_normal as normal


class gauss_mixture:
    def __init__(self, max_iter, num_gaussians, init_means=None, init_cov=None, init_weights=None, init_method='random',
                 init_seed=123456):  # todo finish?
        if init_method not in ['random', 'kmeans']:
            sys.exit("undefined initialization method")
        elif init_method == 'kmeans':
            sys.exit("kmeans initialization not yet implemented")

        self.init_means = init_means
        self.init_cov = init_cov
        self.init_weights = init_weights
        self.init_method = init_method
        self.X = None
        self.ndim = None
        self.max_iter = max_iter
        self.num_gaussians = num_gaussians
        self.random_seed = init_seed

    def fit(self, X):  # todo: finish by rechecking correctness and saving probabilities
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
                    # sigma[k] += np.random.rand(self.ndim, self.ndim)
                    # sigma[k] = np.dot(sigma[k], sigma[k].T)
                    # sigma[k] /= sigma[k].sum()

                    lowerbound = k * self.N / self.num_gaussians
                    upperbound = lowerbound + 20
                    sigma[k] = np.cov(X[lowerbound:upperbound, :].T)
            if self.init_weights is not None:
                lmbda = self.init_weights
            else:
                lmbda = np.random.rand(self.num_gaussians)
                lmbda /= lmbda.sum()

        for iter in range(self.max_iter):
            phat = np.zeros((self.N, self.num_gaussians))
            N = np.zeros(self.num_gaussians)

            # E step
            for n in range(0, self.N):  # for each training point...
                for k in range(0, self.num_gaussians):
                    normalx = normal(mean=mu[k], cov=sigma[k]).pdf(X[n, :])
                    phat[n, k] = lmbda[k] * normalx
                phat[n, :] /= phat[n, :].sum()

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

    def pred(self, test_data):
        num_pts = len(test_data)
        likelihood = np.zeros(num_pts)
        p = np.zeros((num_pts, self.num_gaussians), dtype=np.float64)
        pred_labels = np.zeros(num_pts)
        for n in range(num_pts):
            for k in range(self.num_gaussians):
                normal_var = normal(mean=self.mu[k], cov=self.sigma[k])
                p[n, k] = self.lmbda[k] * normal_var.pdf(test_data[n])
            pred_labels[n] = p[n, :].argmax()
            likelihood[n] = p[n, :].sum()
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

def make_plot():
    log_likelihoods = np.zeros(30)
    for num_gauss in range(1, 30):
        model = gauss_mixture(17, num_gaussians=num_gauss)
        model.fit(train_data)  # fitting
        # train_labels, train_likelihoods = model.pred(train_data)
        test_labels, test_likelihoods = model.pred(test_data)

        log_likelihoods[num_gauss-1] = np.log(test_likelihoods).sum()
        print log_likelihoods[num_gauss-1], 'log likelihood ', num_gauss - 1
    with open('num_gaussians_likelihoods.csv', 'w+') as f:
        f.write("number of gaussians,log likelihood\n")
        for i, row in enumerate(log_likelihoods):
            f.write("{},{}\n".format(i+1, row))

if __name__ == "__main__":

    if len(sys.argv) < 1:
        print 'No dataset. \nAdd data file with program argument.'
        sys.exit()
    else:
        num_feats = 2  # known ahead of time

        data = read_data('../points.dat', num_feats)
        train_data = data[:(len(data) * 9 / 10), :]  # first 90%
        test_data = data[(len(data)) * 9 / 10:, :]  # last 10%
        # means = np.array([[1.3, 2],    # eye-balled means for initialization
        #                   [2, -1.3],
        #                   [-1.5, 1],
        #                   [-1, -1.5]])
        # cov = [np.array([[1, 0],
        #                  [0, 1]])]*4
        model = gauss_mixture(30, num_gaussians=3)
        model.fit(train_data)  # fitting
        train_labels, train_likelihoods = model.pred(train_data)
        test_labels, test_likelihoods = model.pred(test_data)

        log_likelihood = np.log(test_likelihoods).sum()

        print log_likelihood, "log likelihood"

        with open('label_logs_6rand.csv', 'w+') as f:
            f.write("dim1,dim2,label,likelihood\n")
            for row in np.vstack((test_data[:, 0], test_data[:, 1], test_labels, test_likelihoods)).T:
                f.write("{},{},{},{}\n".format(row[0], row[1], row[2], row[3]))

        make_plot()