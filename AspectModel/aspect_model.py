import sys
import numpy as np


class aspect_model:
    def __init__(self, max_iter, num_topics, init_seed=123456):  # todo finish?
        self.X = None
        self.y_pred = None
        self.ndim = None
        self.max_iter = max_iter
        self.num_topics = num_topics
        self.word_vals = None
        self.doc_vals = None
        self.pz = None
        self.pdz = None
        self.pwz = None
        self.random_seed = init_seed

    def fit(self, X):  # todo: finish by rechecking correctness and saving probabilities
        self.X = X
        self.N = X.shape[0]
        self.ndim = X.shape[1]
        self.word_vals = np.unique(X[:, 0])
        self.doc_vals = np.unique(X[:, 1])
        np.random.seed(self.random_seed)
        phat = np.zeros(self.num_topics)  # phat for a given "n", not the total one!
        pz = np.random.rand(self.num_topics)
        pz /= pz.sum()
        pwz = np.random.rand(len(self.word_vals), self.num_topics)
        pwz /= pwz.sum()
        pdz = np.random.rand(len(self.doc_vals), self.num_topics)
        pdz /= pdz.sum()

        for iter in range(self.max_iter):
            ec_k = np.zeros(self.num_topics)
            ec_wk = np.zeros((len(self.word_vals), self.num_topics))
            ec_dk = np.zeros((len(self.doc_vals), self.num_topics))
            # E step
            for n in range(0, self.N):  # for each training point...
                for k in range(0, self.num_topics):  # for each topic...
                    phat[k] = pz[k] * pwz[X[n][0]][k] * pdz[X[n][1]][
                        k]  # Use parameters to get a distribution
                Z = phat.sum()
                phat /= float(Z)
                for k in range(0, self.num_topics):
                    ec_k[k] += phat[k]
                    ec_wk[X[n][0], k] += phat[k]
                    ec_dk[X[n][1], k] += phat[k]
            # M step
            for k in range(0, self.num_topics):  # Use distribution to calculate optimal parameters
                pz[k] = ec_k[k] / float(self.N)
                for i in self.word_vals:
                    pwz[i, k] = ec_wk[i, k] / float(ec_k[k])
                for i in self.doc_vals:
                    pdz[i, k] = ec_dk[i, k] / float(ec_k[k])
            pass  # end of this iteration
        self.pwz = pwz
        self.pdz = pdz
        self.pz = pz

    def pred(self, test_data):
        num_pts = len(test_data)
        pred_topics = np.zeros(num_pts)
        likelihood = np.zeros(num_pts)
        p = np.zeros(self.num_topics)
        for n in range(0, num_pts):
            for k in range(0, self.num_topics):  # for each topic...
                p[k] = self.pz[k] * self.pwz[test_data[n][0]][k] * self.pdz[test_data[n][1]][k]  # Use parameters to get a distribution
            # p /= p.sum()
            pred_topics[n] = p.argmax()
            likelihood[n] = p.sum()
        return pred_topics, likelihood


def read_data(file_str, num_feats):  # todo: verify this is correct
    """ INPUT: string denoting the file containing the dataset
        OUTPUT: matrix of the data """
    with open(file_str, 'r') as data_file:  # Reading number of observations
        for i, l in enumerate(data_file):  # lines from stackoverflow article
            pass  #
        size = i + 1  #
        data = np.zeros((size, num_feats), dtype=np.int64)

    with open(file_str, 'r') as data_file:  # Reading data to matrix
        for i, l in enumerate(data_file):
            for j, word in enumerate(l.split(' ')):
                if word == '\n':
                    pass  # if the word is the end of the line, skip it
                else:
                    data[i, j] = int(word)
    return data

if __name__ == "__main__":

    if len(sys.argv) < 1:
        print 'No dataset. \nAdd data file with program argument.'
        sys.exit()
    else:
        num_feats = 2  # known ahead of time

        data = read_data('../pairs.dat', num_feats)
        train_data = data[:(len(data) * 9 / 10), :]
        test_data = data[(len(data)) * 9 / 10:, :]
        model = aspect_model(25, num_topics=5)
        model.fit(train_data)  # fitting
        train_pred_labels, train_pt_likelihood = model.pred(train_data)
        test_pred_labels, test_pt_likelihood = model.pred(train_data)
        test_log_likelihood = np.log(test_pt_likelihood).sum()

        print test_log_likelihood, 'log likelihood of last 10% of data'

