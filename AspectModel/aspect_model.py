import sys
import numpy as np


class aspect_model:  # wrapper for the perceptron utilities
    def __init__(self, max_iter, num_topics):  # todo
        self.X = None
        self.y_pred = None
        self.ndim = None
        self.max_iter = max_iter
        self.num_topics = num_topics
        self.word_vals = None
        self.doc_vals = None

    def fit_pred(self, X):  # todo
        self.X = X
        self.N = X.shape[0]
        self.ndim = X.shape[1]
        self.word_vals = np.unique(X[:,0])
        self.doc_vals = np.unique(X[:,1])

        phat = np.zeros(self.num_topics)  # phat for a given "n", not the total one!
        pz = np.zeros(self.num_topics)
        pwz = np.zeros((len(self.word_vals), self.num_topics))
        pdz = np.zeros((len(self.doc_vals), self.num_topics))

        ec_k = np.zeros(self.num_topics)
        ec_wk = np.zeros((len(self.word_vals), self.num_topics))
        ec_dk = np.zeros((len(self.doc_vals), self.num_topics))

        for iter in range(self.max_iter):
            # E step
            Z = 1  # todo: figure out if this needs to actually be anything
            for n in range(0, self.N):
                for k in range(0, self.num_topics):
                    phat[k] = (1 / Z) * pz[k] * pwz[X[n][0]] * pdz[X[n][1]]  # Use parameters to get a distribution
                    ec_k[k] += phat[k]
                    ec_wk[X[n][0], k] += phat[k]
                    ec_dk[X[n][1], k] += phat[k]

            # M step
            for k in range(0, self.num_topics):  # Use distribution to calculate optimal parameters
                pz[k] = ec_k[k]
                for i in self.word_vals:
                    pwz[i, k] = ec_wk[i, k] / ec_k[k]
                for i in self.doc_vals:
                    pdz[i, k] = ec_dk[i, k] / ec_k[k]


def read_data(file_str, num_feats):
    """ INPUT: string denoting the file containing the dataset
        OUTPUT: matrix of the data """
    with open(file_str, 'r') as data_file:  # Reading number of observations
        for i, l in enumerate(data_file):   # lines from stackoverflow article
            pass                            #
        size = i + 1                        #
        data = np.zeros((size, num_feats), dtype=np.int64)

    with open(file_str, 'r') as data_file:  # Reading data to matrix
        for i, l in enumerate(data_file):
            for j, word in enumerate(l.split(' ')):
                if word == '\n':
                    pass  # if the word is the end of the line, skip it
                else:
                    feat_number, value = word.split(':')
                    data[i, int(feat_number)] = value  # count from 0
    return data

def evaluate_accuracy():  # todo
    pass
# def evaluate_accuracy(y, yhat):
#     int_prediction = np.sign(yhat)
#     num_right = (int_prediction == y).sum()
#     accuracy = float(num_right) / len(y)
#     return accuracy, num_right, len(y)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print 'No test dataset. \nAdd test file with program argument.'
        sys.exit()
    else:
        testfile = sys.argv[1]

        num_feats = 2  # known ahead of time

        train_data, train_labels = read_data('a7a.train', num_feats)
        test_data, test_labels = read_data(testfile, num_feats)

        model = aspect_model()  # declare model with max iterations 1000
        label_pred = model.fit_pred(test_data)  # predict

        # accuracy, num_right, total_pts = evaluate_accuracy(test_labels,
        #                                                    predictions)


        # todo
        # print num_right, 'correct predictions for', total_pts, '.'
        # print 'The accuracy is', accuracy