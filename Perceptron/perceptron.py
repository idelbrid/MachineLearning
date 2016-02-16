import sys
import numpy as np


class perceptron:  # wrapper for the perceptron utilities
    def __init__(self, learn_rate=1, max_iter=250, rate_decay=None):
        self.max_iter = max_iter  # repetitions

        self.learn_rate = learn_rate
        if rate_decay == "quadratic":
            self.rate_update = lambda x: x * (1 - float(1) / self.max_iter)
        elif rate_decay == "linear":
            self.rate_update = lambda x: x - float(self.learn_rate) / self.max_iter  # Todo
        elif type(rate_decay) is None:
            self.rate_update = lambda x: x
        else:
            pass
            # print ("Invalid learning rate input")
        self.w = None

    def fit(self, X, y, add_constant=True):
        if add_constant:
            self.add_constant = True
            self.X = np.zeros((len(X), len(X[0]) + 1))
            self.X[:, :-1] = X
            self.X[:, -1] = 1
        self.y = y
        self.N = len(self.X)
        self.K = len(self.X[0])
        self.w = np.zeros(self.K)

        for i in range(0, self.max_iter):
            if i % 100 == 0:
		print i, 'iterations'
	    any_updates = False
            for n in range(0, self.N):
                if np.sign(np.dot(self.w.T, self.X[n])) != self.y[n]:
                    self.w += self.learn_rate * self.y[n] * self.X[n]
                    any_updates = True
                self.learn_rate = self.rate_update(self.learn_rate)
            if not any_updates:
                break

    def predict(self, X):
        if self.add_constant:
            self.test = np.zeros((len(X), len(X[0]) + 1))
            self.test[:, :-1] = X
            self.test[:, -1] = 1
        return np.dot(self.w, self.test.T)


def read_data(file_str, num_feats):
    """ INPUT: string denoting the file containing the dataset
        OUTPUT: matrix of the data """
    with open(file_str, 'r') as data_file:  # Reading number of observations
        for i, l in enumerate(data_file):   # lines from stackoverflow article
            pass                            #
        size = i + 1                        #
        data = np.zeros((size, num_feats), dtype=np.int64)
        labels = np.zeros(size, dtype=np.int16)

    with open(file_str, 'r') as data_file:  # Reading data to matrix
        for i, l in enumerate(data_file):
            for j, word in enumerate(l.split(' ')):
                if j == 0:
                    labels[i] = word
                elif word == '\n':
                    pass  # if the word is the end of the line, skip it
                else:
                    feat_number, value = word.split(':')
                    data[i, int(feat_number) - 1] = value  # count from 0
    return data, labels


def evaluate_accuracy(y, yhat):
    int_prediction = np.sign(yhat)
    num_right = (int_prediction == y).sum()
    accuracy = float(num_right) / len(y)
    return accuracy, num_right, len(y)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print 'No test dataset. \nAdd test file with program argument.'
        sys.exit()
    else:
        testfile = sys.argv[1]

        num_feats = 123  # known ahead of time

        train_data, train_labels = read_data('../a7a.train', num_feats)
        test_data, test_labels = read_data(testfile, num_feats)

        model = perceptron(max_iter=250, learn_rate=150, rate_decay='linear')  # declare model with max iterations 1000
        model.fit(train_data, train_labels)  # fit the model
        predictions = model.predict(test_data)  # predict

        accuracy, num_right, total_pts = evaluate_accuracy(test_labels,
                                                           predictions)

        print num_right, 'correct predictions for', total_pts, '.'
        print 'The accuracy is', accuracy
