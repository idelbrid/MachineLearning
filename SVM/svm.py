
import sys
import numpy as np

class SVM:  # wrapper for the perceptron utilities
    def __init__(self, capacity=250, learn_rate=1, max_iter=250, rate_decay=None):
        self.max_iter = max_iter  # repetition
        self.learn_rate = learn_rate
        self.C = capacity
        self.rate_decay = rate_decay
        
        if rate_decay == "quadratic":
            self.rate_update = lambda x: x * (1 - float(1) / self.max_iter)
        elif rate_decay == "linear":
            self.rate_update = lambda x: x - float(self.learn_rate) / self.max_iter
        elif rate_decay is None:
            self.rate_update = lambda x: x
        else:
            print ("Invalid learning rate input")
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.K = self.X[0]
        self.w = np.zeros(len(self.K))
        self.b = 0

        epsilon = 0.5
        # Stochastic Gradient Descent
        for i in range(0, self.max_iter):
            if i % 100 == 0:
                pass
                # print i, 'iterations'
            run_w = self.w
            for n in range(0, self.N):
                if 1 - self.y[n] * (np.dot(self.w.T, self.X[n]) + self.b) > 0:
                    self.w += - self.learn_rate * (float(1) / self.N * self.w - self.C *
                                                   self.y[n] * self.X[n])
                    self.b += self.learn_rate * self.C * self.y[n]
                else:
                    self.w = self.w - self.learn_rate * (float(1) / self.N * self.w)
            self.learn_rate = self.rate_update(self.learn_rate)
            if (np.abs(run_w - self.w).sum()) < epsilon:  # good enough right here...
                break

    def predict(self, X):
        self.test = X

        return np.dot(self.w, self.test.T) + self.b


def read_data(file_str, num_feats):
    """ INPUT: string denoting the file containing the dataset
        OUTPUT: matrix of the data """
    with open(file_str, 'r') as data_file:  # Reading number of observations
        for i, l in enumerate(data_file):  # lines from stackoverflow article
            pass  #
        size = i + 1  #
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
        print('No test dataset. \nAdd test file with program argument.')
        sys.exit()
    else:
        test_capacity = 10
        test_max_iter = 500
        test_learn_rate = 50
        test_rate_decay = 'linear'

        testfile = sys.argv[1]
        if len(sys.argv) > 2:
            test_capacity = int(sys.argv[2])

        num_feats = 123  # known ahead of time

        train_data, train_labels = read_data('../a7a.train', num_feats)
        test_data, test_labels = read_data(testfile, num_feats)

        model = SVM(max_iter=test_max_iter, capacity=test_capacity, learn_rate=test_learn_rate, rate_decay=test_rate_decay)  # declare model

        model.fit(train_data, train_labels)  # fit the model
        predictions = model.predict(test_data)  # predict

        accuracy, num_right, total_pts = evaluate_accuracy(test_labels,
                                                           predictions)

        print "capacity:", test_capacity, "max_iter:", test_max_iter, 'learn_rate:', test_learn_rate
        print num_right, 'correct predictions for', total_pts, '.'
        print 'The accuracy is', accuracy
        # print accuracy