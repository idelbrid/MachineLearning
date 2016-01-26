__author__ = 'idelbrid'

import numpy as np
import sys


class naive_bayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        """INPUT: training data matrix, traning data labels
        OUTPUT: weights of the model (ndarray)"""
        self.X = X
        self.y = y
        y_has_1 = (y == 1)
        X_filtered = X[y_has_1]  # Not sure if this will run
        cx = X_filtered.sum(axis=1)


def read_data(file_str, num_feats):
    """ INPUT: string denoting the file containing the dataset
        OUTPUT: matrix of the data """
    with open(file_str, 'r') as data_file:   # Reading number of observations
        for i, l in enumerate(data_file): # lines from stackoverflow article
            pass                          #
        size = i + 1                      #
        data = np.zeros((size, num_feats), dtype = np.int64)
        labels = np.zeros(size, dtype = np.int16)

    with open(file_str, 'r') as data_file:   # Reading data to matrix
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
        print 'No test dataset. \nAdd test file with program argument'
        sys.exit()

    testfile = sys.argv[1]
    num_feats = 123  # Known ahead of time

    train_data, train_labels = read_data('a7a.train', num_feats)
    test_data, test_labels = read_data(testfile, num_feats)

    model = naive_bayes(alpha=1)
    model.fit(np.asmatrix(train_data), train_labels)
    predictions = model.predict(np.asmatrix(test_data))

    accuracy, num_right, total_pts = evaluate_accuracy(test_labels, predictions)

    print num_right, 'correct predictions for', total_pts, '.'
    print 'The accuracy is ', accuracy
