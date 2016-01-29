__author__ = 'idelbrid'
import numpy as np
import sys
import time


class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.num_atts, self.num_labels, self.num_recrds = None, None, None
        self.yvals, self.ycts, self.xval_cts = None, None, None

    def fit(self, X, y):
        """INPUT: training data matrix, traning data labels
        OUTPUT: weights of the model (ndarray)"""
        # p(y=a | x1,...,xn) = p(x1,...,xn | y=a)*p(y=a) / p(x1,...xn)  bayes rule
        # p(x1,...,xn) same for all labels so ... don't need it
        # p(y=a) = frequency of y=a
        # p(x1,...,xn | y=a) = p(x1|y=a)...p(xn|y=a) (assumed)
        # p(y=a, x1,...,xn) goal
        # 
        self.X = X
        self.y = y
        unique_yvals, yval_cts = np.unique(y, return_counts=True)
        # unique_yvals = unique_yvals_w_freq[:,0]
        # yvals_cts = unique_yvals_w_freq[:,1]
        has_yval_mask = [y == yval for yval in unique_yvals]

        X_has_yval = [X[has_yval_mask[yval_indx]] for yval_indx in range(len(unique_yvals))]
        xval_cts = [dict() for x in unique_yvals]
        for i, yval in enumerate(unique_yvals):

            frequencies = np.array([np.unique(x_column, return_counts=True) for x_column in X_has_yval[i].T])
            # one frequency for each attribute
            # inline_freq = np.array(np.stack(np.unique(x_column, return_counts=True) for x_column in X_has_yval[i].T))
            xval_cts[i] = [{att_values[indx]: freqs[indx] for indx in range(len(att_values))} for
                           (att_values, freqs) in frequencies]  # stats for each attribute
        self.num_atts = len(X[0])
        self.num_recrds = len([X])
        self.yvals = unique_yvals
        self.num_labels = len(self.yvals)
        self.ycts = yval_cts
        self.xval_cts = xval_cts
         
    def predict(self, X):
        scores = np.zeros((len(X), self.num_labels))
        predictions = np.zeros((len(X)))
        for x_indx, record in enumerate(X):
            max_score, best_y = -np.inf, -1
            for y_indx, yval in enumerate(self.yvals):
                xval_cts = self.xval_cts[y_indx]
                cursum = 0
                this_rec_freqs = np.array([xval_cts[att].get(val, 0) for (att, val) in enumerate(record)], dtype=np.float64)
                marginal_scores = np.log((this_rec_freqs + self.alpha) / (self.ycts[y_indx] + self.num_labels *
                                                                               self.alpha))
                s = np.sum(marginal_scores) + np.log(self.ycts[y_indx])
                # for att, val in enumerate(record):
                #     cursum += np.log(float(xval_cts[att].get(val, 0) + self.alpha)/(self.ycts[y_indx]+self.num_labels *
                #                                                              self.alpha))
                # s = cursum + np.log(self.ycts[y_indx])
                scores[x_indx, y_indx] = s
                if s > max_score:
                    # best_y_indx = indx
                    # best_y = yval
                    max_score = s
                    predictions[x_indx] = yval
        self.scores = scores
        self.predictions = predictions

        return predictions
        
def read_data(file_str, num_feats):
    """ INPUT: string denoting the file containing the dataset
        OUTPUT: matrix of the data """
    with open(file_str, 'r') as data_file:   # Reading number of observations
        for i, l in enumerate(data_file): # lines from stackoverflow article
            pass                          #
        size = i + 1                      #
        data = np.zeros((size, num_feats), dtype=np.int64)
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

def batch_tests(train_data, train_labels, test_data, test_labels, alpha_vals):
    """
    :param alpha_vals: list of values to test alpha for
    :return: list of accuracies corresponding to the alpha vals
    """
    accuracies = np.zeros(len(alpha_vals))
    for index, alpha in enumerate(alpha_vals):
        model = NaiveBayes(alpha=alpha)
        model.fit(train_data, train_labels)
        predictions = model.predict(test_data)
        accuracy, num_right, total_pts = evaluate_accuracy(test_labels, predictions)
        accuracies[index] = accuracy
    return accuracies

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'No test dataset. \nAdd test file with program argument'
        sys.exit()

    testfile = sys.argv[1]
    num_feats = 123  # Known ahead of time

    train_data, train_labels = read_data('a7a.train', num_feats)
    test_data, test_labels = read_data(testfile, num_feats)

    model = NaiveBayes(alpha=305.5)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)

    accuracy, num_right, total_pts = evaluate_accuracy(test_labels, predictions)

    print num_right, 'correct predictions for', total_pts, '.'
    print 'The accuracy is ', accuracy

    # alphas = np.arange(0.5, 1000, 0.5)
    # accuracies = batch_tests(train_data, train_labels, test_data, test_labels, alphas)
    # with open('record_accuracies.txt', 'w') as f:
    #     for alpha, accuracy in zip(alphas, accuracies):
    #         f.write('%f, %f\n' % (alpha, accuracy))
