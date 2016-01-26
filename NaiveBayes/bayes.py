__author__ = 'idelbrid'

import numpy as np
import sys

class naive_bayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        """INPUT: training data matrix, traning data labels
        OUTPUT: weights of the model (ndarray)"""
        #p(y=a | x1,...,xn) = p(x1,...,xn | y=a)*p(y=a) / p(x1,...xn)        
        # p(x1,...,xn) same for all classes so ... don't need it
        # p(y=1) = relative frequency of y=1
        # p(x1,...,xn | y=1) = p(x1|y=1)...p(xn|y=1)
        # p(y=1, x1,...,xn) goal 
        # 
        self.X = X
        self.y = y
        unique_yvals, yval_cts = np.unique(y, return_counts=True)
        # unique_yvals = unique_yvals_w_freq[:,0]
        # yvals_cts = unique_yvals_w_freq[:,1]
        has_yval_mask = [y == yval for yval in unique_yvals]
        # y_has_yval = [y[has_yval_mask[yval]] for yval in unique_yvals]
        # y = np.concatenate(y_has_0, y_has_1, axis=1)
        # x_cts = [np.ndarray((1,1)) for yval in unique_yvals]
        X_has_yval = [X[has_yval_mask[yval_indx]] for yval_indx in range(len(unique_yvals))]
        xval_cts = [dict() for x in unique_yvals]
        for i, yval in enumerate(unique_yvals):
            frequencies = np.array([np.unique(x_column, return_counts=True) for 
                                x_column in X_has_yval[i].T]) #freq for each attribute
            xval_cts[i] = [{att_values[indx]: freqs[indx] for indx in range(len(att_values))} for 
                                 (att_values, freqs) in frequencies] # stats for each attribute
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
            max_score, best_y = (0, -1)
            for indx, yval in enumerate(self.yvals):
                xval_cts = self.xval_cts[indx]
                cursum = 0
                for att, val in enumerate(record):
                    try:
                        cursum += np.log((xval_cts[att][val] + self.alpha)/(
                            self.ycts[indx]+self.num_labels * self.alpha))
                    except KeyError:
                        cursum += np.log(self.alpha * 1.0 / (self.ycts[indx] + 
                                         self.num_labels * self.alpha))
                        # never seen this y with this attribute's value - 0 ct
                        # print x_indx, record, indx, yval, att, val
                s = cursum * self.ycts[indx]
                scores[x_indx, indx] = s
                if s > max_score:
                    # best_y_indx = indx
                    # best_y = yval
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
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)

    accuracy, num_right, total_pts = evaluate_accuracy(test_labels, predictions)

    print num_right, 'correct predictions for', total_pts, '.'
    print 'The accuracy is ', accuracy
