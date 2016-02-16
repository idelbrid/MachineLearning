# linear regression for input file

import sys
import numpy as np

class linear_regression:  # wrapper for the linear regression utilities
    def __init__(self, l=0, add_constant=True):
        self.l = l
        self.add_constant = add_constant
        
    def fit(self, X, y):
        """ INPUT: training data matrix, training data labels
            OUTPUT: weights of the model (ndarray)"""
        if self.add_constant:
            self.X = np.zeros((len(X), len(X[0])+1))
            self.X[:, :-1] = X
            self.X[:, -1] = 1
        else:
            self.X = X
        self.X = np.asmatrix(self.X)
        self.y = y        
        # (X.T * X).inverse() * X.T * y
        term1 = (np.dot(self.X.T, self.X) + self.l * np.identity(len(self.X.T))).I
        mat_w = np.dot(np.dot(term1, self.X.T), self.y)
        self.w = np.asarray(mat_w).flatten()  # Restablish as a 1 dim ndarray
        return self.w  
        
    def predict(self, X):
        if self.add_constant:
            self.test = np.zeros((len(X), len(X[0]) + 1))
            self.test[:, :-1] = X
            self.test[:, -1] = 1
        yhat = np.dot(self.test, self.w.T)
        self.prediction = yhat
        return yhat

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
    
    if len(sys.argv)<2:
        print('No test dataset. \nAdd test file with program argument.')
        sys.exit()
    else:
        testfile = sys.argv[1]

        num_feats = 123 # known ahead of time 

        train_data, train_labels = read_data('../a7a.train', num_feats)
        test_data, test_labels = read_data(testfile, num_feats)
        
        model = linear_regression(-1.04)  # declare model with lambda -1.04 for regularization
        model.fit(train_data, train_labels)  # fit the model
        predictions = model.predict(test_data)  # predict
        
        accuracy, num_right, total_pts = evaluate_accuracy(test_labels,  
                                                           predictions)
           
        print num_right, 'correct predictions for', total_pts, '.'
        print 'The accuracy is', accuracy                          
        

