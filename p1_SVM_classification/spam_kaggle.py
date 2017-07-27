import matplotlib.pyplot as plt


import numpy as np
import scipy as sp
from scipy import io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import csv

# Purpose: Trains SVM
# Params:
# Return: list of valid_errors from each experiment
#REMEMBER TO UPDATE THE RETURN VALID_ERROR, TRAIN_ERROR, need to do train_svm[0] to get valid_error
#poor naming for what's returned, says error but should be score
def train_svm_no_plot(samples, clf, train_set, train_y, valid_set, valid_y, name):
    train_error = list()
    valid_error = list()
    for sample_size in samples:
        print("Sample_size: " + str(sample_size))
        clf.fit(train_set[:sample_size], train_y[:sample_size])
        train_score = clf.score(train_set, train_y)
        train_error.append(train_score)
        print("train_score: " + str(train_score))
        valid_score = clf.score(valid_set, valid_y)
        valid_error.append(valid_score)
        print("valid_score: " + str(valid_score))
    return valid_error, train_error

# --------------SPAM DataSet (sklearn fn)----------------
spam_dict = sp.io.loadmat('spam/spam_data.mat')
spam_trainX= spam_dict['training_data']
spam_labels = spam_dict['training_labels']


spam_train_set, spam_valid_set, spam_train_y, spam_valid_y = train_test_split(spam_trainX, spam_labels.T, test_size=0.2, random_state=42)


clf_spam = SVC(kernel='linear')
spam_data = io.loadmat('spam/spam_data.mat')['test_data']

clf_spam.fit(spam_train_set, spam_train_y)
predicted_labels = clf_spam.predict(spam_data)
valid_error, train_error = train_svm_no_plot([spam_data.shape[0]], clf_spam, spam_train_set, spam_train_y, spam_valid_set, spam_valid_y,
          'MNIST')

with open('thurs_spam.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(predicted_labels)):
        writer.writerow([i, predicted_labels[i]])
csvfile.close()
