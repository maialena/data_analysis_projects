import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
from scipy import io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#plots accuracy, returns None
def plot_accuracy(x, y, name):
    plt.plot(x, y, 'bo')
    plt.xlabel('num_samples')
    plt.ylabel('accuracy')
    plt.title(name)
    plt.show()


def train_svm(samples, clf, train_set, train_y, valid_set, valid_y, name):
    valid_error, train_error = train_svm_no_plot(samples, clf, train_set, train_y, valid_set, valid_y, name)
    plot_accuracy(samples, train_error, name + ' Training_Accuracy')
    plot_accuracy(samples, valid_error, name + ' Validation_Accuracy')
    return valid_error, train_error


# Purpose: Trains SVM
# Params:
# Return: list of valid_errors from each experiment
#REMEMBER TO UPDATE THE RETURN VALID_ERROR, TRAIN_ERROR, need to do train_svm[0] to get valid_error
#poor naming for what's returned, says error but should be score
def train_svm_no_plot(samples, clf, train_set, train_y, valid_set, valid_y, name):
    train_error = list()
    valid_error = list()
    for sample_size in samples:
        # print("Sample_size: " + str(sample_size))
        clf.fit(train_set[:sample_size], train_y[:sample_size])
        train_score = clf.score(train_set, train_y)
        train_error.append(train_score)
        # print("train_score: " + str(train_score))
        valid_score = clf.score(valid_set, valid_y)
        valid_error.append(valid_score)
        # print("valid_score: " + str(valid_score))
    return valid_error, train_error


#Splits data set into training and validation set of given size
#SHould this return a 2-d numpy array for labels? or keep it at one? clf requires 1-d, but concatenate requires 2.
def split_train_and_valid_sets(data, size):
    np.random.shuffle(data)
    shape = data.shape[0] 
    valid_set = data[:size, :-1] 
    valid_y = data[:size, -1] 
    train_set = data[size:, :-1] 
    train_y = data[size:, -1]
    return shape, valid_set, valid_y, train_set, train_y

# -----------MNIST SET-------- (sklearn fn)
mnist_dict = io.loadmat('mnist/train.mat')
mnist_trainX = mnist_dict['trainX']


mnist_train_set, mnist_valid_set, mnist_train_y, mnist_valid_y = train_test_split(mnist_trainX[:, :-1], mnist_trainX[:, -1], test_size=10000, random_state=42)
print('mnist_train_set ' + str(mnist_train_set.shape))
print('mnist_valid_set ' + str(mnist_valid_set.shape))
print('mnist_train_y ' + str(mnist_train_y.shape))
print('mnist_valid_y ' + str(mnist_valid_y.shape))


# *********************PROBLEM 2 --> TRAIN DATA ***********************

# ----------TRAIN MNIST DATA------------#
print("Training MNIST")
clf_mnist = SVC(kernel="linear")
experiments = [100, 200, 500, 1000, 2000, 5000, 10000]
# expect between 70-90% accuracy
valid_error, train_error = train_svm(experiments, clf_mnist, mnist_train_set, mnist_train_y, mnist_valid_set, mnist_valid_y,
          'MNIST')
print('Valid_error is: ' + str(valid_error))
print('Train_error is: ' + str(train_error))


# ************************** PROBLEM 3: BEST C Value ********************************
from sklearn.svm import SVC
C_range = [.01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001]
experiments = [10000]
for C in C_range:
    clf_mnist = SVC(kernel="linear", C=C)
    scores = train_svm_no_plot(experiments, clf_mnist, mnist_train_set, mnist_train_y, mnist_valid_set, mnist_valid_y, 'MNIST')
    print('C value: ' + str(C))
    print('valid_score ' + str(scores[0]))