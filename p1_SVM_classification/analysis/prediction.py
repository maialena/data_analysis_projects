import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np


def plot_accuracy(data, labels, name):
    plt.plot(data, labels)
    plt.xlabel('num_samples')
    plt.ylabel('accuracy')
    plt.title(name + '_Linear_SVM')
    plt.show()


# returns plot of error rate
def train_svm(samples, clf, train_set, train_y, valid_set, valid_y, name):
    error = list() # need error to be an array in order to plot it (See matplotlib.pyplot)
    print('Training ' + name + '\n')
    for sample_size in samples:
        # fit the training data to the labels
        # print('Train_data shape, train_label shape:')
        # print(train_set.shape, train_y.shape)
        clf.fit(train_set[:sample_size], train_y[:sample_size])
        y_hat = clf.predict(valid_set)
        error.append(accuracy_score(valid_y, y_hat))
    plot_accuracy(samples, error, name)


def split_train_and_valid_sets(data, size):
    np.random.shuffle(data)
    shape = data.shape[0]
    valid_set = data[:size, :-1]
    valid_y = data[:size, -1]
    train_set = data[size:, :-1]
    train_y = data[size:, -1]
    return shape, valid_set, valid_y, train_set, train_y

