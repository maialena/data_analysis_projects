import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
from scipy import io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# -----------MNIST SET-------- (sklearn fn)
mnist_dict = io.loadmat('mnist/train.mat')
mnist_trainX = mnist_dict['trainX']


mnist_train_set, mnist_valid_set, mnist_train_y, mnist_valid_y = train_test_split(mnist_trainX[:, :-1], mnist_trainX[:, -1], test_size=10000, random_state=42)



# # *********************PROBLEM 2 --> TRAIN DATA ***********************

# # ----------TRAIN MNIST DATA------------#
# print("Training MNIST")
# clf_mnist = SVC(kernel="linear")
# experiments = [100, 200, 500, 1000, 2000, 5000, 10000]
# # expect between 70-90% accuracy
# valid_error, train_error = train_svm(experiments, clf_mnist, mnist_train_set, mnist_train_y, mnist_valid_set, mnist_valid_y,
#           'MNIST')
# print('Valid_error is: ' + str(valid_error))
# print('Train_error is: ' + str(train_error))


# # ************************** PROBLEM 3: BEST C Value ********************************
# from sklearn.svm import SVC
# C_range = [.01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001]
# experiments = [10000]
# for C in C_range:
#     clf_mnist = SVC(kernel="linear", C=C)
#     scores = train_svm_no_plot(experiments, clf_mnist, mnist_train_set, mnist_train_y, mnist_valid_set, mnist_valid_y, 'MNIST')
#     print('C value: ' + str(C))
#     print('valid_score ' + str(scores[0]))


# ****************** PROBLEM 5: KAGGLE COMPETITION ***************
import csv

# mnist submission
mnist_data = io.loadmat('mnist/test.mat')['testX']
clf_mnist = SVC(C=.000001, kernel="linear")
clf_mnist.fit(mnist_train_set, mnist_train_y)
predicted_labels = clf_mnist.predict(mnist_data)
# file = open('mnist_submission.csv', 'w')
# w = csv.writer(f)

with open('mnist_submit_to_kaggle.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(predicted_labels)):
        writer.writerow([i, predicted_labels[i]])
csvfile.close()