#!/usr/bin/env python3
# coding: utf8

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# import sys

from mat_to_np import load_np_file
from data_class import DataGenerator

if __name__ == "__main__":
    colors = {
            0: "b",
            1: "r",
            }
    x_file = "10-9-18-uv_X.npy"
    y_file = "10-9-18-uv_Y.npy"
    
    # index = -100
    # for index in range(y_data.shape[0]):
        # plt.plot(x_data[index], color=colors[y_data[index]])
    # plt.show()

    data_gen = DataGenerator()
    print("Splitting data...")
    X_train, Y_train, X_test, Y_test = data_gen.train_test_split(train_percentage=0.8)
    print("Loaded {0} examples.".format(X_train.shape[0]))
    print("Data Ready, beginning to train...")
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)

    Y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: \033[1;49;32m{:.2f}\033[0m'
            .format(logreg.score(X_test, Y_test)))

    confusion_matrix = confusion_matrix(Y_test, Y_pred)
    print("Printing Confusion Matrix:")
    print(confusion_matrix, end="\n\n")
    
    print("Displaying Classification Report:")
    print(classification_report(Y_test, Y_pred))

    print("Plotting ROC curves...")
    logit_roc_auc = roc_auc_score(Y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    plt.show()
