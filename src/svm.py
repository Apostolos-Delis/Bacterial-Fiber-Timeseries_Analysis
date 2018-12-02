#!/usr/bin/env python3
# coding: utf8

import numpy as np
import pickle
import matplotlib.pyplot as plt

from os import path
from time import strftime
from glob import glob

from sklearn import svm
from sklearn import metrics
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from mat_to_np import load_np_file
from data_class import DataGenerator
from constants import MODEL_DIR


def train_model(verbose=True, training_percentage=0.8, 
        plot_roc=False, save_model=True, svm_kernel='rbf'):
    """
    Trains the logistic regression model.

    :param verbose: bool for whether to display progress reports
    :training_percentage: float from 0 to 1 of what percent of the 
                          data will be use for training purposes as 
                          opposed to testing
    :param plot_roc: bool for whether to display ROC curves for the model
    :param save_mode: bool for whether to save the trained model
    :param svm_kernel: svm kernel, possible choices are: "rbf", "poly"
                       "sigmoid", and "precomputed"
    """
    data_gen = DataGenerator()
    if verbose:
        print("Splitting data...")

    X_train, Y_train, X_test, Y_test = data_gen.train_test_split(training_percentage)

    if verbose:
        print("Loaded {0} examples.".format(X_train.shape[0]))
        print("Data Ready, beginning to train...")


    svm_classifier = svm.SVC(kernel=svm_kernel, gamma="scale")
    svm_classifier.fit(X_train, Y_train)

    Y_pred = svm_classifier.predict(X_test)
    print('Accuracy of SVM regression classifier on test set: \033[1;49;32m{:.2f}\033[0m'
            .format(svm_classifier.score(X_test, Y_test)))

    confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    if verbose:
        print("Printing Confusion Matrix:")
    print(confusion_matrix, end="\n\n")
    
    if verbose:
        print("Displaying Classification Report:")
    print(classification_report(Y_test, Y_pred))

    if plot_roc:
        if verbose:
            print("Plotting ROC curves...")
        logit_roc_auc = roc_auc_score(Y_test, svm_classifier.predict(X_test))
        fpr, tpr, thresholds = roc_curve(Y_test, svm_classifier.predict_proba(X_test)[:,1])
        plt.figure()
        plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.savefig('Log_ROC')
        plt.show()

    if save_model:
        if verbose:
            print("Serializing the SVM classifier")
        time_str = strftime("%Y-%m-%d_%H-%M-%S")
        file_name = "svm_" + svm_kernel + '_' + time_str + ".sav"
        serialize_model(file_name, model=svm_classifier)


def serialize_model(file_name: str, model):
    """
    Serialize the SVM model using pickle
    
    :param file_name: the name of the file of the model
    :param model: the sklearn logistic regression model
    """
    file_name = path.join(MODEL_DIR, file_name)
    print(file_name)
    pickle.dump(model, open(file_name, 'wb'))


def load_model(file_name: str, full_path: bool = False):
    """
    Load a serialized model from a file

    :param file_name: the name of the file of the model
    :param full_path: bool on whether or not the file_name
                     is an absolute or relative path
    :returns the model stored in the file_name 
    """
    if not full_path:
        file_name = path.join(MODEL_DIR, file_name)

    if not path.isfile(file_name):
        raise OSError("{0} does not exist!".format(file_name))

    return pickle.load(open(file_name, 'rb'))


def svm_classifier(ts: list) -> bool:
    """
    Returns true or false based on whether the support vector machine
    returns 1 or 0 with ts as the input data 

    :param ts: list of time series to be classified
    """
    classification_map = {1: True, 0: False}
    list_of_files = glob(MODEL_DIR + "/svm_*.sav")

    if list_of_files == []:
        raise Exception("Error: No logistic regression model saved")

    latest_file = max(list_of_files, key=path.getctime)
    classifier = load_model(latest_file, full_path=True)

    try:
        prediction = int(np.array(classifier.predict(ts)).reshape(1)[0])
    except ValueError:
        prediction = int(np.array(classifier.predict([ts])).reshape(1)[0])
    return classification_map[prediction]


if __name__ == "__main__":
    colors = {
            0: "b",
            1: "r",
            }
    x_file = "10-9-18-uv_X.npy"
    y_file = "10-9-18-uv_Y.npy"
    # dg = DataGenerator(100)
    # test_data, y = dg.get_data()
    # print(test_data[0], y[0])
    # print(logistic_reg_prediction(test_data[0]))

    train_model(verbose=True, training_percentage=0.8, 
        plot_roc=False, save_model=True, svm_kernel='rbf')
    # index = -100
    # for index in range(y_data.shape[0]):
        # plt.plot(x_data[index], color=colors[y_data[index]])
    # plt.show()

