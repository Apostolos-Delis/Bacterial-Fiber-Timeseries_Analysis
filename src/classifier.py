#!/usr/bin/env python3
# coding: utf8

import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Create Plots
import pandas as pd  # For data-frames
from os import path  # For robust pathing
import os  # For general os functions

from utilities import sliding_window, classify_ts, make_directory
from utilities import load_from_mat, process_mat, error
from derivative import generate_derivatives
from constants import DATA_DIR, IMAGE_DIR


class Classifier:

    def __init__(self, classifier=None, gen_derivatives=False):
        """
        :param classifier: func of the classifying function that will take in a 
                           time series as a list or pd.Series and return a bool
        """
        self._classifier = classifier
        self._gen_derivatives = gen_derivatives

    def __repr__(self):
        return "Classifier Object <{0}> with classifier: {1}".format(
                id(self), self._classifier.__name__)
    
    def save_image(self, time_series, file_name: str, lines: dict):
        """
        Generates the first and second derivative, generates the graphs,
        and then saves the graph to IMAGE_DIR/filename
        Also adds a line where the current classifier 

        :param time_series: either a list or a np.array that will be
        """
        if self._gen_derivatives:
            df = generate_derivatives(time_series)
        else:
            index = list(range(len(time_series)))
            df = pd.Series(time_series, index)

        df.plot()

        for line in lines.keys():
            plt.axvline(x=line, color=lines[line])
        plt.savefig(file_name)
        plt.close()

    def classify_file(self, file_name):
        """
        Runs the classifier function on all the filters in the file_name
        :param file_name: the file_name in the form of %d-%d-%d-[uv|blue].mat
        :rtype: pd.Series of all the 
        """
        mat = load_from_mat(file_name)
        data = process_mat(mat)

        index = list(range(1, 1+len(data)))
        classifier_indexes = [classify_ts(ts, self._classifier) for ts in data]
        df = pd.Series(classifier_indexes, index)
        return df

    def create_image_directory(self, directory_name: str, limit: int = 100, verbose: bool = True):
        """
        Creates an image directory structured like this:
        
        IMAGE_DIR/directory_name {
            FILE_1 {
                FILTER_1.png
                FILTER_2.png
                ...
            }
            FILE_2 {
                ...
            }
        }
        
        Where each png file is a graph of a time series of an individual filter, along with
        its first and second derivatives.
        
        :param directory_name: name that the directory will be called inside the IMAGE_DIR
        :param verbose: bool for whether to print the progress made in creating the images
        """
        directory_name = path.join(IMAGE_DIR, directory_name)

        if not path.isdir(directory_name):
            if verbose:
                print("Creating Image Directory...")
            make_directory(IMAGE_DIR)

        # Add the mat files into the queue to be processed
        for m, mat_file in enumerate(os.listdir(DATA_DIR)[:limit]):
            print("Processing file: {0}, file {1}/{2}"
                    .format(mat_file, m+1, len(os.listdir(DATA_DIR)[:limit])))
            mat = load_from_mat(mat_file)
            data = process_mat(mat)
            image_name = mat_file.split('.')[-2] 
            make_directory(path.join(directory_name, image_name))
            for i, ts in enumerate(data):
                ts_name = path.join(directory_name, 
                        path.join(image_name, "filter_{0}.png".format(i)))

                # Find the point where the graph is classified at so it can be added as a line
                lines = dict()
                if self._classifier is not None:
                    classifying_point = classify_ts(ts, self._classifier, window_size=10)
                    if classifying_point != -1:
                        lines[classifying_point] = "r"

                self.save_image(ts, ts_name, lines)
                if verbose and (i+1) % 10 == 0:
                    print("Processing fiber: {0}/{1}"
                        .format(i+1, len(data)))
            if verbose: 
                print("Finished processing {0}...\n".format(mat_file))

    @staticmethod
    def compare(classifier_1, classifier_2, save_images=True, verbose=True):
        """
        TODO: Implement Compare function
        """

        for m, mat_file in enumerate(os.listdir(DATA_DIR)[:limit]):
            print("Processing file: {0}, file {1}/{2}"
                    .format(mat_file, m+1, len(os.listdir(DATA_DIR)[:limit])))
            mat = load_from_mat(mat_file)
            data = process_mat(mat)
            image_name = mat_file.split('.')[-2] 
            make_directory(path.join(directory_name, image_name))
            for i, ts in enumerate(data):
                ts_name = path.join(directory_name, 
                        path.join(image_name, "filter_{0}.png".format(i)))
                self.save_image(ts, ts_name)
                if verbose and (i+1) % 10 == 0:
                    print("Processing fiber: {0}/{1}"
                        .format(i+1, len(data)))
            if verbose: 
                print("Finished processing {0}...\n".format(mat_file))

    @staticmethod
    def compare(classifier_1, classifier_2, file_name: str, create_image_dir=False):
        """
        TODO: Write Documentation for this
        pass"""
        if not isinstance(classifier_1, Classifier):
            class_1 = Classifier(classifier_1)
        if not isinstance(classifier_2, Classifier):
            class_2 = Classifier(classifier_2)

        series_1 = test.classify_file(test_file)
        series_2 = test2.classify_file(test_file)

def temp(*args, **kwargs):
    return True


if __name__ == "__main__":
    from derivative import series_threshold, percentage_threshold
    test = Classifier(series_threshold, gen_derivatives=True)
    """
    test_file = "10-19-18-uv.mat"
    test2 = Classifier(percentage_threshold)
    series_1 = test.classify_file(test_file)
    series_2 = test2.classify_file(test_file)
    # df = pd.concat(series_1, series_2)
    print("Filter | 2nd_Deriv | 1.2 Threshold | Difference ")
    for i in range(1, len(series_1)+1):
        print(" {0}\t  {1}\t\t{2}\t\t{3}".format(
            i, series_1[i], series_2[i], series_1[i]-series_2[i]))
    """
    test.create_image_directory("test", verbose=True)
 
    # print(test)
