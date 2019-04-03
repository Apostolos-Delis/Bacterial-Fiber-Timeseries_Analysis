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
        try:
            return "Classifier Object <{0}> with classifier: {1}".format(
                    hex(id(self)), self._classifier.__name__)
        except AttributeError:
            return "Classifier Object <{0}> with no classifier".format(hex(id(self)))


    
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

        for color in lines.keys():
            for line in lines[color]:
                plt.axvline(x=line, color=color)
        plt.xlabel("TODO complete the plotting", fontsize=16)
        plt.ylabel('Loss', fontsize=16)
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

        # Go through all the files in the DATA_DIR, classify them, and save the image
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
                lines = {'r': [], 'b': []}
                if self._classifier is not None:
                    classifying_point = classify_ts(ts, self._classifier, window_size=10)
                    if classifying_point != -1:
                        lines['r'].append(classifying_point)

                self.save_image(ts, ts_name, lines)
                if verbose and (i+1) % 10 == 0:
                    print("Processing fiber: {0}/{1}"
                        .format(i+1, len(data)))
            if verbose: 
                print("Finished processing {0}...\n".format(mat_file))


    @staticmethod
    def compare(classifier_1, classifier_2, directory_name: str, 
                save_images=True, verbose=True, limit=1000,
                ignore_blue=True):
        """
        Compare the results of classifier_1 and classifier_2

        :param classifer_1 and classifier_2: need to be functions
                                             that return true or false
                                             after classifying a time series
        :param directory_name: the name of where the images will be stored
                               note: they will be stored in IMAGE_DIR
        :param save_images: whether to save the images or not
        :param verbose: display the progress report
        :param limit: how many matlab files at max to compare
        :param ignore_blue: bool for whether to test against 
                            blue light data
        """
        directory_name = path.join(IMAGE_DIR, directory_name)
        if not path.isdir(directory_name):
            if verbose:
                print("Creating Image Directory...")
            make_directory(IMAGE_DIR)

        plotter = Classifier(None, gen_derivatives=True)
        
        classifier_1_vals = []
        classifier_2_vals = []
        for m, mat_file in enumerate(os.listdir(DATA_DIR)[:limit]):
            print("Processing file: {0}, file {1}/{2}"
                    .format(mat_file, m+1, len(os.listdir(DATA_DIR)[:limit])))

            if ignore_blue and "blue" in mat_file:
                if verbose:
                    print("{0} is a blue light dataset, ignoring...\n".format(mat_file))
                continue

            mat = load_from_mat(mat_file)
            data = process_mat(mat)
            image_name = mat_file.split('.')[-2] 
            if save_images:
                make_directory(path.join(directory_name, image_name))

            for i, ts in enumerate(data):
                ts_name = path.join(directory_name, 
                        path.join(image_name, "filter_{0}.png".format(i)))

                lines = {'r': [], 'b': []}
                classifying_point_1 = classify_ts(ts, classifier_1, window_size=10)
                if classifying_point_1 != -1:
                    classifier_1_vals.append(classifying_point_1)
                    lines['r'].append(classifying_point_1)

                classifying_point_2 = classify_ts(ts, classifier_2, window_size=10)
                if classifying_point_2 != -1:
                    classifier_2_vals.append(classifying_point_2)
                    lines['b'].append(classifying_point_2)

                if classifying_point_2 == -1 and classifying_point_1 != -1:
                    print("Classifying point 2 classified!")
                    make_directory(path.join(directory_name, image_name))
                    plotter.save_image(ts, ts_name, lines)

                if classifying_point_2 != -1 and classifying_point_1 == -1:
                    print("Classifying point 1 classified!")
                    make_directory(path.join(directory_name, image_name))
                    plotter.save_image(ts, ts_name, lines)

                if save_images:
                    plotter.save_image(ts, ts_name, lines)
                if verbose and (i+1) % 10 == 0:
                    print("Processing fiber: {0}/{1}"
                        .format(i+1, len(data)))
            if verbose: 
                print("Finished processing {0}...\n".format(mat_file))

        if verbose:
            print("Generating Classification Report..")
        
        classifier_1_vals = np.array(classifier_1_vals)
        average_1 = np.average(classifier_1_vals)
        classifier_2_vals = np.array(classifier_2_vals)
        average_2 = np.average(classifier_2_vals)

        print("Average of {0}: ".format(classifier_1.__name__))
        print("Average of {0} over {1} items".format(average_1, classifier_1_vals.size))
        print("----------------------------")
        print("Average of {0}: ".format(classifier_2.__name__))
        print("Average of {0} over {1} items".format(average_2, classifier_2_vals.size))
        

if __name__ == "__main__":
    from derivative import series_threshold, percentage_threshold
    from logistic_reg import logistic_reg_classifier
    from svm import svm_classifier

    directory_name = "svm_classifier_vs_standard_threshold"
    # Classifier.compare(svm_classifier,
                    # percentage_threshold,
                    # directory_name,
                    # save_images=False,
                    # limit=100,
                    # verbose=False,
                    # ignore_blue=True)
    
    test = Classifier(percentage_threshold, gen_derivatives=True)

    test.create_image_directory("rbf_kernel", limit=1)
