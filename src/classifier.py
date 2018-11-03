#!/usr/bin/env python3
# coding: utf8

import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Create Plots
import pandas as pd  # For data-frames
from os import path  # For robust pathing
import os  # For general os functions

from utilities import sliding_window, classify_ts, make_directory
from utilities import load_from_mat, process_mat
from derivative import generate_derivatives
from constants import DATA_DIR, IMAGE_DIR


class Classifier:

    def __init__(self, classifier=None):
        self._classifier = classifier

    def __repr__(self):
        return "Classifier Object <{0}> with classifier: {1}".format(
                id(self), self._classifier.__name__)
    
    def save_image(self, time_series, file_name: str):
        """
        Generates the first and second derivative, generates the graphs,
        and then saves the graph to IMAGE_DIR/filename
        Also adds a line where the current classifier 

        :param time_series: either a list or a np.array that will be
        """
        df = generate_derivatives(time_series, verbose=False)
        # index = list(range(len(time_series)))
        # df = pd.Series(time_series, index)
        assert(isinstance(df, pd.DataFrame) or isinstance(df, pd.Series))
        df.plot()
        if self._classifier is not None:
            classifying_point = classify_ts(df, self._classifier, window_size=10)

            if classifying_point != -1:
                plt.axvline(x=classifying_point, color="r")

        plt.savefig(file_name)
        plt.close()

    
    def create_image_directory(self, directory_name: str, limit: int = 100, verbose: bool =True):
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
                self.save_image(ts, ts_name)
                if verbose and (i+1) % 10 == 0:
                    print("Processing fiber: {0}/{1}"
                        .format(i+1, len(data)))
            if verbose: 
                print("Finished processing {0}...\n".format(mat_file))
        

if __name__ == "__main__":
    from derivative import series_threshold
    test = Classifier(series_threshold)

    test.create_image_directory("test", verbose=True)
 
    print(test)
