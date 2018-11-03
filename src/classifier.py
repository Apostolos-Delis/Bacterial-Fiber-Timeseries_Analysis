#!/usr/bin/env python3
# coding: utf8

import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Create Plots
import pandas as pd  # For data-frames

DATA_DIR = "../data/extracted_data"
IMAGE_DIR = "../data/images/"

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
        assert(isinstance(df, pd.DataFrame) or isinstance(df, pd.Series))
        df.plot()
        if classifier is not None:
            classifying_point = classify_ts(time_series, series_threshold, window_size=10)

            if classifying_point != -1:
                plt.axvline(x=classifying_point, color="r")

        plt.savefig(file_name)
        plt.close()

    

if __name__ == "__main__":
    
    test = Classifier(f)
 
    print(test)
