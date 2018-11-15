#!/usr/bin/env python3
# coding: utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("..") # Adds higher directory to python modules path.
from utilities import process_mat, load_from_mat, sliding_window
from derivative import generate_derivatives, series_threshold

NUMPY_DIR = "../../data/numpy"
DATA_DIR = "../../data/extracted_data"

def save_input_matrix():
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
    pass

def save_output_matrix():
    f = open("test.npy", 'ab+')
    arr = np.arange(100000)
    np.save(f, arr)
    f.close()
    pass

if __name__ == "__main__":
    # save_output_matrix()
    with open("test.npy", "rb") as f:
        arr = np.load(f)
        print(arr)
    pass

     
