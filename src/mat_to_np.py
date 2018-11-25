#!/usr/bin/env python3
# coding: utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from utilities import process_mat, load_from_mat, sliding_window
# from utilities import classify_ts
from derivative import generate_derivatives
from constants import DATA_DIR, NUMPY_DIR


def get_correct_y(ts, threshold=0.05):
    """
    TODO: Write documentation for get_correct_y
    """

    ts = generate_derivatives(ts)

    upper_limit = threshold
    lower_limit = -threshold
    for i, val in enumerate(ts["derivative2"]):
        if val >= upper_limit or val <= lower_limit:
            return i
    return -1

def save_to_np(X: np.array, y: np.array, file_name: str):
    """
    TODO: write documentation for save_to_np 
    """
    file_name = os.path.join(NUMPY_DIR, file_name)

    # Save the X input matrix
    with open(file_name + "_X.npy", "wb") as f:
        np.save(f, X)

    # Save the Y output matrix (this is assuming f: X -> Y)
    with open(file_name + "_Y.npy", "wb") as f:
        np.save(f, y)

def load_np_file(file_name: str, full_path=False) -> np.array:
    """
    Load a numpy file and return in as a numpy array 

    :param full_path: bool to specify if the file_name already has NUMPY_DIR 
                      concatinated to it
    """
    if not full_path:
        file_name = os.path.join(NUMPY_DIR, file_name)

    if not os.path.isfile(file_name):
        raise OSError("{0} does not exist!".format(file_name))
    f = open(file_name, "rb")
    arr = np.load(f)
    f.close()
    return arr
        
def convert_mat_to_np(limit=100, verbose=True):
    """
    TODO: Write documentation for save_output_matrix
    :param limit: int of 
    """

    for m, mat_file in enumerate(os.listdir(DATA_DIR)[:limit]):
        
        if verbose:
            print("Processing file: {0}, file {1}/{2}"
                    .format(mat_file, m+1, len(os.listdir(DATA_DIR)[:limit])))
        if "blue" in mat_file:
            print("Blue file not normalized, skipping\n")
            continue

        mat = load_from_mat(mat_file)
        data = process_mat(mat)
        file_name = mat_file.split('.')[-2] 

        X = []
        y = []
        for i, ts in enumerate(data):
            # Get the index for where the time series becomes classified as  
            classifying_point = get_correct_y(ts)
            for index, window in enumerate(sliding_window(ts, window_size=10)):
                X.append(window)
                if classifying_point == -1:
                    y.append(0)
                # elif index - classifying_point > 25:
                    # y.append(0)
                else:
                    y.append(1 if index >= classifying_point else 0) 
            if verbose and (i+1) % 10 == 0:
                print("Processing fiber: {0}/{1}"
                    .format(i+1, len(data)))
        if verbose: 
            print("Finished processing {0}...\n".format(mat_file))

        X = np.array(X)
        y = np.array(y)
        save_to_np(X, y, file_name)


if __name__ == "__main__":
    convert_mat_to_np()
    file_name = "10-3-18-uv_X.npy"
    arr = load_np_file(file_name)
    print(arr.shape)

     
