#!/usr/bin/env python3 
# coding: utf8

import pandas as pd  # For data-frames
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Create Plots
import scipy.io as sio  # To load matlab files
from os import path  # For robust pathing
import os  # For general os functions
import pylab  # For saving plots


DEFAULT_THRESHOLD = 0.05


def generate_derivatives(ts: [list, np.array], verbose=False) -> pd.DataFrame:
    """
    if ts is a time series, returns a pandas dataframe with the following
    format:
        
        Index   Values  derivative1 derivative2
        ...     ...     ...         ...
        
    :param ts: list of the time series that will be modeled
    :param verbose: bool for whether to display the dataframe to stdout
    """
    index = list(range(len(ts)))
    tmp = pd.Series(ts, index)

    dxdt = np.gradient(tmp.values)
    dxdt2 = np.gradient(dxdt)

    derivative1 = pd.Series(dxdt, tmp.index, name="derivative1")
    derivative2 = pd.Series(dxdt2, tmp.index, name="derivative2")

    df = pd.concat([tmp.rename('values'), derivative1, derivative2], axis=1)

    if verbose:
        print(df)

    return df

def create_image_directory(verbose=True):
    """
    Creates an image directory structured like this:
    
    IMAGE_DIR {
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

    :param verbose: bool for whether to print the progress made in creating the images
    """
    if not path.isdir(IMAGE_DIR):
        if verbose:
            print("Creating Image Directory...")
        make_directory(IMAGE_DIR)

    # Add the mat files into the queue to be processed
    for m, mat_file in enumerate(os.listdir(DATA_DIR)):
        print("Processing file: {0}, file {1}/{2}"
                .format(mat_file, m+1, len(os.listdir(DATA_DIR))))
        mat = load_from_mat(mat_file)
        data = process_mat(mat)
        image_name = mat_file.split('.')[-2] 
        make_directory(path.join(IMAGE_DIR, image_name))
        for i, ts in enumerate(data):
            ts_name = path.join(IMAGE_DIR, 
                    path.join(image_name, "filter_{0}.png".format(i)))
            save_image(ts, ts_name)
            if verbose and (i+1) % 10 == 0:
                print("Processing fiber: {0}/{1}"
                    .format(i+1, len(data)))
        if verbose: 
            print("Finished processing {0}...\n".format(mat_file))

def series_threshold(ts, threshold: float = DEFAULT_THRESHOLD, derivative=2) -> bool:
    """
    Returns true or false based on whether the threshold was exceded by the time series
    """
    series = ts["derivative{0}".format(derivative)]
    upper_limit = threshold
    lower_limit = -threshold
    for val in series:
        if val >= upper_limit or val <= lower_limit:
            return True
    return False

if __name__ == "__main__":

    create_image_directory(verbose=True)
    # file_name = os.listdir(DATA_DIR)[0]
    # mat = load_from_mat(file_name)      
    # data = process_mat(mat)

    # classifying_point = classify_ts(data[0], series_threshold, window_size=10)
    # print("FINAL VAL:", classifying_point)
    # df = generate_derivatives(data[0], verbose=True)
    # # print(df["values"][0])
    # # df.plot()
    # # plt.axvline(x=classifying_point, color="r")
    # plt.show()
    
    # # classify_ts(df["derivative1"], None, window_size=5)
   
