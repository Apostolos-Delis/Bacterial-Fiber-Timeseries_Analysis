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


def series_threshold(ts, threshold: float = DEFAULT_THRESHOLD, derivative=2) -> bool:
    """
    Returns true or false based on whether the threshold was exceded by the time series
    """
    try:
        series = ts["derivative{0}".format(derivative)]
    except KeyError:
        ts = generate_derivatives(ts)
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
   
