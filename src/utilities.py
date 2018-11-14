#!/usr/bin/env python3
# coding: utf8

import pandas as pd  # For data-frames
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Create Plots
import scipy.io as sio  # To load matlab files
from os import path  # For robust pathing
import os  # For general os functions
from sys import stderr  # For Error msges

from constants import DATA_DIR, IMAGE_DIR

def error(output, *args, interupt=False, **kwargs):
    print("\033[0;49;31m{0}\033[0m".format(output), *args, file=stderr **kwargs)
    if interupt:
        exit(-1)


def make_directory(file_path: str):
    """
    Creates the directory at the path:
    :param file_path: the path of the directory that you want to create
    """
    if file_path == "\\":
        return 0
    try:
        os.makedirs(file_path, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(file_path):
            pass
        else:
            print("Error while attempting to create a directory.")
            exit(3)

def load_from_mat(file_name: str) -> dict:
    """
    Loads the data from a matlab .mat file into python

    :param file_name: str of the relative file name
                      DO NOT INCLUDE THE PATH, 
                      the function will auto-append the
                      data directory path to the file
    """
    file_path = path.join(DATA_DIR, file_name)
    if path.isfile(file_path):
        return sio.loadmat(file_path)
    else:
        raise OSError("{0} is not a file!".format(file_path))

def process_mat(mat: dict):
    """
    Process the .mat file as its loaded in from a dictionary

    :param data: dict of all the values from the .mat file.
                 depending on the file opened (uv or blue) 
                 the data will be contained in the key "B"
                 or in "B3"
    """
    
    # First average the 13 bacterial fibers
    try:
        if "B3" in mat.keys():
            data = mat["B3"]
        elif "B" in mat.keys():
            data = mat["B"]
        else:
            error(mat.keys(), interrupt=True) 
    except KeyError as err:
        error(err, interupt=True)

    NUM_FILTERS = len(data[0][0])

    averaged_list = []
    
    # Average over the bacterial fiber
    # The current Matrix is 93 x 13 x 40, want it 40 x 93
    for filter_index in range(NUM_FILTERS):
        filter_time_series = []
        for time_index, time_stamp in enumerate(data):
            l = []
            for fiber in time_stamp:
                l.append(fiber[filter_index])
            filter_time_series.append(np.average(l))
        averaged_list.append(filter_time_series)

    return averaged_list


def sliding_window(sequence,window_size=10,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
 
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(window_size) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(window_size) and type(step) must be int.")
    if step > window_size:
        raise Exception("**ERROR** step must not be larger than window_size.")
    if window_size > len(sequence):
        raise Exception("**ERROR** window_size must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    num_chunks = int(((len(sequence)-window_size)/step)+1)
 
    # Do the work
    for i in range(0,num_chunks*step,step):
        yield sequence[i:i+window_size]

def classify_ts(ts: [list, np.array, pd.Series], classifier,
                window_size: int = 10, step: int = 1, **kwargs):
    """
    Attempt to classify the time series as malignant or not. 
    
    :param ts: list or np.array of the time series
    :param classifier: the function that will classify the function
    :param window_size: how large you want the sliding window panel to be
    :param step: how many indexes to jump with each iteration of sliding window
    :param kwargs: any additional arguements for the classifier function
    :return 1 if the time series identifies as malignant, 0 otherwise
    """
    # if isinstance(ts, list):
        # index = list(range(len(ts)))
        # ts = pd.Series(ts, index)

    for index, window in enumerate(sliding_window(ts, window_size, step)):
        if classifier(window, **kwargs):
            return index + window_size - 1

    return -1


if __name__ == "__main__":
    pass

