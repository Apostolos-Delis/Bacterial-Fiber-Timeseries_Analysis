#!/usr/bin/env python3 
# coding: utf8

import pandas as pd  # For data-frames
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Create Plots
import scipy.io as sio  # To load matlab files
from os import path  # For robust pathing
import os  # For general os functions
from sys import stderr  # For Error msges
import pylab  # For saving plots


DATA_DIR = "extracted_data"
IMAGE_DIR = "images"
DEFAULT_THRESHOLD = 0.05

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

def save_image(time_series, file_name: str):
    """
    Generates the first and second derivative, generates the graphs,
    and then saves the graph to IMAGE_DIR/filename
    Also adds a line where the current classifier 

    :param time_series: either a list or a np.array that will be
    """
    df = generate_derivatives(time_series, verbose=False)
    classifying_point = classify_ts(time_series, series_threshold, window_size=10)

    df.plot()
    if classifying_point != -1:
        plt.axvline(x=classifying_point, color="r")
    plt.savefig(file_name)
    plt.close()

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
    if not isinstance(ts, pd.Series):
        ts = generate_derivatives(ts)

    for index, window in enumerate(sliding_window(ts, window_size, step)):
        if classifier(window, **kwargs):
            return index + window_size - 1

    return -1

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
   
