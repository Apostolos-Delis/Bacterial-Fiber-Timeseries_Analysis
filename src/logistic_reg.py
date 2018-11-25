#!/usr/bin/env python3
# coding: utf8

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import os
# import sys

from mat_to_np import load_np_file
from constants import NUMPY_DIR


if __name__ == "__main__":
    colors = {
            0: "b",
            1: "r",
            }
    x_file = "10-9-18-uv_X.npy"
    y_file = "10-9-18-uv_Y.npy"
    x_data = load_np_file(x_file)
    y_data = load_np_file(y_file)
    
    print(y_data)
    index = -100
    for index in range(y_data.shape[0]):
        plt.plot(x_data[index], color=colors[y_data[index]])
    plt.show()

