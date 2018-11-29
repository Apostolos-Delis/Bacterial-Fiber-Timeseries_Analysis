#!/usr/bin/env python3
# coding: utf8

import numpy as np
import os

from mat_to_np import load_np_file
from constants import NUMPY_DIR

class DataGenerator:
    """
    DataGenerator is a class that is usefull for dealing with the numpy data
    for the bacterial fibers.

    To use: 
    
        o Specify how many data points you will load in the constructor, with a default
        of -1 for infinite values

        o Load the data with the load_data function
        
        o If you want to access the data you can with get_data()

        o train_test_split() will let you split the data for machine learning purposes

    """

    def __init__(self, size=-1):
        """
        :param size: how many data points you want in the generator object, d
                     the default of -1 is infinite
        """
        self._size = size
        self._x_data = None
        self._y_data = None


    def __str__(self):
        return "DataGenerator object at <{0}>".format(hex(id(self)))


    def get_data(self):
        if self._x_data is None:
            self.load_data()
        return (self._x_data, self._y_data)


    def train_test_split(self, train_percentage=0.8):
        """
        Splits the data into X_train, Y_train, X_test, Y_test
        
        :param train_percentage: what percentage do you want for training
        """

        if self._x_data is None:
            self.load_data()

        num_data_points = self._x_data.shape[0]
        num_training = int(num_data_points * train_percentage)

        X_train = self._x_data[:num_training]
        Y_train = self._y_data[:num_training]

        X_test = self._x_data[num_training:]
        Y_test = self._y_data[num_training:]

        return (X_train, Y_train, X_test, Y_test)


    def load_data(self) -> tuple:
        """
        Load all the data from the NUMPY files
        """
        
        x_data = []
        y_data = []

        for numpy_file in os.listdir(NUMPY_DIR):
            if 'Y' in numpy_file:
                continue
            
            x_data.append(load_np_file(numpy_file, full_path=False))
            y_file_name = numpy_file.replace('X', 'Y')
            y_data.append(load_np_file(y_file_name, full_path=False))

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        x_data = x_data.reshape((x_data.shape[0] * x_data.shape[1], x_data.shape[2]))
        y_data = y_data.reshape((y_data.shape[0] * y_data.shape[1],))

        if self._size >= -1 and x_data.shape[0] > self._size:
            x_data = x_data[:self._size]
            y_data = y_data[:self._size]

        self._x_data = x_data
        self._y_data = y_data

        return (x_data, y_data)



if __name__ == "__main__":
    dg = DataGenerator()
    X_data, Y_data, _, _, = dg.train_test_split()
    print(X_data.shape)
    print(Y_data.shape)
    print(Y_data[600:700])


