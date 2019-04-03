#!/usr/bin/env python3
# coding: utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_class import DataGenerator


class BatchLoader:
    """
    Class that splits data up into batches
    """

    def __init__(self, X_data, Y_data, batch_size=32):
        if X_data.shape[0] != Y_data.shape[0]:
            raise ValueError("X and Y data lists must have the same number of datapoints!")

        one_hot_encoding = {
                0: [1,0],
                1: [0,1],
        }
        self._x = X_data
        self._y = np.array(list(map(lambda x: one_hot_encoding[x], Y_data)))

        self._batch_index = 0
        self.batch_size = batch_size

    def next_batch(self):
        """ 
        returns as a tuple the next x and y batch of the dataset,
        if batches run out, returns None, None and resets the batch
        loader
        """
        if self._batch_index >= self._x.shape[0]:
            self._batch_index = 0
            return None, None

        x_batch = self._x[self._batch_index:self._batch_index+self.batch_size]
        y_batch = self._y[self._batch_index:self._batch_index+self.batch_size]
        self._batch_index += self.batch_size
        return x_batch, y_batch


if __name__ == "__main__":

    dg = DataGenerator()
    X_data, Y_data, X_test, Y_test, = dg.train_test_split()
    print("X_data:", X_data.shape)
    print("Y_data:", Y_data.shape)
    print("X_test:", X_test.shape)
    print("y_test:", Y_test.shape)
    print()
    
    bl = BatchLoader(X_data, Y_data, 64)
    for _ in range(0, 1):
        x_batch, y_batch = bl.next_batch()
        print(x_batch.shape)
        print(y_batch)

