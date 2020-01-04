'''
This module contains utilities for the brain2brain project.
'''
import os
import numpy as np
import glob


class Utils:
    '''
    This class contains functions and utilities for the brain2brain
    project.
    '''
    @staticmethod
    def generator(data: np.ndarray, lookback: int, delay: int,
                  min_index: int, max_index: int, batch_size: int, step: int, shuffle: bool = False):
        '''
        
        A generator function that parses multiple brain signals from the
        NYU dataset. Code from Deep Learning with Python, pag. 211 (Author: François Chollet).

        :param data: The original array of floating-point data, which is normalized.
        :type data: class: `numpy.ndarray`
        :param lookback: The number of timesteps the input data should go back.
        :type lookback: int
        :param delay: How many timesteps in the future the target should be.
        :type delay: int
        :param min_index: Index in the data array that delimite which time steps to draw from. This is useful for keeping a segment of the data for validation and another for testing.
        :type min_index: int
        :param max_index: Index in the data array that delimite which time steps to draw from. This is useful for keeping a segment of the data for validation and another for testing.
        :type max_index: int
        :param shuffle: Whether to shuffle the samples or draw them  in chronological order, defaults to False.
        :type shuffle: bool
        :param batch_size: The number of samples per batch.
        :type batch_size: int
        :param step: The period, in timesteps, at which you sample data. 
        :type step: int

        :return: A generator object.
        :rtype: generator

        '''
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while True:
            if shuffle:
                # Select batch_size random ints between min_index+lookback
                # and max_index in order to select rows randomly. 
                rows = np.random.randint(min_index + lookback, max_index,
                                         size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                # Return evenly spaced values within a given interval 
                # for the rows.
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)
            
            samples = np.zeros((len(rows),
                               lookback // step,
                               data.shape[-1]))
            
            targets = np.zeros((len(rows), ))
            for j, _ in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]
            yield samples, targets

    @staticmethod
    def get_data(patient_number: int):
        '''
        TODO: Add description.
        '''
        # This is where the conversations are stored.
        top_of_path = "/projects/HASSON/247/data/"
        conversations_path = top_of_path + patient_number + "-conversations/"
        # Getting all the numpy arrays .npy files based on matching pattern (*.npy)
        file_paths = glob.glob(os.path.join(conversations_path, '*.npy'))
        print(file_paths)
