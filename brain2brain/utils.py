'''
This module contains utilities for the brain2brain project.
'''
import os
import numpy as np
import glob


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

def get_file_paths(patient_number: int, shuffle: bool = True):
    '''
    TODO: Add description.
    '''
    # This is where the conversations are stored.
    top_of_path = "/projects/HASSON/247/data/"
    conversations_path = os.path.join(top_of_path,
                                      str(patient_number) + "-conversations/**/",
                                       "*.npy")
    # Getting all the numpy arrays .npy files based on matching pattern (*.npy)
    file_paths = glob.glob(conversations_path, recursive=True)
    # Sort the file paths.
    file_paths.sort()
    return file_paths

def create_ecog_array(file_paths: list, verbose: bool = True):
    '''
    Create
    '''
    # Read each numpy array and append it to the ecogs list.
    ecogs = []
    for path in file_paths:
        vprint(f"Reading {path}", verbose)
        try:
            with open(path, 'r') as file:
                data = np.load(path)
                vprint(f"Finished reading {path}. Loaded into numpy array of shape {data.shape}", verbose)
        except IOError:
            print(f"IOError: {path} cannot be opened. Reading will continue")
            continueß
        # Append to list.
        ecogs.append(data)
        vprint(f"Added rows to ecogs. Ecogs size: {len(ecogs)}", verbose)
    np_ecogs = np.asarray(ecogs)
    vprint(f"Reading DONE! Final shape: {np_ecogs.shape}")
    return np_ecogs

def vprint(istring, verbose: bool = True):
    if verbose:
        print(istring)