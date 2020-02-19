'''
This module contains utilities for the brain2brain project.

Created by Theodor Marcu 2019-2020
tmarcu@princeton.edu

'''
import os
import random
import numpy as np
import glob
import tensorflow.keras as keras

def generator(data: np.ndarray, lookback: int, delay: int,
              min_index: int, max_index: int, batch_size: int, step: int, shuffle: bool = False):
    '''
    DEPRECATED
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
            # Get these timesteps from the data.
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            # Multiple electrodes? Only one here. [1]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

def get_file_paths(patient_number: int, shuffle: bool = True):
    '''
    Get filepaths for a specific patient number and shuffle them.
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
    # Shuffle the filepaths.
    if shuffle:
        random.shuffle(file_paths)
    return file_paths

def split_file_paths(file_paths: list, split_ratio: float = 0.8):
    '''
    Split the list of file paths into a training and test list based on the split_ratio.

    Note: The data is sampled at 512Hz. 512 timesteps are equal to one second.

    Args:
        file_paths (list): A list of file paths (shuffled or not).
        split_ratio (float): The split for the training set. (Default=0.8)
    
    Returns:
        list, list: Two lists that represent the training and testing datasets, respectively.
    '''
    training_file_paths = list()
    testing_file_paths = list()

    total_timestep_count = get_total_timestep_count(file_paths)

    current_timestep_count = 0
    for path in file_paths:
        # Open the file to only read the header
        data = np.load(path, mmap_mode='r')
        # Get the number of rows (i.e. timesteps)
        file_timestep_count = data.shape[0]
        current_timestep_count += file_timestep_count
        if current_timestep_count >= (split_ratio * total_timestep_count):
            testing_file_paths.append(path)
        else:
            training_file_paths.append(path)

    return training_file_paths, testing_file_paths

def get_total_timestep_count(file_paths: list):
    '''
    Returns total sample count for the given file paths.

    Args:
        file_paths (list): A list of file paths (shuffled or not).
    Returns:
        int: Total sample count.
    '''
    total_timestep_count = 0
    for path in file_paths:
        # Open the file to only read the header
        data = np.load(path, mmap_mode='r')
        # Get the number of rows (i.e. timesteps)
        file_timestep_count = data.shape[0]
        total_timestep_count += file_timestep_count
    return total_timestep_count

def get_total_sample_count(file_paths: list, lookback: int, delay: int, length: int):
    '''
    Args:
        file_paths (list): A list of file paths (shuffled or not).
        lookback (int): The number of timesteps to predict on.
        delay (int): The number of timesteps between sample and prediction.
        length (int): The length of the sequence to predict (in timesteps).
    Returns:
        int: Total sample count.

    '''
    total_sample_count = 0
    for path in file_paths:
        # Open the file to only read the header
        data = np.load(path, mmap_mode='r')
        # Get the number of rows (i.e. timesteps)
        file_timestep_count = data.shape[0]
        # Calculate the number of total samples in this file.
        file_sample_count = file_timestep_count // (lookback + delay + length)
        print(file_sample_count)
        total_sample_count += file_sample_count
    return total_sample_count

def print_file_shape(file_paths: list):
    '''
    Prints the shape of each file in file_paths.

    Args:
        file_paths (list): A list of file paths.
    '''
    for path in file_paths:
        # Open the file to only read the header
        data = np.load(path, mmap_mode='r')
        shape = data.shape
        print(shape)

def create_ecog_array(file_paths: list, verbose: bool = True):
    '''
    Create an array from .npy files. Does not work if the dataset
    cannot fit into memory. Refer to the Generator function if that is the case.
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
