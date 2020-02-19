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
