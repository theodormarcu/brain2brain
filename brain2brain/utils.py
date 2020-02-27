# Theodor Marcu
# tmarcu@princeton.edu
# Created December 2019
# Computer Science Senior Thesis

import os
import random
import numpy as np
import glob
import tensorflow.keras as keras
import time
from pathlib import Path


def get_file_paths(filename: str):
    '''
    Read file paths from a file.

    Args:
        filename (str): Path to file that contains file paths.

    Returns:
        (list): List of file paths.

    The format of the file is:

    path1
    path2
    path3

    (One path per line.)
    '''
    with open(filename) as f:
        content = f.readlines()
    # Remove whitespace characters (like '\n') at the end of the lines.
    content = [x.strip() for x in content]
    return content


def get_file_paths_from_root(patient_number: int, sort: bool = False, shuffle: bool = False):
    '''
    Get filepaths for a specific patient number and shuffle them.

    Args:
        patient_number (int): The number of the patient.
        sort (bool): Sort the files. (Default = False)
        shuffle(bool): Shuffle the files. (Default = False)

    Returns:
        (list): List of file paths.
    '''
    # This is where the conversations are stored.
    top_of_path = "/projects/HASSON/247/data/"
    conversations_path = os.path.join(top_of_path,
                                      str(patient_number) +
                                      "-conversations/**/",
                                      "*.npy")
    # Getting all the numpy arrays .npy files based on matching pattern (*.npy)
    file_paths = glob.glob(conversations_path, recursive=True)
    # Sort the file paths.
    if sort:
        file_paths.sort()
    if shuffle:
        random.shuffle(file_paths)
    return file_paths


def get_file_paths_from_dir(dir: str, sort: bool = False, shuffle: bool = False):
    '''
    Get filepaths for a specific patient number and shuffle them.

    Args:
        dir (str): Where the file paths are stored.
        sort (bool): Sort the files. (Default = False)
        shuffle(bool): Shuffle the files. (Default = False)

    Returns:
        (list): List of file paths.
    '''
    # This is where the conversations are stored.
    top_of_path = dir
    conversations_path = os.path.join(top_of_path, "*/**/*.npy")
    # Getting all the numpy arrays .npy files based on matching pattern (*.npy)
    file_paths = glob.glob(conversations_path, recursive=True)
    # Sort the file paths.
    if sort:
        file_paths.sort()
    # Shuffle the file paths.
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
        # print(file_sample_count)
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

    Args:
        file_paths (list): A list of file paths.
        verbose (bool): Verbose.

    Returns:
        (numpy array): Array of ecogs.
    '''
    # Read each numpy array and append it to the ecogs list.
    ecogs = []
    for path in file_paths:
        vprint(f"Reading {path}", verbose)
        try:
            with open(path, 'r') as file:
                data = np.load(path)
                vprint(
                    f"Finished reading {path}. Loaded into numpy array of shape {data.shape}", verbose)
        except IOError:
            print(f"IOError: {path} cannot be opened. Reading will continue")
            continueß
        # Append to list.
        ecogs.append(data)
        vprint(f"Added rows to ecogs. Ecogs size: {len(ecogs)}", verbose)
    np_ecogs = np.asarray(ecogs)
    vprint(f"Reading DONE! Final shape: {np_ecogs.shape}")
    return np_ecogs


def normalize_files(file_paths: list, output_directory: str = "/tmp/tmarcu/normalized-conversations/",
                    file_prefix: str = "norm_"):
    '''
    Normalizes npy files and saves them in the specified directory.

    Args:
        files_paths (list): List of one or more file paths.
        output_directory (str): Path to the output directory. (Default = "/tmp/tmarcu/normalized-conversations/")
        file_prefix (str): Prefix to add to directories and files. (Default = "norm_")

    '''
    n_files = len(file_paths)
    for ix, old_path in enumerate(file_paths):
        # Open the file, normalize the data, and save it to a new path.
        start_time = time.time()
        data = np.load(old_path)
        electrode_means = np.mean(data, axis=0)
        _, broadcast_electrode_means = np.broadcast_arrays(
            data, electrode_means)
        new_data = data - broadcast_electrode_means
        electrode_stds = np.std(data, axis=0)
        _, broadcast_electrode_stds = np.broadcast_arrays(data, electrode_stds)
        new_data /= broadcast_electrode_stds
        # Save to new path.
        # Ensure that the directory exists.
        # Respect old directory structure. Add prefix.
        old_file_name = old_path.split('/')[6]
        new_path = os.path.join(output_directory + file_prefix + old_file_name)
        Path(new_path).mkdir(parents=True, exist_ok=True)
        new_path = os.path.join(new_path, file_prefix + old_file_name)
        np.save(new_path, new_data)
        print(f"Normalized file and saved at {new_path}.npy\n"
              f"Elapsed time: {time.time() - start_time:.2f}s {ix + 1}/{n_files} done")
