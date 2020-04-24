#!/usr/bin/env python3
# Theodor Marcu
# tmarcu@princeton.edu
# Created December 2019
# Computer Science Senior Thesis

import os
import time
import random
import numpy as np
import math
import glob
import scipy
import tensorflow.keras as keras
import json
from brain2brain import models
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

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

def get_file_shape(file_path: str, print_flag: bool = True):
    '''
    Prints the shape of each file in file_paths.

    Args:
        file_paths (list): A list of file paths.
    '''
    # Open the file to only read the header
    data = np.load(file_path, mmap_mode='r')
    shape = data.shape
    if print_flag:
        print(shape)
    return shape

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
    total_file_count = len(file_paths)
    for ix, old_path in enumerate(file_paths):
        # Open the file, normalize the data, and save it to a new path.
        start_time = time.time()
        data = np.load(old_path)
        print(f"Original Shape: {data.shape}\n")
        electrode_means = np.mean(data, axis=0)
        _, broadcast_electrode_means = np.broadcast_arrays(
            data, electrode_means)
        new_data = data - broadcast_electrode_means
        electrode_stds = np.std(data, axis=0)
        _, broadcast_electrode_stds = np.broadcast_arrays(data, electrode_stds)
        new_data /= broadcast_electrode_stds
        print(f"New Shape: {new_data.shape}\n")
        # Save to new path.
        # Ensure that the directory exists.
        # Respect old directory structure. Add prefix.
        old_file_name = old_path.split('/')[6]
        new_path = os.path.join(output_directory + file_prefix + old_file_name)
        Path(new_path).mkdir(parents=True, exist_ok=True)
        new_path = os.path.join(new_path, file_prefix + old_file_name)
        np.save(new_path, new_data)
        print(f"Normalized file and saved at {new_path}.npy\n"
              f"Elapsed time: {time.time() - start_time:.2f}s {ix + 1}/{total_file_count} done")

def generate_binned_data(file_paths: list,
                         avg_timestep_count: int=25,
                         output_directory: str="/tmp/tmarcu/binned_conversation_data/",
                         file_prefix: str="binned_",
                         normalize: bool=False,
                         order: str="bin_norm"):
    '''
    Performs data binning through averaging over a list of files, i.e. averages 
    every 25 timesteps to create a new timesteps.

    This downsamples the data.

    The default is 25 timesteps, which is roughly equivalent to 50ms for 
    data at 512Hz.

    Args:
        files_paths (list): List of one or more file paths.
        avg_timestep_count (int): The number of timesteps to average to create one data point. Default: 25.
        output_directory (str): Path to the output directory. (Default = "/tmp/tmarcu/normalized-conversations/")
        file_prefix (str): Prefix to add to directories and files. (Default = "binned_")
        normalize (bool): Default False. Normalizes the data as well (after binning).
        order (str): Default "bin_norm", which means bin and normalize. Other settings are: "norm_bin".
    '''
    total_file_count = len(file_paths)
    for ix, old_path in enumerate(file_paths):
        start_time = time.time()
        # Open the file, average the data, and save it to a new path.
        data = np.load(old_path)
        print(f"Original Shape: {data.shape}")
        if normalize:
            if order == "bin_norm":
                new_data = bin_data(data, avg_timestep_count)
                print(f"New Shape: {new_data.shape}")
                new_data = normalize_file(new_data)
                print(f"New Shape: {new_data.shape}")
            elif order == "norm_bin":
                new_data = normalize_file(data)
                print(f"New Shape: {new_data.shape}")
                new_data = bin_data(new_data)
                print(f"New Shape: {new_data.shape}")
            else:
                raise Exception("Order not well specified. Aborting...")
        else:
            # Only bin the data.
            new_data = bin_data(data)
            print(f"New Shape: {new_data.shape}")
        # Save to new path.
        # Ensure that the directory exists.
        # Respect old directory structure. Add prefix.
        old_file_name = old_path.split('/')[6]
        new_path = os.path.join(output_directory + file_prefix + old_file_name)
        Path(new_path).mkdir(parents=True, exist_ok=True)
        new_path = os.path.join(new_path, file_prefix + old_file_name)
        np.save(new_path, new_data)
        print(f"Binned file and saved at {new_path}.npy\n"
              f"Elapsed time: {time.time() - start_time:.2f}s {ix + 1}/{total_file_count} done")

def normalize_file(data: np.array):
    '''
    Normalize data and return a normalized numpy array.

    Args:
        data (np.array): Numpy array that contains data.
    Returns:
        (np.array) Normalized data.
    '''
    electrode_means = np.mean(data, axis=0)
    _, broadcast_electrode_means = np.broadcast_arrays(data, electrode_means)
    new_data_normalized = data - broadcast_electrode_means
    electrode_stds = np.std(data, axis=0)
    _, broadcast_electrode_stds = np.broadcast_arrays(data, electrode_stds)
    new_data_normalized /= broadcast_electrode_stds
    return new_data_normalized

def normalize_data_arr(data: np.array, mean: np.array, std: np.array):
    '''
    Normalize data and return a normalized numpy array.

    Args:
        data (np.array): Numpy array that contains data.
        mean (double): Double arr for each electrode to normalize data with. (Must be computed using get_mean_std())
        std (double): Double arr for each electrode to normalize data with. (Must be computed using get_mean_std())
    Returns:
        (np.array) Normalized data.
    '''
    new_data_normalized = np.subtract(data, mean)
    new_data_normalized = np.divide(new_data_normalized, std)
    return new_data_normalized

def bin_data(data: np.array, avg_timestep_count: int=25):
    '''
    Bin data in a np.array and return a new array
    containing the resulting data.

    Args:
        data (np.array): Numpy array that contains data.
        avg_timestep_count (int): The number of timesteps to average
                                  to create one data point. Default: 25.
    Returns:
        (np.array) Normalized data.
    '''
    new_data = data[:(data.shape[0] // avg_timestep_count) * avg_timestep_count]
    new_data = new_data.reshape(-1, avg_timestep_count, data.shape[1]).mean(axis=1)
    return new_data

def get_mean(file_paths: list):
    """
    Return the mean of a list of file paths. Useful for normalizing datasets.
    """
    electrode_count = np.load(file_paths[0]).shape[1]
    total_row_count = 0
    current_sum = np.zeros(shape=(electrode_count))
    total_file_count = len(file_paths)
    for ix, path in enumerate(file_paths, start=1   ):
        print(f"File {ix}/{total_file_count}", end="\r", flush=True)
        data = np.load(path)
        sum = np.sum(data, axis=0)
        current_sum += np.sum(data, axis=0)
        total_row_count += data.shape[0]
    mean = current_sum / total_row_count
    return mean

def get_mean_std(file_paths: list):
    """
    Return the mean and std of a list of file paths. Useful for normalizing datasets.
    """
    electrode_count = np.load(file_paths[0]).shape[1]
    total_row_count = 0
    current_sum = np.zeros(shape=(electrode_count),  dtype=np.double)
    total_file_count = len(file_paths)
    print("Calculating mean per electrode")
    for ix, path in enumerate(file_paths, start=1):
        print(f"File {ix}/{total_file_count}", end="\r", flush=True)
        data = np.load(path)
        current_sum += np.sum(data, axis=0, dtype=np.double)
        total_row_count += data.shape[0]
    mean = current_sum / total_row_count
    print("Calculating std per electrode")
    running_sum = np.zeros(shape=(electrode_count), dtype=np.double)
    for ix, path in enumerate(file_paths, start=1):
        print(f"File {ix}/{total_file_count}", end="\r", flush=True)
        data = np.load(path)
        a = np.square(abs(data - mean), dtype=np.double)
        sum = np.sum(a, axis=0, dtype=np.double)
        running_sum += sum
    std = np.sqrt(running_sum/total_row_count)
    return mean, std

def normalize_dataset_partition(file_paths: list,
                                output_directory: str="/tmp/tmarcu/binned_conversation_data/",
                                file_prefix: str="binned_",
                                binned: bool=False,
                                avg_timestep_count: int=25):
    '''
    Normalizes a list of files.
    '''
    print("Normalizing dataset partition")
    mean, std = get_mean_std(file_paths)
    total_file_count = len(file_paths)
    file_path_list = list()
    for ix, path in enumerate(file_paths, start=1):
        print(f"File {ix}/{total_file_count}", end="\r", flush=True)
        start_time = time.time()
        data = np.load(path)
        if binned:
            print("Binning dataset partition.")
            data = bin_data(data, avg_timestep_count)
        data = normalize_data_arr(data, mean, std)
        # Save to new path.
        # Ensure that the directory exists.
        # Respect old directory structure. Add prefix.
        old_file_name = path.split('/')[6]
        new_path = os.path.join(output_directory + file_prefix + old_file_name)
        Path(new_path).mkdir(parents=True, exist_ok=True)
        new_path = os.path.join(new_path, file_prefix + old_file_name + ".npy")
        np.save(new_path, data)
        print(f"Normalized file and saved at {new_path}\n"
              f"Elapsed time: {time.time() - start_time:.2f}s {ix}/{total_file_count} done")
        file_path_list.append(new_path)
    return file_path_list

def normalize_dataset(file_paths: list,
                      path_list_out: str="bin_norm_dataset",
                      split_data: bool=False,
                      split_ratio: float=0.8,
                      output_directory: str = "/tmp/tmarcu/normalized-conversations/",
                      file_prefix: str = "norm_",
                      binned: bool=False,
                      avg_timestep_count: int=25):
    '''
    Normalizes entire data set and saves them into the output directory.
    Generates txt files that contain the paths to the train and validation dataset if necessary.

    Args:
        file_paths (list): List of one or more file paths.
        output_directory (str): Path to the output directory. (Default = "/tmp/tmarcu/normalized-conversations/")
        file_prefix (str): Prefix to add to directories and files. (Default = "norm_")
        split_data (bool): Whether to split data.
        split_ratio (str): Split ratio for train/val. Default=0.8.
        binned (bool): Whether to bin data before normalization.
        avg_timestep_count (int): Default=25. How many timesteps to average at once.
    '''
    # Split files into train/val/test.
    # file_paths = file_paths[:5]
    if split_data:
        print("Splitting data in train, val, test.")
        train, test = split_file_paths(file_paths, split_ratio=split_ratio)
        train, val = split_file_paths(train, split_ratio=split_ratio)
        print("Training set normalization.")
        train_file_path_list = normalize_dataset_partition(file_paths=train,
                                                           output_directory=output_directory,
                                                           file_prefix=file_prefix,
                                                           binned=binned,
                                                           avg_timestep_count=avg_timestep_count)
        path_list_out_train = "train_" + path_list_out + ".txt"
        save_file_paths(train_file_path_list, path_list_out_train)
        print("Validation set normalization.")
        val_file_path_list = normalize_dataset_partition(file_paths=val,
                                                         output_directory=output_directory,
                                                         file_prefix=file_prefix,
                                                         binned=binned,
                                                         avg_timestep_count=avg_timestep_count)
        path_list_out_val = "val_" + path_list_out + ".txt"
        save_file_paths(val_file_path_list, path_list_out_val)
        print("Test set normalization.")
        test_file_path_list = normalize_dataset_partition(file_paths=test,
                                                          output_directory=output_directory,
                                                          file_prefix=file_prefix,
                                                          binned=binned,
                                                          avg_timestep_count=avg_timestep_count)
        path_list_out_test = "test_" + path_list_out + ".txt"
        save_file_paths(test_file_path_list, path_list_out_test)
    else:
        print("All file set normalization.")
        all_file_path_list = normalize_dataset_partition(file_paths=file_paths,
                                                         output_directory=output_directory,
                                                         file_prefix=file_prefix,
                                                         binned=binned,
                                                         avg_timestep_count=avg_timestep_count)
        path_list_out_all = "all_" + path_list_out + ".txt"
        save_file_paths(all_file_path_list, path_list_out_all)

def save_json_file(data, file_path):
    '''
    Saves data in file_path.

    Args:
        data (json-able object): Object to save.
        file_path (str): Path where file should be saved.
    '''
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

def save_file_paths(file_paths: list,
                    target_file: str):
    '''
    Save files in file_paths in a file.
    '''
    # Save the files
    with open(target_file, 'w') as filehandle:
        for path in file_paths:
<<<<<<< HEAD
            filehandle.write('%s\n' % path)
=======
            filehandle.write('%s\n' % path)

def predict_and_plot(inference_encoder_model,
                     inference_decoder_model,
                     encoder_input_data,
                     decoder_target_data,
                     sample_ix,
                     pred_steps: int,
                     enc_tail_len=50):
    encode_series = encoder_input_data[0][sample_ix:sample_ix+1,:,:] 
    pred_series = models.decode_sequence_o2o(inference_encoder_model,
                                             inference_decoder_model, 
                                             input_seq=encode_series,
                                             pred_steps=pred_steps)
    encode_series = encode_series.reshape(-1,1)
    pred_series = pred_series.reshape(-1,1)   
    target_series = decoder_target_data[sample_ix,:,:1].reshape(-1,1)
    encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])
    x_encode = encode_series_tail.shape[0]
    r, p = pearsonr(pred_series.reshape(-1), target_series.reshape(-1))
    print(f"Correlation: {r}. P: {p}")
    plt.figure(figsize=(10,6))   
    
    plt.plot(range(1,x_encode+1),encode_series_tail)
    plt.plot(range(x_encode,x_encode+pred_steps),target_series,color='orange')
    plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='teal',linestyle='--')
    
    plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
    plt.legend(['Encoding Series','Target Series','Predictions'])

def get_corr_mae(inference_encoder_model,
                 inference_decoder_model,
                 encoder_input_data,
                 decoder_target_data,
                 sample_ix,
                 pred_steps: int,
                 enc_tail_len=50, verbose=True):
    encode_series = encoder_input_data[0][sample_ix:sample_ix+1,:,:] 
    pred_series = models.decode_sequence_o2o(inference_encoder_model,
                                             inference_decoder_model, 
                                             input_seq=encode_series,
                                             pred_steps=pred_steps)
    encode_series = encode_series.reshape(-1,1)
    pred_series = pred_series.reshape(-1,1)   
    target_series = decoder_target_data[sample_ix,:,:1].reshape(-1,1)
    encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])
    x_encode = encode_series_tail.shape[0]
    r, p = pearsonr(pred_series.reshape(-1), target_series.reshape(-1))
    mae = np.mean(np.abs(pred_series.reshape(-1) - target_series.reshape(-1)))
    if verbose:
        print(f"Correlation: {r}. P: {p}. MAE: {mae}")
    return r, mae
>>>>>>> new_test
