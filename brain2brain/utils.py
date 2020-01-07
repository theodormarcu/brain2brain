'''
This module contains utilities for the brain2brain project.
'''
import os
import numpy as np
import glob
import keras

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
            # Get these timesteps from the data.
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            # Multiple electrodes? Only one here. [1]
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

def vprint(istring, verbose: bool = True):
    if verbose:
        print(istring)

class Generator(keras.utils.Sequence):
    '''
    Generator class that creates batches so Keras can load them into memory.
    Inspiration: Eric Ham, 
    https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
    '''

    def __init__(self, file_paths: list, lookback: int, delay: int,
                 min_index: int, max_index: int, batch_size: int,
                 step: int, num_electrodes: int, shuffle: bool = False):
        ''' Initialization function for the object. Call when Generator() is called.

        :param file_paths: List of file paths created by `get_file_paths`.
        :type data: list
        :param lookback: The number of timesteps the input data should go back.
        :type lookback: int
        :param delay: How many timesteps in the future the target should be.
        :type delay: int
        :param shuffle: Whether to shuffle the samples or draw them  in chronological order,
        defaults to False.
        :type shuffle: bool
        :param batch_size: The number of samples per batch.
        :type batch_size: int
        :param step: The period, in timesteps, at which you sample data. 
        :type step: int

        :return: A generator object.
        :rtype: generator
        '''
        self.file_paths = file_paths
        self.lookback = lookback
        self.delay = delay
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.step = step
        self.num_electrodes = num_electrodes
        # The on_epoch_end method gets triggered at the very
        # beginning and the very end of each epoch.
        self.on_epoch_end()

    def __len__(self):
        ''' Called by len(a = Generator(...)).

        This function computes the number of batches that this
        generator is supposed to produce. So, we divide the 
        number of total samples by the batch_size and return that value. 
        '''
        # Load data using mmap so that we only load the file header to
        # get array shapes and datatype. We can use this to calculate 
        # the number of timesteps per 
        batch_count = 0
        for path in self.file_paths:
            try:
                with open(path, 'r') as npy_file:
                    data = np.load(npy_file, mmap_mode='r')
                    # Get the number of rows (i.e. timesteps)
                    timestep_count = data.shape[0]
                    # Divide it by the step size and then batch_size
                    file_batch_count = np.ceil(timestep_count // self.step // self.batch_size)
                    assert(type(file_batch_count) == int)
                    batch_count += file_batch_count
            except IOError:
                print(f"IOError: {path} cannot be opened. Reading will STOP")

        return batch_count

    def on_epoch_end():
        ''' Update indices after each epoch.
        '''
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle = True
            np.random.shuffle(self.indices)

    def __data_generation(self, file_paths_tmp: list):
        '''
        Generates data containing batch_size samples.
        ''' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        samples = np.empty(self.batch_size,
                           self.lookback // self.step,
                           self.num_electrodes))
        targets = np.empty((self.batch_size), dtype=int)

        # Generate data from each file. 
        # Each file is not equivalent to one datum, instead
        # it is equivalent to multiple steps of data,
        # where a step is the period, in timesteps,
        # at which the data is sampled.
        for i, path in enumerate(file_paths_tmp):
            # Open the file
            try:
                with open(path, 'r') as npy_file:
                    # Load data into memory.
                    data = np.load(npy_file)
            except IOError:
                print(f"IOError: {path} cannot be opened. Reading will STOP")
            
            

            # Store sample
            samples[i,] = np.load(path)

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getitem__(self, idx: int):
        ''' Given batch number idx, create a list of data.
        
        :param idx: Batch number.
        :type idx: int
        '''

        file_batch = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
