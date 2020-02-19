'''
Created by Theodor Marcu 2019-2020
tmarcu@princeton.edu
'''

import os
import random
import numpy as np
import glob
import tensorflow.keras as keras

class Generator(keras.utils.Sequence):
    '''
    Generator class that creates batches so Keras can load them into memory.

    Sample-based implementation. 
    '''

    def __init__(self, file_paths: list, lookback: int, length: int, delay: int,
                 batch_size: int, sample_period: int, num_electrodes: int,
                 shuffle: bool = False):
        ''' 
        
        Initialization function for the object. Call when Generator() is called.

        Args:
            file_paths (list): List of file paths.
            lookback (int): The number of timesteps the input data should go back.
            delay (int): The number of timesteps the input data should predict in the future.
            length (int): The number of timesteps in the future we should predict.
            shuffle (bool): Shuffle the samples or draw them in chronological order.
            batch_size (int): The number of samples per batch.
            sample_period (int): The period, in timesteps, at which you sample data. 
                                E.g. If you set this to 256, it will sample 256 timesteps from the interval.
        Returns:
            A generator object.
        
        The ecog signals are sampled at 512Hz. This means we have 512 values per second.
        This means that a second of data is equal to 512 timesteps.
        '''
        self.file_paths = file_paths
        self.lookback = int(lookback)
        self.delay = int(delay)
        self.shuffle = shuffle
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.sample_period = int(sample_period)
        self.file_map, self.total_sample_count = self.__get_file_map()
        # At the very beginning and very end of each epoch 
        # we generate list of indices for the files.
        self.__on_epoch_end()
        # TODO: Better way to select electrodes
        self.num_electrodes = num_electrodes
    
    def __len__(self):
        ''' Called by len(a = Generator(...)).

        This function computes the number of batches that this
        generator is supposed to produce. So, we divide the 
        number of total samples by the batch_size and return that value. 
        '''
        # Calculate the total sample count and divide it by the batch_size.
        total_sample_count = self.total_sample_count
        return total_sample_count // self.batch_size

    def __get_file_map(self):
        '''
        Calculate the number of total samples and create a map between samples and files.
        '''
        file_map = dict()
        total_sample_count = 0
        for file_ix, path in enumerate(self.file_paths):
            # Open the file to only read the header
            data = np.load(path, mmap_mode='r')
            # Get the number of rows (i.e. timesteps)
            file_timestep_count = data.shape[0]
            # Calculate the number of total samples in this file.
            file_sample_count = int(file_timestep_count // (self.lookback + self.delay + self.length))
            for ix in np.arange(total_sample_count, total_sample_count + file_sample_count):
                file_map[ix] = (file_ix, ix - total_sample_count)
            total_sample_count += file_sample_count
        return file_map, total_sample_count
    
    def __on_epoch_end(self):
        ''' Update indices after each epoch.
        
        This function creates a list of indices that
        we can use to refer to files in order.
        
        '''
        self.sample_indices = np.arange(self.total_sample_count)
        if self.shuffle == True:
            np.random.shuffle(self.sample_indices)

    def __getitem__(self, index: int):
        ''' 
        Given batch number idx, create a list of data.
        X, y: (batch_size, self.lookback, num_electrodes)
        '''
        indices = self.sample_indices[index * self.batch_size : (index+1) * self.batch_size]
        sample_length = int(self.lookback + self.delay + self.length)
        X = np.empty((self.batch_size, self.lookback, self.num_electrodes))
        y = np.empty((self.batch_size, self.length, self.num_electrodes))

        for ix, k in enumerate(indices):
            # Very expensive implementation. 
            # Opens a file per sample!
            file_ix = self.file_map[k][0]
            sample_ix = self.file_map[k][1]
            filename = self.file_paths[file_ix]
            file_contents = np.load(filename)
            sample = file_contents[sample_ix * sample_length : (sample_ix+1) * sample_length]
            X[ix, ] = sample[:sample_length - self.length - self.delay]
            y[ix,] = sample[sample_length - self.length:]
        
        return X, y

class FGenerator(keras.utils.Sequence):
    '''
    Generator class that creates batches so Keras can load them into memory.

    This generator uses the file-based implementation.
    '''

    def __init__(self, file_paths: list, lookback: int, length: int, delay: int,
                 batch_size: int, sample_period: int, num_electrodes: int,
                 shuffle: bool = False):
        ''' 
        
        Initialization function for the object. Call when Generator() is called.

        Args:
            file_paths (list): List of file paths.
            lookback (int): The number of timesteps the input data should go back.
            delay (int): The number of timesteps the input data should predict in the future.
            length (int): The number of timesteps in the future we should predict.
            shuffle (bool): Shuffle the samples or draw them in chronological order.
            batch_size (int): The number of samples per batch.
            sample_period (int): The period, in timesteps, at which you sample data. 
                                E.g. If you set this to 256, it will sample 256 timesteps from the interval.
        Returns:
            A generator object.
        
        The ecog signals are sampled at 512Hz. This means we have 512 values per second.
        This means that a second of data is equal to 512 timesteps.
        '''
        self.file_paths = file_paths
        self.lookback = int(lookback)
        self.delay = int(delay)
        self.shuffle = shuffle
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.sample_period = int(sample_period)
        # TODO: Better way to select electrodes
        self.num_electrodes = num_electrodes
        # Calculate the total sample count and create a map
        # of files to samples.
        self.file_map, self.total_sample_count = self.__get_file_map()
        # At the very beginning and very end of each epoch 
        # we generate list of indices for the files.
        self.__on_epoch_end()
    
    def __len__(self):
        ''' Called by len(a = Generator(...)).

        This function computes the number of batches that this
        generator is supposed to produce. So, we divide the 
        number of total samples by the batch_size and return that value. 
        '''
        # Calculate the total sample count and divide it by the batch_size.
        return self.total_sample_count // self.batch_size

    def __get_file_map(self):
        '''
        Calculate the number of total samples and create a map between samples and files.
        '''
        file_map = dict()
        total_sample_count = 0
        for file_ix, file_path in enumerate(self.file_paths):
            # Open the file to only read the header
            data = np.load(file_path, mmap_mode='r')
            # Get the number of rows (i.e. timesteps)
            file_timestep_count = data.shape[0]
            # Calculate the number of total samples in this file.
            file_sample_count = int(file_timestep_count // (self.lookback + self.delay + self.length))
            file_map[file_ix] = file_sample_count
            total_sample_count += file_sample_count
        print(file_map)
        print(total_sample_count)
        return file_map, total_sample_count
    
    def __on_epoch_end(self):
        ''' 
        Update indices after each epoch.
        
        This function creates a list of indices that
        we can use to refer to files in order.
        '''
        self.sample_indices = np.arange(len(self.file_paths))
        if self.shuffle == True:
            np.random.shuffle(self.sample_indices)

        # Map batches to files and samples
        self.batch_map = dict()
        # Remainder store
        delta = 0
        # File index
        sample_ix = 0
        file_ix = self.sample_indices[sample_ix]
        for batch_index in range(self.total_sample_count // self.batch_size):
            current_batch_sample_count = 0
            self.batch_map[batch_index] = dict()
            while current_batch_sample_count < self.batch_size:
                # While the batch is not filled with samples.
                if delta > 0:
                    # If there are samples left in a previous file.
                    current_batch_sample_count += delta
                    self.batch_map[batch_index][file_ix] = (self.file_map[file_ix] - delta, self.file_map[file_ix] - 1)
                else:
                    # Add new file samples.
                    current_batch_sample_count += self.file_map[file_ix]
                    self.batch_map[batch_index][file_ix] = (0, self.file_map[file_ix] - 1)

                if current_batch_sample_count > self.batch_size:
                    # Overflow. Save delta for next batch.
                    delta = current_batch_sample_count - self.batch_size
                    prev_start = self.batch_map[batch_index][file_ix][0]
                    self.batch_map[batch_index][file_ix] = (prev_start, self.file_map[file_ix] - delta - 1)
                else:
                    # Exact match or underflow.
                    delta = 0
                    sample_ix += 1
                    file_ix = self.sample_indices[sample_ix]

    def __getitem__(self, index: int):
        ''' 
        Given batch number idx, create a list of data.
        X, y: (batch_size, self.lookback, num_electrodes)
        '''
        X = np.empty((self.batch_size, self.lookback, self.num_electrodes))
        y = np.empty((self.batch_size, self.length, self.num_electrodes))

        # Get ƒiles from batch_map
        batch = self.batch_map[index]
        curr_ix = 0
        sample_length = self.lookback + self.delay + self.length
        # For each file, get the samples and add them to the data
        for file_ix in batch:
            file_path = self.file_paths[file_ix]
            file_contents = np.load(file_path)
            for sample_ix in range(batch[file_ix][0], batch[file_ix][1]):
                sample = file_contents[sample_ix * sample_length : (sample_ix + 1) * sample_length]
                X[curr_ix, ] = sample[:sample_length - self.length - self.delay]
                y[curr_ix, ] = sample[sample_length - self.length:]
                curr_ix += 1
        return X, y

        