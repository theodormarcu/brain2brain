#!/usr/bin/env python3
# Theodor Marcu
# tmarcu@princeton.edu
# Created January 2020
# Computer Science Senior Thesis

import os
import random
import numpy as np
import glob
import tensorflow.keras as keras
import math
import time


class FGenerator(keras.utils.Sequence):
    '''
    Generator class that creates batches so Keras can load them into memory.

    This generator uses the file-based implementation. The advantage of this
    implementation is that it's very fast compared to the sample-based
    implementation below. This is because this generator minimizes the
    number of files opened per get operation.
    '''

    def __init__(self, file_paths: list, lookback: int, length: int, delay: int,
                 batch_size: int, sample_period: int, electrodes: int,
                 electrode_output_ix: int=None, shuffle: bool = False,
                 debug: bool = False, ratio: float = 1.0):
        '''
        Initialization function for the object. Call when Generator() is called.
        Args:
            file_paths (list): List of file paths.
            lookback (int): The number of timesteps the input data should go back.
            length (int): The number of timesteps in the future we should predict.
            delay (int): The number of timesteps the input data should predict in the future.
            batch_size (int): The number of samples per batch.
            sample_period (int): The period, in timesteps, at which you sample data. 
                                E.g. If you set this to 256, it will sample 256 timesteps from the interval.
            electrodes (list[int]): List of electrode indices for one-to-one and many-to-many prediction.
            electrode_output_ix (int): Default = None. If present, the index of the electrode to
                                       be predicted. Useful for many-to-one prediction.
            shuffle (bool): Shuffle the samples or draw them in chronological order.
            normalize (bool): Deprecated. Should the sample values be normalized. Normalization
                              should take place at the file level. See utils.py for a normalize
                              function.
            debug (bool): Whether we should be in debug mode or not. 
                         Debug mode limits the number of batches to 1/4.
            data_ratio (float): How much of the data should the generator use. E.g. 0.5
                               is equal to half the data.
        Returns:
            A generator object.

        The ecog signals are sampled at 512Hz. This means we have 512 values per second.
        This means that a second of data is equal to 512 timesteps.

        Note: For one-to-one and many-to-many prediction, simply specify the list of electrodes
        using the `electrodes` list. For many-to-one prediction, use `electrodes` to
        specify the number of input electrodes and `electrode_output_ix` to specify
        which electrode should be predicted.

        The values of the ecosignals are in microVolts.
        '''
        self.file_paths = file_paths
        self.lookback = int(lookback)
        self.delay = int(delay)
        self.shuffle = shuffle
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.sample_period = int(sample_period)
        # self.normalize = normalize
        self.debug = debug
        self.ratio = ratio
        if self.ratio > 1.0 or self.ratio <= 0.0:
            raise Exception(f"The ratio ({self.ratio}) should be between 1.0 and 0.0 (1.0 >= ratio > 0.0).")
        self.electrodes = electrodes
        self.electrode_output_ix = electrode_output_ix
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
        length = int(np.floor(self.total_sample_count // self.batch_size))
        if self.debug:
            length = int(length // 5)
        elif self.ratio < 1.0:
            length = int(math.floor(length * self.ratio))
        return length

    def __get_file_map(self):
        '''
        Calculate the number of total samples and create a map between samples and files.
        '''
        file_map = dict()
        total_sample_count = 0
        for file_ix, file_path in enumerate(self.file_paths):
            # Open the file to only read the header using mmap_mode.
            data = np.load(file_path, mmap_mode='r')
            # Get the number of rows (i.e. timesteps).
            file_timestep_count = data.shape[0]
            # Calculate the number of total samples in this file.
            file_sample_count = int(
                file_timestep_count // (self.lookback + self.delay + self.length))
            file_map[file_ix] = file_sample_count
            total_sample_count += file_sample_count
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

        # Map batches to files and samples. This reduces the number of
        # file open operations when the generator produces a batch.
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
                    # If there are samples left in a previous file, 
                    # add them to this batch.
                    current_batch_sample_count += delta
                    self.batch_map[batch_index][file_ix] = (
                        self.file_map[file_ix] - delta, self.file_map[file_ix] - 1)
                else:
                    # Add new file samples.
                    current_batch_sample_count += self.file_map[file_ix]
                    self.batch_map[batch_index][file_ix] = (
                        0, self.file_map[file_ix] - 1)

                if current_batch_sample_count > self.batch_size:
                    # Overflow. Save delta (remaining samples) for next batch.
                    delta = current_batch_sample_count - self.batch_size
                    prev_start = self.batch_map[batch_index][file_ix][0]
                    self.batch_map[batch_index][file_ix] = (
                        prev_start, self.file_map[file_ix] - delta - 1)
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
        X = np.empty((self.batch_size, self.lookback //
                      self.sample_period, len(self.electrodes)))

        if self.electrode_output_ix is not None:
            y = np.empty((self.batch_size, math.ceil(
                self.length / self.sample_period)))
        else:
            y = np.empty((self.batch_size, math.ceil(
                self.length / self.sample_period), len(self.electrodes)))

        # Get files from batch_map
        batch = self.batch_map[index]
        curr_ix = 0
        sample_length = self.lookback + self.delay + self.length
        # For each file, get the samples and add them to the data.
        for file_ix in batch:
            file_path = self.file_paths[file_ix]
            file_contents = np.load(file_path)
            file_contents = file_contents[:, self.electrodes]
            # if self.normalize:
            #     print("Normalizing.")
            #     start_time = time.time()
            #     # Normalize the sample (for each electrode).
            #     for electrode in self.electrodes:
            #         mean = np.mean(file_contents[:, electrode])
            #         std = np.std(file_contents[:, electrode])
            #         file_contents[:, electrode] = (file_contents[:, electrode] - mean)/std
            #     print(f"Elapsed time: {time.time() - start_time}")
            for sample_ix in range(batch[file_ix][0], batch[file_ix][1]):
                sample = file_contents[sample_ix *
                                       sample_length: (sample_ix + 1) * sample_length]
                # Sample at sample_period.
                sampled_indices_data = range(
                    0, len(sample) - self.length - self.delay, self.sample_period)
                # X[curr_ix, ] = sample[:sample_length - self.length - self.delay]
                X[curr_ix, ] = sample[sampled_indices_data]
                sampled_indices_target = range(
                    len(sample) - self.length, len(sample), self.sample_period)
                # y[curr_ix, ] = sample[sample_length - self.length:]
                if self.electrode_output_ix is not None:
                    y[curr_ix, ] = sample[sampled_indices_target, [self.electrode_output_ix]]
                else:
                    y[curr_ix, ] = sample[sampled_indices_target]
                # Select just one electrode if the index of an output electrode is specified.
                # This is useful for many-to-one prediction.
                    # y[curr_ix, ] = y[curr_ix, ][:, self.electrode_output_ix]
                curr_ix += 1
        return X, y


class Generator(keras.utils.Sequence):
    '''
    Generator class that creates batches so Keras can load them into memory.

    Sample-based implementation. WARNING: Very slow. Avoid using this implementation.
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
            shuffle (bool): Shuffle the files in file_paths or draw them in the given order. 
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
            file_sample_count = int(
                file_timestep_count // (self.lookback + self.delay + self.length))
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
        indices = self.sample_indices[index *
                                      self.batch_size: (index+1) * self.batch_size]
        sample_length = int(self.lookback + self.delay + self.length)
        X = np.empty((self.batch_size, self.lookback, 114))
        y = np.empty((self.batch_size, self.length, 114))

        for ix, k in enumerate(indices):
            # Very expensive implementation.
            # Opens a file per sample!
            file_ix = self.file_map[k][0]
            sample_ix = self.file_map[k][1]
            filename = self.file_paths[file_ix]
            file_contents = np.load(filename)
            sample = file_contents[sample_ix *
                                   sample_length: (sample_ix+1) * sample_length]
            # print(sample.shape)
            X[ix, ] = sample[:sample_length - self.length - self.delay]
            y[ix, ] = sample[sample_length - self.length:]

        return X, y
