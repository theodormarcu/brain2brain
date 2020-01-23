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
                 sample_period: int, num_electrodes: int,
                 shuffle: bool = False):
        ''' Initialization function for the object. Call when Generator() is called.

        :param file_paths: List of file paths created by `get_file_paths`.
        :type data: list
        :param lookback: The number of timesteps the input data should go back. 1 timestep is equal to a 1/512th of a second.
        :type lookback: int
        :param delay: How many timesteps in the future the target should be. 1 timestep is equal to a 1/512th of a second.
        :type delay: int
        :param shuffle: Whether to shuffle the samples or draw them  in chronological order, 
        defaults to False.
        :type shuffle: bool
        :param batch_size: The number of samples per batch.
        :type batch_size: int
        :param sample_period: The period, in timesteps, at which you sample data. E.g. If you set this to 512, it will sample 512 timesteps per second, which might be highly redundant. 
        :type step: int

        :return: A generator object.
        :rtype: generator
        
        The ecog signals are sampled at 512Hz. This means we have 512 values per second.
        This means that a second of data is equal to 512 timesteps.
        
        How this generator works:
        Each batch of samples contains `batch_size` sets of samples and targets.
        A sample contains timesteps between an index i and i + `lookback`. The number 
        of timesteps is based on the `sample_period`.
        
        Since the data is spread across multiple files, a generator might have
        to extract batches from a file repeatedly or from multiple files.
        
        This means we need a way to map sample indices to files and then create batches
        based on this.
        
        E.g. Batch 1 refers to file A and indices x through z. 
        
        We can do this on the fly, but we need a way to remember where we stopped and
        also the case when the rest of the file does not contain enough timesteps for 
        a new sample.
    
        
        1. Randomize file indices.
        2. For each file, randomly draw samples.
        OR
        1. 
        2. For each batch id:
            a. If we have no timesteps left in a previous file, 
            open a new file.
                1. Calculate how many samples we can draw from this file.
                2. Draw the samples. (Randomly if shuffle. TODO: Not sure if this is necessary.)
                3. Open next file if needed. Repeat until we have enough samples
                in the batch.
                4. Save the filename and timestep index where we ended.
            b. If we have timesteps left in a previous file, open
            that file first.
                Draw timesteps from the file. Open next file if needed. 
        
        
        Mapping in the beginning:
        Random Draws:
        1. Randomize files. Calculate how many samples per file and then create a map:
        1 -> X indices for the files given.
        Select the arange thing    
        
        
        How to create this mapping?
        1. For each file, read the header to understand structure. Then, 
        divide the number of timesteps by the timestep_period. This will yield a sample_count. Each file_name should be associated with a sample_count. Warning! This is not as simple, we need to take into accont the lookback and the delay too.
        2. For each batch_id, get the files based on their sample_count.
        E.g. We need 128 samples, which means that files A (45 samples), B(55 samples), and C(60 samples) will be used. We need to make sure we store where the last index was used so the next batch can contain some samples from C. This also solves the problem if a file contains more samples than a batch would yield. To accomplish this, we have two options:
        1. dictionary that maps sample ID to filename and indices, but this might lead to a lot of useless opening/closing of files.
        2. a variable that stores the last timestep and the last accessed file.
        
        Two cases:
        
        * A batch size of 10 samples and a file that contains 20 samples. Say this is batch_id == 1, then we open the file, but we save that we should look at this index again. For batch_id == 2, the file is opened again and the last 10 samples are retrieved. To do this, we need to keep in mind that last_file_used is XYZ and we stopped reading samples at timestep X.
        * A batch_size of 10 samples and a file that contains 5 samples. Say that batch_id == 1, then we retrieve file A.npy, but we also need to retrieve file B.npy, which has 20 samples. 
        
        
        

        Questions to ask:
        Should the files be randomized or kept in order?
        Should the test/validation files be from similar days,
        or is it ok if they are from only a specific range of days/conversations?
        Check whether we can extract from multiple files/conversations in the same batch?
        Two ways to do this: global drawing: hard, since we need to associated indices with files and
        then the indices can't cross.
        Local file random drawing.
        
        What if batch is bigger than the file? We need to assert that this is not true!!!
        E.g. a 10s signal is 5120 rows. If a file contains 20 seconds, we might only get a sample.
        If a batch is 15, we might need to take samples from multiple files!!!
        This means that an internal mapping is very important. The mapping should take into consideration
        sample ID -> File ID so that regardless of the batch size we know when to open other files or not.
        I should check this assumption?
        
        '''
        self.file_paths = file_paths
        self.lookback = lookback
        self.delay = delay
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.timestep_period = timestep_period
        self.current_file = None
        self.current_file_ix = None
        # At the very beginning and very end of each epoch 
        # we generate list of indices for the files.
        self.arrange_indices()
        # TODO: FIND OUT WHY BEGINNING AND END
        self.next_file = self.file_indices[0]
        # TODO: Better way to select electrodes
        self.num_electrodes = num_electrodes
    
    
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

    def arrange_indices():
        ''' Update indices after each epoch.
        
        This function creates a list of indices that
        we can use to refer to files in order.
        
        '''
        self.file_indices = np.arange(len(self.file_paths))
        if self.shuffle = True
            np.random.shuffle(self.file_indices)

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
        # samples contains `batch_size` samples
        # of `lookback` // `sample_period` timesteps each.
        # Each sample contains `num_electrodes`.
        samples = np.empty(self.batch_size,
                           self.lookback // self.sample_period,
                           self.num_electrodes))
        # targets contains `batch_size` values
        # of `num_electrodes` each.
        targets = np.empty((self.batch_size,
                            self.num_electrodes), dtype=int)
        
        sample_count = 0
        while sample_count != self.batch_size:    
            if self.current_file is None:
                # Read from the next file.
                try:
                    with open(path, 'r') as npy_file:
                        # Load data into memory.
                        data = np.load(npy_file)
                except IOError:
                    print(f"IOError: {path} cannot be opened. Reading will STOP")          
                # Calculate number of samples that
                # can be extracted from this file.
                file_timestep_count = data.shape[0]
                if self.shuffle:
                    # Select batch_size random ints between min_index+lookback
                    # and max_index in order to select rows randomly. 
                    rows = np.random.randint(self.lookback, file_timestep_count, size=batch_size)
                    for j, _ in enumerate(rows):
                        # Get these timesteps from the data.
                        indices = range(rows[j] - self.lookback, rows[j], self.sample_period)
                        samples.append(data[indices])
                        # Multiple electrodes? Only one here
                        targets.append(data[rows[j] + self.delay])
                else:
                    # Read sequentially.
            else:
                # Read from the current file.
                
        

        file_batch = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
 