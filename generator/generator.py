from brain2brain import utils

class Generator(keras.utils.Sequence):
    '''
    Generator class that creates batches so Keras can load them into memory.
    Inspiration: Eric Ham, 
    https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
    '''

    def __init__(self, file_paths: list, lookback: int, length: int, delay: int,
                 min_index: int, max_index: int, batch_size: int,
                 sample_period: int, num_electrodes: int,
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
        '''
        self.file_paths = file_paths
        self.lookback = lookback
        self.delay = delay
        self.shuffle = shuffle
        self.length = length
        self.batch_size = batch_size
        self.timestep_period = timestep_period
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
        # Calculate the total sample count and divide it by the batch_size.
        total_sample_count = utils.get_total_sample_count()
        return total_sample_count // self.batch_size



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
        
 