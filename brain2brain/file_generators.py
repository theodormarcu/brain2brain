class File_Generator(keras.utils.Sequence):
    '''
    Generator class that creates batches so Keras can load them into memory.
    Inspiration: Eric Ham, 
    https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
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
        self.lookback = int(lookback)
        self.delay = int(delay)
        self.shuffle = shuffle
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.sample_period = int(sample_period)
        # Calculate the total sample count and create a map
        # of files to samples.
        self.batch_map, self.total_sample_count = self.__get_file_map()
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
        return self.total_sample_count // self.batch_size

    def __get_file_map(self):
        '''
        Calculate the number of total samples and create a map between samples and files.
        '''
        batch_map = dict()
        total_sample_count = 0
        curr_batch_ix = 0
        curr_batch_sample_count = 0
        for ix, path in enumerate(self.file_paths):
            # Open the file to only read the header
            data = np.load(path, mmap_mode='r')
            # Get the number of rows (i.e. timesteps)
            file_timestep_count = data.shape[0]
            # Calculate the number of total samples in this file.
            file_sample_count = int(file_timestep_count // (self.lookback + self.delay + self.length))

            # If the batch is empty, create a dictionary
            if curr_batch_sample_count == 0:
                batch_map[curr_batch_ix] = dict()
            if file_sample_count + curr_batch_sample_count == self.batch_size:
                # Case 1: We fill the current batch and move on.
                # Add to the batch and increment batch id.
                batch_map[curr_batch_ix][path] = (0, file_sample_count)
                curr_batch_ix += 1
                curr_batch_sample_count = 0
            elif file_sample_count + curr_batch_sample_count > self.batch_size:
                # Case 2: Fill batches with samples from this file while we have samples remaining in the file.
                curr_file_sample_count = 0
                while curr_file_sample_count < file_sample_count:
                    # While we have samples left in the file.
                    if (file_sample_count - curr_file_sample_count) >= (self.batch_size - curr_batch_sample_count):
                        # Fill in current batch if we have as many or more samples than fit in the current batch.
                        batch_map[curr_batch_ix][path] = (curr_file_sample_count, curr_file_sample_count + 
                                                        (self.batch_size - curr_batch_sample_count))
                        # Create a new batch.
                        curr_batch_ix += 1
                        curr_batch_sample_count = 0
                        # Update the file counter with the sample ids 
                        curr_file_sample_count += (self.batch_size - curr_batch_sample_count) + 1
                    else:
                        # Add remaining samples to a new batch and update curr_batch_sample_count.
                        batch_map[curr_batch_ix][path] = (curr_file_sample_coun, file_sample_count)
                        curr_batch_sample_count = file_sample_count - curr_file_sample_count
                        # Update the file counter.
                        curr_file_sample_count += file_sample_count - curr_file_sample_count + 1
            elif file_sample_count + curr_batch_sample_count < self.batch_size:
                # Case 3: Add entire file.
                batch_map[curr_batch_ix][path] = (0, file_sample_count)
                curr_batch_sample_count += file_sample_count
            total_sample_count += file_sample_count
        return batch_map, total_sample_count
    
    def __on_epoch_end(self):
        ''' Update indices after each epoch.
        
        This function creates a list of indices that
        we can use to refer to files in order.
        
        '''
        self.file_indices = np.arange(len(self.file_paths))
        if self.shuffle == True:
            np.random.shuffle(self.file_indices)

    def __getitem__(self, index: int):
        ''' 
        Given batch number idx, create a list of data.
        X, y: (batch_size, self.lookback, num_electrodes)
        '''
        X = np.empty((self.batch_size, self.lookback, self.num_electrodes))
        y = np.empty((self.batch_size, self.length, self.num_electrodes))
        
        # Get the next file in the queue.
        # If random, select the samples randomly. 

        current_batch_sample_count = 0
        batch_remaining_sample_count = self.batch_size
        prev_file = None
        prev_file_ix = 0
        prev_file_sample_count = 0
        sample_length = self.lookback + self.delay + self.length
        
        while current_batch_sample_count < self.batch_size:
            # While the batch is not empty.
            if prev_file is not None:
                # If there's a previous file that still has sample, add them to the batch.
                if self.file_map[prev_file] - prev_file_sample_count > batch_remaining_sample_count:
                    # Overflow case.
                    # Add samples.
                    samples = self.__get_samples(prev_file,
                                                 prev_file_ix,
                                                 sample_length)
                    
                    for sample in samples:
                        X[current_batch_sample_count, ] = sample[:sample_length - self.length - self.delay]
                        y[current_batch_sample_count,] = sample[sample_length - self.length:]
                        current_batch_sample_count += 1
                    # Update counters.
                    batch_remaining_sample_count = self.batch_size - current_batch_sample_count
                    # Update prev_file counters.
                    prev_file_ix += batch_remaining_sample_count * sample_length + 1
                    prev_file_sample_count = prev_file_ix // sample_length
                elif prev_file_sample_count - prev_file_sample_ix <= batch_remaining_sample_count:
                    # Exact case and underflow case.
                    # Add samples.
                    samples = self.__get_samples(prev_file,
                                                 prev_file_ix,
                                                 sample_length)
                    
                    for sample in samples:
                        X[current_batch_sample_count, ] = sample[:sample_length - self.length - self.delay]
                        y[current_batch_sample_count,] = sample[sample_length - self.length:]
                        current_batch_sample_count += 1
                    # Update counters.
                    batch_remaining_sample_count = self.batch_size - current_batch_sample_count
                    # Update prev_file counters.
                    prev_file_ix = 0
                    prev_file_sample_count = 0
                    prev_file = None
            else:
                # Get the next file.
                if file_sample_count > batch_remaining_sample_count:
                    # Overflow case.
                    # Add samples.
                    samples = self.__get_samples(prev_file,
                                                 prev_file_ix,
                                                 sample_length)
                    # Update prev_file for the next batch.
                    batch_remaining_sample_count = self.batch_size - current_batch_sample_count
                elif file_sample_count <= batch_remaining_sample_count:
                    # Underflow and exact cases.
                    # Add samples.
                    prev_file = None

        return X, y
