from keras.utils import Sequence, to_categorical
from tensorflow.random import set_random_seed
import numpy as np
import scipy.io as sio
import math
import glob
#from sklearn.preprocessing import KBinsDiscretizer

np.random.seed(42)
set_random_seed(42)



#pass it a sequence when you create it --> makes this into the sequence you want to use. 
class TCN_Seq(Sequence):
	#x_set --> list of paths
	def __init__(self, num_conversations, patient, electrodes, min_index_frac, max_index_frac, input_size, output_size, lag, step_size, shuffle, batch_size, purpose):
		min_conv_idx = 0
		limit = num_conversations
		if purpose == 'test':
			num_convos_to_use = math.ceil(num_conversations*0.1)
			min_conv_idx = num_conversations
			limit = num_conversations + num_convos_to_use

		#top_of_path = '/mnt/bucket/labs/hasson/ariel/247/conversation_space/conversations/NY*'
		top_of_path = '/scratch/gpfs/zzada/conversations/NY*'
		count = 0
		data_set = []
		
		#print(limit)
		#print(num_convos_to_use)
		#print(min_conv_idx)
		for filepath in glob.iglob(top_of_path):
			#print(count)
			if count >= limit:
				break
			#print(filepath.split('/'))
			patient_num = int(filepath.split('/')[5][2:5]) #[5] for tiger, [9] for scotty (same below)
			file_d = None
			if patient_num == patient:
				print('file_path: ' + str(filepath))
				if count >= min_conv_idx:
					print('chosen file_path: ' + str(filepath)) #<-- to make sure test data doesn't overlap with train/val
					file_d = filepath + '/preprocessed/' + filepath.split('/')[5] + '_electrode_preprocess_file_'
					data_set.append(file_d)
				count+=1
				

		self.purpose = purpose
		self.in_len = input_size
		self.out_len = output_size
		self.lag =lag
		self.electrodes = electrodes
		self.sampling_multiple = step_size

		x_vals = []
		y_vals = []		   
		
		by_convo_x = []
		by_convo_y = []
		num_electrodes = len(electrodes)
		for j in range(len(data_set)):

			assert ((step_size*(input_size + self.out_len + lag/step_size - 1) - (step_size-1)) >=2)

			path = data_set[j]
			mat_file_contents = sio.loadmat(str(path) + str(1) + '.mat')
			size = len(mat_file_contents['p1st'])
			data_array = np.zeros((len(electrodes), size)) #array to hold all electrodes (64) and their values over all time steps (1802762)

			



			for i in range(num_electrodes):
				#electrodes[0] if only electrode in electrodes is '59' is the 59th electrode file.
				#data_array[0] is that eleectrode's data
				mat_file_contents = sio.loadmat(str(path) + str(electrodes[i]) +'.mat')
				array = mat_file_contents['p1st']
				data_array[i, :] = array.flatten()#purpose is to take (x,) array and make it (x) (list)



			#need to comment/uncomment following for evaluation.
			#sinusoid —> sine
			#sine_in = np.arange(0, size)
			#data_array[0] = np.sin(sine_in/(2*np.pi))
			#random signal —> rand

			#sinusoid + random noise. —> sine_plus_rand
			#sine_in = np.arange(0, size)
			#data_array[0] = np.sin(sine_in/(2*np.pi))
			#noise = np.random.randn(size)
			#data_array[0] += noise

			#noise
			#noise = np.random.randn(size).astype(np.float32)
			#data_array[0] = noise

			#FFT Phase scrambling. Shuffle the FFT, then take the inverse FFT to get phase scrambled signal. 
			#y = np.fft.fft(data_array)
			#np.random.shuffle(y)
			#data_array = np.fft.ifft(y)



			down_sampled_data = np.array_split(data_array, size/5, axis = 1) 
			#down_sampled_data = np.array_split(big_data_array, num_samples/5, axis = 1) #Convert to 100Hz.
			data_array = []
			for val in down_sampled_data:
				#print(val.shape)
				data_array.append(np.mean(val, axis = 1))
			data_array = np.asarray(np.transpose(data_array))



			self.std = np.std(data_array)
			self.mean = np.mean(data_array)

			

			size = data_array.shape[1]
			print('Number of Samples after Down Sampling: ' + str(size))

			if min_index_frac == 0 or min_index_frac == None:
				min_index = 0
			else:
				min_index = int(size*min_index_frac)+1
			if max_index_frac == 1 or max_index_frac == None:
				max_index = size - 1
			else:
				max_index = int(max_index_frac*size)
			samps = np.arange(min_index, int(max_index + 1 -(step_size*(input_size + self.out_len + lag/step_size - 1) - (step_size-1))))#, 21) ##

			samples = np.zeros((len(samps), input_size, data_array.shape[0]))  
			targets = np.zeros((len(samps), self.out_len, data_array.shape[0]))
			for j, samp in enumerate(samps):
				sample_indices = np.arange(samps[j], (samps[j] + input_size)*step_size-(samps[j])*(step_size-1), step_size)
				target_indices = np.arange((samps[j] + input_size)*step_size-(samps[j])*(step_size-1)-step_size+lag, ((samps[j] + input_size)*step_size-(samps[j])*(step_size-1)-step_size+lag+self.out_len)*step_size - ((samps[j] + input_size)*step_size-(samps[j])*(step_size-1)-step_size+lag)*(step_size-1), step_size) 
				
				if sample_indices[0] == 128:
					print('samp test: ' + str(sample_indices))
					print('target test: ' + str(target_indices))
					print('Electrode 1 Input: ' + str(data_array[0,sample_indices]))
					print('Electrode 1 Target: ' + str(data_array[0,target_indices]))
					
				#print('in: ' + str(sample_indices))
				#print('out: ' + str(target_indices))
				x_vals.append(np.transpose(data_array[:, sample_indices]))
				y_vals.append(np.transpose(data_array[:, target_indices]))
				#x_vals.append(data_array[:, sample_indices])								##linear model
				#y_vals.append(data_array[:, target_indices])								##linear model
			
			by_convo_x.append(x_vals)
			by_convo_y.append(y_vals)
	
		self.by_conv_x = np.asarray(by_convo_x)
		self.by_conv_y = np.asarray(by_convo_y)
	
		self.x, self.y = np.asarray(x_vals), np.asarray(y_vals)
		#min and max for classification TCN normalization.
		x_min = self.x.min()
		x_max = self.x.max()

		y_min = self.y.min()
		y_max = self.y.max()
		if y_min < x_min:
			self.min = y_min
		else:
			self.min = x_min
		if y_max > x_max:
			self.max = y_max
		else:
			self.max = x_max
		#use below for random output with nonrandom input.
		#self.y = np.random.randn(self.y.shape)
		self.batch_size = batch_size
		self.num_steps = math.ceil(len(self.x)/self.batch_size) - 1 # if have 105 points, batch 10 --> want 10 steps. not math.ceil(10.5) = 11.
		self.data_size = len(self.x)
		self.indices = np.arange(math.ceil(len(self.x)/self.batch_size)) 
		print('Actual # samples: ' + str(self.data_size))
		if shuffle == True:
			np.random.shuffle(self.indices)

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)
	
	def __getitem__(self, idx):
		batch_x = self.x[self.indices[idx]*self.batch_size : (self.indices[idx] + 1)*self.batch_size]
		batch_y = self.y[self.indices[idx]*self.batch_size : (self.indices[idx] + 1)*self.batch_size]
		#print('batch x: ' + str(batch_x.shape))
		#print('batch y: ' + str(batch_y.shape))
		
		#return batch_x, batch_y #do this for regular data
		#return batch_x.reshape(-1, self.in_len), batch_y.reshape(-1, self.out_len) 				#do this for linear data
		#return batch_x, batch_y.reshape(-1, self.num_bins)								#do this for class
		return batch_x, batch_y.reshape(-1,self.out_len)
		#return batch_x, batch_y[:,-1,:]
	def standardize_data(self, mean, std):
		#if using random output: replace mean and std self.y.mean, self.y.std
		self.x = self.x - mean
		self.y = self.y - mean
		self.x = self.x / std
		self.y = self.y / std
		

	#functions below for classification TCN. 
	def normalize_data(self, da_min, da_max):

		#import pdb; pdb.set_trace()
		self.x = -1 + 2*(self.x - da_min)/(da_max-da_min)
		self.y = -1 + 2*(self.y - da_min)/(da_max-da_min)
	'''
	def quantize_data(self):
		#approach 1: linear
		#bins are numerically ordered 1,2,3,4,5
		#import pdb; pdb.set_trace()
		bins = np.array([-1.1, -0.6,-0.2, 0.2, 0.6, 1.0])
		x = np.digitize(self.x, bins, right=True)
		self.x = x - 1
		y = np.digitize(self.y, bins, right=True)
		self.y = y - 1
		#approach 2: nonlinear. 

	def one_hot(self):
		import pdb; pdb.set_trace()
		self.x = to_categorical(self.x, 5)
		self.y = to_categorical(self.y, 5)
	'''
	def quantize_one_hot(self, disc,out_disc, num_bins):
		self.num_bins = num_bins
		#disc = KBinsDiscretizer(5, 'onehot-dense', 'uniform') #linear
		#import pdb; pdb.set_trace()
		#print('x before disc: ' + str(self.x.shape))
		#print('y before disc: ' + str(self.y.shape))
		self.x = self.x.reshape(-1, self.in_len)
		self.y = self.y.reshape(-1, self.out_len)
		#print(self.x.shape)
		#print(self.y.shape)
		self.x = np.transpose(disc.transform(self.x))
		self.y = np.transpose(out_disc.transform(self.y))
		self.x = self.x.reshape(-1, self.in_len, num_bins)
		self.y = self.y.reshape(-1, self.out_len, num_bins)
		#print(self.x.shape)
		#print(self.y.shape)

if __name__ == '__main__':
	conv_num = 1
	patient = 625
	electrodes = [28]
	min_index_train = 0
	max_index_train = 0.5
	in_len = 20
	out_len = 20
	sampling_multiple = 1
	shuffle_val = True
	batch_size = 128
	purpose = 'train'
	lag_val = 19
	lag = lag_val - (in_len-1)
	adj_lag = sampling_multiple*lag
	test_data_gen = TCN_Seq(int(conv_num), patient, electrodes, min_index_train, max_index_train,20, 20, adj_lag, sampling_multiple, shuffle_val, batch_size, purpose)
	test_batch = test_data_gen[0]
	for i in range(batch_size):
		print(test_batch[0][i,:,:])
		print(test_batch[1][i,:,:])


