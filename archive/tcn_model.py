from keras import models, layers
import keras.backend as K
from keras.models import load_model
from tensorflow.random import set_random_seed
import argparse
from os import path, mkdir
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

np.random.seed(42)
set_random_seed(42)

#Model Metrics for Evaluation in Main
def pearson_r_ip(y_true, y_pred):
	y_in = K.reshape(model.inputs[0], (-1, in_len)) # (batch,time_steps)			#should be x in test use test to verify. (check)
	y_p = K.reshape(model.outputs[0], (-1,out_len)) # (batch,time_steps)			#should be y_pred use test to verify (check)

	yi_centered, yp_centered = y_in - K.mean(y_in), y_p - K.mean(y_p)
	r_num = K.sum(yi_centered*yp_centered, axis = 1)
	r_den = K.sqrt(K.sum(yi_centered*yi_centered, axis = 1) * K.sum(yp_centered*yp_centered, axis = 1))
	return r_num/r_den

def pearson_r(y_true,y_pred):
	#import pdb; pdb.set_trace()
	y_tru = K.reshape(y_true, (-1, out_len))
	y_pre = K.reshape(y_pred, (-1,out_len))
	yt_centered, yp_centered = y_tru - K.mean(y_tru), y_pre - K.mean(y_pre)
	r_num = K.sum(yt_centered*yp_centered, axis = 1)
	r_den = K.sqrt(K.sum(yt_centered*yt_centered, axis = 1) * K.sum(yp_centered*yp_centered, axis = 1))
	return r_num/r_den


def pearson_r_it(y_true,y_pred):
	y_in = K.reshape(model.inputs[0], (-1, in_len))
	y_tru = K.reshape(y_true, (-1,out_len))
	yt_centered, yi_centered = y_tru - K.mean(y_tru), y_in - K.mean(y_in)
	r_num = K.sum(yt_centered*yi_centered, axis = 1)
	r_den = K.sqrt(K.sum(yt_centered*yt_centered, axis = 1) * K.sum(yi_centered*yi_centered, axis = 1))
	return r_num/r_den

#Models
def TCN_model(filter_size, num_electrodes):
	model = models.Sequential()
	model.add(layers.Conv1D(64, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=1, activation='relu', input_shape=(None,num_electrodes)))
	model.add(layers.Conv1D(128, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=2, activation='relu'))
	model.add(layers.Conv1D(256, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=4, activation='relu'))
	#model.add(layers.Conv1D(64, 1, strides = 1, padding='causal', data_format='channels_last',dilation_rate=1, activation='relu')) #tried 1D conv.
	#next 2 together.
	#model.add(layers.Conv1D(256, 3, strides=1, padding='causal', data_format='channels_last', dilation_rate=8, activation='relu'))
	#model.add(layers.Dense(num_electrodes, activation='linear'))
	model.add(layers.Conv1D(num_electrodes, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=8, activation='linear'))
	model.summary()
	return model

def classification_TCN_model(filter_size, num_bins):
	model = models.Sequential()
	model.add(layers.Conv1D(64, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=1, activation='relu', input_shape=(None,num_bins)))
	model.add(layers.Conv1D(128, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=2, activation='relu'))
	model.add(layers.Conv1D(256, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=4, activation='relu'))
	model.add(layers.Conv1D(num_bins, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=8, activation='softmax'))
	return model

def linear_model(num_electrodes, in_len, out_len):
	model = models.Sequential()
	model.add(layers.Dense(out_len, input_shape=(in_len,), activation='linear'))#, kernel_regularizer = regularizers.l2(0.001))) #add regularization --> makes worse.
	model.summary()
	return model

def non_linear_model(num_electrodes, in_len, out_len):
	model = models.Sequential()
	model.add(layers.Dense(out_len, input_shape=(in_len, ), activation='relu'))
	model.add(layers.Dense(out_len, activation='linear'))
	model.summary()
	return model

def TCN_model_w_Dense_Layer(filter_size, num_electrodes, input_length):
	model = models.Sequential()
	model.add(layers.Conv1D(64, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=1, activation='relu', input_shape=(input_length,num_electrodes)))
	model.add(layers.Conv1D(128, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=2, activation='relu'))
	model.add(layers.Conv1D(256, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=4, activation='relu'))
	model.add(layers.Conv1D(256, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=8, activation='relu'))
	model.add(layers.Conv1D(512, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=16, activation='relu'))
	#option 1: flatten
	model.add(layers.Flatten())
	#option 2: 1d conv --> reduce to 1 filter (anticipate it to reduce data to (batch_size, 20) --> dense layer can make (batch_size, 1)
	#model.add(layers.Conv1D(1, 1, strides = 1, data_format = 'channels_last', activation='relu')) #causality unnecessary since only convolves along filter axis
	#model.add(layers.Flatten())
	model.add(layers.Dense(num_electrodes, activation = 'linear'))
	model.summary()
	return model

def TCN_model_w_Dense_Layer_and_residuals(filter_size, num_electrodes, input_length):
	input_tensor = Input(shape=(input_length, num_electrodes))
	x = layers.Conv1D(64, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=1, activation='relu')(input_tensor)
	x = layers.Conv1D(128, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate = 2, activation='relu')(x)

	#residual reshapes x to depth of 128 so it can be added to y.
	#activation is linear by default
	residual = layers.Conv1D(128, 1, strides=1, padding='causal', data_format='channels_last', dilation_rate = 1)(input_tensor)
	x = layers.add([x, residual])

	y = layers.Conv1D(256, filter_size, strides=1, padding='causal',data_format='channels_last', dilation_rate = 4, activation ='relu')(x)
	y = layers.Conv1D(256, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate = 8, activation='relu')(y)
	y = layers.Conv1D(512, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate = 16, activation='relu')(y)

	residual2 = layers.Conv1D(512, 1, strides=1, padding='causal', data_format='channels_last', dilation_rate=1)(x)
	y = layers.add([y, residual2])
	y = layers.Flatten()(y)
	output_tensor = layers.Dense(num_electrodes, activation='linear')(y)

	model= Model(input_tensor, output_tensor)
	model.summary()

	return model

#below too small
def TCN_model_w_Dense_Layer_prev(filter_size, num_electrodes, input_length):
	model = models.Sequential()
	model.add(layers.Conv1D(64, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=1, activation='relu', input_shape=(input_length,num_electrodes)))
	model.add(layers.Conv1D(128, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=2, activation='relu'))
	model.add(layers.Conv1D(256, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=4, activation='relu'))
	model.add(layers.Conv1D(256, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=8, activation='relu'))
	#option 1: flatten
	model.add(layers.Flatten())
	#option 2: 1d conv --> reduce to 1 filter (anticipate it to reduce data to (batch_size, 20) --> dense layer can make (batch_size, 1)
	#model.add(layers.Conv1D(1, 1, strides = 1, data_format = 'channels_last', activation='relu')) #causality unnecessary since only convolves along filter axis
	#model.add(layers.Flatten())
	model.add(layers.Dense(num_electrodes, activation = 'linear'))
	model.summary()
	return model

def tcn_minimalist_10(filter_size, num_electrodes, input_length):
	model = models.Sequential()
	model.add(layers.Conv1D(64, filter_size, strides = 1, padding='causal', data_format='channels_last', dilation_rate=1, activation='relu', input_shape=(input_length,num_electrodes)))
	model.add(layers.Conv1D(128, filter_size, strides = 1, padding='causal', data_format='channels_last', dilation_rate=2, activation='relu'))
	model.add(layers.Conv1D(256, filter_size, strides = 1, padding='causal', data_format='channels_last', dilation_rate=4, activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(num_electrodes, activation = 'linear'))
	model.summary()
	return model

def classification_TCN_w_Dense_Layer(filter_size, num_bins, input_length):
	model = models.Sequential()
	model.add(layers.Conv1D(64, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=1, activation='relu', input_shape=(input_length,num_bins)))
	model.add(layers.Conv1D(128, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=2, activation='relu'))
	model.add(layers.Conv1D(256, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=4, activation='relu'))
	model.add(layers.Conv1D(256, filter_size, strides=1, padding='causal', data_format='channels_last', dilation_rate=8, activation='relu'))
	#option 1: flatten (use all filter values)
	model.add(layers.Flatten())
	#option 2: 1d conv --> reduce to 1 filter (needs revision)
	#model.add(layers.conv1D(1, 1, strides = 1, data_format = 'channels_last', activation='relu')) #causality unnecessary since only convolves along filter axis
	model.add(layers.Dense(num_bins, activation = 'softmax'))

	model.summary()
	return model

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--filter_size', type=int, default=3)
	parser.add_argument('--num_electrodes', type=int, default=1)
	parser.add_argument('--input_length', type=int, default=20)
	parser.add_argument('--output_length', type=int, default=20)
	args = parser.parse_args()

	save_dir = './Results/test_model/'

	filter_size = args.filter_size
	num_electrodes = args.num_electrodes
	in_len = args.input_length
	out_len = args.output_length
	#model = TCN_model(filter_size, num_electrodes)
	#model = linear_model(1, in_len, out_len)
	#x = np.arange(40).reshape(2,20)
	#x = (x - x.mean()) / x.std()
	#preds = model.predict_on_batch(x)
	#print(preds)
	#print(np.sum((x - preds)**2) / 40)
	num = str(4)
	model = linear_model(5,20,20) #placeholder
	model = load_model('./Results/linear_model_w5_no_reg_' + num + '_0.01_5/MODEL_linear_model_w5_no_reg_' + num + '_0.01_5.h5', custom_objects={'pearson_r': pearson_r, 'pearson_r_ip':pearson_r_ip, 'pearson_r_it':pearson_r_it})
	weights = model.get_weights()
	nb = weights[0]
	#x = np.arange(1,21).reshape(1,20)	#20
	#y = (x @ nb).astype(int)			#20
	x = np.arange(1, 6).reshape(1,5)
	y = (x @ nb).astype(int)
	#y1 = (x @nb[:,-1]).astype(int)
	#z = (nb - nb.mean(axis=0)) / nb.std(axis=0) #weight arrays are vertical (weight[0][0] is not the first weight vector, but its first entry is the first entry of the first weight vector)

	plt.figure(figsize = (8,8))
	#c = plt.matshow(z, fignum=1)		#normalized
	c = plt.matshow(nb, fignum=1)		#not normalized
	plt.colorbar(c)
	plt.xlabel('nodes')
	plt.ylabel('time')
	plt.title(str(y))
	print(x)
	print(y)
	#print(y1)
	plt.savefig('./Results/linear_model_w5_no_reg_'  + num + '_0.01_5/weights_plot_not_norm_test.png')
