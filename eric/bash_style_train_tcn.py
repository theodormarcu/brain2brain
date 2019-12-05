#import pdb; pdb.set_trace() <--- debugging.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--min_train_frac', type=float, default=0)
parser.add_argument('--max_train_frac', type=float, default=0.8)
parser.add_argument('--min_val_frac', type=float, default=0.8)
parser.add_argument('--max_val_frac', type=float, default=0.9)
parser.add_argument('--input_length', type=int, default=20)
parser.add_argument('--output_length', type=int, default=1)
parser.add_argument('--sampling_multiple', type=int, default=1)
parser.add_argument('--lag', type=int, default=0)
parser.add_argument('--electrode', type=int, default=28)
parser.add_argument('--learning_rate', type=float, default=.01)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--shuffle_model', type=bool, default=True)
parser.add_argument('--shuffle_train', type=bool, default=False)
parser.add_argument('--shuffle_val', type=bool, default=False)
parser.add_argument('--model_name', type=str, default='Current_Model')
parser.add_argument('--filter_size', type=int, default=3)
parser.add_argument('--num_conversations', type=int, default=5)
parser.add_argument('--patient', type=int, default=625)
parser.add_argument('--model_type', type = str, default = 'tcn_d')
parser.add_argument('--num_hours', type=int,default=None)
args = parser.parse_args()
print(args)


#print(lags)
#exit()

#from eval_tcn_update import get_conv_pearson_r
from scipy.stats import pearsonr
import tensorflow as tf
from keras import models, optimizers
from keras.utils import Sequence
import keras.backend as K
from tensorflow.random import set_random_seed
from keras.callbacks import EarlyStopping
import keras.callbacks
from keras.models import load_model
import json
from os import path, mkdir
from tcn_model import TCN_model, linear_model, non_linear_model, TCN_model_w_Dense_Layer, tcn_minimalist_10
from data_generator import TCN_Seq
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)
set_random_seed(42)
tf.reset_default_graph()


batch_size = args.batch_size
min_index_train = args.min_train_frac
max_index_train = args.max_train_frac

min_index_val = args.min_val_frac
max_index_val = args.max_val_frac

in_len = args.input_length
out_len = args.output_length
print(out_len)
sampling_multiple = args.sampling_multiple
filter_size = args.filter_size
model_type = args.model_type

# sampling multiple = 20 #for skipping 20 (instead of averaging)
#rand_data_test = np.random.random_sample((64, len(mat_file_contents['p1st'])))

lag_val = args.lag #[5, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 50]

#electrodes = [13, 28, 46]
electrode = args.electrode
#electrodes = np.range(0,64) #do this for all 64
learning_rate = args.learning_rate
num_epochs = args.num_epochs
patience_val = args.patience
shuffle_model = args.shuffle_model
shuffle_train = args.shuffle_train
shuffle_val = args.shuffle_val
model_name = args.model_name
num_hours=args.num_hours
num_electrodes = 1
num_conversations = args.num_conversations
patient=args.patient

#updated file id with bash script
file_id = str(model_name)
#file_id = str(model_name)  + '_' + str(learning_rate) + '_' + str(num_conversations) + '_' + str(lag_val)
print('lag_val ' + str(lag_val))
save_dir = './Results/' + file_id + '/'
if not path.isdir(save_dir): mkdir(save_dir)


#Handle Lag.
if out_len == 1:
	lag = lag_val
else:
	lag = lag_val - (in_len-1)
adj_lag = sampling_multiple*lag



																		##added
args_file = open(save_dir + 'model_info_file.txt', 'w')
args_file.write(str(args) + '\n')
args_file.close()




#Training Data Generation
train_data_gen = TCN_Seq(int(num_conversations),num_hours,patient, [electrode], min_index_train, max_index_train,in_len, out_len, adj_lag, sampling_multiple, shuffle_train, batch_size, 'train')
train_std = train_data_gen.std
train_mean = train_data_gen.mean
np.save('./Results/' +  file_id +  '/' + file_id + '_mean_std.npy', [train_mean, train_std])
train_steps = train_data_gen.num_steps
train_data_gen.standardize_data(train_mean, train_std)

#Validation Data Generation
val_data_gen = TCN_Seq(int(num_conversations), num_hours,patient, [electrode], min_index_val, max_index_val,in_len, out_len, adj_lag, sampling_multiple, shuffle_val, batch_size, 'validate')
val_steps = val_data_gen.num_steps
val_data_gen.standardize_data(train_mean, train_std)




#callback
#es_vrsquare = EarlyStopping(monitor='val_r_square', mode = 'max', min_delta = .01, patience = 5)
#es_trsquare = EarlyStopping(monitor='pearson_r', mode = 'max', min_delta = .01, patience = patience_val)
#add model checkpoint to save best model if applicable in future

#in_to_pred_mse = InToPredMSE()
rmsprop = optimizers.RMSprop(lr=learning_rate)

##uncomment below.
#model = TCN_model(args.filter_size, num_electrodes)
#model = non_linear_model(num_electrodes, in_len, out_len)
if model_type == 'linear':
	model = linear_model(num_electrodes, in_len, out_len)
	print('Linear Model')
elif model_type == 'tcn':
	model = TCN_model(filter_size, num_electrodes)
	print('TCN')
elif model_type == 'tcn_d':
	print('tcn_d')
	model = TCN_model_w_Dense_Layer(filter_size, num_electrodes, in_len)
	#model = tcn_minimalist_10(filter_size, num_electrodes, in_len)


#Instatiation thing. 500 to 1000 times.
#get initial pearson r.
'''
ip_init_vals = []
pt_init_vals = []
for i in range(0, 500):
	if model_type == 'linear':
		model = linear_model(num_electrodes, in_len, out_len)
		i_to_p_init, _, p_to_t_init = plot_pearson_graph(model, val_data_gen, 'linear', electrode, file_id, False) #assumes only 1 electrode
	else:
		model = TCN_model_w_Dense_Layer(filter_size, num_electrodes, in_len)
		i_to_p_init, _, p_to_t_init = plot_pearson_graph(model, val_data_gen, 'tcn_d', electrode, file_id, False) #assumes only 1 electrode
	ip_init_vals.append(i_to_p_init)
	pt_init_vals.append(p_to_t_init)

np.save(save_dir + file_id + 'i_to_p_init_val.npy', np.mean(ip_init_vals))
np.save(save_dir + file_id + 'p_to_t_init_val.npy', np.mean(pt_init_vals))

plt.figure()
plt.hist(p_to_t_init, bins=100)
plt.savefig(save_dir + file_id + 'p_to_t_distribution.png')
'''
model.compile(optimizer=rmsprop, loss = 'logcosh', metrics = ['mse'])#,pearson_r, pearson_r_ip, pearson_r_it]) #use mean squared error and RMSProp for loss and optimizer respectively. #added
model.summary() #added
#import pdb; pdb.set_trace()
#Train the Model
history = model.fit_generator(train_data_gen, steps_per_epoch= train_steps, epochs = num_epochs, validation_data = val_data_gen, validation_steps = val_steps, shuffle = shuffle_model)#, callbacks=[es_trsquare])


#Evaluation.
model.save(save_dir + 'MODEL_' + file_id + '.h5')
with open(save_dir + file_id + '_history.json', 'w') as f:	#added
	json.dump(history.history, f)
'''
if model_type == 'linear':
	i_to_p, i_to_t, p_to_t = plot_pearson_graph(model, val_data_gen, 'linear', electrode, file_id, True) #assumes only 1 electrode
else:
	i_to_p, i_to_t, p_to_t = plot_pearson_graph(model, val_data_gen, 'tcn_d', electrode, file_id, True) #assumes only 1 electrode

ip_file = open(save_dir + file_id + '_ip_vals.txt','w')
ip_file.write(str(i_to_p))
ip_file.close()

it_file = open(save_dir + file_id + '_it_vals.txt','w')
it_file.write(str(i_to_t))
it_file.close()

pt_file = open(save_dir + file_id + '_pt_vals.txt','w')
pt_file.write(str(p_to_t))
pt_file.close()

#chance_file = open(save_dir + file_id + '_chance_val.txt', 'w')
#chance_file.write(str(chance))
#chance_file.close()

final_mse_file = open(save_dir + file_id + '_final_mse.txt','w')
final_mse_file.write(str(history.history['val_mean_squared_error'][-1]))
final_mse_file.close()

plt.figure()
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.plot(np.arange(0,num_epochs), history.history['val_mean_squared_error'], label='Val MSE')
plt.plot(np.arange(0, num_epochs), history.history['mean_squared_error'], label='MSE')
plt.savefig(save_dir + 'MSE_over_Epochs.png')
plt.legend()
plt.close()

#add MSE/pearson R Plot.

'''
