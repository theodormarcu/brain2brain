import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=128)
parser.add_argument('--min_val_frac', type=float, default=0.8)
parser.add_argument('--max_val_frac', type=float, default=0.9)
parser.add_argument('--electrode', type=int, default=28)
parser.add_argument('--num_conversations', type=int, default=5)
parser.add_argument('--shuffle_val', type=bool, default=False)
parser.add_argument('--sampling_multiple',type=int, default=1)
parser.add_argument('--input_length', type=int, default=20)
parser.add_argument('--output_length', type=int, default=1)
parser.add_argument('--lag',type=int,default=0)
parser.add_argument('--model_type', type = str, default ='tcn_d')
parser.add_argument('--learning_rate', type = float, default = '.001')
parser.add_argument('--model_name', type=str, default = 'tcn')
parser.add_argument('--patient', type = int, default = 625)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--num_hours',type=int, default=None)
args = parser.parse_args()
print(args)

from eval_tcn_update import get_conv_pearson_r, get_actual_signals
from keras.models import load_model
import numpy as np
from tensorflow.random import set_random_seed
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from data_generator import TCN_Seq
from tcn_model import TCN_model_w_Dense_Layer
import json

np.random.seed(42)
set_random_seed(42)

batch_size = args.batch_size
min_index_val = args.min_val_frac
max_index_val = args.max_val_frac
electrode = args.electrode
num_conversations = args.num_conversations
shuffle_val = args.shuffle_val
sampling_multiple = args.sampling_multiple
in_len = args.input_length
out_len = args.output_length
lag_val = args.lag
model_name = args.model_name
patient = args.patient
model_type = args.model_type
num_epochs = args.num_epochs
#print(model_type)
if out_len == 1:
	lag = lag_val
else:
	lag = lag_val - (in_len -1)
adj_lag = sampling_multiple*lag

file_id = str(model_name)
save_dir = './Results/' + file_id + '/'


model = load_model(save_dir + 'MODEL_' + file_id + '.h5')	

val_data_gen = TCN_Seq(int(num_conversations),None, patient, [electrode], min_index_val, max_index_val, in_len, out_len, adj_lag, sampling_multiple, shuffle_val, batch_size, 'validate')

train_mean = np.load('./Results/' +  file_id +  '/' + file_id + '_mean_std.npy')[0]
train_std =   np.load('./Results/' +  file_id +  '/' + file_id + '_mean_std.npy')[1]
val_data_gen.standardize_data(train_mean, train_std)

if model_type == 'linear':
	i_to_p, i_to_t, p_to_t, ip_stde, it_stde, pt_stde = get_conv_pearson_r(model, val_data_gen, 'linear', electrode) #assumes only 1 electrode
else: 
	i_to_p, i_to_t, p_to_t, ip_stde, it_stde, pt_stde = get_conv_pearson_r(model, val_data_gen, 'tcn_d', electrode) #assumes only 1 electrode

ip_file = open(save_dir + file_id + '_ip_vals.txt','w')
ip_file.write(str(i_to_p))
ip_file.close()

it_file = open(save_dir + file_id + '_it_vals.txt','w') 
it_file.write(str(i_to_t))
it_file.close()

pt_file = open(save_dir + file_id + '_pt_vals.txt','w') 
pt_file.write(str(p_to_t))
pt_file.close()

ip_stde_f = open(save_dir + file_id + '_ip_stde.txt', 'w')
ip_stde_f.write(str(ip_stde))
ip_stde_f.close()

it_stde_f = open(save_dir + file_id + '_it_stde.txt', 'w')
it_stde_f.write(str(it_stde))
it_stde_f.close()

pt_stde_f = open(save_dir + file_id + '_pt_stde.txt','w')
pt_stde_f.write(str(pt_stde))
pt_stde_f.close()


in_sig, tru_sig, pred_sig = get_actual_signals(model, val_data_gen, file_id)
np.save(save_dir + file_id + 'true_signal.npy', tru_sig)
np.save(save_dir + file_id + 'pred_signal.npy', pred_sig)
np.save(save_dir + file_id + 'in_put_signal.npy', in_sig)




#chance_file = open(save_dir + file_id + '_chance_val.txt', 'w')
#chance_file.write(str(chance))
#chance_file.close()
history = json.load(open(save_dir + file_id +  '_history.json','r'))
final_mse_file = open(save_dir + file_id + '_final_mse.txt','w') 
final_mse_file.write(str(history['val_mean_squared_error'][-1]))
final_mse_file.close()

plt.figure()
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.plot(np.arange(0,num_epochs),history['val_mean_squared_error'], label='Val MSE')
plt.plot(np.arange(0, num_epochs), history['mean_squared_error'], label='MSE')
plt.savefig(save_dir + 'MSE_over_Epochs.png')
plt.legend()
plt.close()

#add MSE/pearson R Plot.






