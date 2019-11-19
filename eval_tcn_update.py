#import pdb; pd.set_trace() <--- debugging.


from tensorflow.random import set_random_seed
import numpy as np
from scipy.stats import pearsonr, zscore
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from os import mkdir, path
from keras.models import load_model
import keras.backend as K
from keras import models
import json
np.random.seed(42)
set_random_seed(42)
#from sklearn.metrics import r2_score

from tcn_model import TCN_model, linear_model, non_linear_model
#model = None
def over_lag_R_graph(lags_i,window, electrode, file_path, purpose, m_type):
	#import pdb; pdb.set_trace()
	i_to_p_vals = []
	i_to_t_vals = []
	p_to_t_vals = []
	metric_vals = []
	#ip_init_vals = []
	for val in lags_i:
		#file_parts = file_path.split('_')
		#print(file_parts)
		#regression
		#file_id = file_parts[0] + '_' + file_parts[1] + '_' + file_parts[2] + str(val) + '_' + file_parts[3] + '_' + file_parts[4] + '_' + str(val)
		#classification verion:
		#file_id = file_parts[0] + '_' + file_parts[1] + '_' + file_parts[2] + '_' + file_parts[3] + str(val) + '_' + str(val) + '_' +  file_parts[4] + '_' + file_parts[5] + '_' + file_parts[6]
		#file_id = file_path + '_' + str(val) #fix naming in train file for this
		#print(file_path)
		fp = file_path.split('-')
		#class		
		#file_id = fp[0] + '-' + fp[1] + '-'  + fp[2] + '-'  + fp[3] + '-'  + fp[4]  + '-' + fp[5] + '-l' + str(val)
		#reg no fp[5]
		file_id = fp[0] + '-e' + str(electrode) + '-w' + str(window) + '-'  + fp[3] + '-'  + fp[4]  + '-l' + str(val)
		save_dir = './Results/' + file_id + '/'     #new models
		print(file_id)
		ip_file = open(save_dir + file_id + '_ip_vals.txt','r')
		i_to_p_vals.append(float([val for val in ip_file][0]))
		it_file = open(save_dir + file_id + '_it_vals.txt','r') 
		i_to_t_vals.append(float([val for val in it_file][0]))
		pt_file = open(save_dir + file_id + '_pt_vals.txt','r') 
		p_to_t_vals.append(float([val for val in pt_file][0]))
		if m_type == 'tcn_class_d': 
			metric_file = open(save_dir + file_id + '_final_acc.txt','r') 
			metric_vals.append(float([val for val in metric_file][0]))
		else:
			metric_file = open(save_dir + file_id + '_final_mse.txt','r') 
			metric_vals.append(float([val for val in metric_file][0]))
		#ip_init_file = open(save_dir + file_id + 'i_to_p_init_val.npy', 'r')
		#ip_init_vals.append(float([val for val in ip_init_file][0]))

		ip_file.close()
		it_file.close()
		pt_file.close()
		metric_file.close()
		#ip_init_file.close()

	#file_id = file_path + '_' + str(lags_i[-1])
	#regression
	#file_id = file_parts[0] + '_' + file_parts[1] + '_' + file_parts[2] + str(lags_i[-1]) + '_' + file_parts[3] + '_' + file_parts[4] + '_' + str(lags_i[-1])
	#classification verion:
	
	#file_id = file_parts[0] + '_' + file_parts[1] + '_' + file_parts[2] + '_' + file_parts[3] + str(lags_i[-1]) + '_' + str(val) + '_' +  file_parts[4] + '_' + file_parts[5] + '_' + file_parts[6]
	#file_id = file_parts[0] + 'l' + str(lags_i[-1])
	#class
	#file_id = fp[0] + '-' + fp[1] + '-'  + fp[2] + '-'  + fp[3] + '-'  + fp[4]  + '-' +fp[5] +  '-l' + str(lags_i[-1])
	#reg: no fp[5]
	file_id = fp[0] + '-e' + str(electrode) + '-w'  + str(window) + '-'  + fp[3] + '-'  + fp[4] +  '-l' + str(lags_i[-1])
	#print(file_id)
	save_dir = './Results/' + file_id + '/'

	fig, ax1 = plt.subplots(figsize=(12, 12))
	ax1.set_ylabel('R')
	
	ax1.plot(lags_i, i_to_p_vals, 'bo',linestyle = '-',label='Input to Pred (R)', color='blue')
	ax1.plot(lags_i, i_to_t_vals, 'bo',linestyle = '-',label='Input to True (R)', color='orange')
	ax1.plot(lags_i, p_to_t_vals, 'bo',linestyle = '-',label='Pred to True (R)', color='green')
	#Ax2 commented out for backwards compat
	ax2 = ax1.twinx()
	#ax2.plot(lags_i, ip_init_vals, linestyle = ':', label = 'Pre Training Input to Pred (R)')
	if m_type == 'tcn_class_d':
		ax_label = 'Val ACC'
	else:	
		ax_label = 'Val MSE'
	ax2.plot(lags_i, metric_vals, 'bo',linestyle  = '--', label = ax_label)
	
	if m_type == 'tcn_class_d':
		ax2.set_ylabel('ACC')
	else:
		ax2.set_ylabel('MSE')
	#ax2.set_xlabel('Lags')
	plt.xlabel('Lags')
	plt.title('Pearson R for Different Lags ' + str(file_id))

	ax1.legend(loc = 'upper left')
	ax2.legend(loc = 'upper right')

	plt.savefig(save_dir + str(file_id) + '_' + str(purpose) + '_R_Over_diff_lags.png')

def over_window_R_graph(windows_i, lag, electrode,file_path, purpose, m_type):
	#import pdb; pdb.set_trace()
	i_to_p_vals = []
	i_to_t_vals = []
	p_to_t_vals = []
	metric_vals = []
	#ip_init_vals = []
	for val in windows_i:
		#file_parts = file_path.split('-')
		#print(file_parts)
		#regression
		#file_id = file_parts[0] + '-' + file_parts[1] + '-' + 'w' + str(val) + '-' + file_parts[3] + '-' + file_parts[4] + '-' + file_parts[5]
		#classification verion:
		#file_id = file_parts[0] + '-' + file_parts[1] + '-' + 'w' + str(val) + '-' + file_parts[3] + '-' + file_parts[4] + '-' + file_parts[5] + '-' + file_parts[6]
		#file_id = file_path + '_' + str(val) #fix naming in train file for this
		fp = file_path.split('-')
		
		file_id = fp[0] + '-e' + str(electrode) + '-w' + str(val) + '-'  + fp[3] + '-'  + fp[4]  + '-l' + str(lag)
		save_dir = './Results/' + file_id + '/'     #new models

		ip_file = open(save_dir + file_id + '_ip_vals.txt','r')
		i_to_p_vals.append(float([val for val in ip_file][0]))
		it_file = open(save_dir + file_id + '_it_vals.txt','r') 
		i_to_t_vals.append(float([val for val in it_file][0]))
		pt_file = open(save_dir + file_id + '_pt_vals.txt','r') 
		p_to_t_vals.append(float([val for val in pt_file][0]))
		if m_type == 'tcn_class_d': 
			metric_file = open(save_dir + file_id + '_final_acc.txt','r') 
			metric_vals.append(float([val for val in metric_file][0]))
		else:
			metric_file = open(save_dir + file_id + '_final_mse.txt','r') 
			metric_vals.append(float([val for val in metric_file][0]))
		#ip_init_file = open(save_dir + file_id + 'i_to_p_init_val.npy', 'r')
		#ip_init_vals.append(float([val for val in ip_init_file][0]))

		#ip_file.close()
		it_file.close()
		pt_file.close()
		metric_file.close()
		#ip_init_file.close()

	#file_id = file_path + '_' + str(windows_i[-1])
	#regression
	#file_id = file_parts[0] + '-' + file_parts[1] + '-' + 'w' + str(windows_i[-1]) + '-' + file_parts[3] + '-' + file_parts[4] + '-' + file_parts[5]
	#classification verion:
	#file_id = file_parts[0] + '-' + file_parts[1] + '-' + 'w' + str(windows_i[-1]) + '-' + file_parts[3] + '-' + file_parts[4] + '-' + file_parts[5] + '-' + file_parts[6]
		
	file_id = fp[0] + '-e' + str(electrode) + '-w' + str(windows_i[-1]) + '-'  + fp[3] + '-'  + fp[4]  + '-l' + str(lag)
	save_dir = './Results/' + file_id + '/'

	fig, ax1 = plt.subplots(figsize=(12, 12))
	ax1.set_ylabel('R')
	
	ax1.plot(windows_i, i_to_p_vals, 'bo',linestyle = '-',label='Input to Pred (R)', color = 'blue')
	ax1.plot(windows_i, i_to_t_vals, 'bo',linestyle = '-',label='Input to True (R)', color = 'orange')
	ax1.plot(windows_i, p_to_t_vals, 'bo',linestyle = '-',label='Pred to True (R)', color = 'green')
	#Ax2 commented out for backwards compat
	ax2 = ax1.twinx()
	#ax2.plot(lags_i, ip_init_vals, linestyle = ':', label = 'Pre Training Input to Pred (R)')
	if m_type == 'tcn_class_d':
		ax_label = 'Val ACC'
	else:	
		ax_label = 'Val MSE'
	ax2.plot(windows_i, metric_vals, 'bo',linestyle  = '--', label = ax_label)
	
	if m_type == 'tcn_class_d':
		ax2.set_ylabel('ACC')
	else:
		ax2.set_ylabel('MSE')
	#ax2.set_xlabel('Lags')
	plt.xlabel('Windows')
	plt.title('Pearson R for Different Windows ' + str(file_id))

	ax1.legend(loc = 'upper left')
	ax2.legend(loc = 'upper right')

	plt.savefig(save_dir + str(file_id) + '_' + str(purpose) + '_R_Over_diff_windows.png')


def over_lag_many_windows_graph(lags_i, windows_i, electrode, file_path):
	plt.figure(figsize = (22,10))
	for val in windows_i:
		over_lags = []
		#ip_init_vals = []
		#pt_init_vals = []
		#chance_vals = []
		i_to_t_vals = []
		for lag in lags_i:		
			fp = file_path.split('-')
			file_id = fp[0] + '-e' + str(electrode) + '-w'  + str(val) + '-'  + fp[3] + '-'  + fp[4]  + '-l' + str(lag)
			save_dir = './Results/' + file_id + '/'     #new models			
			pt_file = open(save_dir + file_id + '_pt_vals.txt','r') 
			over_lags.append(float([item for item in pt_file][0]))
			pt_file.close()
			#ip_init_val = np.load(save_dir + file_id + 'i_to_p_init_val.npy').reshape(1,)[0]
			#ip_init_vals.append(ip_init_val)
			#uninitialized model prediction to true
			#pt_init_val = np.load(save_dir + file_id + 'p_to_t_init_val.npy').reshape(1,)[0]
			#pt_init_vals.append(pt_init_val)
			#chance
			#chance_file = open(save_dir + file_id + '_chance_val.txt', 'r')
			#chance_vals.append(float([item for item in chance_file][0]))
			#chance_file.close()
			
			#autocorrelation
			it_file = open(save_dir + file_id + '_it_vals.txt','r') 
			i_to_t_vals.append(float([item for item in it_file][0]))
		
		#print(ip_init_vals)
		color_val = tuple(np.random.random_sample((3,)))				
		plt.plot(lags_i, over_lags,'bo', linestyle = '-',label = 'Window Size: ' + str(val), color = color_val)
		#plt.plot(lags_i, ip_init_vals,linestyle = '--', label= 'Window Size: ' + str(val) +'Pre Train', color=color_val)	
		#plt.plot(lags_i, pt_init_vals, 'bo',linestyle = '--', label= 'Window Size: ' + str(val) +'Pre Train', color=color_val)
		#plt.plot(lags_i, chance_vals, 'bo',linestyle = ':', 	label='Window Size: ' + str(val) + 'Shuffled Pred to True', color = color_val)				
	plt.plot(lags_i, i_to_t_vals, 'bo', linestyle = '-.', label = 'Window Size: ' + str(val) + 'AutoCorrelation', color=color_val)
	plt.title('Prediction to True Over Diff Window Sizes')
	plt.xlabel('Lags')
	plt.ylabel('Pearson R')
	plt.legend(bbox_to_anchor=(1.001, 1),loc=2)
	plt.savefig(save_dir + str(file_id) + 'Same_E_Diff_Window' + str(window) + '_Plot.png')
	plt.xlim([0,128])
	plt.savefig(save_dir + str(file_id) + 'Same_E_Diff_Window_' + str(window) + 'Zoom_Plot.png')
def over_electrodes_same_window(lags_i, electrodes_i, window, file_path):
	plt.figure(figsize = (22,10))
	for val in electrodes_i:
		over_lags = []
		over_lags_stde = []
		#ip_init_vals = []
		#pt_init_vals = []
		#chance_vals = []
		i_to_t_vals = []
		it_stde = []
		for lag in lags_i:		
			fp = file_path.split('-')
			file_id = fp[0]  + '-e' + str(val) + '-w' + str(window) + '-'  + fp[3] + '-'  + fp[4]  + '-l' + str(lag)
			save_dir = './Results/' + file_id + '/'     #new models			
			pt_file = open(save_dir + file_id + '_pt_vals.txt','r') 
			over_lags.append(float([item for item in pt_file][0]))
			pt_file.close()
			pt_stde_file = open(save_dir + file_id + '_pt_stde.txt','r')
			over_lags_stde.append(float([item for item in pt_stde_file][0]))
			pt_stde_file.close()
			#ip_init_val = np.load(save_dir + file_id + 'i_to_p_init_val.npy').reshape(1,)[0]
			#ip_init_vals.append(ip_init_val)
			#pt_init_val = np.load(save_dir + file_id + 'p_to_t_init_val.npy').reshape(1,)[0]
			#pt_init_vals.append(pt_init_val)
			#chance_file = open(save_dir + file_id + '_chance_val.txt', 'r')
			#chance_vals.append(float([item for item in chance_file][0]))
			#chance_file.close()
			
			#autocorrelation
			it_file = open(save_dir + file_id + '_it_vals.txt','r') 
			i_to_t_vals.append(float([item for item in it_file][0]))
			it_file.close()
			it_stde_file = open(save_dir + file_id + '_it_stde.txt','r')
			it_stde.append(float([item for item in it_stde_file][0]))
			it_stde_file.close()

		#print(ip_init_vals)
		color_val = tuple(np.random.random_sample((3,)))				
		plt.plot(lags_i, over_lags,'bo',linestyle = '-',label = 'Electrode: ' + str(val) + 'P to T', color = color_val)
		plt.fill_between(lags_i, np.asarray(over_lags)-np.asarray(over_lags_stde), np.asarray(over_lags) + np.asarray(over_lags_stde), alpha = 0.1, color = color_val)
		#plt.plot(lags_i, ip_init_vals,linestyle = '--', label= 'Window Size: ' + str(val) +'Pre Train', color=color_val)	
		#plt.plot(lags_i, pt_init_vals, 'bo',linestyle = '--', label= 'Electrode: ' + str(val) +'Pre Train', color=color_val)
		#plt.plot(lags_i, chance_vals, 'bo',linestyle = ':', 	label='Electrode: ' + str(val) + 'Shuffled Pred to True', color = color_val)				
		plt.plot(lags_i, i_to_t_vals, 'bo', linestyle = '-.', label = 'Window Size: ' + str(val) + 'AutoCorrelation', color=color_val)
		plt.fill_between(lags_i, np.asarray(i_to_t_vals) - np.asarray(it_stde), np.asarray(i_to_t_vals) + np.asarray(it_stde), alpha = 0.1, color=color_val)
	plt.title('Prediction to True Over Diff Electrodes with Window Size: ' + str(window))
	plt.xlabel('Lags')
	plt.ylabel('Pearson R')
	plt.legend(bbox_to_anchor=(1.001, 1),loc=2)
	plt.savefig(save_dir + str(file_id) + 'Diff_Electrodes_Same_Window' + str(window) + '_Plot.png')
	plt.xlim([0, 257])
	plt.savefig(save_dir + str(file_id) + 'Diff_Electrodes_Same_Window' + str(window) + '_Zoom_Plot.png')		
	plt.xlim([0, 128])
	plt.savefig(save_dir + str(file_id) + 'Diff_Electrodes_Same_Window' + str(window) + '_Zoom2_Plot.png')
	plt.xlim([0, 32])
	plt.savefig(save_dir + str(file_id) + 'Diff_Electrodes_Same_Window' + str(window) + '_Zoom3_Plot.png')
def over_electrodes_same_lag(windows_i, electrodes_i, lag, file_path):
	plt.figure(figsize = (22,10))
	for electrode in electrodes_i:
		over_windows = []
		#ip_init_vals = []
		#pt_init_vals = []
		#chance_vals = []
		i_to_t_vals = []
		for window in windows_i:		
			fp = file_path.split('-')
			file_id = fp[0]  + '-e' + str(electrode) + '-w'  + str(window) + '-'  + fp[3] + '-'  + fp[4]  + '-l' + str(lag)
			save_dir = './Results/' + file_id + '/'     #new models			
			pt_file = open(save_dir + file_id + '_pt_vals.txt','r') 
			over_windows.append(float([val for val in pt_file][0]))
			pt_file.close()
			#ip_init_val = np.load(save_dir + file_id + 'i_to_p_init_val.npy').reshape(1,)[0]
			#ip_init_vals.append(ip_init_val)
			#pt_init_val = np.load(save_dir + file_id + 'p_to_t_init_val.npy').reshape(1,)[0]
			#pt_init_vals.append(pt_init_val)
			#chance_file = open(save_dir + file_id + '_chance_val.txt', 'r')
			#chance_vals.append(float([val for val in chance_file][0]))
			#chance_file.close()
			
			#autocorrelation
			it_file = open(save_dir + file_id + '_it_vals.txt','r') 
			i_to_t_vals.append(float([item for item in it_file][0]))
		

			
		#print(ip_init_vals)
		color_val = tuple(np.random.random_sample((3,)))				
		plt.plot(windows_i, over_windows,'bo', linestyle = '-',label = 'Electrode: ' + str(electrode) + 'P to T', color = color_val)
		#plt.plot(lags_i, ip_init_vals,linestyle = '--', label= 'Window Size: ' + str(val) +'Pre Train', color=color_val)	
		#plt.plot(windows_i, pt_init_vals, 'bo',linestyle = '--', label= 'Electrode: ' + str(electrode) +'Pre Train', color=color_val)
		#plt.plot(windows_i, chance_vals, 'bo',linestyle = ':',label='Electrode: ' + str(electrode) + 'Shuffled Pred to True', color = color_val)				
		plt.plot(windows_i, i_to_t_vals, 'bo', linestyle = '-.', label = 'Window Size: ' + str(electrode) + 'AutoCorrelation', color=color_val)

	plt.title('Prediction to True Over Diff Electrodes with Lag: ' + str(lag))
	plt.xlabel('Windows')
	plt.ylabel('Pearson R')
	plt.legend(bbox_to_anchor=(1.001, 1),loc=2)
	plt.savefig(save_dir + str(file_id) + 'Diff_Electrodes_Same_Lag_' + str(lag) + '_Plot.png')
		
def over_diff_num_convs(convos_i, window, lags_i, electrode, file_path):
	plt.figure(figsize=(22,10))
	for num_convo in convos_i:
		pt_vals = []
		it_vals = []
		for lag in lags_i:
			fp = file_path.split('-')
			file_id = fp[0] + '-e' + str(electrode) + '-w' + str(window) + '-n' + str(num_convo) + '-' + fp[4] + '-l' + str(lag)
			save_dir = './Results/' + file_id + '/'
			pt_file = open(save_dir + file_id + '_pt_vals.txt','r')
			pt_vals.append(float([val for val in pt_file][0]))
			pt_file.close()
			it_file = open(save_dir + file_id + '_it_vals.txt','r')
			it_vals.append(float([item for item in it_file][0]))

		color_val = tuple(np.random.random_sample((3,)))
		plt.plot(lags_i, pt_vals, 'bo', linestyle='-', label='Num Convs: ' + str(num_convo) + 'PT', color=color_val)
		plt.plot(lags_i, it_vals, 'bo', linestyle-'-.', label = 'Num Convs: ' + str(num_convo) + 'Auto Correlation', color=color_val)

	plt.title('Prediction to True and Auto Correlation over diff # Convos for training, Diff Lags with Window ' + str(window) + 'E' + str(electrode))
	plt.xlabel('Lags')
	plt.ylabel('Pearson R')
	plt.legend(bbox_to_anchor=(1.001,1),loc=2)
	plt.savefig(save_dir + file_id + 'Diff_Num_Conv_Diff_Lag_Same_Window' + str(window) + '_plot.png')

def get_conv_pearson_r(model, data_gen, m_type, test_e):
	print(m_type)
	out_len = 1
	in_len = data_gen.in_len
	num_steps = data_gen.num_steps
	batch_size = data_gen.batch_size
	electrodes = data_gen.electrodes
	num_electrodes = len(electrodes)
	test_electrode = test_e     
	electrode_dict = {}
	for idx in range(len(electrodes)):
		electrode_dict[int(electrodes[idx])] = int(idx)

	if m_type == 'tcn_d':
		conv_sizes = data_gen.conv_indices
		left_over_samps = ([],[])
		next_batch_index = 0
		i_to_p = []
		i_to_t = []
		p_to_t = []
		for conv_size in conv_sizes:
			inputs = []
			true_outputs = []
			if conv_size > len(left_over_samps[0]):
				inputs.extend(list(left_over_samps[0]))
				true_outputs.extend(list(left_over_samps[1]))
			
				new_conv_size = conv_size - len(left_over_samps[0])
				num_batches = int(new_conv_size/batch_size)
				num_extra_samps = new_conv_size%batch_size
	
				for i in range(num_batches):
					inputs.extend(list(data_gen[i+ next_batch_index][0]))
					true_outputs.extend(list(data_gen[i + next_batch_index][1]))
				
				inputs.extend(list(data_gen[num_batches+ next_batch_index][0][0:num_extra_samps]))
				true_outputs.extend(list(data_gen[num_batches + next_batch_index][1][0:num_extra_samps]))
				
				left_over_samps = (data_gen[num_batches + next_batch_index][0][num_extra_samps:],data_gen[num_batches + next_batch_index][1][num_extra_samps:])
				
					
				next_batch_index = num_batches + 1

			elif conv_size == len(left_over_samps[0]):
				inputs.extend(list(left_over_samps[0]))
				true_outputs.extend(list(left_over_samps[1]))
				left_over_samps = ([],[])
			else:	
				inputs.extend(list(left_over_samps[0][0:len(left_over_samps[0])-conv_size]))
				true_outputs.extend(list(left_over_samps[1][0:len(left_over_samps[0])-conv_size]))
				left_over_samps = (left_over_samps[0][len(left_over_samps[0])-conv_size:], left_over_samps[len(left_over_samps[0])-conv_size])
			inputs = np.asarray(inputs)
			true_outputs = np.asarray(true_outputs)
			print(inputs.shape)
			print(true_outputs.shape)	
			preds = model.predict_on_batch(inputs)
			input_pts = inputs[:,-1, electrode_dict[test_electrode]]
			pred_pts = preds[:,electrode_dict[test_electrode]]
			true_pts = true_outputs[:,electrode_dict[test_electrode]].reshape(true_outputs.shape[0],)
			
			i_to_p.append(pearsonr(input_pts, pred_pts)[0])
			i_to_t.append(pearsonr(input_pts, true_pts)[0])
			p_to_t.append(pearsonr(pred_pts, true_pts)[0])
					
		ip_mean = np.mean(i_to_p)
		ip_stde = np.std(i_to_p)/(np.sqrt(len(i_to_p)))
		it_mean = np.mean(i_to_t)
		it_stde = np.std(i_to_t)/(np.sqrt(len(i_to_t)))
		pt_mean = np.mean(p_to_t)
		pt_stde = np.std(p_to_t)/(np.sqrt(len(p_to_t)))
	
	return ip_mean, it_mean, pt_mean, ip_stde, it_stde, pt_stde	
		
def get_actual_signals(model, data_gen, file_id):	
	num_batches = data_gen.num_steps
	input_pts = []
	true_pts = []
	pred_pts = []
	for i in range(num_batches):
		inputs = data_gen[i][0]
		true_pts.extend(data_gen[i][1])
		pred_pts.extend(model.predict_on_batch(inputs))
		input_pts.extend(inputs)

	input_pts = np.asarray(input_pts)[:,-1,0]
	pred_pts = np.asarray(pred_pts)[:,0]
	true_pts = np.asarray(true_pts)[:,0]
	true_pts = true_pts.reshape(true_pts.shape[0],)
	
	return input_pts, true_pts, pred_pts

def plot_actual_signals(file_id):
	save_dir = './Results/' + file_id + '/'
	true_signal = np.load(save_dir + file_id + 'true_signal.npy')
	pred_signal = np.load(save_dir + file_id + 'pred_signal.npy')
	input_signal = np.load(save_dir + file_id + 'input_signal.npy')
	plt.figure()
	num_pts= len(true_signal)
	plt.plot(np.arange(num_pts),true_signal, label='True Output')
	plt.plot(np.arange(num_pts), pred_signal, label='Predicted Output')
	plt.plot(np.arange(num_pts), input_signal, label ='Input')
	plt.legend()
	plt.savefig(save_dir + file_id +  '_True_and_Pred_Signals.png')
	plt.xlim([0,500])
	plt.savefig(save_dir + file_id + '_True_and_Pred_Signals_Zoomed.png')
	plt.xlim([0, 250])
	plt.savefig(save_dir + file_id +  '_True_and_Pred_Signals_Zoomed2.png')
	plt.xlim([0, 125])
	plt.savefig(save_dir + file_id  + '_True_and_Pred_Signals_Zoomed4.png')
	plt.close()

	
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--min_test_frac', type=float, default=0)
	parser.add_argument('--max_test_frac', type=float, default=1)
	parser.add_argument('--test_electrodes', nargs= '*', default=[28])		#depends on training
	parser.add_argument('--electrodes', nargs= '*', default=[28])			#depends on training
	parser.add_argument('--input_length', nargs='*', default=[20])				#depends on training
	parser.add_argument('--output_length', type=int, default=20)			#depends on training
	parser.add_argument('--lags', nargs= '*', default=[0, 10, 20, 30, 40])
	parser.add_argument('--learning_rates', nargs='*', default=[.01])
	parser.add_argument('--num_points', type=int, default=250)
	parser.add_argument('--shuffle_test', type=bool, default=False)
	parser.add_argument('--shuffle_val', type=bool, default=False)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--model_name', type=str, default='Current_Model')
	parser.add_argument('--sampling_multiple', type=int, default=1)			#depends on training
	parser.add_argument('--num_conversations', nargs='*', default=[5])
	parser.add_argument('--patient', type=int, default=625)
	parser.add_argument('--min_val_frac', type=float, default=0.8)
	parser.add_argument('--max_val_frac', type=float, default=0.9)
	parser.add_argument('--model_type', type = str, default = 'TCN')
	parser.add_argument('--num_bins', type = int, default=8)
	args = parser.parse_args()
	print(args)

	shuffle_val = args.shuffle_val
	model_name = args.model_name
	test_electrodes = list(map(int, args.test_electrodes))
	num_points = args.num_points
	min_index_test = args.min_test_frac
	max_index_test = args.max_test_frac
	electrodes = list(map(int, args.electrodes))
	num_electrodes = len(electrodes)
	in_len = list(map(int, args.input_length))
	out_len = args.output_length

	model_type = args.model_type

	min_index_val = args.min_val_frac
	max_index_val = args.max_val_frac
	num_bins = args.num_bins
	lags = list(map(int, args.lags))

	sampling_multiple = args.sampling_multiple
	shuffle_test = args.shuffle_test
	batch_size = args.batch_size
	num_conversations = args.num_conversations
	patient=args.patient
	learning_rates = list(map(float, args.learning_rates))

	#regression
	#file_id = str(model_name)  + '_' + str(learning_rates[0]) + '_' + str(num_conversations[0])
	#classification
	#file_id = str(model_name) + '_' + str(learning_rates[0]) + '_' + str(num_conversations[0]) + '_' +  str(num_bins)
	#file_id = str(model_name)
	#print(file_id)
	convs = [1,6]
	for electrode in electrodes:
		fp = str(model_name).split('-') 
		#print(fp)
		file_id = fp[0]  + '-e' + '-w'  + '-'  + fp[3] + '-'  + fp[4]  + '-l'
		for window in in_len:
			#over_diff_num_convos(convs, window, lags, electrode, file_id) 	
			over_lag_R_graph(lags, window,electrode,  file_id, 'val', model_type) #make model type class if doing classification. 	
		for lag in lags:
			over_window_R_graph(in_len,lag, electrode, file_id, 'val', model_type)
		over_lag_many_windows_graph(lags, in_len, electrode, file_id)
	for window in in_len:
		#over_electrodes_same_window(lags, electrodes, window, file_id)
		over_electrodes_same_window(lags, [13], window, file_id)
	for lag in lags:
		over_electrodes_same_lag(in_len, electrodes, lag, file_id)

	for electrode in electrodes:
		for lag in lags:
			for window in in_len:
				fp = str(model_name).split('-')
				file_id = fp[0] + '-e' + str(electrode) + '-w' + str(window) + '-n6' + '-r0.001' + '-l' + str(lag)
				plot_actual_signals(file_id)					
