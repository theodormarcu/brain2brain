#!/usr/bin/env python3
# Theodor Marcu
# tmarcu@princeton.edu
# Created on 04/23/2020
# Computer Science Senior Thesis


import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam



def define_enc_dec_model(n_input: int, n_output: int, n_units: int):
    '''
    Args:
        n_input (int): Number of input features (electrodes) for each timestep.
        n_output (int): Number of output features (electrodes) for each timestep.
        n_units (int): Number of cells to create in the encoder and decoder models. E.g.
                      128 or 256.
    Returns three models: train, inference_encoder, and inference_decoder.

    Modeled after: https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
    '''
    # Define Training Encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # Define Training Decoder 
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    # Finish Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

def predict_sequence(infenc, infdec, input_sequence, n_steps, cardinality):
    '''
    Encoder-decoder framework for inference and validation.
    Args:
    Returns the inferred output.
    '''
    # encode
    state = infenc.predict(input_sequence)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
	    # update target sequence
        target_seq = yhat
    return np.array(output)

def define_enc_dec_model_jeddy_o2o(latent_dim: int, dropout: float):
    '''
    Args:
        latent_dim (int): Number of hidden layers.
        dropout (float): Dropout for LSTM layers.
    Returns three models: train, inference_encoder, and inference_decoder.

    Modeled after: https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Intro.ipynb
    '''
    # Define an input series and encode it with an LSTM. 
    encoder_inputs = Input(shape=(None, 1)) 
    encoder = LSTM(latent_dim, dropout=dropout, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the final states. These represent the "context"
    # vector that we use as the basis for decoding.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    # This is where teacher forcing inputs are fed in.
    decoder_inputs = Input(shape=(None, 1)) 

    # We set up our decoder using `encoder_states` as initial state.  
    # We return full output sequences and return internal states as well. 
    # We don't use the return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, dropout=dropout, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)

    decoder_dense = Dense(1) # 1 continuous output at each timestep
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # from our previous model - mapping encoder sequence to state vectors
    encoder_model = Model(encoder_inputs, encoder_states)

    # A modified version of the decoding stage that takes in predicted target inputs
    # and encoded state vectors, returning predicted target outputs and decoder state vectors.
    # We need to hang onto these state vectors to run the next step of the inference loop.
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                        [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model

def decode_sequence_o2o(encoder_model: Model,
                        decoder_model: Model,
                        input_seq: np.array,
                        pred_steps: int):
    '''
    Decoder sequnece loop for inference (one-to-one).
    Args:
        encoder_model:
        decoder_model:
        input_seq:
        pred_steps:

    Returns predicted sequence.
    '''
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 1))
    
    # Populate the first target sequence with end of encoding series pageviews
    target_seq[0, 0, 0] = input_seq[0, -1, 0]

    # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
    # (to simplify, here we assume a batch of size 1).
    decoded_seq = np.zeros((1,pred_steps,1))
    
    for i in range(pred_steps):
        
        output, h, c = decoder_model.predict([target_seq] + states_value)
        
        decoded_seq[0,i,0] = output[0,0,0]

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, 0] = output[0,0,0]

        # Update states
        states_value = [h, c]

    return decoded_seq

def define_baseline_nn_model_o2o(latent_dim: int, input_len: int, output_dim: int):
    '''
    Simple baseline nn model to compare results.
    Args:
        latent_dim (int): Hidden inputs.
        output_dim (int): Number of timesteps to predict.
    '''
    model = Sequential()
    model.add(layers.Dense(latent_dim, input_shape=(input_len, 1)))
    model.add(Flatten())
    model.add(layers.Dense(output_dim))
    return model

def predict_sequence_o2o(input_sequence, pred_steps, model):
    history_sequence = input_sequence.copy()
    pred_sequence = np.zeros((1,pred_steps,1)) # initialize output (pred_steps time steps)  
    
    for i in range(pred_steps):
        
        # record next time step prediction (last time step of model output) 
        last_step_pred = model.predict(history_sequence)[0,-1,0]
        pred_sequence[0,i,0] = last_step_pred
        
        # add the next time step prediction to the history sequence
        history_sequence = np.concatenate([history_sequence, 
                                           last_step_pred.reshape(-1,1,1)], axis=1)

    return pred_sequence