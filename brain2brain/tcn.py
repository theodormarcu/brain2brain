# Theodor Marcu
# tmarcu@princeton.edu
# Created on 02/23/2020
# Computer Science Senior Thesis
#
# Inspired/Adapted from https://github.com/philipperemy/keras-tcn
#
# The code is based on the TCN model described in https://arxiv.org/abs/1803.01271
# "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
# by Bai et al.


import inspect
from typing import List

from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization

def is_power_of_two(num):
    return num != 0 and ((num & (num - 1)) == 0)

def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations

class ResidualBlock(Layer):
    def __init__(self, 
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str='relu',
                 dropout_rate: float=0.0,
                 kernel_initializer: str='he_normal',
                 use_batch_norm: bool=False,
                 use_layer_norm: bool=False,
                 last_block: bool=True,
                 **kwargs):

        """
        Defines the residual block for the WaveNet TCN.

        Args:
            x: The previous layer in the model.
            training (bool): Should the layer be in training or inference mode.
            dilation_rate (int): The dilation power of 2 we are using for this
                                residual block.
            nb_filters (int): The number of convolutional filters to use in
                             this block.
            kernel_size (int): The size of the convolutional kernel.
            padding (str): The padding used in convolutional layers, "same"
                          or "causal".
            activation (str): The final activation used in o = Activation(x+F(x))
            dropout_rate (float): Between 0 and 1. Fraction of the input units to drop.
            kernel_initializer (str): Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm (bool): Whether to use batch normalization in the residual layers
                                  or not.
            use_layer_norm (bool): Whether to use layer normalization in the residual layers
                                  or not.
            kwargs: Any initializers for Layer class.
        """
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.last_block = last_block
        self.layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """
        Helper function for building layer.

        Args:
            layer: Appens layer to internal layer list and builds it based
                    on the current output shape of ResidualBlock. Updates current
                    output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):
        """
        TODO: Add docstring.
        """
        # name scope used to make sure weights get unique names
        with K.name_scope(self.name):
            self.layers=[]
            self.res_output_shape = input_shape

            # Add two layers, like in the paper An Empirical Evaluation...
            for k in range(2):
                # Add a Dilated Causal Convolution Layer
                name = "conv1D_{}".format(k)
                with K.name_scope(name):
                    self._add_and_activate_layer(Conv1D(filters=self.nb_filters,
                                                        kernel_size=self.kernel_size,
                                                        dilation_rate=self.dilation_rate,
                                                        padding=self.padding,
                                                        name=name,
                                                        kernel_initializer=self.kernel_initializer))
                # Add a Weight Normalization Layer
                if self.use_batch_norm:
                    self._add_and_activate_layer(BatchNormalization())
                elif self.use_layer_norm:
                    self._add_and_activate_layer(LayerNormalization())

                # Add a ReLU Activation Layer
                self._add_and_activate_layer(Activation('relu'))
                # Add a Dropout Layer
                self._add_and_activate_layer(SpatialDropout1D(rate=self.dropout_rate))
            if not self.last_block:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'conv1D_{}'.format(k+1)
                with K.name_scope(name):
                    # make and build this layer separately because it 
                    # directly uses input_shape
                    self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                                    kernel_size=1,
                                                    padding='same',
                                                    name=name,
                                                    kernel_initializer=self.kernel_initializer)
            else:
                self.shape_match_conv=Lambda(lambda x : x, name='identity')
            
            self.shape_match_conv.build(input_shape)
            self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """
        Returns:
            A list where the first element is the residual model tensor,
            and the second is the skip connection tensor.
        """

        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            # TODO: What is this doing?
            training_flag = "training" in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)
        x2 = self.shape_match_conv(inputs)
        self.layers_outputs.append(x2)
        res_x = layers.add([x2, x])
        self.layers_outputs.append(res_x)

        res_act_x = self.final_activation(res_x)
        self.layers_outputs.append(res_act_x)
        return [res_act_x, x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]

class TCN(Layer):
    '''
    Creates a TCN Layer.

    Input shape:
        A tensor of shape (batch_size, timesteps, input_dim).
    Args:
        nb_filters (int): The number of filters to use in convolutional layers.
        kernel_size (int): The size of the kernel to use in each convolutional layer.
        dilations (list): The list of the dilations. 
                          Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks (int): The number of stacks for residual blocks to use.
        padding (str): The padding to use in convolutional layers, "causal" or "same".
        use_skip_connections (bool): If we want to add skip 
                                     connections from input to each residual block.
        return_sequences (bool): Whether to return the last output 
                                 in the output sequence, or the full sequence.
        activation (str): The activation used in the residual blocks o = Activation(x + F(x)).
        dropout_rate (float): Between 0 and 1. Fraction of the input units to drop.
        kernel_initializer (str): Initializer for the kernel weights matrix (Conv1D).
        use_batch_norm (bool): Whether to use batch normalization 
                               in the residual layers or not.
        kwargs: Any other arguments for configuring parent class layer. 
                For example "name=str", name of the model. Use
                unique names when using multiple TCNs.

    Returns:
        A TCN layer.
    '''

    def __init__(self,
                 nb_filters: int = 64,
                 kernel_size: int = 2,
                 dilations: list = [1, 2, 4, 8, 16, 32],
                 nb_stacks: int = 1,
                 padding = 'causal',
                 use_skip_connections = True,
                 return_sequences: bool = False,
                 activation: str = "linear",
                 dropout_rate: float = 0.0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 **kwargs):

        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.padding = padding
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.main_conv1D = None
        self.build_output_shape = None
        self.lambda_layer = None
        self.lambda_ouput_shape = None

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")
        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()
        
        # Initialize parent class with kwargs.
        super().__init__(**kwargs)

    @property
    def receptive_field(self):
        assert_msg = 'The receptive field formula works only with power of two dilations.'
        assert all([is_power_of_two(i) for i in self.dilations]), assert_msg
        return self.kernel_size * self.nb_stacks * self.dilations[-1]
    
    def build(self, input_shape):
        '''
        Builds the TCN Layer for the input_shape.

        Args:
            input_shape ():

        '''
        self.main_conv1D = Conv1D(filters=self.nb_filters,
                                  kernel_size=1,
                                  padding=self.padding,
                                  kernel_initializer=self.kernel_initializer)
        self.main_conv1D.build(input_shape)
        # Member to hold current output shape of the layer for building purposes.
        self.build_output_shape = self.main_conv1D.compute_output_shape(input_shape)

        # List to hold all the member Residual Blocks.
        self.residual_blocks = []
        # TODO: Why do this?
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1 # Cheap way to false case for below.

        for s in range (self.nb_stacks):
            for d in self.dilations:
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=self.nb_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation,
                                                          dropout_rate=self.dropout_rate,
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          kernel_initializer=self.kernel_initializer,
                                                          last_block=len(self.residual_blocks) + 1 == total_num_blocks,
                                                          name="residual_block_{}".format(len(self.residual_blocks))))
                # Build the newest residual block.
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape
        
        # Force Keras to add the layers to the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)
        
        # Author: @karolbadowski.
        # TODO: What does this mean?
        output_slice_index = int(self.build_output_shape.as_list()[1] / 2) if self.padding == 'same' else -1
        self.lambda_layer = Lambda(lambda tt: tt[:, output_slice_index, :])
        self.lambda_ouput_shape = self.lambda_layer.compute_output_shape(self.build_output_shape)
    
    def compute_output_shape(self, input_shape):
        '''
        Compute the output shape.
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        '''
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            return self.lambda_ouput_shape
        else:
            return self.build_output_shape


    def call(self, inputs, training=None):
        '''
        TODO: Add docstring.
        '''
        x = inputs
        self.layers_outputs = [x]
        try:
            x = self.main_conv1D(x)
            self.layers_outputs.append(x)
        except AttributeError:
            print('The backend of keras-tcn>2.8.3 has changed from keras to tensorflow.keras.')
            print('Either update your imports:\n- From "from keras.layers import <LayerName>" '
                  '\n- To "from tensorflow.keras.layers import <LayerName>"')
            print('Or downgrade to 2.8.3 by running "pip install keras-tcn==2.8.3"')
            import sys
            sys.exit(0)
        
        self.skip_connections = []
        for layer in self.residual_blocks:
            x, skip_out = layer(x, training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            x = layers.add(self.skip_connections)
            self.layers_outputs.append(x)
        if not self.return_sequences:
            x = self.lambda_layer(x)
            self.layers_outputs.append(x)
        return x

    def get_config(self):
        """
        Returns the config of the layer. This is used for saving and loading from a model
        :return: Python dict with specs to rebuild layer.
        """
        config = super(TCN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config
                
def compiled_tcn(num_feat: int,
                 num_classes: int,
                 nb_filters: int,
                 kernel_size: int,
                 dilations: int,
                 nb_stacks: int,
                 max_len: int,
                 output_len: int = 1,
                 padding: str ='causal',
                 use_skip_connections: bool = True,
                 return_sequences: bool = True,
                 regression: bool = False,
                 dropout_rate: float = 0.05,
                 name: str = 'tcn',
                 kernel_initializer: str = "he_normal",
                 activation: str = "linear",
                 opt: str = 'adam',
                 lr: float = 0.0001,
                 use_batch_norm = False,
                 use_layer_norm = False):
    # type(...) -> Model
    """
    Creates a compiled TCN model for a given task (i.e. regression or classification).
    Classification uses a sparse categorical loss. 
    Please input class ids and not one-hot encodings

    Args:
        num_feat (int): The number of features of your input, i.e. the last dimension
                       of: (batch_size, timesteps, input_dim).
        num_classes (int): The size of the final dense layer, how many classes are we
                          predicting.
        nb_filters (int): The number of filters to use in the convolutional layers.
        kernel_size (int): The size of the kernel to use in the convolutional layers.
        dilations (int): The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks (int): The number of stacks of residual blocks to use.
        max_len (int): The maximum sequence length. Use None if the sequence length is dynamic.
        padding (str): The padding to use in the convolutional layer.
        use_skip_connections (bool): If we want to add skip connections from the 
                                    input to each residual block.
        return_sequences (bool): Whether to return the last output in the output sequence,
                                or the full sequence.
        regression (bool): Whether the output should be continous or discrete.
        dropout_rate (float): Between 0 and 1. Fraction of the input units to drop.
        activation (str): The activation used in residual blocks o = Activation(x+F(x)).
        name (str): Name of the model. Useful when having multiple TCNs.
        kernel_initializer (str): Initializer for the kernel weights matrix (Conv1D).
        opt (str): Optimizer name.
        lr (float): Learning rate.
        use_batch_norm (bool): Whether to use batch normalization in the residual layers or not.
        use_layer_norm (bool): Whether to use layer normalization in the residual layers or not.
    Returns:
        A compiled Keras TCN.
    """

    dilations = adjust_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, kernel_initializer, use_batch_norm, 
            use_layer_norm, name=name)(input_layer)
    print("x.shape=", x.shape)

    def get_opt():
        if opt == "adam":
            return optimizers.Adam(lr=lr, clipnorm=1.0)
        elif opt == "rmsprop":
            return optimizers.RMSprop(lr=lr, clipnorm=1.0)
        else:
            raise Exception("Only Adam and RMSProp are available here.")
    
    if not regression:
        # Classification.
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

        model.compile(get_opt(), loss='sparse_categorical_crossentropy', metrics=[accuracy])
    else:
        # regression
        x = Dense(output_len)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(get_opt(), loss='mean_squared_error')
    print('model.x = {}'.format(input_layer.shape))
    print('model.y = {}'.format(output_layer.shape))
    return model
