import sys
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Flatten, Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from keras.models import Sequential
import six

def conv_batch_relu_3D(**conv_params):
    """Helper to build the 3D convolution -> Batch Normalization -> relu activation module
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    padding = conv_params.setdefault("padding", 'same')
    dropout_rate = conv_params.setdefault("dropout_rate", None)
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    initializer = conv_params.setdefault("kernel_initializer", 'glorot_uniform')

    def f(input_layer):
        output_layer = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides,
                              padding=padding, kernel_initializer=initializer,
                              kernel_regularizer=kernel_regularizer)(input_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('relu')(output_layer)
        if dropout_rate:
            output_layer = Dropout(dropout_rate)(output_layer)
        return output_layer

    return f


def vgg_module_3D(block_function, filters, kernel_size, strides, pool_size, kernel_regularizer=None):
    """Helper to build a VGG like module: conv -> conv -> average_pooling
       block_function: e.g. = conv_batch_relu_3D
    """

    block_function = _get_block_function(block_function)

    def f(input_layer):
        output_layer = block_function(filters=filters, kernel_size=kernel_size, strides=strides,
                                      kernel_regularizer=kernel_regularizer)(input_layer)
        output_layer = block_function(filters=filters, kernel_size=kernel_size, strides=strides,
                                      kernel_regularizer=kernel_regularizer)(output_layer)
        output_layer = MaxPooling3D(pool_size=pool_size)(output_layer)

        return output_layer

    return f

def _bn_relu(input_layer):
    """Helper to build a BN -> relu block
       the activation function could either be relu or prelu
    """
    norm = BatchNormalization()(input_layer)
    return Activation("relu")(norm)


def _bn_prelu(input_layer):
    """Helper to build a BN -> prelu block
       the activation function could either be relu or prelu
    """
    norm = BatchNormalization()(input_layer)
    return PReLU()(norm)


def dense_bn_activation(**dense_params):
    """Helper to build a Dense -> BN > activation block
     This is a block function for dense layer
    """
    units = dense_params["units"]
    kernel_initializer = dense_params.setdefault("kernel_initializer", "glorot_uniform")
    kernel_regularizer = dense_params.setdefault("kernel_regularizer", None)
    activation = dense_params.setdefault("activation", "relu")
    dropout_rate = dense_params.setdefault("dropout_rate", None)

    def f(input_layer):
        dense_1 = Dense(units=units, kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer)(input_layer)
        if activation == "relu":
            output_layer = _bn_relu(dense_1)
        else:
            output_layer = _bn_prelu(dense_1)

        if dropout_rate:
            output_layer = Dropout(dropout_rate)(output_layer)
        return output_layer

    return f


def dense_selu(**dense_params):
    """Helper to build a Dense -> selu block
       Self_Normalization
       This is a block function for dense layer
    """

    units = dense_params["units"]

    kernel_initializer = dense_params.setdefault("kernel_initializer", "he_normal")
    kernel_regularizer = dense_params.setdefault("kernel_regularizer", None)
    dropout_rate = dense_params.setdefault("dropout_rate", None)

    def f(input_layer):
        dense_1 = Dense(units=units, kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer)(input_layer)
        output_layer = Activation("selu")(dense_1)
        if dropout_rate:
            output_layer = Dropout(dropout_rate)(output_layer)
        return output_layer
    return f


def task_module(block_function, task_name, num_dense_units, dropout_rate, task_mod, init='glorot_uniform',
                activation='relu', num_of_class=2):
    """Helper to build a task module, start from the last second layer till the final output
       This module only has two layers, last second layer and final task layer. This design is to
       make the just connection easier

    """
    block_function = _get_block_function(block_function)

    def f(input_layer):
        # all layers of the task module, except the first and last layers

        if activation == "prelu":
            output_layer = block_function(units=num_dense_units, dropout_rate=dropout_rate, kernel_initializer=init,
                                              activation=activation)(input_layer)
        else:
            output_layer = block_function(units=num_dense_units, dropout_rate=dropout_rate, kernel_initializer=init)(input_layer)

            # the last (output) layer for current task

        if task_mod == 'regression':
            output_layer = Dense(units=1, kernel_initializer=init, name=task_name)(output_layer)
        else: # 'classification'
            output_layer = Dense(units=num_of_class, activation='softmax', kernel_initializer=init, name=task_name)(output_layer)
        return output_layer

    return f


def _get_block_function(identifier):
    """Helper to get block function from string if needed
    """
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier






class CNN_HSCNN_builder(object):
    """Helper class to build HSCNN model
    """

    @staticmethod
    def build_no_direct_connection(input_shape,
                                   conv_block_function,
                                   conv_module_function,
                                   num_conv_module,
                                   filters_list,
                                   kernel_size_list,
                                   strides_list,
                                   pool_size_list,
                                   task_mod,
                                   task_module,
                                   major_task_name,
                                   sub_task_name_list,
                                   task_weights_list,
                                   drop_out_task_base_list,
                                   fully_connected_block_function,
                                   num_middle_layers_task_module,
                                   num_dense_units_task_base_list,
                                   num_dense_units_subtask_module,
                                   num_dense_units_major_task_module,
                                   dropout_rate_task_module,
                                   dropout_rate_major_task_module,
                                   num_subtasks,
                                   init='glorot_uniform',
                                   activation='relu',
                                   num_of_class=2,
                                   kernel_regularizer=None,
                                   droput_rate_flatten=None):

        """
        the general builder for CNN
        :param input_shape: the shape of the input
        :param conv_block_function: the convlution unit used, e.g. conv_batch_relu_3D
        :param conv_module_function: the module unit used, e.g. vgg_module_3D
        :param num_conv_module: the number of convolutional modules
        :param filters_list: a list storing the number of filers for the convolution for each module
        :param kernel_size_list: a list storing the kernel size for the convolution for each module
        :param strides_list: a list storing the strides for the convoluion for each module
        :param pool_size_list: a list storing the pooling size for each module
        :param task_mod: either 'regression' or 'classification'
        :param task_module: the module function for task, e.g. task_module
        :param major_task_name: name for major task, e.g. 'maligancy'
        :param sub_task_name_list: name list for all the sub task
        :param task_weights_list: list of weights for each task
        :param drop_out_task_base_list: list of dropout rates for each layers before the last second layer
        :param fully_connected_block_function: block unit defined as the basic building blocks for task module and task
        base layers
        :param num_middle_layers_task_module: number of base layers for each task before the last second layer
        :param num_dense_units_task_base_list: list of number of neurons for each base layers before the last second
        layer, and this will apply to every task
        :param num_dense_units_subtask_module: number of nerons for the last second task layer for each sub_task,
        the same for every subtask
        :param num_dense_units_major_task_module: number of nerons for the last second major task layer
        :param dropout_rate_task_module: the dropout rate for the task
        :param init: initializer for all layers
        :param activation: activation function for all layers
        :param num_of_class: number of classes for classification mode
        :num_subtasks: number of subtasks
        :param kernel_regularizer: regularization for both conv and fully-connected layers
        :param droput_rate_flatten: droput rate for the dropout connecting the fallten layer to the first fully connected layer
        :return: Keras model
        """
        conv_block_function = _get_block_function(conv_block_function)
        conv_module_function = _get_block_function(conv_module_function)
        fully_connected_block_function = _get_block_function(fully_connected_block_function)
        task_module = _get_block_function(task_module)

        first_input_layer = Input(input_shape)

        conv_layer_output = first_input_layer
        # build all the convolution layers before connected to fully connected layer
        for i in range(num_conv_module):
            conv_layer_output = conv_module_function(block_function=conv_block_function, filters=filters_list[i],
                                                     kernel_size=kernel_size_list[i], strides=strides_list[i],
                                                     pool_size=pool_size_list[i],
                                                     kernel_regularizer=kernel_regularizer)(conv_layer_output)
        # build all the fully connected layers
        flattened_output = Flatten()(conv_layer_output)
        flattened_output = Dropout(droput_rate_flatten)(flattened_output)

        final_output_list = []
        task_base_layer_list = []
        for i in range(num_subtasks):
            task_name = sub_task_name_list[i]
            task_base_layer = flattened_output
            for j in range(num_middle_layers_task_module):
                task_base_layer = fully_connected_block_function(units=num_dense_units_task_base_list[j],
                                                                 dropout_rate=drop_out_task_base_list[j],
                                                                 kernel_regularizer=kernel_regularizer,
                                                                 activation=activation)(task_base_layer)
            task_base_layer_list.append(task_base_layer)
            output_subtask = task_module(block_function=fully_connected_block_function,
                                         task_name=task_name,
                                         num_dense_units=num_dense_units_subtask_module,
                                         dropout_rate=dropout_rate_task_module,
                                         task_mod=task_mod,
                                         init=init,
                                         activation=activation,
                                         num_of_class=2)(task_base_layer)
            final_output_list.append(output_subtask)
        
        
        task_base_layer_list.append(flattened_output)
        ##########build the major task module

        # build the merged input for the major task module
        merged_input_major_task = Concatenate()(task_base_layer_list)
        major_task_output = task_module(block_function=fully_connected_block_function,
                                         task_name=major_task_name,
                                         num_dense_units=num_dense_units_major_task_module,
                                         dropout_rate=dropout_rate_major_task_module,
                                         task_mod=task_mod,
                                         init=init,
                                         activation=activation,
                                         num_of_class=2)(merged_input_major_task)
        final_output_list.append(major_task_output)

        # define outputs and inputs
        model = Model(inputs=first_input_layer, outputs=final_output_list)
        adam = Adam(lr=0.00003)
        # build the last output layer
        if task_mod == 'regression':
            model.compile(loss='mse', optimizer=adam, metrics=['mse'], loss_weights=task_weights_list)
        else:
            model.compile(loss='binary_crossentropy', optimizer= adam, metrics=['accuracy'], loss_weights=task_weights_list)
        return model


    @staticmethod
    def build_multi_label_classification(input_shape):
        return CNN_multi_task_builder.build_no_direct_connection(input_shape=input_shape,
                                   conv_block_function='conv_batch_relu_3D',
                                   conv_module_function='vgg_module_3D',
                                   num_conv_module=2,
                                   filters_list=[16, 32],
                                   kernel_size_list=[(3 , 3, 3), (3, 3, 3)],
                                   strides_list=[(1, 1, 1), (1, 1, 1)],
                                   pool_size_list=[(2, 2, 2), (2, 2, 2)],
                                   task_mod='classification',
                                   task_module='task_module',
                                   major_task_name='main_malignancy_output',
                                   sub_task_name_list=['calcification_output', 'margin_output',
                                                       'sphericity_output', 'subtlety_output',
                                                       'texture_output'],
                                   task_weights_list=[0.1, 0.1, 0.2, 0.2, 0.1, 0.8],
                                   drop_out_task_base_list=[0.6],
                                   fully_connected_block_function='dense_bn_activation',
                                   num_middle_layers_task_module=1,
                                   num_dense_units_task_base_list=[256],
                                   num_dense_units_subtask_module=64,
                                   num_dense_units_major_task_module=256,
                                   dropout_rate_task_module=0.6,
                                   num_subtasks=5,
                                   init='glorot_uniform',
                                   activation='relu',
                                   num_of_class=2,
                                   kernel_regularizer=None,
                                   droput_rate_flatten=None)

    