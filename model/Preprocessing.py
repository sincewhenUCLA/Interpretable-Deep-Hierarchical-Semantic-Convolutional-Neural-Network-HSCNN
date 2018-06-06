from __future__ import print_function
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import backend as K
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from keras.models import load_model
import scipy.sparse as sp
import sys
import warnings

def save_2_pickle(input_data, output_dir, output_name):
    """Helper to save input data into pickle file (e.g. test_data.pkl)
    """
    output_file_name = os.path.join(output_dir, output_name)
    output = open('{0}'.format(output_file_name), 'wb')
    # Pickle the data
    pickle.dump(input_data, output)
    output.close()
    print('Data hase been saved to pickle file: {0}'.format(output_file_name))


def load_pickle(input_dir_name, input_file_name):
    """Helper to load data from pickle file
    """
    input_file_name = os.path.join(input_dir_name, input_file_name)
    pkl_file = open('{0}'.format(input_file_name), 'rb')
    data = pickle.load(pkl_file)
    #pprint.pprint(data)
    pkl_file.close()
    print('Data hase been loaded from pickle file: {0}'.format(input_file_name))
    return(data)

def load_all_pkl_only_label(data_load_folder, num_pkl, name_pkl):
    """ load only label from seris pickle files, e.g. train_data_1.pkl, train_data_2.pkl
    input:
         data_load_folder: pickle files location folder
         name_pkl: common name of all the pickle files, e.g. train_data
         num_pkl: total number of pickle files, which is used to infer the file names
    output:
         all_labels: list of all labels for each nodule; could be coverted into pandas dataframe by
         pd.DataFrame(all_labels)
    """
    all_labels = []
    for i in range(1, num_pkl + 1):
        file_name = name_pkl + '_' + str(i) + '.pkl'
        data_loaded = load_pickle(data_load_folder, file_name)
        loaded_values = data_loaded.values()
        loaded_labels = [x[0] for x in loaded_values]
        all_labels = all_labels + loaded_labels
    return(all_labels)


def load_all_pkl_label_image(data_load_folder, num_pkl, name_pkl, no_mask=True):
    """ load label and images from seris pickle files, e.g. train_data_1.pkl, train_data_2.pkl
    input:
         data_load_folder: pickle files location folder
         name_pkl: common name of all the pickle files, e.g. train_data
         num_pkl: total number of pickle files, which is used to infer the file names
    output:
         all_labels: list of all labels for each nodule; could be coverted into pandas dataframe by
         pd.DataFrame(all_labels)
         all_ims: 4D image array [n_sample, z, x, y]
    """
    all_labels = []
    all_ims = []
    all_mask = []
    for i in range(1, num_pkl + 1):
        file_name = name_pkl + '_' + str(i) + '.pkl'
        data_loaded = load_pickle(data_load_folder, file_name)
        loaded_values = data_loaded.values()
        loaded_labels = [x[0] for x in loaded_values]
        loaded_image =  [x[1] for x in loaded_values]
        if not no_mask:
            loaded_mask = [x[2] for x in loaded_values]
            all_mask = all_mask + loaded_mask
        all_labels = all_labels + loaded_labels
        all_ims = all_ims + loaded_image
    all_ims = np.stack(all_ims, axis=0)
    if no_mask:
        return(all_labels, all_ims)
    else:
        all_mask = np.stack(all_mask, axis=0)
        return(all_labels, all_ims, all_mask)


def get_binary_labels(label_list, t_dic):
    """ Take list of labels and output selected labels thresholded by the given threshold for each label
    Input:
        label_list: list of labels for each nodule
        t_dic: threshold dictionaries for each label, only labels contained in the key list will be selected
    Output:
        label_df: binay label dataframe
    """
    label_df = pd.DataFrame(label_list)
    selected_category = t_dic.keys()
    label_df = label_df[selected_category]
    for key, value in t_dic.iteritems():
        label_df[key] = np.where(label_df[key] < value, 0, 1)
    return(label_df)


def get_labels(label_list, label_name_list):
    """ Take list of labels and output selected labels thresholded by the given threshold for each label
    Input:
        label_list: list of labels for each nodule
        t_dic: threshold dictionaries for each label, only labels contained in the key list will be selected
    Output:
        label_df: binay label dataframe
    """
    label_df = pd.DataFrame(label_list)
    label_df = label_df[label_name_list].astype('float')
    return(label_df)


def normalize_image(array_in, upper_bound=500, lower_bound=-1000):
    """
    normalize images using the give
    :param array_in:
    :return: normalized images
    """
    array_in[array_in > upper_bound] = upper_bound
    array_in[array_in < lower_bound] = lower_bound
    array_in = (array_in - lower_bound) / (upper_bound - lower_bound)
    return array_in


def preprocess_image_label(data_load_folder,
                           img_file_name,
                           label_file_name,
                           label_name_list=None,
                           upper_bound=500,
                           lower_bound=-1000,
                           nb_classes=2,
                           if_label_catogerial=True):
    """
    convert data into the format the could directly be feed into keras model
    :param data_load_folder:
    :param img_file_name:
    :param label_file_name:
    :param upper_bound:
    :param lower_bound:
    :return:
    """
    image_data = load_pickle(data_load_folder, img_file_name)
    label_data = load_pickle(data_load_folder, label_file_name)

    # the loaded image data has this dimension [n_sample, z, x, y], it works for theano backend
    # If backend is tensorflow, it need to be converted to [n_sample, x, y, z]
    if K.image_dim_ordering() == 'tf':
        image_data = np.moveaxis(image_data, 1, 3)
    
    image_data = image_data.astype('float32')
    image_data = normalize_image(image_data, upper_bound, lower_bound)
    image_data = image_data.reshape(image_data.shape[0], image_data.shape[1], image_data.shape[2], image_data.shape[3], 1)   

    if not if_label_catogerial:
        if label_name_list:
            if len(label_name_list) == 1:
                new_label = label_data[label_name_list].values
            else:
                new_label = []
                for label_name in label_name_list:
                    this_label = label_data[label_name].values
                    new_label.append(this_label)
        else:
            new_label = label_data.values
    else:
        if label_name_list:
            new_label = []
            for label_name in label_name_list:
                this_label = np_utils.to_categorical(label_data[label_name].values, nb_classes)
                new_label.append(this_label)
        else:
            new_label = np_utils.to_categorical(label_data.values, nb_classes)

    print('{0} and {1} has been processed'.format(img_file_name, label_file_name))
    return(image_data, new_label)


def preprocess_image_only(data_load_folder,
                           img_file_name,
                           upper_bound=500,
                           lower_bound=-1000):
    """
    convert data into the format the could directly be feed into keras model
    :param data_load_folder:
    :param img_file_name:
    :param label_file_name:
    :param upper_bound:
    :param lower_bound:
    :return:
    """
    image_data = load_pickle(data_load_folder, img_file_name)

    # the loaded image data has this dimension [n_sample, z, x, y], it works for theano backend
    # If backend is tensorflow, it need to be converted to [n_sample, x, y, z]
    if K.image_dim_ordering() == 'tf':
        image_data = np.moveaxis(image_data, 1, 3)

    image_data = image_data.astype('float32')
    image_data = normalize_image(image_data, upper_bound, lower_bound)
    image_data = image_data.reshape(image_data.shape[0], image_data.shape[1], image_data.shape[2], image_data.shape[3], 1)



    print('{0} and has been processed'.format(img_file_name))
    return(image_data)

def combine_two_fold_image_label(img1, img2, label1, label2, is_muti_lable):
    """
    combine preprocessed image and label data for n_fold cross_validation
    :param img1: numpy array of image data 1
    :param img2: numpy array of image data 2
    :param label1: numpy arrray of image 1 lable for single label, list of numpy array of image 1 for muti_label
    :param label2: numpy arrray of image 2 lable for single label, list of numpy array of image 2 for muti_label
    :param is_muti_lable: boolean indicator to indicate if the label is multi_label or single label
    :return: combined image and label
    """
    new_img = np.concatenate((img1, img2), axis=0)
    if not is_muti_lable:
        new_label = np.concatenate((label1, label2), axis=0)
    else:
        new_label = []
        for i in range(len(label1)):
            new_label.append(np.concatenate((label1[i], label2[i]), axis=0))

    return(new_img, new_label)


def generate_3D_flow(image_data,
                     label,
                     rotation_range=0,
                     batch_size=32,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True):
    """ Generate 3D image patches for 3D convolution neural net
    """
    datagene = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                 samplewise_center=False,  # set each sample mean to 0
                                 featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                 samplewise_std_normalization=False,  # divide each input by its std
                                 zca_whitening=False,  # apply ZCA whitening
                                 rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
                                 width_shift_range=width_shift_range,  # randomly shift images horizontally (fraction of total width)
                                 height_shift_range=height_shift_range,  # randomly shift images vertically (fraction of total height)
                                 horizontal_flip=horizontal_flip,  # randomly flip images
                                 vertical_flip=vertical_flip)# randomly flip images
    reshapes_image = image_data.reshape(image_data.shape[0], image_data.shape[1], 
                                        image_data.shape[2], image_data.shape[3])
    
    gen_flow = datagene.flow(reshapes_image, label, batch_size=batch_size)
    while True:
            data_cur = gen_flow.next()
            image_data_5D = data_cur[0].reshape(data_cur[0].shape[0], data_cur[0].shape[1], data_cur[0].shape[2], 
                                                data_cur[0].shape[3], 1)
            yield image_data_5D, data_cur[1]


def generate_3D_multi_label_flow(image_data,
                                 label,
                                 rotation_range=0,
                                 batch_size=32,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 is_contibous_label=False):
    """ Generate 3D image patches for 3D convolution neural net
        label: this is a list containing multiple labels
    """
    datagene = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                 samplewise_center=False,  # set each sample mean to 0
                                 featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                 samplewise_std_normalization=False,  # divide each input by its std
                                 zca_whitening=False,  # apply ZCA whitening
                                 rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
                                 width_shift_range=width_shift_range,  # randomly shift images horizontally (fraction of total width)
                                 height_shift_range=height_shift_range,  # randomly shift images vertically (fraction of total height)
                                 horizontal_flip=horizontal_flip,  # randomly flip images
                                 vertical_flip=vertical_flip)# randomly flip images
    reshapes_image = image_data.reshape(image_data.shape[0], image_data.shape[1],
                                        image_data.shape[2], image_data.shape[3])
    label_new_array = np.stack(label, axis=-1)

    gen_flow = datagene.flow(reshapes_image, label_new_array, batch_size=batch_size)
    while True:
            data_cur = gen_flow.next()
            image_data_5D = data_cur[0].reshape(data_cur[0].shape[0], data_cur[0].shape[1], data_cur[0].shape[2],
                                                data_cur[0].shape[3], 1)
            if is_contibous_label:
                label_list = [data_cur[1][:, i] for i in range(data_cur[1].shape[-1])]
            else:
                label_list = [data_cur[1][:, :, i] for i in range(data_cur[1].shape[-1])]
            yield image_data_5D, label_list


class roc_callback(Callback):
    """
    call back class to calculate AUC after each training epoch
    """
    def __init__(self,training_data,validation_data, label_index=None):

        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.label_index = label_index


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        if self.label_index:
            y_pred = y_pred[self.label_index]
        roc = roc_auc_score(self.y, y_pred)
        logs['roc-auc'] = roc

        y_pred_val = self.model.predict(self.x_val)
        if self.label_index:
            y_pred_val = y_pred_val[self.label_index]
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc-auc_val'] = roc_val

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class ExponentialMovingAverage(Callback):
    """create a copy of trainable weights which gets updated at every
       batch using exponential weight decay. The moving average weights along
       with the other states of original model(except original model trainable
       weights) will be saved at every epoch if save_mv_ave_model is True.
       If both save_mv_ave_model and save_best_only are True, the latest
       best moving average model according to the quantity monitored
       will not be overwritten. Of course, save_best_only can be True
       only if there is a validation set.
       This is equivalent to save_best_only mode of ModelCheckpoint
       callback with similar code. custom_objects is a dictionary
       holding name and Class implementation for custom layers.
       At end of every batch, the update is as follows:
       mv_weight -= (1 - decay) * (mv_weight - weight)
       where weight and mv_weight is the ordinal model weight and the moving
       averaged weight respectively. At the end of the training, the moving
       averaged weights are transferred to the original model.
       """
    def __init__(self, decay=0.999, filepath='temp_weight.hdf5',
                 save_mv_ave_model=True, verbose=0,
                 save_best_only=False, monitor='val_loss', mode='auto',
                 save_weights_only=False, custom_objects={}):
        self.decay = decay
        self.filepath = filepath
        self.verbose = verbose
        self.save_mv_ave_model = save_mv_ave_model
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.custom_objects = custom_objects  # dictionary of custom layers
        self.sym_trainable_weights = None  # trainable weights of model
        self.mv_trainable_weights_vals = None  # moving averaged values
        super(ExponentialMovingAverage, self).__init__()

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs={}):
        self.sym_trainable_weights = self.model.trainable_weights
        # Initialize moving averaged weights using original model values
        self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in
                                          self.sym_trainable_weights}
        if self.verbose:
            print('Created a copy of model weights to initialize moving'
                  ' averaged weights.')

    def on_batch_end(self, batch, logs={}):
        for weight in self.sym_trainable_weights:
            old_val = self.mv_trainable_weights_vals[weight.name]
            self.mv_trainable_weights_vals[weight.name] -= \
                (1.0 - self.decay) * (old_val - K.get_value(weight))

    def on_epoch_end(self, epoch, logs={}):
        """After each epoch, we can optionally save the moving averaged model,
        but the weights will NOT be transferred to the original model. This
        happens only at the end of training. We also need to transfer state of
        original model to model2 as model2 only gets updated trainable weight
        at end of each batch and non-trainable weights are not transferred
        (for example mean and var for batch normalization layers)."""
        if self.save_mv_ave_model:
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best moving averaged model only '
                                  'with %s available, skipping.'
                                  % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('saving moving average model to %s'
                                  % (filepath))
                        self.best = current
                        model2 = self._make_mv_model(filepath)
                        if self.save_weights_only:
                            model2.save_weights(filepath, overwrite=True)
                        else:
                            model2.save(filepath, overwrite=True)
                        model2 = None
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving moving average model to %s' % (epoch, filepath))
                model2 = self._make_mv_model(filepath)
                if self.save_weights_only:
                    model2.save_weights(filepath, overwrite=True)
                else:
                    model2.save(filepath, overwrite=True)
                model2 = None

    def on_train_end(self, logs={}):
        for weight in self.sym_trainable_weights:
            K.set_value(weight, self.mv_trainable_weights_vals[weight.name])

    def _make_mv_model(self, filepath):
        """ Create a model with moving averaged weights. Other variables are
        the same as original mode. We first save original model to save its
        state. Then copy moving averaged weights over."""
        self.model.save(filepath, overwrite=True)
        model2 = load_model(filepath, custom_objects=self.custom_objects)

        for w2, w in zip(model2.trainable_weights, self.model.trainable_weights):
            K.set_value(w2, self.mv_trainable_weights_vals[w.name])
       # for w2 in model2.trainable_weights:
       #     K.set_value(w2, self.mv_trainable_weights_vals[w2.name])

        return model2

class BestMetric(Callback):
    def __init__(self, monitor, mode):
        self.monitor = monitor
        self.mode = mode

    def on_train_begin(self, logs={}):
        self.metric_value = []

    def on_epoch_end(self, epoch, logs={}):
        self.metric_value.append(logs.get(self.monitor))

    def on_train_end(self, logs={}):
        if self.mode == 'max':
            self.best_value = max(self.metric_value)
        else:
            self.best_value = min(self.metric_value)


def get_label_class_weights(all_label, is_multi_label):
    """
    calculate the label weights to solve unbalanced data problem for input labels
    :param all_label: is list for multi_lable, is single variable for single label
    :param is_multi_label: boolean vairaible to indicate if the input label multi-label
    :return:
    """
    if is_multi_label:
        class_weights = []
        for cur_label in all_label:
            count_0 = np.sum(cur_label[:, 0])
            count_1 = np.sum(cur_label[:, 1])
            weight_0 = float(count_1) / (count_1 + count_0)
            weight_1 = count_0 / (count_1 + count_0)
            class_weights.append({0 : weight_0, 1 : weight_1})
    else:
        count_0 = np.sum(all_label[:, 0])
        count_1 = np.sum(all_label[:, 1])
        weight_0 = float(count_1) / (count_1 + count_0)
        weight_1 = count_0 / (count_1 + count_0)
        class_weights = {0 : weight_0, 1 : weight_1}
    return class_weights




