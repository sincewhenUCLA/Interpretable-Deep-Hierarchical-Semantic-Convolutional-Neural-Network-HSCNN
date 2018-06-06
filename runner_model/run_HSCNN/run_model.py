from __future__ import print_function
from model.Preprocessing import combine_two_fold_image_label, get_label_class_weights, preprocess_image_label, generate_3D_multi_label_flow,  ExponentialMovingAverage
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from model.cnn_model_module import CNN_multi_task_builder
from keras import regularizers
from keras import backend as K
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_index(input_index, num_fold):
    return(input_index % num_fold)

data_folder = '../../Data/data_for_model_3mm_corrected_n_fold/'
fold_0_image_name = 'fold_0_img.pkl'
fold_0_label_name = 'fold_0_all_binary_label.pkl'
fold_1_image_name = 'fold_1_img.pkl'
fold_1_label_name = 'fold_1_all_binary_label.pkl'
fold_2_image_name = 'fold_2_img.pkl'
fold_2_label_name = 'fold_2_all_binary_label.pkl'
fold_3_image_name = 'fold_3_img.pkl'
fold_3_label_name = 'fold_3_all_binary_label.pkl'

nb_classes = 2
batch_size = 32
nb_epoch = 300
data_augmentation = True
label_name_list = ['calcification', 'margin', 'sphericity', 'subtlety', 'texture', 'malignancy']

fold_0_image, fold_0_label = preprocess_image_label(data_folder, fold_0_image_name, fold_0_label_name, label_name_list, if_label_catogerial=True)
fold_1_image, fold_1_label = preprocess_image_label(data_folder, fold_1_image_name, fold_1_label_name, label_name_list, if_label_catogerial=True)
fold_2_image, fold_2_label = preprocess_image_label(data_folder, fold_2_image_name, fold_2_label_name, label_name_list, if_label_catogerial=True)
fold_3_image, fold_3_label = preprocess_image_label(data_folder, fold_3_image_name, fold_3_label_name, label_name_list, if_label_catogerial=True)

img_folds = [fold_0_image, fold_1_image, fold_2_image, fold_3_image]
label_folds = [fold_0_label, fold_1_label, fold_2_label, fold_3_label]

for test_fold_index in range(3, 7):
    test_ith = get_index(test_fold_index, 4)
    train_ith_1 = get_index(test_fold_index + 1, 4)
    train_ith_2 = get_index(test_fold_index + 2, 4)
    val_ith = get_index(test_fold_index + 3, 4)
    save_folder = 'saved_weights_test_index_{0}/'.format(test_ith)
    train_image, train_label = combine_two_fold_image_label(img1=img_folds[train_ith_1],
                                                            img2=img_folds[train_ith_2],
                                                            label1=label_folds[train_ith_1],
                                                            label2=label_folds[train_ith_2],
                                                            is_muti_lable=True)
    val_image = img_folds[val_ith]
    val_label = label_folds[val_ith]
    weights_class = get_label_class_weights(all_label=train_label, is_multi_label=True)
    print(weights_class)
    input_shape = train_image.shape[1:]
    model = CNN_HSCNN_builder.build_no_direct_connection(input_shape=input_shape,
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
                                                              task_weights_list=[0.1, 0.1, 0.2, 0.2, 0.1, 1],
                                                              drop_out_task_base_list=[0.6],
                                                              fully_connected_block_function='dense_bn_activation',
                                                              num_middle_layers_task_module=1,
                                                              num_dense_units_task_base_list=[256],
                                                              num_dense_units_subtask_module=64,
                                                              num_dense_units_major_task_module=256,
                                                              dropout_rate_task_module=None,
                                                              dropout_rate_major_task_module=0.6,
                                                              num_subtasks=5,
                                                              init='glorot_uniform',
                                                              activation='relu',
                                                              num_of_class=2,
                                                              kernel_regularizer=regularizers.l2(0.048),
                                                              droput_rate_flatten=0.6)
    
    model.summary()


    # serialize model to JSON
    model_json = model.to_json()
    with open(save_folder + "lidc_model.json", "w") as json_file:
        json_file.write(model_json)
    print("model saved to disk.")

    csv_logger = CSVLogger(save_folder + 'ann_logger.csv')
    filepath=save_folder + "best_weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_main_malignancy_output_acc', verbose=1, save_best_only=True,mode='max')
    em_ensemble = ExponentialMovingAverage(decay=0.999,
                                           filepath=save_folder + 'average_weight.hdf5',
                                           save_mv_ave_model=False,
                                           verbose=0,
                                           save_best_only=False,
                                           monitor='val_main_malignancy_output_acc',
                                           mode='auto',
                                           save_weights_only=True)
    callbacks_list = [csv_logger, checkpoint, em_ensemble]

    print('Using data augmentation 3D')
    history_this = model.fit_generator(generate_3D_multi_label_flow(train_image, train_label, batch_size=batch_size),
                                      nb_epoch=nb_epoch, steps_per_epoch=int(train_image.shape[0] / batch_size),
                                      verbose=2, validation_data=(val_image, val_label), 
                                      callbacks=callbacks_list,
                                      class_weight=weights_class)
    model.save_weights(save_folder + 'average_weight.hdf5', overwrite=True)
    del history_this
    del model
    K.clear_session()
