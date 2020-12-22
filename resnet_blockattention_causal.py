# resnet model 
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
from utils import save_test_duration

matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils import save_logs
from utils import calculate_metrics
import selfattention



class Classifier_RESNET_BLOCKATTENTION_CAUSAL:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        if build == True:
            print("创建模型。。。")
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64
# =============================================================================
#         blocknum = 32
#         paddnum=input_shape[0]%blocknum
#         blocks=[]
#         input_layer = keras.layers.Input(input_shape)
#         #print("input_layer.shape",input_layer.shape)
#         
#         #pad_input=tf.pad(input_layer,tf.constant([[0,0],[0,paddnum],[0,0]]),"CONSTANT")
#         #print("input_layer.shape",pad_input.shape)
#         
#         #for i in range(0,(input_shape[0]+paddnum),blocknum):
#         for i in range((input_shape[0]-blocknum+1)):
#             print(i)
#             #block_data=keras.layers.Lambda(lambda x: x[:,i:i+blocknum,:])(pad_input)
#             block_data=keras.layers.Lambda(lambda x: x[:,i:i+blocknum,:])(input_layer)
#             block_rem=keras.layers.SimpleRNN(1,return_state=True,return_sequences=False)(block_data)
#             blocks.append(block_rem[1])
#         concat_layer = keras.layers.Concatenate(axis=-1)(blocks)
#         concat_layer=tf.expand_dims(concat_layer,axis=1)
#         concat_layer=tf.transpose(concat_layer,[0,2,1])
#         print("concat_layer.shape",concat_layer.shape)
# 
# =============================================================================
        blocknum=32
        input_layer = keras.layers.Input(input_shape)
        blocksrem=keras.layers.Conv1D(filters=n_feature_maps, kernel_size=blocknum, padding='same')(input_layer)
        ###################################################

        O_seq = selfattention.Self_Attention(128)(blocksrem)
        
        #O_seq = keras.layers.BatchNormalization()(O_seq)
        #O_seq=keras.layers.add([input_layer,O_seq])
        
        #conv1 = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        #O_seq = selfattention.Self_Attention(64)(conv1)
        #O_seq=keras.layers.add([conv1,O_seq])
        ########################################################

        # BLOCK 1

        #conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(O_seq)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        #shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(O_seq)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        ##############################
        print(model.summary())#ljw修改
        #############################

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        #nb_epochs = 1500
        nb_epochs = 1500

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        print("mini_batch_size",mini_batch_size)

        start_time = time.time()
        print("开始训练。。。")
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        print("训练结束。。。")
        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')
        print("开始预测。。。")
        
        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)
        print("预测结束。。。")
        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        #####################################################################
        _custom_objects = {"Self_Attention" : selfattention.Self_Attention}
        model = keras.models.load_model(model_path,custom_objects=_custom_objects)
        ##########################################################################
        print("x.text.shape########################",x_test.shape)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
