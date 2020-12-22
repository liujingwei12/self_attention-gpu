# -*- coding: utf-8 -*-
# resnet model 局部因果自注意力，并且因果扩张卷积
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
import local_causal_attention
import block_feature



class Classifier_RESNET_IN_OUT:

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

        blocknum=32
        blocksdata=[]
        input_layer = keras.layers.Input(input_shape)
        cov_n=keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        
        for i in range((input_shape[0]-blocknum+1)):
            #print(i)
            block_data=keras.layers.Lambda(lambda x: x[:,i:i+blocknum,:])(cov_n)
            pos_val=local_causal_attention.local_causal_attention(n_feature_maps)(block_data)
            #print("pos_val.shape",pos_val.shape)
            WQ= keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(pos_val)
            WK= keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(pos_val)
            WV= keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(pos_val)
            W=tf.matmul( WQ,tf.transpose(WK, perm=[0, 2, 1]))
            W = W / (n_feature_maps**0.5)
            W = keras.layers.Activation('softmax')(W)
            block_attention =tf.matmul(W,WV)
            blockrem=block_feature.block_feature(n_feature_maps)(block_attention)#输出特征维度必须与输入的时刻特征维度相同
            #print("blockrem.shape",blockrem.shape)
            blockrem=tf.transpose(blockrem, perm=[0, 2, 1])
            blocksdata.append(blockrem)
        blocksrem=keras.layers.Concatenate(axis=1)(blocksdata)
        #print("blocksrem.shape",blocksrem.shape)
        
        pos_val=local_causal_attention.local_causal_attention(n_feature_maps)(blocksrem)#返回引入位置编码的新输入
        ###################################################
        #块间自注意的kqv
        WQ= keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(pos_val)
        WK= keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(pos_val)
        WV= keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(pos_val)
        W=tf.matmul( WQ,tf.transpose(WK, perm=[0, 2, 1]))
        #W = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
        W = W / (n_feature_maps**0.5)
        W = keras.layers.Activation('softmax')(W)
        O_seq =tf.matmul(W,WV)#得到经过自注意之后的新序列
        
        #时间维度上哪一块最重要
        W_Block=tf.reduce_sum(W, 1)#自注意权值矩阵行加和
        W_Block = keras.layers.Activation('softmax')(W_Block)#激活函数，行的和为一
        W_Block=tf.expand_dims(W_Block,axis=1)#为向量增加维度（None*1*T）
        W_Blocks=[]
        for i in range(pos_val.shape[2]):
            W_Blocks.append(W_Block)
        W_Blocks=keras.layers.Concatenate(axis=1)(W_Blocks)#拼接为与每一时刻的特征维度相同（None*n_feature_maps*T）
        W_Blocks=tf.transpose(W_Blocks, perm=[0, 2, 1])#转置为（None*T*n_feature_maps），便于与原矩阵对应元素相乘
        W_time=W_Blocks*blocksrem
        O_seq=O_seq+W_time #（None*T*n_feature_maps）
        
        
        #O_seq=keras.layers.add([input_layer,O_seq])
        ########################################################

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(O_seq)
        #conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='causal',dilation_rate=1)(O_seq)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(O_seq)
        #shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(O_seq)
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
        
        # BLOCK 4

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_3)

        output_block_4 = keras.layers.add([shortcut_y, conv_z])
        output_block_4 = keras.layers.Activation('relu')(output_block_4)
        
        # BLOCK 5

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_4)

        output_block_5 = keras.layers.add([shortcut_y, conv_z])
        output_block_5 = keras.layers.Activation('relu')(output_block_5)
        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_5)

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
        #model = keras.models.load_model(model_path)
        #####################################################################
        _custom_objects = {"local_causal_attention" : local_causal_attention.local_causal_attention,
                           "block_feature":block_feature.block_feature}
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
        
