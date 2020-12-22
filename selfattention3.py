# -*- coding: utf-8 -*-
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 


from utils import save_logs
from utils import calculate_metrics
import selfattention

class Classifier_SELF:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=True,build=True):
		self.output_directory = output_directory
		if build == True:
			print("创建模型。。。")
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory+'model_init.hdf5')
		return

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)
		print("input_layer.shape",input_layer.shape)
		##########################################ljw修改
		O_seq = selfattention.Self_Attention(128)(input_layer)
		print("O_seq.shape",O_seq.shape)
		##############################################
		O_seq=keras.layers.Reshape((O_seq.shape[1]*O_seq.shape[2],1))(O_seq)
		gap_layer = keras.layers.GlobalAveragePooling1D()(O_seq)
		O_seq = keras.layers.Dropout(0.5)(gap_layer)
		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(O_seq)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)
		##############################
		print(model.summary())#ljw修改
		#############################
		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training
		batch_size = 1 #16
		nb_epochs = 500 #nb_epochs = 2000


		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))#确保一次更新的数据足够

		start_time = time.time() 
		print("训练数据。。。")
		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		print("训练结束。。。")
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')
		print("加载模型。。。")

		#ljw修改#############################
		_custom_objects = {"Self_Attention" : selfattention.Self_Attention}
		###################################################
		#model = keras.models.load_model(self.output_directory+'best_model.hdf5')
		model = keras.models.load_model(self.output_directory+'best_model.hdf5',custom_objects=_custom_objects)
		print("开始预测。。。")
		y_pred = model.predict(x_val)
		print("预测结束。。。")
		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()

	def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
		model_path = self.output_directory + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		print("0000000000000000000000000000开始预测。。。")
		y_pred = model.predict(x_test)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)#返回沿着行的最大值索引
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred
