
from matplotlib import pyplot as plt
import pandas as pd

#from keras import backend as K
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

class block_feature(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(block_feature, self).__init__(**kwargs)
        
    def get_config(self):
        config = {"output_dim":self.output_dim}
        base_config = super(block_feature, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        print("#############input_shape",input_shape)#输入的一个样本的形状
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,input_shape[1],1),
                                      initializer='uniform',
                                      trainable=True)#创建参数W的形状
        super(block_feature, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        print("x.shape",x.shape)
        y=tf.transpose(x, perm=[0, 2, 1])
        V=tf.matmul(y,self.kernel[0])
        #V = K.dot(K.permute_dimensions(x, [0, 2, 1]),self.kernel[0])
        #print("V.shape",V.shape)
        return V
