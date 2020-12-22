# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import pandas as pd

from keras import backend as K
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import math

class Pos(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Pos, self).__init__(**kwargs)
        
    def get_config(self):
        config = {"output_dim":self.output_dim}
        base_config = super(Pos, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        print("input_shape",input_shape)#输入的一个样本的形状
# =============================================================================
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(3,input_shape[2], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
# =============================================================================

        super(Pos, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        dmodel=x.shape[2]
        num_time=x.shape[1]
        position=np.zeros([num_time,dmodel])
        for pos in range(num_time):
            for i in range(0,dmodel//2,2):
                position[pos][i]=math.sin(pos / (10000 ** ((2 * i)/dmodel)))
                position[pos][i+1]=math.cos(pos / (10000 ** ((2 * i)/dmodel)))
        postion=tf.convert_to_tensor(position)
        print("postion.shape",postion.shape)
# =============================================================================
#         for pos in range(num_time):
#             for i in range(0,dmodel,2):
#                 for j in range(x.shape[0]):
#                     x_v = tf.Variable(x)
#                     x_v[j,pos, i].assign(math.sin(pos / (10000 ** ((2 * i)/dmodel))))
#                     x_v[j,pos, i+1].assign(math.sin(pos / (10000 ** ((2 * i)/dmodel))))
#                     x=tf.convert_to_tensor(x_v)
# =============================================================================
        print("######x.shape",postion.shape)
        return postion



