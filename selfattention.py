#经典自注意机制+因果自注意机制
from matplotlib import pyplot as plt
import pandas as pd

from keras import backend as K
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)
        
    def get_config(self):
        config = {"output_dim":self.output_dim}
        base_config = super(Self_Attention, self).get_config()
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
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2],self.output_dim),
                                      initializer='uniform',
                                      trainable=True)#创建参数W(K/Q/V)的形状

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        y=Position_Embedding(x)
        x=x+y
        WQ = K.dot(x, self.kernel[0])#q
        WK = K.dot(x, self.kernel[1])#k
        WV = K.dot(x, self.kernel[2])#v


        print("x.shape",x.shape)
        
        print("kernel[0].shape",self.kernel[0].shape)
        print("kernel.shape",self.kernel.shape)

        print("WQ.shape",WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)#转置
        #print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [1, 0]).shape)#转置

        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
        #QK = K.dot(WQ,K.permute_dimensions(WK, [1, 0]))

        QK = QK / (self.output_dim**0.5)
        #####################################################
        #QK=tf.linalg.band_part(QK,-1,0)#与经典的区别，取下三角
        #####################################################
        QK = K.softmax(QK)

        print("QK.shape",QK.shape)

        V = K.batch_dot(QK,WV)
        #V = K.dot(QK,WV)
        print("V.shape",V.shape)
        return V

        
def Position_Embedding(inputs):
    """
    :param inputs: shape=(batch_size,timestep,word_size)
    :param position_size: int_
    :return: shape=(1,seq_len.size,position_size)
    """

    # inputs: shape=(batch_size,timestep,word_size)
    batch_size,seq_len,position_size=tf.shape(inputs)[0],tf.shape(inputs)[1],tf.shape(inputs)[2]

    # shape=(position_size,)
    #position_j=1./tf.pow(10000.,2*tf.range(position_size,dtype=tf.int32)/position_size)
    position_j=1./tf.pow(10000.,2*tf.range(position_size,dtype=tf.float32)/tf.cast(position_size,tf.float32))

    # shape=(1,position_size)
    position_j=tf.expand_dims(position_j,axis=0)

    # shape=(seq_len.size,)
    position_i=tf.range(tf.cast(seq_len,tf.float32),dtype=tf.float32)

    # shape=(seq_len.size,1)
    position_i=tf.expand_dims(position_i,axis=1)

    # 这是上面维度扩展的原因
    position_ij=tf.matmul(position_i,position_j)

    # shape=(seq_len.size,position_size)
    # 在axis=1，即：seq_len.size 拼接
    #position_ij=tf.concat([tf.cos(position_ij),tf.sin(position_ij)],axis=1)
    position_ij=tf.sin(position_ij)

    # shape=(1,seq_len.size,position_size)
    position_embedding=tf.expand_dims(position_ij,axis=0)+tf.zeros((batch_size,seq_len,position_size))

    return position_embedding
