#仅仅进行了取上三角的操作
from tensorflow.keras.layers import Layer
import tensorflow as tf


class local_causal_attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(local_causal_attention, self).__init__(**kwargs)
        
    def get_config(self):
        config = {"output_dim":self.output_dim}
        base_config = super(local_causal_attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        #print("input_shape",input_shape)#输入的一个样本的形状
        super(local_causal_attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        #V=tf.linalg.band_part(x,-1,0)
        #tf.print(V)
        V=Position_Embedding(x)
        return V+x

def Position_Embedding(inputs):
        """
        :param inputs: shape=(batch_size,timestep,word_size)
        :param position_size: int_
        :return: shape=(1,seq_len.size,position_size)
        """

        # inputs: shape=(batch_size,timestep,word_size)
        batch_size,seq_len,position_size=tf.shape(inputs)[0],tf.shape(inputs)[1],tf.shape(inputs)[2]
        #batch_size,seq_len,position_size=inputs.shape[0],inputs.shape[1],inputs.shape[2]

        # shape=(position_size,)
        #position_j=1./tf.pow(10000.,2*tf.range(position_size,dtype=tf.int32)/position_size)
        position_j=1./tf.pow(10000.,2*tf.range(position_size,dtype=tf.float32)/tf.cast(position_size,tf.float32))
        #position_j=1./(10000.**(2*tf.range(position_size,dtype=tf.float32)/tf.cast(position_size,tf.float32)))
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
