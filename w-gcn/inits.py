import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


#  glorot初始化方法：它为了保证前向传播和反向传播时每一层的方差一致:在正向传播时，每层的激活值的方差保持不变；
#       在反向传播时，每层的梯度值的方差保持不变。根据每层的输入个数和输出个数来决定参数随机初始化的分布范围，
#  是一个通过该层的输入和输出参数个数得到的分布范围内的均匀分布。
def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)