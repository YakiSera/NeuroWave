from keras import backend as K
import numpy as np
import keras
import math
import tensorflow as tf


def default_func(x):
    return x


def gauss_func(x):
    return -x * tf.math.exp(-x/2)


real_func = default_func


def make_data_sets(range_a, range_b, amount):
    #x = [random.uniform(range_a, range_b) for i in range(amount)]
    div = (math.fabs(range_a) + math.fabs(range_b)) / amount
    x = np.arange(range_a, range_b, div)
    print('Data set in ({range_a}, {range_b}) with {amount} points are ready.'.format(
        range_a=range_a, range_b=range_b, amount=amount))
    return x, [real_func(x[i]) for i in range(amount)]

#init_rand = keras.initializers.random_uniform(minval=-1, maxval=1)
init_rand = keras.initializers.lecun_normal(111)

# https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, infunc='gauss'):
        super(Linear, self).__init__()
        self.units = units
        dict_func = {'sinf': tf.sin, 'gauss': gauss_func}
        self.curr_func = dict_func[infunc]


    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), initializer=init_rand, trainable=True
        )
        self.b = self.add_weight(shape=(self.units,), initializer=init_rand, trainable=True)
        super(Linear, self).build(1)

    def call(self, inputs):
        return self.activation_function((inputs * self.w) + self.b)

    def activation_function(self, inputs):
        return self.curr_func(inputs)


class PresumLayer(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(PresumLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), initializer=init_rand, trainable=True
        )
        super(PresumLayer, self).build(1)

    def call(self, inputs):
        return inputs * self.w


class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total = K.sum(inputs)
        return self.total
