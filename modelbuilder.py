import keras
from keras.layers import Input
import layers
import tensorflow as tf


class ModelBuilder():
    def __init__(self, num_of_neurons=10, infunc='sinf', data_x=[], data_y=[]):
        self.in_val = data_x
        self.out_val = data_y
        self.x = Input(shape=(1,))
        self.num_of_neurons = num_of_neurons

        self.linear_layer = layers.Linear(num_of_neurons, 1, infunc)
        self.presum = layers.PresumLayer(num_of_neurons, 1)
        self.my_sum = layers.ComputeSum(num_of_neurons)
        self.lin = self.linear_layer(self.x)
        self.pres = self.presum(self.lin)
        self.y = self.my_sum(self.pres)
        self.model = keras.Model(self.x, self.y)

    def start_training(self, epochs=21):
        self.model.compile(optimizer='sgd', loss='mse')
        self.model.summary()
        self.model.fit(self.in_val, self.out_val, epochs=epochs, batch_size=1, verbose=1)

    def predict(self, value):
        return self.model.predict(value)
