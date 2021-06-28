import keras
from keras.layers import Input
import layers
import tensorflow as tf


class ModelBuilder():
    def __init__(self, num_of_neurons=10, infunc='sinf', data_x=[], data_y=[]):
        self.bat = 1
        self.in_val = data_x
        self.out_val = data_y
        self.x = Input(shape=(self.bat,))
        self.num_of_neurons = num_of_neurons

        self.linear_layer = layers.Linear(num_of_neurons, self.bat, infunc)
        self.presum = layers.PresumLayer(num_of_neurons, self.bat)
        self.my_sum = layers.ComputeSum(num_of_neurons)
        self.lin = self.linear_layer(self.x)
        self.pres = self.presum(self.lin)
        self.y = self.my_sum(self.pres)
        self.model = keras.Model(self.x, self.y)

    def start_training(self, epochs=21):
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.008)
        self.model.compile(optimizer, loss=tf.keras.losses.logcosh)#loss='mse')
        self.model.summary()
        self.model.fit(self.in_val, self.out_val, epochs=epochs, batch_size=self.bat, verbose=1)

    def predict(self, value):
        return self.model.predict(value)
