import tensorflow as tf


class myLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha = 0.1):
        super().__init__()
        self.alpha = alpha

    # Create the state of the layer (weights)
    # def build(self, input_shape):
    #     self.kernel = self.add_weight(
    #         shape=(input_shape[-1], self.units),
    #         initializer="glorot_uniform",
    #         trainable=True,
    #         name="kernel",
    #     )
    #     self.bias = self.add_weight(
    #         shape=(self.units,),
    #         initializer="zeros",
    #         trainable=True,
    #         name="bias",
    #     )

    # Defines the computation
    def call(self, inputs):
        return tf.keras.activations.leaky_relu(inputs, negative_slope = self.alpha)
    



x = tf.keras.Input(shape = [5, ])
y = tf.keras.layers.Dense(10, activation='sigmoid')(x)
y = tf.keras.layers.Dense(30, activation='linear')(y)
y = myLeakyReLU()(y)
y = tf.keras.layers.Dense(22, activation='linear')(y)
y = tf.keras.layers.Dense(3)(y)
model = tf.keras.Model(inputs = x, outputs = y)


model.summary(expand_nested = True)