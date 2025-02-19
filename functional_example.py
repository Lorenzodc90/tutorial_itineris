import tensorflow as tf

x = tf.keras.Input(shape = [5, ])
y = tf.keras.layers.Dense(10, activation='sigmoid')(x)
y = tf.keras.layers.Dense(30, activation='relu')(y)
y = tf.keras.layers.Dense(22, activation='linear')(y)
y = tf.keras.layers.Dense(3)(y)
model = tf.keras.Model(inputs = x, outputs = y)


model.summary(expand_nested = True)