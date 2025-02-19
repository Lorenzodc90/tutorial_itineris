import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(5,)))
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(22, activation='linear'))
model.add(tf.keras.layers.Dense(3))

model.summary(expand_nested = True)
