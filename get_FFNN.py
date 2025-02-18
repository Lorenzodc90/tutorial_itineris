import tensorflow as tf
from keras.layers import Dense
from keras import Sequential


def get_FFNN_sequential(n_var_in, n_var_out, n_depth, n_units1, n_units2, activ):
	model = Sequential()
	model.add(Dense(n_units1, activation=activ, input_shape=(n_var_in,)))
	model.add(Dense(n_units2, activation=activ))
	model.add(Dense(n_var_out*n_depth))
	return model

def get_FFNN_functional(n_var_in, n_var_out, n_depth, n_units1, n_units2, activ):


	X = tf.keras.Input(shape = [n_var_in])
	Y = tf.keras.layers.Dense(n_units1, activation=activ)(X)
	Y = tf.keras.layers.Dense(n_units2, activation=activ)(Y)
	Y = tf.keras.layers.Dense(n_var_out*n_depth)(Y)
	model = tf.keras.Model(inputs = X, outputs = Y)
	
	return model