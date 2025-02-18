
from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
import tensorflow as tf


def get_LSTM_sequential(n_var_in, n_var_out, n_depth, n_units1, n_units2, activ):
	model = Sequential()
	# Add a LSTM layers
	model.add(LSTM(n_units1, return_sequences=True, activation=activ,input_shape=(n_depth, n_var_in)))
	model.add(LSTM(n_units2, return_sequences=True, activation=activ))
	model.add(TimeDistributed(Dense(n_var_out)))
	return model

def get_LSTM_functional(n_var_in, n_var_out, n_depth, n_units1, n_units2, activ):


	X = tf.keras.Input(shape = [n_depth, n_var_in])
	Y = LSTM(n_units1, return_sequences=True, activation=activ)(X)
	Y = LSTM(n_units2, return_sequences=True, activation=activ)(Y)
	Y = TimeDistributed(Dense(n_var_out))(Y)

	model = tf.keras.Model(inputs = X, outputs = Y)

	return model

