

import tensorflow as tf

def get_Att(n_var_in, n_var_out, n_depth, m_dim, n_heads):




	xx = tf.keras.Input(shape = [n_depth, n_var_in])
	yy_list = []
	for _ in range(n_heads):
		Q = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(m_dim))(xx)
		K = Q #tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(m_dim))(xx)

		yy_t = tf.keras.layers.Attention()([Q, K])
		yy_list.append(yy_t)

	yy = tf.keras.layers.Concatenate(axis = -1)(yy_list)
	yy = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_var_out))(yy)

	model = tf.keras.Model(inputs = xx, outputs = yy)