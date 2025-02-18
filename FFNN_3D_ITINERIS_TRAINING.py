"""Python procedure for the 3D reconstruction of Chl and Temperature profile from surface observation based on the codes github from Bruno Buongiorno Nardelli, 2020

Bibliography reference, Bruno Buongiorno Nardelli, 2020 and Sammartino et al., 2020
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from subprocess import call
import warnings
import random
import math
warnings.filterwarnings("ignore")# specify to ignore warning messages
from keras import optimizers
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
import glob
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from keras.models import load_model

from get_FFNN import get_FFNN_sequential, get_FFNN_functional



import load_dataset
from load_dataset import load_dataset

model_dir = './'
file_train = model_dir+'BIOARGO_MATCHUP_2018_2021.nc'


# loading the data cubes
X_cube, Y_cube, X_test_cube, Y_test_cube, minMaxDict = load_dataset(file_train)

# keep only the "surface" value for network input
X = X_cube[:, 0, :]
X_test = X_test_cube[:, 0, :]

Y = tf.concat( tf.unstack(Y_cube, axis = -1), axis = -1)
Y_test = tf.concat( tf.unstack(Y_test_cube, axis = -1), axis = -1)





####################################
# FFNN model configuration parameters
####################################

activ = 'sigmoid'#'softsign'#
# opt='Adam'
opt = Adam(learning_rate = 0.0001)
pat=30
epochs = 20
val_split=.30
n_units1 =1000
n_units2= 1000

n_var_in = X_cube.shape[2]
n_var_out = 2
n_depth = X_cube.shape[1]

# names for save files
model_name_generic = model_dir + 'FFNN_' + str(n_units1) + '_' + str(n_units2) + '_MODEL_v1'

model_name = model_name_generic + '.keras'
model_name_es = model_name_generic + '_es'

model_name_history = model_dir + 'FFNN_' + str(n_units1) + '_' + str(n_units2) + '_MODEL_v1.npy'


# get the model (see the function to see what it does)
model = get_FFNN_sequential(n_var_in, n_var_out, n_depth, n_units1, n_units2, activ)

# model = get_FFNN_functional(n_var_in, n_var_out, n_depth, n_units1, n_units2, activ)


model.summary()



# compile model with loss and optimizer
model.compile(loss='mse', optimizer=opt)

# fit model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=pat)

history=model.fit(X, Y, epochs=epochs, verbose=1, shuffle=False, validation_split=val_split, callbacks=[es])


# once training is done, save everything

np.save(model_name_history, history)
model.save(model_name)
np.save(model_name_es, es)
print("Model saved to disk")



model = load_model(model_name)
print("Model loaded from disk")

OUT_test = model.predict(X_test)

Tmax = minMaxDict['Tmax']
Tmin = minMaxDict['Tmin']

OUT_Ttest=(OUT_test[:,0:n_depth]*(Tmax-Tmin))+Tmin

CHLmax = minMaxDict['CHLmax'] 
CHLmin = minMaxDict['CHLmin']

OUT_CHLtest=(OUT_test[:,n_depth:n_depth*2]*(CHLmax-CHLmin))+CHLmin
OUT_CHLtest=10**(OUT_CHLtest)


T_testtot= Y_test[:, 0:n_depth]*(Tmax - Tmin) + Tmin
CHL_testtot= np.power(10., Y_test[:, n_depth:2*n_depth] *(CHLmax-CHLmin)+CHLmin)


RMSE_T = tf.sqrt(tf.reduce_mean(tf.square(T_testtot - OUT_Ttest), axis = 0))


plt.ioff
plt.figure(10)
plt.plot(RMSE_T, -np.arange(148) - 3, 'r', label='rmse temperature (°C)')
plt.xlabel('RMSE_T')
plt.title('Temperature')
plt.ylabel('Depth')
plt.savefig('RMSE_T.png')


RMSE_CHL = tf.sqrt(tf.reduce_mean(tf.square(CHL_testtot - OUT_CHLtest), axis = 0))

plt.figure(3)
plt.plot(RMSE_CHL, -np.arange(148) - 3, 'g', label='rmse chlorophyll (mg/kg)')
plt.xlabel('RMSE_CHL')
plt.title('Chlorophyll')
plt.ylabel('Depth')
plt.savefig('RMSE_CHL.png')

print("Losses printed to images!")

# plt.show()


