"""Python procedure for the 3D reconstruction of Chl and Temperature profile from surface observation based on the codes github from Bruno Buongiorno Nardelli, 2020

Bibliography reference, Bruno Buongiorno Nardelli, 2020
"""


import os

import matplotlib.pyplot as plt
import numpy as np
# from subprocess import call
import warnings

warnings.filterwarnings("ignore")  # specify to ignore warning messages
from keras.optimizers import SGD, Adam


from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Lambda

# import seawater as sw
import glob
from keras.models import load_model

import tensorflow as tf

import load_dataset
from load_dataset import load_dataset

from get_LSTM import get_LSTM_sequential, get_LSTM_functional


model_dir = './'
file_train = model_dir+'BIOARGO_MATCHUP_2018_2021.nc'
# loading the data cubes
X, Y, X_test, Y_test, minMaxDict = load_dataset(file_train)


####################################
activ = 'tanh'#'tanh'  # 'softsign'#
opt = 'Adam'#AdamW, AMSgrad=True
pat = 40
n_epochs = 20
val_split = .30
n_units1 = 35
n_units2 = 35
# n_units3 =n_unit

batch_size = 68

##################################

n_depth = X.shape[1]
n_samples= X.shape[0]
n_steps_out = 1#fixed

n_var_in=11
n_var_out=2


n_depth_test = X_test.shape[1]
n_samples_test= X_test.shape[0]
n_steps_out_test = 1#fixed

n_var_in_test=11
n_var_out_test=2





##################################################################
#
#   model definition/fit
#
#############################################################################################################
n_depth = X.shape[1]
n_samples= X.shape[0]
n_steps_out = 1#fixed


model_name_generic=model_dir + 'LSTM_' + str(n_units1) + '_' + str(n_units2) + '_MODEL_v1'

model_name = model_name_generic + '.keras'
model_name_es = model_name_generic + '_es'

model_name_history=model_dir+'LSTM_'+str(n_units1)+'_'+str(n_units2)+'_MODEL_v1.npy'

# model = get_LSTM_sequential(n_var_in, n_var_out, n_depth, n_units1, n_units2, activ)
model = get_LSTM_functional(n_var_in, n_var_out, n_depth, n_units1, n_units2, activ)

model.summary()

model.compile(loss='mse', optimizer=opt)

# fit model
es2 = [EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=pat,restore_best_weights=True),ModelCheckpoint(model_dir+'best-weights_v1_training.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)]
history=model.fit(X, Y, batch_size=batch_size, epochs=n_epochs, verbose=1, shuffle=False, validation_split=val_split, callbacks=[es2])


np.save(model_name_history, np.array(history))


model.save(model_name)
np.save(model_name_es, np.array(es2))
print("Saved model to disk")



from keras.models import load_model
model = load_model(model_name)

OUT_test = model.predict(X_test)

Tmax = minMaxDict['Tmax']
Tmin = minMaxDict['Tmin']
OUT_Ttest=(OUT_test[:,:,0]*(Tmax-Tmin))+Tmin

CHLmax = minMaxDict['CHLmax'] 
CHLmin = minMaxDict['CHLmin']
OUT_CHLtest=(OUT_test[:,:,1]*(CHLmax-CHLmin))+CHLmin
OUT_CHLtest=10**(OUT_CHLtest)


T_testtot= Y_test[:, :, 0]*(Tmax - Tmin) + Tmin
CHL_testtot= np.power(10., Y_test[:, :, 1] *(CHLmax-CHLmin)+CHLmin)


RMSE_T = tf.sqrt(tf.reduce_mean(tf.square(T_testtot - OUT_Ttest), axis = 0))

plt.figure(10)
plt.plot(RMSE_T, -np.arange(148) - 3, 'r', label='rmse temperature (°C)')
plt.xlabel('RMSE_T')
plt.title('Temperature')
plt.ylabel('Depth')
plt.savefig('RMSE_T_LSTM.png')

RMSE_T = tf.sqrt(tf.reduce_mean(tf.square(T_testtot - OUT_Ttest), axis = 0))
RMSE_CHL = tf.sqrt(tf.reduce_mean(tf.square(CHL_testtot - OUT_CHLtest), axis = 0))

plt.figure(3)
plt.plot(RMSE_CHL, -np.arange(148) - 3, 'g', label='rmse chlorophyll (mg/kg)')
plt.xlabel('RMSE_CHL')
plt.title('Chlorophyll')
plt.ylabel('Depth')
plt.savefig('RMSE_CHL_LSTM.png')




