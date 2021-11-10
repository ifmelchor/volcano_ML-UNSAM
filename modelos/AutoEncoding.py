#!/usr/bin/env python3
# coding=utf-8

#%%
import os, sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow import keras

sys.path.append( '..' )
from utils import LP_PSDs
#%%

# %load_ext tensorboard

#%%
train, test = LP_PSDs(test_size=0.3)
PSD_length = train.shape[1]
# hiperparametros
encoding_dim = 18 
compresion_factor = PSD_length/encoding_dim
#%%

# simple AE
#%%
input_img = keras.layers.Input(shape=(PSD_length,))
encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_img)
decoded = keras.layers.Dense(PSD_length, activation='sigmoid')(encoded)
AE = keras.models.Model(input_img, decoded)
keras.utils.plot_model(AE, show_shapes=True)
#%%

# AE.compile(optimizer='adam', loss='binary_crossentropy')

# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# # %tensorboard --logdir logs

# history = AE.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test),
#                 callbacks=[tensorboard_callback]
#                 )

# x_pred = AE.predict(x_test)

# loss = np.zeros(len(x_test))
# for nx, x in enumerate(x_test):
#   loss[nx]=(1/PSD_length)*np.sum([-x[j]*np.log(x_pred[nx][j]) for j in range(PSD_length)])

# plt.hist(loss)
# plt.axvline(0.08,color='black',linestyle='dotted')
# # plt.show() # a partir de loss = 0.08, para mi es anomalia