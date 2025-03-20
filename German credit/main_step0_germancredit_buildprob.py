# -*- coding: utf-8 -*-
"""
Step 0. Construct feasible region
"""


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.models import Model
from keras import regularizers
from tensorflow.python.keras.utils.generic_utils import class_and_config_for_serialized_keras_object
from tqdm import tqdm
import numpy as np
import shap as shap_util
from itertools import combinations


import math
import tensorflow as tf
import keras
import copy

from keras import backend as K
from keras.models import load_model

import random
import numpy as np
import pandas as pd
import os
path = '...\GermanCredit'
print(path)
os.chdir(path)


import matplotlib.pyplot as plt
import seaborn as sns
#%%
#import shapley_regression

def build_dataset(random_seed):
    dataset = pd.read_csv('german_processed.csv')
    #gender = dataset['Gender']
    dataset = dataset.drop(columns = ['PurposeOfLoan','OtherLoansAtStore']) #
    GoodCustomer = {-1: 0, 1:1}
    dataset = dataset.replace({'GoodCustomer': GoodCustomer})
    Gender = {'Female': 0, 'Male':1}
    dataset = dataset.replace({'Gender': Gender})
    y_raw = dataset['GoodCustomer']
    X_raw = dataset.drop(columns = ['GoodCustomer'])


    # normalize data (this is important for model convergence)
    for k in ['Age','LoanDuration','LoanAmount']:
        X_raw[k] -= X_raw[k].mean()
        X_raw[k] /= X_raw[k].std()
    X_train, X_valid, y_train, y_valid = train_test_split(X_raw, y_raw, test_size=0.2, random_state=random_seed-3)
    dtypes = list(zip(X_train.dtypes.index, map(str, X_train.dtypes)))
    X_train_array = np.array([X_train[k].values for k,t in dtypes]).T
    X_valid_array = np.array([X_valid[k].values for k,t in dtypes]).T
    return X_train, X_valid, y_train, y_valid, X_train_array, X_valid_array


### build a simple 4-layer regression model
def build_regression(X_train_array, X_valid_array, y_train, y_valid, random_seed): 
    input_els = Input(shape=(X_train_array.shape[1],))
    layer1 = (Dense(100, activation="relu")(input_els))
    layer2 = Dropout(0.5)(Dense(100, activation="relu")(layer1))
    layer3 = Dropout(0.5)(Dense(100, activation="relu")(layer2))
    layer4 = Dropout(0.5)(Dense(100, activation="relu")(layer3))
    out = Dense(1, activation='sigmoid')(layer4)
    # train model
    regression = Model(inputs=input_els, outputs=[out])
    regression.compile(optimizer="adam", loss='mse')

    trained = False #use False to enable the training

    # train model
    if not trained:
        regression.fit(
            X_train_array,
            y_train,
            epochs=20,
            batch_size=512,
            shuffle=True,
            validation_data=(X_valid_array, y_valid)
        )
        regression.save_weights('./german_model_deep_{}.ckpt'.format(random_seed))
        #regression.save('original_model_deep_{}.h5'.format(random_seed))
    else:
        #regression = load_model('original_model_deep_{}.h5'.format(random_seed))
        regression.load_weights('./german_model_deep_{}.ckpt'.format(random_seed))
    return regression



def build_prob_data(regression, X_train_array, y_valid):
    sh1 = X_train_array.shape[0]
    X_train_array_fake = np.zeros((X_train_array.shape[0]*100, X_train_array.shape[1]))
    y_train_fake = np.zeros((X_train_array.shape[0]*100,3))
    # y_fake has size of (N, 3)
    # first dimension is prediction of regression
    # second dimension is whether on-manifold or off-manifold, 1 is on-manifold
    # third dimension is set to 0 now
    y_pred = regression.predict(X_train_array)[:,0]

    for i in range(50):
        X_train_array_fake[sh1*i: sh1*(i+1), :] = X_train_array
        y_train_fake[sh1*i: sh1*(i+1),0] = y_pred
        y_train_fake[sh1*i: sh1*(i+1),1] = 0

    for i in range(50):
        X_train_array_fake[(50+i)*sh1:(51+i)*sh1 , :] = X_train_array
        y_train_fake[(50+i)*sh1:(51+i)*sh1 ,1] = 1

    mask = np.reshape(np.random.choice(2,size=(X_train_array.shape[0])*50*X_train_array.shape[1]),((X_train_array.shape[0])*50,X_train_array.shape[1]))
    mask = np.concatenate([mask,np.zeros((X_train_array.shape[0]*50,X_train_array.shape[1]))],axis=0)

    X_train_array_fake[np.where(mask>0)] = 0
    for i in range(100):
        # X_train_array_fake[sh1*i: sh1*(i+1), :] = X_train_array
        y_train_fake[sh1*i: sh1*(i+1),0] = regression.predict(X_train_array_fake[sh1*i: sh1*(i+1), :])[:,0]

    return X_train_array_fake, y_train_fake


def build_prob_model(X_train_array_fake, y_train_fake, random_seed):
    # build prob_model (ooD detector), which predicts whether x is 1  (on-mainfold) 0 (off-manifold)
    input_prob = Input(shape=(X_train_array.shape[1],))
    layer1_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(input_prob)
    layer2_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer1_prob)
    layer3_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer2_prob)
    out_prob = Dense(1)(layer3_prob)

    trained = False #True
    model_prob = Model(inputs=input_prob, outputs=[out_prob])
    opt = keras.optimizers.SGD(learning_rate=1e-3, momentum = 0.9)
    #model_prob.compile(optimizer=opt, loss='mean_squared_error')
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    model_prob.compile(optimizer=opt, loss=loss)
    # train model
    if not trained:
        model_prob.fit(
            X_train_array_fake,
            y_train_fake[:,1],
            epochs=30,
            batch_size=1000,
            shuffle=True,
        )
        model_prob.save_weights('./german_prob_model_deep_{}.ckpt'.format(random_seed))
        #model_prob.save('prob_model_deep_{}.h5'.format(random_seed))
    else:
        #model_prob = load_model('prob_model_deep_{}.h5'.format(random_seed))
        model_prob.load_weights('./german_prob_model_deep_{}.ckpt'.format(random_seed))
    return model_prob

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# transfer ODD prob to density, 1/17 is an arbitrary constant
def prob_NCE(x):     #----> calculate prob!
    h = model_prob.predict(x)
    G =  np.clip(sigmoid(h),0.01,0.99)
    return G / (1-G) / 100


def pdf(X):
    return prob_NCE(X).flatten()

#%% Build feasible region and save the indicator function

random_seed = 1234
random.seed(random_seed)
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
#os.environ['PYTHONHASHSEED']=str(random_seed)

X_train, X_valid, y_train, y_valid, X_train_array, X_valid_array = build_dataset(random_seed)

regression = build_regression(X_train_array, X_valid_array, y_train, y_valid, random_seed)
X_train_array_fake, y_train_fake = build_prob_data(regression, X_train_array, y_valid)
model_prob= build_prob_model(X_train_array_fake, y_train_fake, random_seed)
    
plt.hist(pdf(X_train))
plt.show()
plt.hist(pdf(X_valid))
plt.show()
