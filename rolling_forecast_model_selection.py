# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:30:00 2021

@author: Cedric Yu
"""

#%%
"""
#####################################


The Client is a marine electronics company. They produce hardware and software for sailing yachts. They have installed some sensors on one of the boats and provide the dataset they’ve collected. They don’t have any experience with machine learning, and they don’t have a solid understanding of what exactly they want to do with this data. The .csv file with the data and data dictionary are provided.

Your first task is to analyse the data, give the client some feedback on data collection and handling process, and suggest some ideas of how this data can be used to enhance their product and make it more popular among professional sailors and boat manufacturers.

Your supervisor told you that on top of whatever you come up with, what you should definitely do is ‘tack prediction’. “A tack is a specific maneuver in sailing and alerting the sailor of the necessity to tack in the near future would bring some advantage to them compared to other sailors, who would have to keep an eye out on the conditions all the time to decide when to tack,” he writes in his email. The supervisor, who has some experience in sailing labels the tacks in the data from the client (added as ‘Tacking’ column in the data). 
Your second task is to build a forecasting model that would be alerting sailors of the tacking event happening ahead.

#####################################
Dataset

File descriptions

# test_data.csv - Input features and target 'Tacking' for the training set (220,000 rows).

# Data fields


# Latitude: Latitudinal coordinate
# Longitude: Longitudinal coordinate

# SoG: knots. Speed over ground
# SoS: knots. Speed over surface (water surface)
# AvgSoS: knots. Average speed over surface per minute
# VMG: knots. Velocity made good = speed at which you are making progress directly upwind or downwind. Calculated as (Speed over surface) * cos(True Wind angle)

# CurrentSpeed: knots. speed of water current
# CurrentDir: degrees. wrt. true heading or magnetic heading??
# TWS: knots. True Wind Speed, relative to water
# TWA: degrees. True Wind Angle, angle between [wind relative to water] and [boat direction]
# TWD: degrees. True wind direction, wrt. true heading or magnetic heading??
# AWS: knots. Apparent Wind Speed, relative to the boat
# AWA: degrees. Apparent Wind Angle, angle between [wind relative to boat] and [boat direction]
# WSoG: knots. Wind speed over ground, i.e. relative to ground

# HeadingTrue: degrees. true heading. True heading - heading over ground = Yaw
# HeadingMag: degrees. magnetic heading
# Roll: degrees. Roll, also equals to -Heel
# Pitch: degrees. Pitch angle
# Yaw: degrees. = True heading - heading over ground
# HoG: degrees. heading over ground, i.e. heading of the course over ground
# AirTemp: degrees Celcius. air temperature
# RudderAng: degrees. Rudder angle
# Leeway: degrees. 

# VoltageDrawn: Volts. Voltage drawn from the boat batteries that power the navigation system and (at night) lights
# ModePilote: unclear. whether the autopilot is active or not

# Target
# Tacking: Boolean.

"""

#%% This file

"""
# Feature pre-processing and engineering
model evaluation with a limited choice of features (see below)
# Rolling forecast
"""

#%% Workflow

"""
1. load labeled dataset and keep only a limited choice of features
2. Rolling forecast: 
    sliding training window is 18 hours = 64800 seconds, validation window is 12 hours = 43200 seconds
    preprocessing
    fit models: logistic regression, XGBoost, LSTM
"""

#%% Preamble

import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import matplotlib.pyplot as plt
import gc

# import dask
# import dask.dataframe as dd

import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\tacking')


from tqdm import tqdm    # fancy progress bar for the for-loop
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score, roc_auc_score, f1_score

# f1 and f_beta metrics for XGBoost fitting
def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err

# beta
beta = 1
def fbeta_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-fbeta_score(y_true, np.round(y_pred), beta = beta)
    return 'fbeta_err', err


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

#%% tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_addons as tfa # import tfa which contains metrics for regression
# from tensorflow.keras import regularizers
from keras.initializers import glorot_uniform
from tensorflow.random import set_seed
set_seed(0)

#%% Hyper-parameters

# only use a limited selection of features
cols_to_keep = ['CurrentSpeed', 'CurrentDir', 'AWS', 'AWA', 'Roll', 'Pitch', 'HoG', 'AirTemp', 'SoS', 'AvgSoS', 'VMG', 'Leeway', 'TWD', 'WSoG', 'Yaw', 'Tacking']

# angles to convert to sines and cosines
triglist = ['CurrentDir', 'TWD', 'HoG']

# angles to convert to principal values
ang_list = ['AWA', 'Pitch', 'Roll', 'Yaw', 'Leeway']

# not including present time step
lags = 5

# size of sliding windows in seconds
# 18 hours
window_size_train = 64800
# 12 hours
window_size_valid = 43200  # 21600  # tried 6 and 18 hours: worse

# PCA for dimensionality reduction
n_components = 60

# logistic regression
# regularisation hyperparameter C (smaller means more regularisation)
C=.1

# hyperparameters for LSTM
learning_rate = 5e-6
epochs = 10
class_weight = {0: 5 / (2 * 4), 1: 5 / (2 * 1)}

#%% load dataset

train_df = pd.read_csv('test_data.csv')
# train_df_raw = pd.read_csv('test_data.csv')
train_df['DateTime'] = pd.to_datetime(train_df['DateTime'])
train_df_DateTime = train_df['DateTime']
# train_df_raw.shape

train_df.shape
# (220000, 27)

# keep only columns we use
train_df = train_df[cols_to_keep]

# fillna with preceding value; do this here because validation set always has access to previously values that are in the training set
train_df = train_df.fillna(method='ffill')

#%% pre-processing

"""
preprocessing(df) takes a labeled/unlabeled dataset df and returns a preprocessed one
"""

def preprocessing(df):
    
    df_ = df.copy()
    
    # convert some angles to sines and cosines
    for trig in triglist:
        trig_sin = trig + '_sin'
        trig_cos = trig + '_cos'
        df_[trig_sin] = np.sin(df_[trig])
        df_[trig_cos] = np.cos(df_[trig])
    
    df_.drop(triglist, axis = 1, inplace = True)
    
    # convert other angles (['AWA', 'Pitch', 'Roll', 'Yaw', 'Leeway']) to principal branch (-180,180) degrees
    for ang in ang_list:
        df_[ang] = np.arctan2(np.sin(df_[ang]*np.pi/180),
                             np.cos(df_[ang]*np.pi/180)) * 180 / np.pi
    
    
    feat_columns = list(df_.columns)
    if 'Tacking' in feat_columns:
        labeled = True
    else:
        labeled = False
    
    # generate lag features
    if lags > 0:
        if labeled:
            feat_columns.remove('Tacking')
            # df_['Tacking_lag_1'] = df_['Tacking'].shift(1)  # DO NOT USE 'Tacking_lag_1
        for col in feat_columns:
            for i in range(lags):
                lag_col_name = col + '_lag_' + str(i+1)
                df_[lag_col_name] = df_[col].shift(i+1)
    
    df_ = df_.dropna()
    
    # put target column in front
    if labeled:
        df_ = pd.concat([df_['Tacking'], df_.drop(['Tacking'], axis = 1)], axis=1)

    return df_

# train_df = preprocessing(train_df)

#%% Rolling forecast

"""
train-validation split: sliding training window is 18 hours=64800 seconds, validation and prediction windows are 12 hours = 43200 seconds.

Given an initial dataset, take the first 18 hours as training set, and divide the rest into 3 (complete) chunks of 12 hours. 
If the last chunk does not have 43200 instances, ignore until the sensors gather enough data.
Thereafter, once we gather 12 hours of data, it is appended to the existing dataset (so we can fillna with previous values) and retrain the model with new train/validation windows.
"""

# preprocess dataset
train_df_processed = preprocessing(train_df)
train_df_DateTime = train_df_DateTime.iloc[lags:]

# initial training set
Xy_train = train_df_processed.iloc[:window_size_train]

# anything after the initial training period
Xy_rest = train_df_processed.iloc[window_size_train:]


# partition Xy_rest into chunks of validation sets of 12 hours = 43200 seconds
# if the last chunk does not have 43200 instances, ignore until the sensors gather enough data
Xy_rest_partitioned = []
for i in range(len(Xy_rest)//window_size_valid):
    Xy_chunk = Xy_rest.iloc[window_size_valid*i:window_size_valid*(i+1)]
    Xy_rest_partitioned.append(Xy_chunk)

"""
Rolling forecast
"""

#!!! Start Here
# rolling preprocessed training set
history_rolling = Xy_train.copy()
# rolling forecast predictions
rolling_forecast = np.array([])
# LSTM model history at each time step
histories_nn = []


# model: choose one by uncommenting the apropriate lines
# initialise one model, then re-train the same model in each sliding window
use_logistic = False
use_XGBoost = False
use_nn = False

# logistic regression
use_logistic = True
model = LogisticRegression(max_iter = 10000, class_weight='balanced', C=C)

# XGBoost
# use_XGBoost = True
# model = XGBClassifier(scale_pos_weight = 5., verbosity=0)

# neural network
# use_nn = True
# model = tf.keras.models.Sequential([
#     layers.InputLayer(input_shape=(history_rolling.shape[1]-1, 1)),
#     # layers.Bidirectional(layers.LSTM(16, return_sequences=True)),
#    layers.Bidirectional(layers.LSTM(16, return_sequences=False)),
#    layers.Dropout(0.2),
#    layers.BatchNormalization(),
#   # tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4)),
#     layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4)),
#     layers.Dense(1, activation = 'sigmoid')
# ])
#
# callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', mode = 'min', patience=30, min_delta = 0.00001, restore_best_weights=True)
#
# model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),metrics=[tf.keras.metrics.AUC()])


# for i in range(1):
for i in tqdm(range(len(Xy_rest_partitioned))):
    
    Xy_valid_roll = Xy_rest_partitioned[i]
    
    # separate features and labels
    X_valid_roll = Xy_valid_roll.drop(['Tacking'], axis = 1)
    y_valid_roll = Xy_valid_roll['Tacking']
    
    # Re-shuffle training set to avoid sequence bias
    Xy_cols = history_rolling.columns
    np.random.seed(0)
    Xy_train_roll_shuffled = history_rolling.to_numpy().copy()
    np.random.shuffle(Xy_train_roll_shuffled)
    Xy_train_roll_shuffled = pd.DataFrame(Xy_train_roll_shuffled, columns = Xy_cols)
    
    # separate features and labels
    X_train_roll_shuffled = Xy_train_roll_shuffled.drop(['Tacking'], axis = 1)
    y_train_roll_shuffled = Xy_train_roll_shuffled['Tacking']
    
    # Dimensionality reduction with PCA
    # a little faster for logistic regression (like 3 seconds vs 5 seconds), but detrimental to model scores
    # pca = PCA(n_components=n_components)
    # pca.fit(X_train_roll_shuffled)
    # X_train_roll_shuffled = pca.transform(X_train_roll_shuffled)
    # X_valid_roll = pca.transform(X_valid_roll)
    
    # feature normalisation
    scaler = MinMaxScaler()
    X_train_roll_shuffled_scaled = scaler.fit_transform(X_train_roll_shuffled)
    X_valid_roll_scaled = scaler.transform(X_valid_roll)
    
    
    # re-train the same model and make predictions
    if use_logistic:
        model.fit(X_train_roll_shuffled_scaled, y_train_roll_shuffled)
        
        # forecast
        y_pred = model.predict(X_valid_roll_scaled)
        
    elif use_XGBoost:
        model.fit(X_train_roll_shuffled_scaled, y_train_roll_shuffled, 
                  eval_set = [(X_train_roll_shuffled_scaled, y_train_roll_shuffled), (X_valid_roll_scaled, y_valid_roll)], 
                  eval_metric=fbeta_eval,
                  early_stopping_rounds = 30, 
                  verbose=0)
        
        # forecast
        y_pred = model.predict(X_valid_roll_scaled)
        
    elif use_nn:
        X_train_roll_shuffled_scaled_nn = np.expand_dims(X_train_roll_shuffled_scaled, -1)
        X_valid_roll_scaled_nn = np.expand_dims(X_valid_roll_scaled, -1)
        
        history = model.fit(X_train_roll_shuffled_scaled_nn, y_train_roll_shuffled, 
                    validation_data=(X_valid_roll_scaled_nn, y_valid_roll),
                    class_weight=class_weight,
                    epochs=epochs, 
                    callbacks = [callback])
        histories_nn.append(history)
        
        # forecast
        y_pred = model.predict(X_valid_roll_scaled_nn)
        y_pred = y_pred.squeeze()
        y_pred = y_pred > 0.5
        
    else: 
        print('Pick a model.')
        break;
    
    # append forecast
    rolling_forecast = np.hstack((rolling_forecast,y_pred))
    
    # update training set
    history_rolling = pd.concat([history_rolling.iloc[window_size_valid:window_size_train], Xy_valid_roll])


# save trained LSTM model
# model.save('LSTM_model.h5')

# logistic regression
# 12 hour validation window
# 100%|██████████| 7/7 [00:05<00:00,  1.31it/s]
# XGBoost
# 100%|██████████| 7/7 [00:57<00:00,  8.20s/it]
# LSTM
# 100%|██████████| 3/3 [23:23<00:00, 467.72s/it]

"""
# Summary of model performance
"""

validation_DateTime = train_df_DateTime.iloc[window_size_train:window_size_train+len(Xy_rest_partitioned)*window_size_valid]

# save forecast to file
# validation_DateTime.to_csv('predictions/validation_DateTime.csv')
# rolling_forecast.tofile('predictions/forecast.npy')


# y_valid = Xy_rest_partitioned[0]['Tacking'].to_numpy()
y_valid = pd.concat([X_chunk['Tacking'] for X_chunk in Xy_rest_partitioned]).to_numpy()

print('\nF_beta score (beta=1): ', fbeta_score(y_valid, rolling_forecast, beta=1))
print('F_beta score (beta=2): ', fbeta_score(y_valid, rolling_forecast, beta=2))
print('F_beta score (beta=0.5): ', fbeta_score(y_valid, rolling_forecast, beta=0.5))
print('AUC score: ', roc_auc_score(y_valid, rolling_forecast))

# Without Tacking_lag_1 feature:
# rolling forecast
# logistic regression
# validation window = 6 hours
# F_beta score (beta=1):  0.20442750195195888
# F_beta score (beta=2): 0.30609578301653234
# F_beta score (beta=0.5): 0.1534573587819947
# AUC score:  0.6251476455250841
# C=0.1:
# F_beta score (beta=1):  0.20991266768704422
# F_beta score (beta=2): 0.316831548622058
# F_beta score (beta=0.5): 0.1569484086380527
# AUC score:  0.6336851694782144

# validation window = 12 hours
# C=0.1:
# F_beta score (beta=1):  0.3444506001846723
# F_beta score (beta=2): 0.4145994487418867    <---------
# F_beta score (beta=0.5): 0.29460449835734137
# AUC score:  0.6869299855410966
# favours recall

# XGBoost 
# validation window = 12 hours
# F_beta score (beta=1):  0.0016998370989446843
# AUC score:  0.4851133257939874

# LSTM
# validation window = 12 hours
# F_beta score (beta=1):  0.2563152114222954
# F_beta score (beta=2):  0.21347877789974384
# F_beta score (beta=0.5):  0.32065814784281393  <---------
# AUC score:  0.5836058280502725
# favours precision


# With Tacking_lag_1 feature:
# rolling forecast
# logistic regression
# first validation window
# F_beta score (beta=1):  0.9996153846153846
# F_beta score (beta=2):  0.9996707886668997
# F_beta score (beta=0.5):  0.999609085671961
# AUC score:  0.9997813765182185 
# rolling forecast
# F_beta score:  0.9996399362172728
# AUC score:  0.9998315427386675

# XGBoost 
# first validation window (beta=1)
# F_beta score:  0.9996153846153846
# AUC score:  0.9997813765182185
# rolling forecast (beta=1)
# F_beta score (beta=1):  0.9979972269295948   # worse than naive forecast
# AUC score:  0.9997184525492413
# rolling forecast (beta=2)
# F_beta score (beta=2):  0.9990130158534328
# rolling forecast (beta=0.5)
# F_beta score (beta=0.5):  0.9969835016005909   # worse than naive forecast

"""
# Plot forecast against ground truth
"""
plt.figure(dpi=150)
plt.plot(validation_DateTime, rolling_forecast, color = 'tomato', label = 'Model forecast')
plt.plot(validation_DateTime, y_valid, color = 'black', label='Ground truth')
ax = plt.gca()
ax.set_xlabel('Date-Hour')
ax.legend(loc='upper right')
ax.set_ylabel('Tacking')
ax.set_yticks([0, 1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('predictions/rolling__windows_18_12hrs')


"""
# Confusion matrix
"""

from sklearn.metrics import confusion_matrix
# the count of true negatives is C00, false negatives is C10, true positives is C11 and false positives is C01.
confusion_matrix(y_valid, rolling_forecast)


# Without Tacking_lag_1 feature: 
# # logistic regression
# validation window = 6 hours
# less false negative than false positive 
# array([[112105,  29375],
#        [  5269,   4451]], dtype=int64)
# precision = 4451/(29375+4451) = 0.13158517117010585
# recall= 0.4579218106995885

# validation window = 12 hours
# C=0.1
# array([[107188,  12692],
#        [  5057,   4663]], dtype=int64)
# precision = 4663/(4663+12692) = 0.26868337654854507
# recall= 0.47973251028806585

# XGBoost beta=1
# validation window = 12 hours
# less false positive than false negative 
# array([[137093,   4387],
#        [  9708,     12]], dtype=int64)
# precision = 12/(4387+12) = 0.00272789270288702
# recall= 0.0012345679012345679

# LSTM
# validation window = 12 hours
# array([[116899,   2981],
#        [  7853,   1867]], dtype=int64)
# precision = 1867/(1867+2981) = 0.3851072607260726
# recall= 0.1920781893004115

# With Tacking_lag_1 feature
# logistic regression
# array([[141476,      4],
#        [     3,   9717]], dtype=int64)

# XGBoost beta=1
# array([[141444,     36],
#        [     3,   9717]], dtype=int64)




# free VRAM after using tensorflow
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()



#%% logistic regression: feature coefficients [without target lag 1 feature]

""" feature coefficients"""
# get the feature names as numpy array
feature_names = np.array(list(train_df_processed.drop(['Tacking'], axis = 1).columns))
# Sort the [absolute values] of coefficients from the model
logreg_coef = np.abs(model.coef_[0]).copy()
logreg_coef.sort()
sorted_coef_logreg = np.abs(model.coef_[0]).argsort()

# Find the 20 smallest and 10 largest absolute-coefficients
print('Smallest Coefs:\n{}'.format(feature_names[sorted_coef_logreg[:20]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_logreg[:-21:-1]]))
# Smallest Coefs:
# ['HoG_cos' 'HoG_cos_lag_2' 'HoG_cos_lag_3' 'HoG_cos_lag_1' 'HoG_cos_lag_5'
#  'HoG_cos_lag_4' 'HoG_sin_lag_5' 'HoG_sin_lag_3' 'HoG_sin_lag_2'
#  'HoG_sin_lag_4' 'WSoG_lag_2' 'HoG_sin_lag_1' 'HoG_sin'
#  'CurrentDir_cos_lag_4' 'CurrentDir_cos_lag_3' 'CurrentDir_cos_lag_5'
#  'TWD_sin_lag_5' 'CurrentDir_cos_lag_2' 'TWD_cos_lag_1' 'CurrentDir_cos']
# Largest Coefs: 
# ['SoS' 'Yaw_lag_4' 'Yaw_lag_5' 'Yaw_lag_3' 'CurrentSpeed' 'Yaw_lag_2'
#  'Pitch_lag_4' 'Pitch_lag_3' 'Pitch_lag_5' 'AWA_lag_5' 'SoS_lag_1'
#  'Yaw_lag_1' 'AvgSoS' 'AWS_lag_5' 'Pitch_lag_2' 'AvgSoS_lag_5' 'Pitch'
#  'Pitch_lag_1' 'AirTemp' 'AirTemp_lag_1']

# smallest absolute-coefficients
logreg_coef[:10]
# array([0.00048453, 0.00190568, 0.00277005, 0.00664827, 0.01515842,
#        0.01549623, 0.01875815, 0.02201448, 0.0226287 , 0.02425764])
# largest absolute-coefficients
logreg_coef[:-11:-1]
# array([3.17048214, 3.04464682, 2.85167851, 2.79872434, 2.39916501,
#        2.38034591, 2.23961572, 2.02752777, 1.91030257, 1.89853359])




#%% Models: Rolling forecast [with target lag 1 feature]

from sklearn.metrics import fbeta_score, roc_auc_score

#################################
# Naive forecast: y_pred(t) = y_true(t-1)

y_pred_naive = train_df['Tacking'].shift(1).iloc[window_size_train:window_size_train+window_size_valid*len(Xy_rest_partitioned)]


print('F_beta score: ', fbeta_score(y_valid, y_pred_naive, beta=1))
# F_beta score:  0.9981481481481481
print('F_beta score: ', fbeta_score(y_valid, y_pred_naive, beta=2))
# F_beta score:  0.998148148148148
print('F_beta score: ', fbeta_score(y_valid, y_pred_naive, beta=0.5))
# F_beta score:  0.998148148148148
print('AUC score: ', roc_auc_score(y_valid, y_pred_naive))
# AUC score:  0.9990104608425218

#################################

# logistic regression

""" feature coefficients"""
# get the feature names as numpy array
feature_names = np.array(list(train_df_processed.drop(['Tacking'], axis = 1).columns))
# Sort the [absolute values] of coefficients from the model
logreg_coef = np.abs(model.coef_[0]).copy()
logreg_coef.sort()
sorted_coef_logreg = np.abs(model.coef_[0]).argsort()

# Find the 20 smallest and 10 largest absolute-coefficients
print('Smallest Coefs:\n{}'.format(feature_names[sorted_coef_logreg[:20]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_logreg[:-21:-1]]))
# Smallest Coefs:
# ['AWA_lag_2' 'AWA_lag_5' 'CurrentDir_cos_lag_2' 'VMG'
#  'CurrentDir_sin_lag_4' 'VMG_lag_3' 'AWA_lag_3' 'TWD_sin_lag_2'
#  'WSoG_lag_5' 'AWS_lag_1' 'AWA_lag_4' 'Roll' 'HoG_sin_lag_3' 'AWA'
#  'AWA_lag_1' 'WSoG_lag_2' 'Leeway' 'AWS' 'Pitch' 'Pitch_lag_5']
# Largest Coefs: 
# ['Tacking_lag_1' 'CurrentSpeed' 'CurrentSpeed_lag_5' 'CurrentSpeed_lag_1'
#  'CurrentSpeed_lag_3' 'CurrentSpeed_lag_4' 'CurrentSpeed_lag_2'
#  'Yaw_lag_1' 'Yaw' 'Yaw_lag_2' 'Yaw_lag_4' 'AirTemp_lag_5' 'AirTemp'
#  'AirTemp_lag_2' 'AirTemp_lag_1' 'Yaw_lag_5' 'AirTemp_lag_4'
#  'AirTemp_lag_3' 'Yaw_lag_3' 'CurrentDir_cos_lag_4']

# smallest absolute-coefficients
logreg_coef[:10]
# array([0.00564613, 0.00707776, 0.0079667 , 0.00967982, 0.01727488,
#        0.01771446, 0.01844673, 0.02493154, 0.02676159, 0.0294121 ])
# largest absolute-coefficients
logreg_coef[:-11:-1]
# array([13.63693977,  1.56552944,  1.53659163,  1.50086433,  1.47638369,
#         1.46183099,  1.45842771,  0.73787632,  0.73123569,  0.64945981])

# largest coefficient is 'Tacking_lag_1'; ~ naive forecasting


#################################

XGBC_model_feature_importances = pd.Series(model.feature_importances_, index = train_df_processed.drop(['Tacking'],axis=1).columns).sort_values(ascending = False)
XGBC_model_feature_importances = XGBC_model_feature_importances / XGBC_model_feature_importances.max()

# essentially just naive forecasting

# Tacking_lag_1           1.000000  <----------
# AWA_lag_4               0.000046
# CurrentSpeed_lag_4      0.000044
# CurrentSpeed_lag_3      0.000043
# AirTemp                 0.000031
# AirTemp_lag_4           0.000010
# WSoG_lag_4              0.000000
# CurrentDir_sin_lag_3    0.000000


#################################




