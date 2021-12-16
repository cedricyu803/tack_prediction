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
# Rolling forecast with the good features and lags
# Commented out tensorflow and LSTM network
"""

#%% Workflow

"""
1. load labeled dataset and keep only columns we want to use
2. Rolling forecast: 
    sliding training window is 18 hours = 64800 seconds, validation window is 12 hours = 43200 seconds
    preprocessing
    fit model: logistic regression
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
from sklearn.decomposition import PCA   # for dimensionality reduction
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# f1 and f_beta metrics for XGBoost fitting
def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err

# beta
beta = 2
def fbeta_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-fbeta_score(y_true, np.round(y_pred), beta = beta)
    return 'fbeta_err', err


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

#%% tensorflow

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# # import tensorflow_addons as tfa # import tfa which contains metrics for regression
# # from tensorflow.keras import regularizers
# from keras.initializers import glorot_uniform
# from tensorflow.random import set_seed
# set_seed(0)

#%% Hyper-parameters

cols_to_keep = ['CurrentSpeed', 'CurrentDir', 'TWS', 'TWA', 'AWS', 'AWA', 'Roll',
       'Pitch', 'HeadingMag', 'HoG', 'HeadingTrue', 'AirTemp', 'SoG', 'SoS', 'AvgSoS', 'VMG', 'Leeway', 'TWD',
       'WSoG', 'VoltageDrawn', 'ModePilote', 'Yaw', 'Tacking']

# angles to convert to sines and cosines
triglist = ['CurrentDir', 'TWD', 'HoG', 'HeadingMag', 'HeadingTrue']

# angles to convert to principal values
ang_list = ['TWA', 'AWA', 'Pitch', 'Roll', 'Yaw', 'Leeway']

# not including present time step
# logistic regression:
# lags = 1 for best F1 and F_0.5 (prefer precision i.e. false negative is OK) scores
# lags = 2 for best F2 (prefer recall i.e. false positive is OK) scores
# lags = 4 for best AUC
lags = 1

# size of sliding windows in seconds
# 18 hours
window_size_train = 64800
# 12 hours
window_size_valid = 43200  # 21600  # tried 6 and 18 hours: worse

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
# train_df = train_df.iloc[21600:]
train_df_DateTime = train_df['DateTime']
# train_df_raw.shape

# train_df.shape
# (220000, 27)

# keep only columns we use
train_df = train_df[cols_to_keep]

# fillna with preceding value; do this here because validation set always has access to previously values that are in the training set
train_df = train_df.fillna(method='ffill')

#%% pre-processing

"""
preprocessing(df) takes a labeled/unlabeled dataset df and returns a preprocessed one
"""

# for nlags in [3,4,5,6,7,8]:
#     lags = nlags
    
def preprocessing(df):
    
    df_ = df.copy()
    
    # convert some angles to sines and cosines
    for trig in triglist:
        trig_sin = trig + '_sin'
        trig_cos = trig + '_cos'
        df_[trig_sin] = np.sin(df_[trig])
        df_[trig_cos] = np.cos(df_[trig])
    
    df_.drop(triglist, axis = 1, inplace = True)
    
    # convert other angles to principal branch (-180,180) degrees
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
# use_nn = False

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
    
    # re-shuffle training set to avoid sequence bias
    Xy_cols = history_rolling.columns
    np.random.seed(0)
    Xy_train_roll_shuffled = history_rolling.to_numpy().copy()
    np.random.shuffle(Xy_train_roll_shuffled)
    Xy_train_roll_shuffled = pd.DataFrame(Xy_train_roll_shuffled, columns = Xy_cols)
    
    # separate features and labels
    X_train_roll_shuffled = Xy_train_roll_shuffled.drop(['Tacking'], axis = 1)
    y_train_roll_shuffled = Xy_train_roll_shuffled['Tacking']
    
    # eimensionality reduction with PCA
    # a little faster for logistic regression (like 3 seconds vs 5 seconds), but detrimental to model scores
    # pca = PCA(n_components=60)
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
        
    # elif use_nn:
    #     X_train_roll_shuffled_scaled_nn = np.expand_dims(X_train_roll_shuffled_scaled, -1)
    #     X_valid_roll_scaled_nn = np.expand_dims(X_valid_roll_scaled, -1)
        
    #     history = model.fit(X_train_roll_shuffled_scaled_nn, y_train_roll_shuffled, 
    #                 validation_data=(X_valid_roll_scaled_nn, y_valid_roll),
    #                 class_weight=class_weight,
    #                 epochs=epochs, 
    #                 callbacks = [callback])
    #     histories_nn.append(history)
        
    #     # forecast
    #     y_pred = model.predict(X_valid_roll_scaled_nn)
    #     y_pred = y_pred.squeeze()
    #     y_pred = y_pred > 0.5
        
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
# without lag features
# 100%|██████████| 3/3 [00:01<00:00,  1.80it/s]
# with lag features
# 100%|██████████| 3/3 [00:02<00:00,  1.03it/s]


#%% Model performance and predictions

"""
# Summary of model performance
"""

validation_DateTime = train_df_DateTime.iloc[window_size_train:window_size_train+len(Xy_rest_partitioned)*window_size_valid]

# save forecast to file
# validation_DateTime.to_csv('predictions/validation_DateTime.csv')
# rolling_forecast.tofile('predictions/forecast.npy')


# y_valid = Xy_rest_partitioned[0]['Tacking'].to_numpy()
y_valid = pd.concat([X_chunk['Tacking'] for X_chunk in Xy_rest_partitioned]).to_numpy()

# !!!
print('\nlags = {}'.format(lags))
print('\nF_beta score (beta=1): ', fbeta_score(y_valid, rolling_forecast, beta=1))
print('F_beta score (beta=2): ', fbeta_score(y_valid, rolling_forecast, beta=2))
print('F_beta score (beta=0.5): ', fbeta_score(y_valid, rolling_forecast, beta=0.5))
print('AUC score: ', roc_auc_score(y_valid, rolling_forecast))

# rolling forecast
# logistic regression
# validation window = 12 hours
# C=0.1:
# lag 1
# F_beta score (beta=1):  0.3662066393136889      <-------
# F_beta score (beta=2):  0.4385385027693407
# F_beta score (beta=0.5):  0.3143570696721312    <-------
# AUC score:  0.7017156044933822
# lag 2
# F_beta score (beta=1):  0.36275502821588773
# F_beta score (beta=2):  0.4413421590028871      <-------
# F_beta score (beta=0.5):  0.30792473223936323
# AUC score:  0.7040762985207428
# lags = 3
# F_beta score (beta=1):  0.3579348824069496
# F_beta score (beta=2):  0.4408643307004418
# F_beta score (beta=0.5):  0.30126498002663116
# AUC score:  0.7042681570459348
# lags = 4
# F_beta score (beta=1):  0.35318444995864356
# F_beta score (beta=2):  0.44038778877887785
# F_beta score (beta=0.5):  0.29480806407069876
# AUC score:  0.704471137804471                   <-------
# lags = 5
# F_beta score (beta=1):  0.3475856487725924
# F_beta score (beta=2):  0.43814608269858546
# F_beta score (beta=0.5):  0.28804882410802113
# AUC score:  0.7034701368034703
# lags = 6
# F_beta score (beta=1):  0.34291410533311184
# F_beta score (beta=2):  0.43540629482744075
# F_beta score (beta=0.5):  0.2828327121245341
# AUC score:  0.7019728061394729
# lags = 7
# F_beta score (beta=1):  0.34005839320276876
# F_beta score (beta=2):  0.43450195328873453
# F_beta score (beta=0.5):  0.27934074936403225
# AUC score:  0.701633578022467
# lags = 8
# F_beta score (beta=1):  0.3377238515442512
# F_beta score (beta=2):  0.4338656975193385
# F_beta score (beta=0.5):  0.2764617149655817
# AUC score:  0.7014347681014348


# Confusion matrix, precision, recall

# the count of true negatives is C00, false negatives is C10, true positives is C11 and false positives is C01.
print(confusion_matrix(y_valid, rolling_forecast))
print('Precision score: ', precision_score(y_valid, rolling_forecast))
print('Recall score: ', recall_score(y_valid, rolling_forecast))
# lag 1
# array([[107699,  12181],
#        [  4811,   4909]], dtype=int64)
# Precision score:  0.28724400234055003    <------
# Recall score:  0.5050411522633745
# lag 2
# [[106970  12910]
#  [  4706   5014]]
# Precision score:  0.27973666592278507
# Recall score:  0.5158436213991769    
# lag 3
# [[106350  13530]
#  [  4652   5068]]
# Precision score:  0.27250241961501237
# Recall score:  0.5213991769547325
# lag 4
# [[105708  14172]
#  [  4596   5124]]
# Precision score:  0.26554726368159204
# Recall score:  0.5271604938271605
# lag 5
# [[105098  14782]
#  [  4566   5154]]
# Precision score:  0.25852728731942215
# Recall score:  0.5302469135802469
# lag 6
# [[104665  15215]
#  [  4560   5160]]
# Precision score:  0.2532515337423313
# Recall score:  0.5308641975308642
# lag 7
# [[104300  15580]
#  [  4537   5183]]
# Precision score:  0.249626739873814
# Recall score:  0.5332304526748971
# lag 8
# [[103981  15899]
#  [  4515   5205]]
# Precision score:  0.24663570887035632
# Recall score:  0.5354938271604939      <------


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
# plt.savefig('predictions/good_features_lag4_logreg_rolling_windows_18_12hrs')


# free VRAM after using tensorflow
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()

#%% logistic regression: feature coefficients [with lag 1 features]

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
# ['WSoG_lag_1' 'HeadingMag_cos_lag_1' 'SoG_lag_1' 'HoG_cos' 'HoG_cos_lag_1'
#  'HeadingMag_sin_lag_1' 'TWD_cos_lag_1' 'HoG_sin_lag_1' 'HeadingMag_cos'
#  'CurrentDir_cos' 'HeadingTrue_cos' 'HoG_sin' 'TWD_cos'
#  'HeadingTrue_cos_lag_1' 'CurrentDir_cos_lag_1' 'HeadingMag_sin'
#  'TWD_sin_lag_1' 'TWD_sin' 'CurrentDir_sin' 'CurrentDir_sin_lag_1']
# Largest Coefs: 
# ['AWA_lag_1' 'Yaw_lag_1' 'Yaw' 'AirTemp' 'AirTemp_lag_1' 'SoS' 'TWS_lag_1'
#  'TWS' 'TWA_lag_1' 'Pitch' 'Pitch_lag_1' 'Roll_lag_1' 'AWA' 'TWA'
#  'AWS_lag_1' 'VMG_lag_1' 'VoltageDrawn_lag_1' 'AWS' 'VoltageDrawn'
#  'CurrentSpeed']

# smallest absolute-coefficients
logreg_coef[:10]
# array([0.00041963, 0.00115628, 0.00697025, 0.00719586, 0.01011405,
#        0.01564169, 0.02760203, 0.02891603, 0.029022  , 0.03127004])
# largest absolute-coefficients
logreg_coef[:-11:-1]
# array([2.73040145, 2.72453614, 2.52166662, 2.39053482, 2.3293054 ,
#        2.04060126, 1.99097316, 1.96664918, 1.73755162, 1.65643076])


#%% logistic regression: feature coefficients [with lags=2 features]

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
# ['WSoG_lag_1' 'HeadingMag_cos_lag_1' 'SoG_lag_1' 'HoG_cos' 'HoG_cos_lag_1'
#  'HeadingMag_sin_lag_1' 'TWD_cos_lag_1' 'HoG_sin_lag_1' 'HeadingMag_cos'
#  'CurrentDir_cos' 'HeadingTrue_cos' 'HoG_sin' 'TWD_cos'
#  'HeadingTrue_cos_lag_1' 'CurrentDir_cos_lag_1' 'HeadingMag_sin'
#  'TWD_sin_lag_1' 'TWD_sin' 'CurrentDir_sin' 'CurrentDir_sin_lag_1']
# Largest Coefs: 
# ['AWA_lag_1' 'Yaw_lag_1' 'Yaw' 'AirTemp' 'AirTemp_lag_1' 'SoS' 'TWS_lag_1'
#  'TWS' 'TWA_lag_1' 'Pitch' 'Pitch_lag_1' 'Roll_lag_1' 'AWA' 'TWA'
#  'AWS_lag_1' 'VMG_lag_1' 'VoltageDrawn_lag_1' 'AWS' 'VoltageDrawn'
#  'CurrentSpeed']

# smallest absolute-coefficients
logreg_coef[:10]
# array([0.00041963, 0.00115628, 0.00697025, 0.00719586, 0.01011405,
#        0.01564169, 0.02760203, 0.02891603, 0.029022  , 0.03127004])
# largest absolute-coefficients
logreg_coef[:-11:-1]
# array([2.73040145, 2.72453614, 2.52166662, 2.39053482, 2.3293054 ,
#        2.04060126, 1.99097316, 1.96664918, 1.73755162, 1.65643076])




