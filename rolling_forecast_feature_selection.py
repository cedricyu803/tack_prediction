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
# Find the good features and lags with Rolling forecast
# use logistic regression with C=0.1
# Commented out tensorflow and LSTM network
# Note: turns out XGBoost performs much better
"""

#%% Workflow

"""
1. load labeled dataset, aggregate by minute and keep only columns we want to use
2. Rolling forecast: 
    sliding training window is 18 hours = 1080 minutes, validation window is 1 minute
    preprocessing
    fit model: logistic regression with C=0.1
    compute evaluation metrics
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

# # f1 and f_beta metrics for XGBoost fitting
# def f1_eval(y_pred, dtrain):
#     y_true = dtrain.get_label()
#     err = 1-f1_score(y_true, np.round(y_pred))
#     return 'f1_err', err

# # beta
# beta = 2
# def fbeta_eval(y_pred, dtrain):
#     y_true = dtrain.get_label()
#     err = 1-fbeta_score(y_true, np.round(y_pred), beta = beta)
#     return 'fbeta_err', err


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


num_cols = ['CurrentSpeed', 'CurrentDir', 'TWS', 'TWA', 'AWS', 'AWA', 'Roll',
       'Pitch', 'HeadingMag', 'HoG', 'HeadingTrue', 'AirTemp', 'SoG', 'SoS', 'AvgSoS', 'VMG', 'Leeway', 'TWD', 'WSoG',
        'VoltageDrawn', 'Yaw']

cat_cols = ['ModePilote', 'Tacking']

cols_to_keep = num_cols + cat_cols

# angles to convert to sines and cosines
triglist = ['CurrentDir', 'TWD', 'HoG', 'HeadingMag', 'HeadingTrue']

# angles to convert to principal values
ang_list = ['TWA', 'AWA', 'Pitch', 'Roll', 'Yaw', 'Leeway']

# not including present time step
# logistic regression:
# lags = 5 for recall oriented
# lags = 10 for precision oriented
lags = 5

# size of sliding windows in minutes
# 18 hours
window_size_train = 1080
# validation (test) window is fixed to 1 minute; DO NOT change it
# we are not forecasting other varibles. given what we know, we can only predict one step ahead.
window_size_valid = 1
# 1 minute
# window_size_test = 1

# logistic regression
# regularisation hyperparameter C (smaller means more regularisation)
C=.1

# hyperparameters for LSTM
learning_rate = 5e-6
epochs = 10
patience = 3   # early stopping based on val_loss
class_weight = {0: 5 / (2 * 4), 1: 5 / (2 * 1)}

#%% load dataset

train_df = pd.read_csv('test_data.csv')

# train_df.shape
# (220000, 27)

# fillna with preceding value; do this here because validation set always has access to previously values that are in the training set
train_df = train_df.fillna(method='ffill')


#%% parse datetime, group by minute


train_df['DateTime'] = pd.to_datetime(train_df['DateTime'])

def get_day(row):
    return row.day

def get_hour(row):
    return row.hour

def get_minute(row):
    return row.minute


# extract datetime features

train_df['day'] = train_df['DateTime'].apply(get_day)
train_df['hour'] = train_df['DateTime'].apply(get_hour)
train_df['minute'] = train_df['DateTime'].apply(get_minute)

train_df_DateTime = train_df[['day', 'hour', 'minute', 'DateTime']]
train_df_DateTime = train_df_DateTime.groupby(['day', 'hour', 'minute']).agg(np.min)

train_df = train_df[['day', 'hour', 'minute'] + cols_to_keep]

# group by minute
# numerical columns are aggregated by the mean
train_df_num = train_df[['day', 'hour', 'minute'] + num_cols].groupby(['day', 'hour', 'minute']).agg(np.nanmean)
# categorical columns are aggregated by the mode
train_df_cat = train_df[['day', 'hour', 'minute'] + cat_cols].groupby(['day', 'hour', 'minute']).agg(lambda x:x.value_counts().index[0])


train_df = pd.concat([train_df_num, train_df_cat], axis = 1)
train_df = train_df.reset_index()
train_df = train_df.drop(['day', 'hour', 'minute'], axis = 1)
train_df_DateTime = train_df_DateTime.reset_index().drop(['day', 'hour', 'minute'], axis = 1)

# train_df.to_csv('test_data_by_minute.csv')


#%% pre-processing

"""
preprocessing(df) takes a labeled/unlabeled dataset df and returns a preprocessed one
"""

# lags_list = np.arange(1,11)
# f1_list = []
# f2_list = []
# fhalf_list = []
# confusion_matrix_list = []
# precision_list = []
# recall_list = []

# for nlags in lags_list:
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
train-test split: sliding training window is 18 hours = 1080 minutes, test=prediction window is 1 minute.
"""

# preprocess dataset
train_df_processed = preprocessing(train_df)
train_df_DateTime = train_df_DateTime.iloc[lags:]

# initial training set
Xy_train = train_df_processed.iloc[:window_size_train]

# anything after the initial training period
Xy_rest = train_df_processed.iloc[window_size_train:]


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
#     layers.Bidirectional(layers.LSTM(16, return_sequences=False)),
#     layers.Dropout(0.2),
#     layers.BatchNormalization(),
#   # tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4)),
#     layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4)),
#     layers.Dense(1, activation = 'sigmoid')
# ])
# model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),metrics=[tf.keras.metrics.AUC()])


# for i in range(1):
for i in tqdm(range(len(Xy_rest))):
    
    Xy_valid_roll = Xy_rest.iloc[i:i+window_size_valid]
    
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
        
    # elif use_XGBoost:
    #     model.fit(X_train_roll_shuffled_scaled, y_train_roll_shuffled, 
    #               eval_set = [(X_train_roll_shuffled_scaled, y_train_roll_shuffled)], 
    #               eval_metric=fbeta_eval,
    #               early_stopping_rounds = 30, 
    #               verbose=0)
    #     # forecast
    #     y_pred = model.predict(X_valid_roll_scaled)
        
    # elif use_nn:
    #     X_train_roll_shuffled_scaled_nn = np.expand_dims(X_train_roll_shuffled_scaled, -1)
    #     X_valid_roll_scaled_nn = np.expand_dims(X_valid_roll_scaled, -1)
        
    #     history = model.fit(X_train_roll_shuffled_scaled_nn, y_train_roll_shuffled,
    #                 class_weight=class_weight,
    #                 epochs=epochs)
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


# logistic regression
# without lag features
# 100%|██████████| 2254/2254 [00:20<00:00, 108.55it/s]
# with lag features
# lag 1
# 100%|██████████| 2253/2253 [00:26<00:00, 86.62it/s]
# lag 10
# 100%|██████████| 2244/2244 [01:07<00:00, 33.09it/s]

#%% Model performance and predictions

"""
# Summary of model performance
"""



# save forecast to file
# validation_DateTime.to_csv('predictions/validation_DateTime.csv')
# rolling_forecast.tofile('predictions/forecast.npy')


# y_valid = Xy_rest_partitioned[0]['Tacking'].to_numpy()
y_valid = Xy_rest['Tacking'].to_numpy()


# f1_list.append(fbeta_score(y_valid, rolling_forecast, beta=1))
# f2_list.append(fbeta_score(y_valid, rolling_forecast, beta=2))
# fhalf_list.append(fbeta_score(y_valid, rolling_forecast, beta=0.5))
# confusion_matrix_list.append(confusion_matrix(y_valid, rolling_forecast))
# precision_list.append(precision_score(y_valid, rolling_forecast))
# recall_list.append(recall_score(y_valid, rolling_forecast))


# !!!
print('\nlags = {}'.format(lags))
print('F_beta score (beta=1): ', fbeta_score(y_valid, rolling_forecast, beta=1))
print('F_beta score (beta=2): ', fbeta_score(y_valid, rolling_forecast, beta=2))
print('F_beta score (beta=0.5): ', fbeta_score(y_valid, rolling_forecast, beta=0.5))
print('AUC score: ', roc_auc_score(y_valid, rolling_forecast))
# Confusion matrix, precision, recall
# the count of true negatives is C00, false negatives is C10, true positives is C11 and false positives is C01.
print(confusion_matrix(y_valid, rolling_forecast))
print('Precision score: ', precision_score(y_valid, rolling_forecast))
print('Recall score: ', recall_score(y_valid, rolling_forecast))

# rolling forecast
# logistic regression
# validation window = 1 min
# C=0.1:
# lags = 0
# all features except DateTime
# F_beta score (beta=1):  0.255659121171771
# F_beta score (beta=2):  0.3889789303079417
# F_beta score (beta=0.5):  0.1904006346687822
# AUC score:  0.6801242236024845
# [[1599  494]
#  [  65   96]]
# Precision score:  0.16271186440677965
# Recall score:  0.5962732919254659
# without latitude and logitude
# F_beta score (beta=1):  0.2655367231638418
# F_beta score (beta=2):  0.3946263643996642
# F_beta score (beta=0.5):  0.20008514261387828
# AUC score:  0.6837075967510751
# [[1640  453]
#  [  67   94]]
# Precision score:  0.17184643510054845
# Recall score:  0.5838509316770186
# also without RudderAng
# F_beta score (beta=1):  0.2646657571623465
# F_beta score (beta=2):  0.39884868421052627
# F_beta score (beta=0.5):  0.1980400163331972
# AUC score:  0.6877687529861444
# [[1618  475]
#  [  64   97]]
# Precision score:  0.16958041958041958
# Recall score:  0.6024844720496895       <-------
# also without WSoG
# F_beta score (beta=1):  0.25885558583106266
# F_beta score (beta=2):  0.39030402629416594
# F_beta score (beta=0.5):  0.19364044027721156
# AUC score:  0.6808408982322025
# [[1615  478]
#  [  66   95]]
# Precision score:  0.16579406631762653
# Recall score:  0.5900621118012422


# all features except DateTime, latitude and logitude, RudderAng
# lags 1 to 10
# f1_list
# [0.2732732732732732,
#  0.2825396825396826,
#  0.2805280528052805,
#  0.28960817717206133,
#  0.30085470085470084,
#  0.2951388888888889,
#  0.29432624113475175,
#  0.3051470588235294,
#  0.3097514340344168,
#  0.32926829268292684]  <-------
# f2_list
# [0.3959965187119234,
#  0.3998203054806829,
#  0.3902662993572084,
#  0.3971962616822429,
#  0.41198501872659177,  <-------
#  0.4013220018885741,
#  0.3963705826170009,
#  0.40408958130477113,
#  0.4025844930417494,
#  0.41538461538461535]  <-------
# fhalf_list
# [0.20861989912883996,
#  0.21845851742758962,
#  0.21895929933024214,
#  0.22788203753351202,
#  0.23694130317716744,
#  0.23338824821526633,
#  0.23406655386350816,
#  0.24512699350265799,
#  0.25170913610938467,
#  0.2727272727272727]  <-------
# confusion_matrix_list
# [array([[1678,  414],
#         [  70,   91]], dtype=int64),
#  array([[1711,  380],
#         [  72,   89]], dtype=int64),
#  array([[1730,  360],
#         [  76,   85]], dtype=int64),
#  array([[1748,  341],
#         [  76,   85]], dtype=int64),
#  array([[1752,  336],
#         [  73,   88]], dtype=int64),
#  array([[1757,  330],
#         [  76,   85]], dtype=int64),
#  array([[1766,  320],
#         [  78,   83]], dtype=int64),
#  array([[1785,  300],
#         [  78,   83]], dtype=int64),
#  array([[1803,  281],
#         [  80,   81]], dtype=int64),
#  array([[1833,  250],
#         [  80,   81]], dtype=int64)]
# precision_list
# [0.1801980198019802,
#  0.18976545842217485,
#  0.19101123595505617,
#  0.19953051643192488,
#  0.20754716981132076,
#  0.20481927710843373,
#  0.20595533498759305,
#  0.21671018276762402,
#  0.22375690607734808,
#  0.24471299093655588]   <-------
# recall_list
# [0.5652173913043478,   <-------
#  0.5527950310559007,
#  0.5279503105590062,
#  0.5279503105590062,
#  0.546583850931677,
#  0.5279503105590062,
#  0.515527950310559,
#  0.515527950310559,
#  0.5031055900621118,
#  0.5031055900621118]

# accuracy and recall decrease with number of lags, while precision increases

"""
# Plot forecast against ground truth
"""

validation_DateTime = train_df_DateTime.iloc[window_size_train:window_size_train+len(Xy_rest)]
plt.figure(dpi=150)
plt.plot(validation_DateTime.squeeze().to_numpy(), rolling_forecast, color = 'tomato', label = 'Model forecast')
plt.plot(validation_DateTime.squeeze(), y_valid, color = 'black', label='Ground truth')
ax = plt.gca()
ax.set_xlabel('Date-Hour')
ax.legend(loc='upper right')
ax.set_ylabel('Tacking')
ax.set_yticks([0, 1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('predictions/good_features_lag5_logreg_rolling_windows_18hrs_1_min.png')


# free VRAM after using tensorflow
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()

#%% logistic regression: feature coefficients [with lag 5 features]

""" feature coefficients"""
# get the feature names as numpy array
feature_names = np.array(list(train_df_processed.drop(['Tacking'], axis = 1).columns))
# Sort the [absolute values] of coefficients from the model
logreg_coef = np.abs(model.coef_[0]).copy()
logreg_coef.sort()
sorted_coef_logreg = np.abs(model.coef_[0]).argsort()

# Find the 20 smallest and 20 largest absolute-coefficients
print('Smallest Coefs:\n{}'.format(feature_names[sorted_coef_logreg[:20]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_logreg[:-21:-1]]))
# Smallest Coefs:
# ['AWA_lag_4' 'CurrentDir_sin_lag_5' 'AWA_lag_3' 'AWA_lag_5' 'AWA_lag_2'
#  'AWA' 'AWA_lag_1' 'TWA_lag_5' 'TWA_lag_4' 'TWA_lag_2' 'TWA_lag_3'
#  'ModePilote_lag_5' 'ModePilote_lag_4' 'ModePilote_lag_3'
#  'ModePilote_lag_2' 'ModePilote_lag_1' 'ModePilote' 'TWA_lag_1' 'TWA'
#  'CurrentDir_sin_lag_4']
# Largest Coefs: 
# ['CurrentSpeed_lag_5' 'CurrentSpeed_lag_2' 'CurrentSpeed'
#  'CurrentSpeed_lag_4' 'CurrentSpeed_lag_1' 'CurrentSpeed_lag_3'
#  'Yaw_lag_5' 'AirTemp_lag_5' 'Yaw_lag_2' 'AirTemp_lag_4' 'Yaw_lag_3'
#  'Yaw_lag_4' 'AirTemp_lag_3' 'Yaw' 'Yaw_lag_1' 'AirTemp_lag_2'
#  'AirTemp_lag_1' 'AirTemp' 'TWD_cos_lag_4' 'TWD_cos_lag_3']

# smallest absolute-coefficients
logreg_coef[:10]
# array([0.00033887, 0.00062423, 0.00216862, 0.0024112 , 0.00274351,
#        0.00421884, 0.00427492, 0.00497861, 0.00503539, 0.00516882])
# largest absolute-coefficients
logreg_coef[:-21:-1]
# array([1.33841698, 1.32404213, 1.32343457, 1.32066259, 1.31739357,
#        1.31309796, 0.57411317, 0.51262244, 0.5047435 , 0.49092369,
#        0.48549615, 0.47570912, 0.46443663, 0.46197697, 0.45794192,
#        0.43718324, 0.41027366, 0.38356504, 0.36826297, 0.34765995])


#%% logistic regression: feature coefficients [with lags=10 features]

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
# ['AWA_lag_10' 'AWA_lag_8' 'AWA_lag_7' 'HeadingTrue_sin_lag_2'
#  'HeadingMag_cos_lag_3' 'AWA_lag_5' 'AWA_lag_9' 'AWA_lag_6'
#  'ModePilote_lag_5' 'ModePilote_lag_10' 'ModePilote_lag_6'
#  'ModePilote_lag_9' 'ModePilote_lag_8' 'ModePilote_lag_7'
#  'ModePilote_lag_4' 'ModePilote_lag_3' 'AWA_lag_4' 'ModePilote_lag_2'
#  'ModePilote_lag_1' 'ModePilote']
# Largest Coefs: 
# ['CurrentSpeed_lag_10' 'CurrentSpeed_lag_9' 'CurrentSpeed_lag_8'
#  'CurrentSpeed_lag_7' 'CurrentSpeed_lag_6' 'CurrentSpeed'
#  'CurrentSpeed_lag_5' 'CurrentSpeed_lag_1' 'CurrentSpeed_lag_2'
#  'CurrentSpeed_lag_4' 'CurrentSpeed_lag_3' 'Yaw_lag_9' 'Yaw_lag_10'
#  'Yaw_lag_8' 'Yaw_lag_2' 'TWD_cos_lag_8' 'TWD_cos_lag_10' 'TWD_cos_lag_7'
#  'Yaw_lag_1' 'Yaw']

# smallest absolute-coefficients
logreg_coef[:10]
# array([1.61272705e-05, 5.22717091e-05, 2.06135651e-04, 2.48135553e-04,
#        5.65218405e-04, 7.11025452e-04, 8.05508102e-04, 1.04922581e-03,
#        1.28879130e-03, 1.29823792e-03])
# largest absolute-coefficients
logreg_coef[:-21:-1]
# array([0.89873726, 0.88114369, 0.86570139, 0.85541802, 0.84915519,
#        0.83957362, 0.83517612, 0.8337572 , 0.83252783, 0.8250982 ,
#        0.81743947, 0.57609711, 0.5740369 , 0.47098254, 0.4288052 ,
#        0.42773008, 0.42481753, 0.41691593, 0.4010457 , 0.39825647])




