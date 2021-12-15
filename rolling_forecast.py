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

# VoltageDrawn: Volts. Voltage drawn by the system of one of its parts ??
# ModePilote: unclear. unclear

# Target
# Tacking: Boolean.

"""

#%% This file

"""
# Feature pre-processing and engineering
# Rolling forecast
"""

#%% Workflow

"""
1. load labeled dataset and keep only columns we want to use
2. Rolling forecast: 
    sliding training window is 18 hours = 64800 seconds, validation window is 12 hours = 43200 seconds
    preprocessing
    fit models: logistic regression, XGBoost, lightGBM
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


#%% Hyper-parameters

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
# 6 hours
window_size_valid = 43200  # 21600  # tried 18 hours: worse


#%% load dataset

train_df = pd.read_csv('test_data.csv')
# train_df_raw['DateTime'] = pd.to_datetime(train_df_raw['DateTime'])
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
    
    # convert angles to sines and cosines
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
    
    # generate lag features
    feat_columns = list(df_.columns)
    labeled = False
    if 'Tacking' in feat_columns:
        labeled = True
        feat_columns.remove('Tacking')
        # df_['Tacking_lag_1'] = df_['Tacking'].shift(1)
    
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
train-validation split: sliding training window is 18 hours=64800 seconds, validation and prediction windows are 6 hours = 21600 seconds.

Given an initial dataset, take the first 18 hours as training set, and divide the rest into chunks of 6 six hours. 
If the last chunk does not have 21600 instances, ignore until the sensors gather enough data.
Thereafter, once we gather 6 hours of data, it is appended to the existing dataset (so we can fillna with previous values) and retrain the model with new train/validation windows.
"""

# preprocess dataset
train_df_processed = preprocessing(train_df)

# initial training set
Xy_train = train_df_processed.iloc[:window_size_train]

# anything after the initial training period
Xy_rest = train_df_processed.iloc[window_size_train:]

# partition Xy_rest into chunks of validation sets of 6 hours = 21600 seconds
# if the last chunk does not have 21600 instances, ignore until the sensors gather enough data
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
# historiesLSTM = []

from tqdm import tqdm    # fancy progress bar for the for-loop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score, roc_auc_score, f1_score

# for XGBoost
def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err

beta = 1
def fbeta_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-fbeta_score(y_true, np.round(y_pred), beta = beta)
    return 'fbeta_err', err

# model: choose one
use_logistic = False
use_XGBoost = False

# logistic regression
from sklearn.linear_model import LogisticRegression
use_logistic = True
model = LogisticRegression(max_iter = 10000, class_weight='balanced', C=.1)

# XGBoost
from xgboost import XGBClassifier
# use_XGBoost = True
# model = XGBClassifier(scale_pos_weight = 5., verbosity=0)


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
    
    
    # feature normalisation
    scaler = MinMaxScaler()
    X_train_roll_shuffled_scaled = scaler.fit_transform(X_train_roll_shuffled)
    X_valid_roll_scaled = scaler.transform(X_valid_roll)
    
    # model fitting and predictions
    if use_logistic:
        model.fit(X_train_roll_shuffled_scaled, y_train_roll_shuffled)
    elif use_XGBoost:
        model.fit(X_train_roll_shuffled_scaled, y_train_roll_shuffled, 
                  eval_set = [(X_train_roll_shuffled_scaled, y_train_roll_shuffled), (X_valid_roll_scaled, y_valid_roll)], 
                  eval_metric=fbeta_eval,
                  early_stopping_rounds = 30, 
                  verbose=0)
    else: 
        print('Pick a model.')
        break;
    
    # forecast
    y_pred = model.predict(X_valid_roll_scaled)
    rolling_forecast = np.hstack((rolling_forecast,y_pred))
    
    # update training set
    history_rolling = pd.concat([history_rolling.iloc[window_size_valid:window_size_train], Xy_valid_roll])



# logistic regression
# 100%|██████████| 7/7 [00:05<00:00,  1.31it/s]
# XGBoost
# 100%|██████████| 7/7 [00:57<00:00,  8.20s/it]


# predict

# y_valid = Xy_rest_partitioned[0]['Tacking'].to_numpy()
y_valid = pd.concat([X_chunk['Tacking'] for X_chunk in Xy_rest_partitioned]).to_numpy()

print('F_beta score (beta={}): '.format(beta), fbeta_score(y_valid, rolling_forecast, beta=beta))
print('AUC score: ', roc_auc_score(y_valid, rolling_forecast))

# validation window = 6 hours
# logistic regression
# with Tacking_lag_1 feature
# first validation window
# F_beta score (beta=1):  0.9996153846153846
# F_beta score (beta=2):  0.9996707886668997
# F_beta score (beta=0.5):  0.999609085671961
# AUC score:  0.9997813765182185 
# rolling forecast
# F_beta score:  0.9996399362172728
# AUC score:  0.9998315427386675

# without Tacking_lag_1 feature
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
# without Tacking_lag_1 feature
# C=0.1:
# F_beta score (beta=1):  0.3444506001846723
# F_beta score (beta=2): 0.4145994487418867
# F_beta score (beta=0.5): 0.29460449835734137
# AUC score:  0.6869299855410966

# XGBoost 
# with Tacking_lag_1 feature
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

# without Tacking_lag_1 feature
# F_beta score (beta=1):  0.0016998370989446843
# AUC score:  0.4851133257939874


plt.plot(np.arange(len(rolling_forecast)), rolling_forecast, color = 'tomato')
# plt.figure()
plt.plot(np.arange(len(rolling_forecast)), y_valid, color = 'skyblue')
plt.axvline(x=len(Xy_train), color = 'black')
plt.xlim(36200, 37000)

from sklearn.metrics import confusion_matrix
# the count of true negatives is C00, false negatives is C10, true positives is C11 and false positives is C01.
confusion_matrix(y_valid, rolling_forecast)

# logistic regression
# with Tacking_lag_1 feature
# array([[141476,      4],
#        [     3,   9717]], dtype=int64)

# without Tacking_lag_1 feature: less false negative than false positive 
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
# with Tacking_lag_1 feature
# array([[141444,     36],
#        [     3,   9717]], dtype=int64)

# without Tacking_lag_1 feature: less false positive than false negative 
# array([[137093,   4387],
#        [  9708,     12]], dtype=int64)
# precision = 12/(4387+12) = 0.00272789270288702
# recall= 0.0012345679012345679

#%% Models: Rolling forecast

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


#################################




