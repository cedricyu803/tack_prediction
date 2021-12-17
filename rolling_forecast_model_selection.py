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
# Find optimal model with rolling forecast, and with the good features and lags
# Commented out tensorflow and LSTM network
"""

#%% Workflow

"""
1. load labeled dataset, aggregate by minute and keep only columns we want to use
2. Rolling forecast: 
    sliding training window is 18 hours = 1080 minutes, validation window is 1 minute
    preprocessing
    fit model: logistic regression, XGBoost
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
lags = 10

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
C=100.

# hyperparameters for LSTM
# learning_rate = 5e-6
# epochs = 10
# patience = 3   # early stopping based on val_loss
# class_weight = {0: 5 / (2 * 4), 1: 5 / (2 * 1)}

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
# use_logistic = True
# model = LogisticRegression(max_iter = 10000, class_weight='balanced', C=C)

# XGBoost
use_XGBoost = True
model = XGBClassifier(scale_pos_weight = 5., verbosity=0)

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
        
    elif use_XGBoost:
        model.fit(X_train_roll_shuffled_scaled, y_train_roll_shuffled, 
                  eval_set = [(X_train_roll_shuffled_scaled, y_train_roll_shuffled)], 
                  eval_metric=fbeta_eval,
                  early_stopping_rounds = 30, 
                  verbose=0)
        # forecast
        y_pred = model.predict(X_valid_roll_scaled)
        
    # elif use_nn:
    #     X_train_roll_shuffled_scaled_nn = np.expand_dims(X_train_roll_shuffled_scaled, -1)
    #     X_valid_roll_scaled_nn = np.expand_dims(X_valid_roll_scaled, -1)
        
    #     history = model.fit(X_train_roll_shuffled_scaled_nn, y_train_roll_shuffled,
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


# logistic regression
# C = 0.1
# without lag features
# 100%|██████████| 2254/2254 [00:20<00:00, 108.55it/s]
# with lag features
# lag 1
# 100%|██████████| 2253/2253 [00:26<00:00, 86.62it/s]
# lag 10
# 100%|██████████| 2244/2244 [01:07<00:00, 33.09it/s]

# C=10
# lag 5
# 100%|██████████| 2249/2249 [02:22<00:00, 15.75it/s]

# XGBoost
# lag 5
# 100%|██████████| 2249/2249 [23:43<00:00,  1.58it/s]
# lag 10
# 100%|██████████| 2244/2244 [37:16<00:00,  1.00it/s]


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
# C=10.
# lag 5
# lags = 5
# F_beta score (beta=1):  0.39148936170212767
# F_beta score (beta=2):  0.4826862539349423
# F_beta score (beta=0.5):  0.32927702219040805
# AUC score:  0.7337506841817186
# [[1871  217]
#  [  69   92]]
# Precision score:  0.2977346278317152
# Recall score:  0.5714285714285714
# C=100.
# lags = 5
# F_beta score (beta=1):  0.4246575342465753
# F_beta score (beta=2):  0.5048859934853419
# F_beta score (beta=0.5):  0.3664302600472813
# AUC score:  0.7447585730944053
# [[1904  184]
#  [  68   93]]
# Precision score:  0.33574007220216606
# Recall score:  0.577639751552795
# lags = 10
# F_beta score (beta=1):  0.4649859943977591
# F_beta score (beta=2):  0.494047619047619
# F_beta score (beta=0.5):  0.43915343915343913
# AUC score:  0.7306396352608963
# [[1970  113]
#  [  78   83]]
# Precision score:  0.42346938775510207
# Recall score:  0.515527950310559


# XGBoost: MUCH BETTER
# lags = 5
# F_beta score (beta=1):  0.8333333333333334
# F_beta score (beta=2):  0.8724428399518652
# F_beta score (beta=0.5):  0.7975797579757975
# AUC score:  0.9402530877418435
# [[2046   42]
#  [  16  145]]
# Precision score:  0.7754010695187166
# Recall score:  0.9006211180124224
# lags = 10
# F_beta score (beta=1):  0.85459940652819
# F_beta score (beta=2):  0.8780487804878048
# F_beta score (beta=0.5):  0.8323699421965318
# AUC score:  0.9395237399474599
# [[2051   32]
#  [  17  144]]
# Precision score:  0.8181818181818182
# Recall score:  0.8944099378881988


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
# plt.savefig('predictions/good_features_lag10_XGBoost_rolling_windows_18hrs_1_min.png')


# free VRAM after using tensorflow
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()

#%% XGBoost: feature importances [with lag 5 features]

XGBC_model_feature_importances = pd.Series(model.feature_importances_, index = train_df_processed.drop(['Tacking'],axis=1).columns).sort_values(ascending = False)
XGBC_model_feature_importances = XGBC_model_feature_importances / XGBC_model_feature_importances.max()

# CurrentSpeed_lag_5       1.000000
# CurrentSpeed_lag_1       0.628808
# CurrentSpeed_lag_2       0.473256
# AvgSoS_lag_5             0.275669
# Pitch_lag_2              0.199019
# CurrentSpeed_lag_3       0.194534
# AvgSoS_lag_3             0.156178
# CurrentSpeed             0.146135
# VMG_lag_4                0.142076
# VMG_lag_3                0.134324
# VoltageDrawn_lag_4       0.091403
# SoS_lag_3                0.080030
# AirTemp_lag_2            0.058322
# AvgSoS                   0.057178
# Yaw_lag_1                0.055551
# VMG_lag_5                0.052296
# Yaw                      0.052286
# SoG_lag_1                0.051618
# VMG_lag_2                0.048541
# AirTemp_lag_3            0.046328
# CurrentDir_sin_lag_2     0.042105
# TWD_sin                  0.041015
# Yaw_lag_4                0.033543
# AirTemp_lag_4            0.030771
# TWD_sin_lag_2            0.029861
# SoG_lag_2                0.026470
# AWS_lag_3                0.026442
# Yaw_lag_3                0.025357
# TWS                      0.022262
# SoG_lag_3                0.020797
# AirTemp_lag_1            0.018938
# HeadingMag_sin           0.018040
# HoG_cos_lag_2            0.016631
# CurrentSpeed_lag_4       0.015452
# CurrentDir_sin           0.014209
# CurrentDir_cos_lag_1     0.013698
# CurrentDir_cos_lag_2     0.012246
# HoG_cos                  0.009531
# VoltageDrawn_lag_5       0.008578
# Roll_lag_2               0.007932
# AWA_lag_4                0.006956
# Yaw_lag_5                0.005963
# CurrentDir_cos_lag_3     0.005184
# SoS_lag_2                0.004649
# HeadingTrue_sin_lag_2    0.002840
# AirTemp                  0.002076
# HoG_cos_lag_5            0.002051
# HeadingMag_sin_lag_2     0.001100
# HeadingTrue_cos_lag_1    0.000692
# TWA_lag_5                0.000187
# CurrentDir_cos_lag_4     0.000000
# Yaw_lag_2                0.000000


#%% XGBoost: feature importances [with lag 10 features]

XGBC_model_feature_importances = pd.Series(model.feature_importances_, index = train_df_processed.drop(['Tacking'],axis=1).columns).sort_values(ascending = False)
XGBC_model_feature_importances = XGBC_model_feature_importances / XGBC_model_feature_importances.max()

# CurrentSpeed_lag_9        1.000000
# CurrentSpeed_lag_10       0.881460
# CurrentSpeed_lag_7        0.810984
# CurrentSpeed_lag_6        0.800780
# CurrentSpeed_lag_8        0.610424
# AvgSoS_lag_6              0.391511
# Pitch_lag_3               0.172527
# SoG_lag_7                 0.137643
# VoltageDrawn_lag_4        0.105405
# AvgSoS_lag_5              0.102505
# AvgSoS                    0.096465
# CurrentSpeed_lag_2        0.078123
# CurrentSpeed              0.073301
# VMG_lag_3                 0.066239
# Yaw_lag_6                 0.060255
# SoG_lag_6                 0.057388
# Roll_lag_5                0.049028
# Yaw_lag_5                 0.045565
# VoltageDrawn_lag_3        0.044834
# CurrentSpeed_lag_1        0.043226
# Yaw_lag_4                 0.038204
# TWS_lag_4                 0.034298
# Yaw_lag_2                 0.024776
# CurrentDir_sin_lag_1      0.023993
# Yaw_lag_3                 0.021306
# Yaw_lag_7                 0.018747
# HoG_cos                   0.017276
# Pitch_lag_5               0.017232
# AWS_lag_2                 0.016904
# CurrentSpeed_lag_3        0.015909
# AirTemp_lag_7             0.015718
# AWS_lag_3                 0.015102
# CurrentSpeed_lag_4        0.015006
# CurrentSpeed_lag_5        0.013468
# SoG_lag_4                 0.013382
# AvgSoS_lag_4              0.013316
# VMG_lag_5                 0.010189
# AirTemp_lag_5             0.010089
# CurrentDir_sin            0.008009
# CurrentDir_sin_lag_2      0.007758
# HoG_cos_lag_1             0.005825
# HeadingTrue_sin_lag_9     0.005236
# HeadingMag_sin            0.003462
# TWD_cos_lag_3             0.001349
# TWD_cos                   0.000908
# CurrentDir_cos_lag_1      0.000000
# CurrentDir_cos_lag_2      0.000000
# CurrentDir_cos_lag_3      0.000000
# TWD_sin_lag_6             0.000000
# CurrentDir_sin_lag_10     0.000000
# CurrentDir_cos_lag_10     0.000000
# TWD_sin_lag_5             0.000000
# CurrentDir_cos_lag_4      0.000000
# CurrentDir_cos_lag_5      0.000000
# CurrentDir_cos_lag_6      0.000000
# CurrentDir_cos_lag_7      0.000000
# CurrentDir_cos_lag_8      0.000000
# CurrentDir_cos_lag_9      0.000000
# TWD_sin_lag_4             0.000000
# TWD_sin_lag_3             0.000000
# CurrentDir_sin_lag_8      0.000000
# TWD_sin_lag_2             0.000000
# TWD_sin_lag_1             0.000000
# CurrentDir_sin_lag_9      0.000000
# ModePilote_lag_6          0.000000
# CurrentDir_sin_lag_7      0.000000
# Yaw_lag_10                0.000000
# WSoG_lag_10               0.000000
# VoltageDrawn_lag_1        0.000000
# VoltageDrawn_lag_2        0.000000
# VoltageDrawn_lag_5        0.000000
# VoltageDrawn_lag_6        0.000000
# VoltageDrawn_lag_7        0.000000
# VoltageDrawn_lag_8        0.000000
# VoltageDrawn_lag_9        0.000000
# VoltageDrawn_lag_10       0.000000
# Yaw_lag_1                 0.000000
# Yaw_lag_8                 0.000000
# Yaw_lag_9                 0.000000
# ModePilote_lag_1          0.000000
# CurrentDir_sin_lag_6      0.000000
# ModePilote_lag_2          0.000000
# ModePilote_lag_3          0.000000
# ModePilote_lag_4          0.000000
# ModePilote_lag_5          0.000000
# TWD_sin_lag_8             0.000000
# ModePilote_lag_7          0.000000
# ModePilote_lag_8          0.000000
# ModePilote_lag_9          0.000000
# ModePilote_lag_10         0.000000
# CurrentDir_sin_lag_3      0.000000
# CurrentDir_sin_lag_4      0.000000
# CurrentDir_sin_lag_5      0.000000
# TWD_sin_lag_7             0.000000
# TWD_cos_lag_8             0.000000
# TWD_sin_lag_9             0.000000
# HeadingMag_cos_lag_5      0.000000
# HeadingTrue_sin_lag_1     0.000000
# HeadingMag_cos_lag_10     0.000000
# HeadingMag_cos_lag_9      0.000000
# HeadingMag_cos_lag_8      0.000000
# HeadingMag_cos_lag_7      0.000000
# HeadingMag_cos_lag_6      0.000000
# HeadingMag_cos_lag_4      0.000000
# HeadingMag_sin_lag_6      0.000000
# HeadingMag_cos_lag_3      0.000000
# HeadingMag_cos_lag_2      0.000000
# HeadingMag_cos_lag_1      0.000000
# HeadingMag_sin_lag_10     0.000000
# HeadingMag_sin_lag_9      0.000000
# HeadingMag_sin_lag_8      0.000000
# HeadingTrue_sin_lag_2     0.000000
# HeadingTrue_sin_lag_3     0.000000
# HeadingTrue_sin_lag_4     0.000000
# HeadingTrue_sin_lag_5     0.000000
# HeadingTrue_sin_lag_6     0.000000
# HeadingTrue_sin_lag_7     0.000000
# HeadingTrue_sin_lag_8     0.000000
# HeadingTrue_sin_lag_10    0.000000
# HeadingTrue_cos_lag_1     0.000000
# HeadingTrue_cos_lag_2     0.000000
# HeadingTrue_cos_lag_3     0.000000
# HeadingTrue_cos_lag_4     0.000000
# HeadingTrue_cos_lag_5     0.000000
# HeadingTrue_cos_lag_6     0.000000
# HeadingTrue_cos_lag_7     0.000000
# HeadingTrue_cos_lag_8     0.000000
# HeadingTrue_cos_lag_9     0.000000
# HeadingMag_sin_lag_7      0.000000
# HeadingMag_sin_lag_5      0.000000
# TWD_sin_lag_10            0.000000
# TWD_cos_lag_9             0.000000
# HoG_sin_lag_5             0.000000
# HoG_sin_lag_4             0.000000
# HoG_sin_lag_3             0.000000
# HoG_sin_lag_2             0.000000
# HoG_sin_lag_1             0.000000
# TWD_cos_lag_10            0.000000
# WSoG_lag_8                0.000000
# HeadingMag_sin_lag_4      0.000000
# TWD_cos_lag_7             0.000000
# TWD_cos_lag_6             0.000000
# TWD_cos_lag_5             0.000000
# TWD_cos_lag_4             0.000000
# TWD_cos_lag_2             0.000000
# TWD_cos_lag_1             0.000000
# HoG_sin_lag_6             0.000000
# HoG_sin_lag_7             0.000000
# HoG_sin_lag_8             0.000000
# HoG_sin_lag_9             0.000000
# HoG_sin_lag_10            0.000000
# HoG_cos_lag_2             0.000000
# HoG_cos_lag_3             0.000000
# HoG_cos_lag_4             0.000000
# HoG_cos_lag_5             0.000000
# HoG_cos_lag_6             0.000000
# HoG_cos_lag_7             0.000000
# HoG_cos_lag_8             0.000000
# HoG_cos_lag_9             0.000000
# HoG_cos_lag_10            0.000000
# HeadingMag_sin_lag_1      0.000000
# HeadingMag_sin_lag_2      0.000000
# HeadingMag_sin_lag_3      0.000000
# WSoG_lag_9                0.000000
# Leeway_lag_2              0.000000
# WSoG_lag_7                0.000000
# AWA_lag_1                 0.000000
# TWA_lag_5                 0.000000
# TWA_lag_6                 0.000000
# TWA_lag_7                 0.000000
# TWA_lag_8                 0.000000
# TWA_lag_9                 0.000000
# TWA_lag_10                0.000000
# AWS_lag_1                 0.000000
# AWS_lag_4                 0.000000
# AWS_lag_5                 0.000000
# AWS_lag_6                 0.000000
# AWS_lag_7                 0.000000
# AWS_lag_8                 0.000000
# AWS_lag_9                 0.000000
# AWS_lag_10                0.000000
# AWA_lag_2                 0.000000
# WSoG_lag_6                0.000000
# AWA_lag_3                 0.000000
# AWA_lag_4                 0.000000
# AWA_lag_5                 0.000000
# AWA_lag_6                 0.000000
# AWA_lag_7                 0.000000
# AWA_lag_8                 0.000000
# AWA_lag_9                 0.000000
# AWA_lag_10                0.000000
# Roll_lag_1                0.000000
# Roll_lag_2                0.000000
# Roll_lag_3                0.000000
# Roll_lag_4                0.000000
# Roll_lag_6                0.000000
# Roll_lag_7                0.000000
# TWA_lag_4                 0.000000
# TWA_lag_3                 0.000000
# TWA_lag_2                 0.000000
# TWA_lag_1                 0.000000
# TWA                       0.000000
# AWS                       0.000000
# AWA                       0.000000
# Roll                      0.000000
# Pitch                     0.000000
# AirTemp                   0.000000
# SoG                       0.000000
# SoS                       0.000000
# VMG                       0.000000
# Leeway                    0.000000
# WSoG                      0.000000
# VoltageDrawn              0.000000
# Yaw                       0.000000
# ModePilote                0.000000
# CurrentDir_cos            0.000000
# TWD_sin                   0.000000
# HoG_sin                   0.000000
# HeadingMag_cos            0.000000
# HeadingTrue_sin           0.000000
# HeadingTrue_cos           0.000000
# TWS_lag_1                 0.000000
# TWS_lag_2                 0.000000
# TWS_lag_3                 0.000000
# TWS_lag_5                 0.000000
# TWS_lag_6                 0.000000
# TWS_lag_7                 0.000000
# TWS_lag_8                 0.000000
# TWS_lag_9                 0.000000
# TWS_lag_10                0.000000
# Roll_lag_8                0.000000
# Roll_lag_9                0.000000
# Roll_lag_10               0.000000
# SoS_lag_10                0.000000
# AvgSoS_lag_2              0.000000
# AvgSoS_lag_3              0.000000
# AvgSoS_lag_7              0.000000
# AvgSoS_lag_8              0.000000
# AvgSoS_lag_9              0.000000
# AvgSoS_lag_10             0.000000
# VMG_lag_1                 0.000000
# VMG_lag_2                 0.000000
# VMG_lag_4                 0.000000
# VMG_lag_6                 0.000000
# VMG_lag_7                 0.000000
# VMG_lag_8                 0.000000
# VMG_lag_9                 0.000000
# VMG_lag_10                0.000000
# Leeway_lag_1              0.000000
# TWS                       0.000000
# Leeway_lag_3              0.000000
# Leeway_lag_4              0.000000
# Leeway_lag_5              0.000000
# Leeway_lag_6              0.000000
# Leeway_lag_7              0.000000
# Leeway_lag_8              0.000000
# Leeway_lag_9              0.000000
# Leeway_lag_10             0.000000
# WSoG_lag_1                0.000000
# WSoG_lag_2                0.000000
# WSoG_lag_3                0.000000
# WSoG_lag_4                0.000000
# WSoG_lag_5                0.000000
# AvgSoS_lag_1              0.000000
# SoS_lag_9                 0.000000
# Pitch_lag_1               0.000000
# SoS_lag_8                 0.000000
# Pitch_lag_2               0.000000
# Pitch_lag_4               0.000000
# Pitch_lag_6               0.000000
# Pitch_lag_7               0.000000
# Pitch_lag_8               0.000000
# Pitch_lag_9               0.000000
# Pitch_lag_10              0.000000
# AirTemp_lag_1             0.000000
# AirTemp_lag_2             0.000000
# AirTemp_lag_3             0.000000
# AirTemp_lag_4             0.000000
# AirTemp_lag_6             0.000000
# AirTemp_lag_8             0.000000
# AirTemp_lag_9             0.000000
# AirTemp_lag_10            0.000000
# SoG_lag_1                 0.000000
# SoG_lag_2                 0.000000
# SoG_lag_3                 0.000000
# SoG_lag_5                 0.000000
# SoG_lag_8                 0.000000
# SoG_lag_9                 0.000000
# SoG_lag_10                0.000000
# SoS_lag_1                 0.000000
# SoS_lag_2                 0.000000
# SoS_lag_3                 0.000000
# SoS_lag_4                 0.000000
# SoS_lag_5                 0.000000
# SoS_lag_6                 0.000000
# SoS_lag_7                 0.000000
# HeadingTrue_cos_lag_10    0.000000
# dtype: float32






