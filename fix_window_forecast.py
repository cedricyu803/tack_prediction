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
# Baseline forecast with fixed-window partitioning
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


#%% Workflow

"""
1. load dataset and keep only columns we want to use
2. (Dimensionality reduction with PCA; try later)
3. convert CurrentDir, TWD and HoG to sines and cosines, 
    convert other angles (['AWA', 'Pitch', 'Roll', 'Yaw', 'Leeway']) to principal branch (-180,180) degrees
4. generate lag features: 5 lags for features and 1 lag for target (NO!)
5. initial train-validation split: training window is 18 hours = 64800 seconds, validation/forecast windows are 6 hours = 21600 seconds
    re-shuffle training set to avoid sequence bias
6. Feature normalisation
7. Modelling with fixed-window partition: naive forecast (lag 1), logistic regression, lightGBM

"""



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
window_size_valid = 21600


#%% load dataset

train_df = pd.read_csv('test_data.csv')
# train_df_raw['DateTime'] = pd.to_datetime(train_df_raw['DateTime'])
# train_df_raw.shape

train_df.shape
# (220000, 27)

# keep only columns we use
train_df = train_df[cols_to_keep]

# fillna with preciding value
train_df = train_df.fillna(method='ffill')

#%% pre-processing

"""
preprocessing(df) takes a labeled dataset df and returns a pre-processed one
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

train_df = preprocessing(train_df)


#%% train-validation split: sliding windows are 18 hours=64800 seconds


Xy_train = train_df.iloc[:window_size_train]
Xy_valid = train_df.iloc[window_size_train:window_size_train+window_size_valid]

"""Re-shuffle training set to avoid sequence bias"""

Xy_cols = Xy_train.columns
np.random.seed(0)
Xy_train_shuffled = Xy_train.to_numpy().copy()
np.random.shuffle(Xy_train_shuffled)
Xy_train_shuffled = pd.DataFrame(Xy_train_shuffled, columns = Xy_cols)

"""separate features and labels"""

X_train_shuffled = Xy_train_shuffled.drop(['Tacking'], axis = 1)
y_train_shuffled = Xy_train_shuffled['Tacking']

X_valid = Xy_valid.drop(['Tacking'], axis = 1)
y_valid = Xy_valid['Tacking']

# all remaining data
Xy_valid2 = train_df.iloc[window_size_train:]
X_valid2 = Xy_valid2.drop(['Tacking'], axis = 1)
y_valid2 = Xy_valid2['Tacking']

#%% feature normalisation

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_shuffled_scaled = scaler.fit_transform(X_train_shuffled)

X_valid_scaled = scaler.transform(X_valid)

X_train_scaled = scaler.transform(Xy_train.drop(['Tacking'], axis = 1))
X_valid_scaled2 = scaler.transform(X_valid2)


#%% Models: Fixed-window partitioning

from sklearn.metrics import fbeta_score, roc_auc_score

#################################
"""Naive forecast: y_pred(t) = y_true(t-1)"""

y_pred_naive = train_df['Tacking'].shift(1).iloc[window_size_train:window_size_train+window_size_valid]
y_pred_naive2 = train_df['Tacking'].shift(1).iloc[window_size_train:]

print('F_beta score: ', fbeta_score(y_valid, y_pred_naive, beta=1))
# F_beta score:  0.9996153846153846
print('AUC score: ', roc_auc_score(y_valid, y_pred_naive))
# AUC score:  0.9997813765182185

print('F_beta score: ', fbeta_score(y_valid2, y_pred_naive2, beta=1))
# F_beta score:  0.9996913580246913
print('AUC score: ', roc_auc_score(y_valid2, y_pred_naive2))
# AUC score:  0.9998353679623371


#################################

# logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter = 10000, class_weight='balanced')
logreg.fit(X_train_shuffled_scaled, y_train_shuffled)

""""# Predict the validation set"""
y_pred_logreg = logreg.predict(X_valid_scaled)


print('F_beta score: ', fbeta_score(y_valid, y_pred_logreg, beta=1))
print('AUC score: ', roc_auc_score(y_valid, y_pred_logreg))

# with Tacking_lag_1 feature
# Fbeta score:  0.9996153846153846
# AUC score:  0.9997813765182185
# without Tacking_lag_1 feature
# F_beta score:  0.0
# AUC score:  0.4993157894736842


# Predict the validation set
y_pred_logreg2 = logreg.predict(X_valid_scaled2)

print('F_beta score: ', fbeta_score(y_valid2, y_pred_logreg2, beta=1))
print('AUC score: ', roc_auc_score(y_valid2, y_pred_logreg2))
# with Tacking_lag_1 feature
# F_beta score:  0.9996913580246913
# AUC score:  0.9998353679623371
# without Tacking_lag_1 feature
# F_beta score:  0.0
# AUC score:  0.495934009279945


""" feature coefficients"""
# get the feature names as numpy array
feature_names = np.array(list(train_df.drop(['Tacking'], axis = 1).columns))
# Sort the [absolute values] of coefficients from the model
logreg_coef = np.abs(logreg.coef_[0]).copy()
logreg_coef.sort()
sorted_coef_logreg = np.abs(logreg.coef_[0]).argsort()

# Find the 20 smallest and 10 largest absolute-coefficients
print('Smallest Coefs:\n{}'.format(feature_names[sorted_coef_logreg[:20]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_logreg[:-21:-1]]))
# Smallest Coefs:
# ['TWD_cos_lag_4' 'TWD_cos_lag_2' 'Pitch_lag_2' 'TWD_sin_lag_2'
#  'WSoG_lag_3' 'WSoG_lag_5' 'TWD_cos' 'TWD_sin_lag_3' 'Pitch_lag_5'
#  'TWD_sin_lag_1' 'Pitch' 'Yaw_lag_2' 'Yaw_lag_3' 'CurrentSpeed_lag_5'
#  'CurrentSpeed_lag_4' 'CurrentDir_cos_lag_5' 'CurrentSpeed_lag_3'
#  'Yaw_lag_1' 'Yaw_lag_5' 'AWA_lag_2']
# Largest Coefs: 
# ['Tacking_lag_1' 'HoG_cos' 'HoG_cos_lag_2' 'AirTemp_lag_5' 'AirTemp_lag_4'
#  'AirTemp' 'AirTemp_lag_3' 'AirTemp_lag_2' 'AirTemp_lag_1' 'AWS_lag_4'
#  'AWS_lag_2' 'AWS_lag_5' 'VMG' 'AWS_lag_1' 'AWS' 'AWS_lag_3'
#  'CurrentDir_sin_lag_3' 'TWD_sin_lag_4' 'Roll_lag_1' 'TWD_cos_lag_5']

# smallest absolute-coefficients
logreg_coef[:10]
# array([0.00771116, 0.0097745 , 0.01210206, 0.01271413, 0.01360897,
#        0.02143326, 0.02230896, 0.02992941, 0.03588308, 0.0445916 ])
# largest absolute-coefficients
logreg_coef[:-11:-1]
# array([12.5295522 ,  0.76751276,  0.51990736,  0.43931924,  0.43753668,
        # 0.42996803,  0.42857113,  0.42685353,  0.42485631,  0.42484942])

# largest coefficient is 'Tacking_lag_1'; ~ naive forecasting


# forecast
y_pred_train_valid_logreg2 = logreg.predict(np.vstack((X_train_scaled, X_valid_scaled2)))


plt.plot(np.arange(len(y_pred_train_valid_logreg2)), y_pred_train_valid_logreg2, color = 'tomato')
# plt.figure()
plt.plot(np.arange(len(y_pred_train_valid_logreg2)), pd.concat([Xy_train['Tacking'], y_valid2]), color = 'skyblue')
plt.axvline(x=len(Xy_train), color = 'black')
plt.xlim(36200, 37000)


#################################


from xgboost import XGBClassifier

# (len(y_train_shuffled)-y_train_shuffled.sum())/y_train_shuffled.sum()
# 28485./ 36315.

# XGBR_model = XGBRegressor(eval_metric = "rmse", 
#                           learning_rate = 0.05, 
#                           max_depth = 8,
#                           n_estimators = 100,
#                           reg_lambda = 0.7,
#                           n_jobs = 6)

XGBC_model = XGBClassifier(scale_pos_weight = 5.)
                           # eval_metric='auc', 
                           # n_estimators = 900, 
                           # max_depth = 7,
                           # learning_rate = 0.01,
                           # n_jobs = -1)

# takes about 5 seconds to fit
XGBC_model.fit(X_train_shuffled_scaled, y_train_shuffled)


# Predict the validation set
y_pred_xgbc = XGBC_model.predict(X_valid_scaled)

print('F_beta score: ', fbeta_score(y_valid, y_pred_xgbc, beta=1))
# F_beta score:  0.9996153846153846
print('AUC score: ', roc_auc_score(y_valid, y_pred_xgbc))
# AUC score:  0.9997813765182185


XGBC_model_feature_importances = pd.Series(XGBC_model.feature_importances_, index = X_valid.columns).sort_values(ascending = False)
XGBC_model_feature_importances = XGBC_model_feature_importances / XGBC_model_feature_importances.max()

# essentially just naive forecasting

# Tacking_lag_1           1.000000  <-------------
# AvgSoS_lag_5            0.001600
# Roll_lag_1              0.001300
# TWD_cos_lag_1           0.001177
# AWS_lag_1               0.000888
# VMG_lag_5               0.000807
# WSoG_lag_3              0.000770
# VMG_lag_4               0.000705
# Roll_lag_2              0.000690
# SoS_lag_1               0.000674
# VMG_lag_3               0.000637
# WSoG_lag_5              0.000596
# TWD_sin                 0.000548
# CurrentDir_sin_lag_4    0.000534
# TWD_cos_lag_5           0.000514
# AvgSoS_lag_3            0.000506
# TWD_cos_lag_2           0.000491
# WSoG_lag_2              0.000488
# HoG_sin_lag_5           0.000482
# Pitch_lag_3             0.000443
# AWA_lag_4               0.000401
# HoG_sin_lag_1           0.000400
# TWD_sin_lag_2           0.000393
# Roll                    0.000360
# VMG                     0.000343
# TWD_sin_lag_5           0.000323
# HoG_cos_lag_2           0.000313
# AvgSoS                  0.000309
# Yaw                     0.000297
# WSoG_lag_1              0.000274
# Pitch                   0.000270
# Yaw_lag_4               0.000270
# Pitch_lag_1             0.000265
# CurrentDir_cos          0.000262
# AWS_lag_4               0.000260
# HoG_cos                 0.000233
# TWD_cos_lag_3           0.000228
# Roll_lag_4              0.000219
# CurrentDir_cos_lag_1    0.000218
# AirTemp_lag_3           0.000203
# HoG_cos_lag_5           0.000203
# Pitch_lag_4             0.000199
# VMG_lag_2               0.000196
# WSoG_lag_4              0.000193
# CurrentSpeed_lag_1      0.000191
# TWD_sin_lag_4           0.000182
# CurrentSpeed_lag_4      0.000177
# Pitch_lag_2             0.000176
# CurrentSpeed_lag_2      0.000163
# AWA_lag_2               0.000158
# HoG_cos_lag_3           0.000141
# CurrentSpeed            0.000138
# AWA_lag_5               0.000134
# TWD_sin_lag_1           0.000125
# CurrentSpeed_lag_5      0.000111
# CurrentDir_sin_lag_3    0.000104
# Yaw_lag_3               0.000101
# VMG_lag_1               0.000099
# Roll_lag_3              0.000091
# WSoG                    0.000062
# AWS_lag_5               0.000060
# CurrentDir_sin_lag_5    0.000037
# TWD_cos_lag_4           0.000035
# CurrentDir_cos_lag_3    0.000031
# CurrentDir_cos_lag_2    0.000028
# AWS                     0.000024
# AWA                     0.000019
# CurrentDir_sin_lag_1    0.000015
# HoG_cos_lag_4           0.000014
# HoG_sin_lag_2           0.000000
# CurrentDir_sin          0.000000
# CurrentDir_cos_lag_4    0.000000
# CurrentDir_cos_lag_5    0.000000
# AWS_lag_3               0.000000
# TWD_sin_lag_3           0.000000
# AirTemp                 0.000000
# AWS_lag_2               0.000000
# SoS                     0.000000
# HoG_cos_lag_1           0.000000
# Leeway                  0.000000
# CurrentSpeed_lag_3      0.000000
# HoG_sin                 0.000000
# HoG_sin_lag_4           0.000000
# HoG_sin_lag_3           0.000000
# TWD_cos                 0.000000
# SoS_lag_5               0.000000
# AWA_lag_1               0.000000
# Leeway_lag_3            0.000000
# SoS_lag_4               0.000000
# AvgSoS_lag_4            0.000000
# SoS_lag_3               0.000000
# SoS_lag_2               0.000000
# AirTemp_lag_5           0.000000
# AirTemp_lag_4           0.000000
# AirTemp_lag_2           0.000000
# Leeway_lag_1            0.000000
# Leeway_lag_2            0.000000
# Leeway_lag_4            0.000000
# AvgSoS_lag_1            0.000000
# Leeway_lag_5            0.000000
# AirTemp_lag_1           0.000000
# Pitch_lag_5             0.000000
# Roll_lag_5              0.000000
# Yaw_lag_1               0.000000
# Yaw_lag_2               0.000000
# AWA_lag_3               0.000000
# Yaw_lag_5               0.000000
# CurrentDir_sin_lag_2    0.000000
# AvgSoS_lag_2            0.000000
# dtype: float32


y_pred_train_valid_xgbc = XGBC_model.predict(np.vstack((X_train_scaled, X_valid_scaled)))

plt.plot(np.arange(len(y_pred_train_valid_xgbc)), y_pred_train_valid_xgbc)
plt.axvline(x=len(Xy_train))
plt.plot(np.arange(len(y_pred_train_valid_xgbc)), pd.concat([Xy_train['Tacking'], y_valid]))


y_pred_train_valid_xgbc2 = XGBC_model.predict(np.vstack((X_train_scaled, X_valid_scaled2)))

plt.plot(np.arange(len(y_pred_train_valid_xgbc2)), y_pred_train_valid_xgbc2)
plt.plot(np.arange(len(y_pred_train_valid_xgbc2)), pd.concat([Xy_train['Tacking'], y_valid2]))



#################################


import lightgbm as lgb


LGBMclf = lgb.LGBMClassifier(class_weight = 'balanced', )
                            #  learning_rate = 0.01, 
                            # num_leaves = 800,
                            # n_estimators = 500, 
                            # num_iterations = 900, 
                            # max_bin = 500, 
                            # feature_fraction = 0.7, 
                            # bagging_fraction = 0.7,
                            # lambda_l2 = 0.5,
                            # max_depth = 7,
                            # silent = False
                            # )

LGBMclf.fit(X_train_shuffled_scaled, y_train_shuffled, 
            eval_metric = 'auc')

# Predict the validation set
y_pred_lgbm = LGBMclf.predict(X_valid_scaled)


print('F_beta score: ', fbeta_score(y_valid, y_pred_lgbm, beta=1))
# F_beta score:  0.9992310649750096
print('AUC score: ', roc_auc_score(y_valid, y_pred_lgbm))
# AUC score:  0.9997287449392712


LGBMclf_feature_importances = pd.Series(LGBMclf.feature_importances_, index = X_valid.columns).sort_values(ascending = False)
LGBMclf_feature_importances = LGBMclf_feature_importances / LGBMclf_feature_importances.max()

# Pitch_lag_1             1.000000
# AvgSoS_lag_5            0.574561
# AvgSoS                  0.565789
# HoG_sin_lag_1           0.548246
# CurrentDir_cos_lag_1    0.539474
# AWA_lag_2               0.530702
# WSoG_lag_4              0.447368
# WSoG_lag_2              0.442982
# Tacking_lag_1           0.438596  <---------
# CurrentDir_cos_lag_2    0.429825
# TWD_sin_lag_5           0.416667
# AWS_lag_5               0.394737
# Yaw_lag_3               0.346491
# WSoG_lag_1              0.324561
# AWS_lag_3               0.302632
# TWD_cos_lag_5           0.289474
# Roll_lag_1              0.285088
# Pitch                   0.276316
# CurrentDir_sin_lag_5    0.263158
# CurrentDir_cos_lag_3    0.245614
# TWD_sin_lag_3           0.171053
# CurrentSpeed_lag_1      0.157895
# HoG_cos_lag_2           0.157895
# HoG_sin_lag_5           0.153509
# CurrentSpeed            0.153509
# HoG_sin_lag_4           0.149123
# WSoG                    0.149123
# TWD_sin                 0.149123
# Roll                    0.144737
# SoS                     0.131579
# AWS                     0.131579
# AvgSoS_lag_4            0.122807
# HoG_sin_lag_3           0.114035
# TWD_sin_lag_2           0.109649
# Yaw                     0.109649
# TWD_sin_lag_4           0.100877
# VMG_lag_4               0.096491
# HoG_cos                 0.092105
# AWS_lag_1               0.087719
# VMG_lag_1               0.078947
# AWS_lag_2               0.078947
# WSoG_lag_3              0.078947
# VMG                     0.078947
# AirTemp_lag_2           0.074561
# CurrentDir_sin_lag_4    0.070175
# AirTemp_lag_5           0.065789
# Pitch_lag_5             0.061404
# CurrentSpeed_lag_2      0.061404
# VMG_lag_5               0.057018
# HoG_cos_lag_4           0.057018
# SoS_lag_4               0.052632
# CurrentSpeed_lag_5      0.052632
# Roll_lag_2              0.048246
# WSoG_lag_5              0.048246
# HoG_sin                 0.048246
# Roll_lag_5              0.048246
# CurrentDir_sin_lag_1    0.048246
# AWA_lag_5               0.043860
# VMG_lag_2               0.043860
# AWS_lag_4               0.043860
# AvgSoS_lag_1            0.039474
# AirTemp                 0.039474
# SoS_lag_3               0.039474
# CurrentDir_sin          0.030702
# Roll_lag_4              0.030702
# VMG_lag_3               0.030702
# AirTemp_lag_1           0.030702
# AWA_lag_1               0.030702
# Roll_lag_3              0.030702
# CurrentDir_cos_lag_5    0.026316
# Pitch_lag_3             0.026316
# TWD_cos_lag_2           0.026316
# AWA_lag_4               0.026316
# CurrentDir_sin_lag_2    0.021930
# HoG_cos_lag_3           0.021930
# Pitch_lag_2             0.021930
# HoG_cos_lag_5           0.021930
# SoS_lag_2               0.017544
# AWA_lag_3               0.017544
# AWA                     0.017544
# SoS_lag_5               0.017544
# CurrentDir_sin_lag_3    0.013158
# TWD_cos_lag_1           0.013158
# Leeway_lag_3            0.013158
# CurrentSpeed_lag_4      0.013158
# Leeway                  0.013158
# Yaw_lag_4               0.008772
# CurrentDir_cos          0.008772
# Leeway_lag_4            0.008772
# CurrentDir_cos_lag_4    0.008772
# TWD_cos_lag_4           0.008772
# TWD_cos_lag_3           0.008772
# Yaw_lag_2               0.004386
# Yaw_lag_1               0.004386
# HoG_cos_lag_1           0.004386
# CurrentSpeed_lag_3      0.004386
# Pitch_lag_4             0.004386
# AirTemp_lag_3           0.004386
# AvgSoS_lag_3            0.004386
# HoG_sin_lag_2           0.004386
# Leeway_lag_5            0.000000
# TWD_sin_lag_1           0.000000
# Yaw_lag_5               0.000000
# TWD_cos                 0.000000
# Leeway_lag_2            0.000000
# Leeway_lag_1            0.000000
# SoS_lag_1               0.000000
# AirTemp_lag_4           0.000000
# AvgSoS_lag_2            0.000000
# dtype: float64

# the plots look terrible despite the high score

y_pred_train_valid_lgbm = LGBMclf.predict(np.vstack((X_train_scaled, X_valid_scaled)))

plt.plot(np.arange(len(y_pred_train_valid_lgbm)), y_pred_train_valid_lgbm)
plt.axvline(x=len(Xy_train))
plt.plot(np.arange(len(y_pred_train_valid_lgbm)), pd.concat([Xy_train['Tacking'], y_valid]))


y_pred_train_valid_lgbm2 = LGBMclf.predict(np.vstack((X_train_scaled, X_valid_scaled2)))

plt.plot(np.arange(len(y_pred_train_valid_lgbm2)), y_pred_train_valid_lgbm2)
plt.plot(np.arange(len(y_pred_train_valid_lgbm2)), pd.concat([Xy_train['Tacking'], y_valid2]))

















