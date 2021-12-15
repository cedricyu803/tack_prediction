# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 20:30:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


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
Exploratory data analysis, plots
Summary at the end of this script
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
# import re
import seaborn as sns

import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\tacking')


#%% load data set

train_df_raw = pd.read_csv('test_data.csv')
train_df_raw['DateTime'] = pd.to_datetime(train_df_raw['DateTime'])
# train_df_raw.shape

train_df = train_df_raw.copy()
train_df.shape
# (220000, 27)

# missing target label
len(train_df[train_df['Tacking'].isnull()])
# 5

# re-order columns
train_df = train_df[['Latitude', 'Longitude', 'CurrentSpeed', 'CurrentDir','WSoG','TWD','TWS', 'TWA', 'AWS', 'AWA', 'HeadingTrue', 'HeadingMag', 'HoG','Pitch', 'Roll','Yaw','Leeway','RudderAng','AirTemp', 'SoG', 'SoS', 'AvgSoS', 'VMG' , 'VoltageDrawn', 'ModePilote', 'DateTime', 'Tacking']]

# fillna with preceding value
train_df_fillna = train_df.fillna(method='ffill')

#%% general observations

train_df.info()
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 219995 entries, 0 to 219999
# Data columns (total 27 columns):
#  #   Column        Non-Null Count   Dtype  
# ---  ------        --------------   -----  
#  0   CurrentSpeed  219828 non-null  float64
#  1   CurrentDir    219827 non-null  float64
#  2   TWS           219832 non-null  float64
#  3   TWA           219828 non-null  float64
#  4   AWS           219829 non-null  float64
#  5   AWA           219833 non-null  float64
#  6   Roll          219828 non-null  float64
#  7   Pitch         219831 non-null  float64
#  8   HeadingMag    219830 non-null  float64
#  9   HoG           219833 non-null  float64
#  10  HeadingTrue   219832 non-null  float64
#  11  AirTemp       219835 non-null  float64
#  12  Longitude     219831 non-null  float64
#  13  Latitude      219835 non-null  float64
#  14  SoG           219837 non-null  float64
#  15  SoS           219835 non-null  float64
#  16  AvgSoS        219833 non-null  float64
#  17  VMG           219832 non-null  float64
#  18  RudderAng     219833 non-null  float64
#  19  Leeway        219834 non-null  float64
#  20  TWD           219833 non-null  float64
#  21  WSoG          219831 non-null  float64
#  22  VoltageDrawn  219834 non-null  float64
#  23  ModePilote    219834 non-null  float64
#  24  DateTime      219990 non-null  object 
#  25  Yaw           219829 non-null  float64
#  26  Tacking       219995 non-null  float64
# dtypes: float64(26), object(1)
# memory usage: 47.0+ MB



# train_df.describe()
# !!! plots below show that there are no noticeable outliers

""" # some missing values (not many)"""
train_df.isnull().sum()
# CurrentSpeed    167
# CurrentDir      168
# TWS             163
# TWA             167
# AWS             166
# AWA             162
# Roll            167
# Pitch           164
# HeadingMag      165
# HoG             162
# HeadingTrue     163
# AirTemp         160
# Longitude       164
# Latitude        160
# SoG             158
# SoS             160
# AvgSoS          162
# VMG             163
# RudderAng       162
# Leeway          161
# TWD             162
# WSoG            164
# VoltageDrawn    161
# ModePilote      161
# DateTime          5
# Yaw             166
# Tacking           0
# dtype: int64



#%% label = 'Tacking'


train_df['Tacking'].describe()
# count    219995.000000
# mean          0.209273  <------
# std           0.406791
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           0.000000
# max           1.000000
# Name: Tacking, dtype: float64

# !!! imbalanced class

# bar chart
fig = plt.figure(dpi = 150)
sns.barplot(x = ['No','Yes'], y = [train_df['Tacking'].sum(), len(train_df['Tacking']) - train_df['Tacking'].sum()], color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Tacking')
ax.set_ylabel(None)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/Tacking.png', dpi = 150)

# time series plot
train_df.plot(x='DateTime', y="Tacking")
ax = plt.gca()
ax.set_xlabel('Date-Hour')
ax.set_ylabel('Tacking')
ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/Tacking_time.png', dpi = 150)


#!!!  test stationarity with augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

ADFresult = adfuller(train_df_fillna['Tacking'])
print('ADF Statistic: %f' % ADFresult[0])
print('p-value: %f' % ADFresult[1])
# ADF Statistic: -4.630132
# p-value: 0.000114
# stationary


#!!! tacking duration, except for the first one (which lasted for hours)
df_diff=train_df['Tacking'].diff()
tack_start = list(df_diff[df_diff!=0].dropna().iloc[1::2].index)
# [36300, 60880, 64200, 71440, 118000, 154380]
tack_end = list(df_diff[df_diff!=0].dropna().iloc[2::2].index)
# [36900, 61160, 64440, 74040, 119600, 159900]
tack_duration = [tack_end[i]-tack_start[i] for i in range(len(tack_end))]
tack_duration # in seconds
# Out[32]: [600, 280, 240, 2600, 1600, 5520]
# in minutes
# array([10.        ,  4.66666667,  4.        , 43.33333333, 26.66666667,
#        92.        ])

"""# correlation matrix"""
cor = train_df.corr()
mask = np.array(cor)
# mask = np.abs(mask)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,15)
sns.heatmap(cor,mask= mask,square=True,annot=False)
# plt.savefig('plots/correlation_matrix.png', dpi = 300)

cor['Tacking'].sort_values(ascending=False)
# Tacking         1.000000
# ModePilote      0.717051
# Roll            0.596005
# HoG             0.503146
# Leeway          0.493317
# AirTemp         0.472660
# TWD             0.452589
# CurrentDir      0.431117
# AWA             0.371277
# Pitch           0.365490
# RudderAng       0.169972
# HeadingMag      0.132467
# TWA             0.110141
# HeadingTrue     0.040227
# VoltageDrawn   -0.162096
# WSoG           -0.269085
# TWS            -0.311944
# Yaw            -0.378158
# CurrentSpeed   -0.396365
# Longitude      -0.407314
# Latitude       -0.452816
# AWS            -0.563277
# VMG            -0.626334
# SoS            -0.720361
# SoG            -0.722064
# AvgSoS         -0.763796
# Name: Tacking, dtype: float64
# !!!
# target is highly (anti-) correlated with speeds


# Absolute value of correlation; for better legibility
cor2 = train_df.corr()
cor2 = np.abs(cor2)
mask2 = np.array(cor2)
mask2[np.tril_indices_from(mask2)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,15)
sns.heatmap(cor2,mask= mask2,square=True,annot=False)
# plt.savefig('plots/correlation_matrix_abs.png', dpi = 300)

# !!! small absolute correlation: TWA (except with AWA), magnetic heading, true heading except with yaw, voltage drawn, lat/long except with speeds and yaw
# large absolute correlations between wind speeds, between boat speeds, between boat speeds and wind speeds

"""auto-correlation"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# plot_acf(train_df_fillna['Tacking']);

plot_pacf(train_df_fillna['Tacking']);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,10.5)
# plt.savefig('plots/Tacking_pacf.png', dpi = 150)
# !!! large for lag 1. consider using lag features


#%% coordinates: Latitude, Longitude

# Coordinate heatmap using Google Maps API
# https://stackoverflow.com/questions/28952112/python-have-gps-coordinates-and-corrsponding-values-generate-a-2d-heat-map
import gmaps
import gmaps.datasets
from ipywidgets.embed import embed_minimal_html

gmaps.configure(api_key="YOUR_GOOGLE_MAPS_API_KEY_HERE")

# Heat map
fig = gmaps.figure()
heatmap_layer = gmaps.heatmap_layer(
    train_df[["Latitude","Longitude"]].dropna() ,point_radius=15
)
fig.add_layer(heatmap_layer)

embed_minimal_html('export.html', views=[fig])

# !!! training set coordinates are localised in the Caribbeans; discard because it does not generalise


#%% CurrentSpeed, CurrentDir


fig = plt.figure('CurrentSpeed', dpi = 150)
train_df[['CurrentSpeed']].hist(bins = 100, grid = False, ax = plt.gca(), color = 'skyblue')
plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Current Speed / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(0,)
# plt.savefig('plots/CurrentSpeed_log.png', dpi = 150)


fig = plt.figure('CurrentDir', dpi = 150)
train_df[['CurrentDir']].hist(bins = 100, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Current Direction / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([0,90,180,270,360])
plt.xlim(0,360)
# plt.savefig('plots/CurrentDir.png', dpi = 150)
# mostly around 100 deg




#!!! auto-correlation

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_pacf(train_df['Tacking']);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,10.5)


#!!! time series plots

train_df_tack = [train_df.iloc[tack_start[i]-2*(tack_end[i]-tack_start[i]):tack_end[i]+2*(tack_end[i]-tack_start[i])] for i in range(len(tack_start))]

for i in range(len(train_df_tack)):
    
    train_df_tack[i]['CurrDir_principal'] = np.arctan2(np.sin(train_df_tack[i]['CurrentDir']*np.pi/180),np.cos(train_df_tack[i]['CurrentDir']*np.pi/180))*180/np.pi
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['CurrentDir'].max()-train_df_tack[i]['CurrentDir'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['CurrentDir'].min()


for i in range(len(train_df_tack)):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="CurrentDir", ax=ax, label = 'Current Direction', color='skyblue', linewidth=0.5)
    # train_df_tack[i].plot(x='DateTime', y="TWS", ax=ax, label = 'True wind speed', color='blue', linewidth=0.5)
    # train_df.plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=0.5)
    # train_df.plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
    ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel('Direction / degree')
    # ax.set_yticks([0,1])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('plots/CurrDir_tacking/Figure_'+str(i+1), dpi = 150)


for i in range(len(train_df_tack)):
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['CurrentSpeed'].max()-train_df_tack[i]['CurrentSpeed'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['CurrentSpeed'].min()

for i in range(len(train_df_tack)):
# for i in range(1):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="CurrentSpeed", ax=ax, label = 'Current speed', color='skyblue', linewidth=0.5)
    # train_df_tack[i].plot(x='DateTime', y="TWA", ax=ax, label = 'Apparent Wind Angle', color='blue', linewidth=0.5)
    # train_df.plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=0.5)
    # train_df.plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
    # ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel('Speed / knot')
    # ax.set_yticks([0,1])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('plots/CurrSpeed_tacking/Figure_'+str(i+1), dpi = 150)
    

#%% Wind

# Wind speed over ground
fig = plt.figure('WSoG', dpi = 150)
train_df[['WSoG']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Wind speed over ground / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
plt.xlim(0,30)
# plt.savefig('plots/WSoG.png', dpi = 150)

# looks Gaussian


# True wind direction, relative to [water]
fig = plt.figure('TWD', dpi = 150)
train_df[['TWD']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('True wind direction / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([0,90,180,270,360])
plt.xlim(0,360)
# plt.savefig('plots/TWD.png', dpi = 150)

# mostly from NE

# True Wind Speed, relative to [water]
fig = plt.figure('TWS', dpi = 150)
train_df[['TWS']].hist(bins = 60, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('True wind speed / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
plt.xlim(0,30)
# plt.savefig('plots/TWS.png', dpi = 150)

train_df['CurrDir_TWD'] = train_df['CurrentDir']-train_df['TWD']


fig = plt.figure('CurrDir_TWD', dpi = 150)
train_df[['CurrDir_TWD']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Curent Direction - True wind direction / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([-180,-90,0,90,180,270,360])
plt.xlim(-180,360)
# plt.savefig('plots/CurrDir_TWD.png', dpi = 150)


# !!! Are the wind speed over ground and water calibrated? Are they from different sources, e.g. radar VS pitot tube?
# In the following examples, WSoG and TWS should be roughly the same because CurrentSpeed is tiny
train_df[['CurrentSpeed', 'WSoG', 'TWS']].head(20)
# Out[194]: 
#     CurrentSpeed  WSoG   TWS
# 0         0.0756  10.5  10.8
# 1         0.0756  10.5  10.8
# 2         0.0756   9.9  10.8
# 3         0.0756   9.9  10.8
# 4         0.0756  10.3  10.8
# 5         0.0756  10.3  10.8
# 6         0.0756  10.1  10.8
# 7         0.0756  10.1  10.8
# 8         0.0756  10.1  10.8
# 9         0.0756  10.3  10.8
# 10        0.0756  10.3  10.8
# 11        0.0756  10.5  10.8
# 12        0.0756  10.5  10.8
# 13        0.0756  10.5  10.8
# 14        0.0756  10.5  10.8
# 15        0.0756   8.9  10.8
# 16        0.0756   8.9  10.8
# 17        0.0756   8.9  10.8
# 18        0.0756  10.2  10.8
# 19        0.0756  10.8  10.8

train_df[['CurrentSpeed', 'WSoG', 'TWS']].iloc[530:560]
# Out[199]: 
#      CurrentSpeed       WSoG   TWS
# 530        0.0918  11.400000  11.9
# 531        0.0810  15.000000  12.5
# 532        0.0756  12.500000  12.7
# 533        0.0864  11.000000  12.4
# 534        0.0972  12.100000  12.1
# 535        0.0918  13.300000  12.0
# 536        0.0972  13.000000  12.5
# 537        0.1080  15.500000  12.9
# 538        0.1134  16.200001  13.7
# 539        0.1080  16.200001  14.6
# 540        0.1134  15.400000  15.0
# 541        0.1026  15.600000  15.2
# 542        0.0972  17.600000  15.5
# 543        0.0918  15.100000  15.8
# 544        0.0864  13.900000  15.5
# 545        0.0918  13.400000  15.2
# 546        0.0864  12.100000  14.6
# 547        0.0972  11.800000  14.0
# 548        0.0864  11.600000  13.3
# 549        0.0810  11.600000  12.9
# 550        0.0756  12.100000  12.5
# 551        0.0702  11.800000  12.3
# 552        0.0810  12.900000  12.2
# 553        0.0756  11.800000  12.2
# 554        0.0864  10.200000  12.1
# 555        0.0972  13.200000  12.3
# 556        0.0918  14.300000  12.7
# 557        0.0864  15.700000  13.3
# 558        0.0972  16.400000  13.5
# 559        0.1026  16.600000  14.3


# True wind angle
fig = plt.figure('TWA', dpi = 150)
train_df[['TWA']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('True wind angle / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([-180,-90,0,90,180])
# plt.xlim(-180,180)
# plt.savefig('plots/TWA.png', dpi = 150)


# Apparent Wind Speed, relative to [water]
fig = plt.figure('AWS', dpi = 150)
train_df[['AWS']].hist(bins = 60, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Apparent wind speed / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
plt.xlim(0,30)
# plt.savefig('plots/AWS.png', dpi = 150)


# Apparent wind angle
fig = plt.figure('AWA', dpi = 150)
train_df[['AWA']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Apparent wind angle / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([-180,-90,0,90,180])
# plt.xlim(-180,180)
# plt.savefig('plots/AWA.png', dpi = 150)

# !!! correlations

cor['WSoG'].sort_values(ascending=False)
# TWS             0.857522
# AWS             0.795332
# ...
# Roll           -0.653443
# Name: WSoG, dtype: float64
cor['TWS'].sort_values(ascending=False)
# AWS             0.907994
# WSoG            0.857520
# ...
# Pitch          -0.491641
# Roll           -0.759107

cor['AWS'].sort_values(ascending=False)
# TWS             0.907994
# WSoG            0.795332
# ...
# Roll           -0.884839

cor['TWA'].sort_values(ascending=False)
# AWA             0.856178
# HoG             0.110552
# ...

fig = plt.figure(dpi = 150)
sns.scatterplot(x=train_df['WSoG'], y=train_df['TWS'])
ax = plt.gca()
ax.set_xlabel('Wind speed over ground / knot')
ax.set_ylabel('True wind speed / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0,35)
# plt.savefig('plots/WSoG_TWS.png', dpi = 150)

fig = plt.figure(dpi = 150)
sns.scatterplot(x=train_df['WSoG'], y=train_df['AWS'], color='orange')
ax = plt.gca()
ax.set_xlabel('Wind speed over ground / knot')
ax.set_ylabel('Apparent wind speed / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('plots/WSoG_AWS.png', dpi = 150)

fig = plt.figure(dpi = 150)
sns.scatterplot(x=train_df['TWA'], y=train_df['AWA'])
ax = plt.gca()
ax.set_xlabel('True wind angle / degree')
ax.set_ylabel('Apparent wind angle / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.ylim(0,35)
# plt.savefig('plots/TWA_AWA.png', dpi = 150)

# !!! group by target

train_df[['AWS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
#                AWS
# Tacking           
# 0.0      21.296804
# 1.0      13.933718

fig = plt.figure(dpi = 150)
sns.barplot(x = ['No','Yes'], y = train_df[['AWS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Tacking')
ax.set_ylabel('Mean apparent wind speed / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(12,)
# plt.savefig('plots/AWS_target.png', dpi = 150)

train_df[['AWA', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
#                AWA
# Tacking           
# 0.0      35.328167
# 1.0      53.792533

fig = plt.figure(dpi = 150)
sns.barplot(x = ['No','Yes'], y = train_df[['AWA', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Tacking')
ax.set_ylabel('Mean apparent wind angle / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(25,)
# plt.savefig('plots/AWA_target.png', dpi = 150)

#!!!
""" time series plots """


train_df_tack = [train_df.iloc[tack_start[i]-2*(tack_end[i]-tack_start[i]):tack_end[i]+2*(tack_end[i]-tack_start[i])] for i in range(len(tack_start))]

for i in range(len(train_df_tack)):
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['AWS'].max()-train_df_tack[i]['AWS'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['AWS'].min()


for i in range(len(train_df_tack)):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="AWS", ax=ax, label = 'Apparent wind speed', color='skyblue', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="TWS", ax=ax, label = 'True wind speed', color='blue', linewidth=0.5)
    # train_df.plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=0.5)
    # train_df.plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
    ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel('Speed / knot')
    # ax.set_yticks([0,1])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('plots/wind_speed_tacking/Figure_'+str(i+1), dpi = 150)


for i in range(len(train_df_tack)):
    train_df_tack[i]['AWA_principal'] = np.arctan2(np.sin(train_df_tack[i]['AWA']*np.pi/180),np.cos(train_df_tack[i]['AWA']*np.pi/180))*180/np.pi
    train_df_tack[i]['TWA'] = np.arctan2(np.sin(train_df_tack[i]['TWA']*np.pi/180),np.cos(train_df_tack[i]['TWA']*np.pi/180))*180/np.pi
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['AWA_principal'].max()-train_df_tack[i]['AWA_principal'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['AWA_principal'].min()

for i in range(len(train_df_tack)):
# for i in range(1):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="AWA", ax=ax, label = 'Apparent Wind Angle', color='skyblue', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="TWA", ax=ax, label = 'Apparent Wind Angle', color='blue', linewidth=0.5)
    # train_df.plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=0.5)
    # train_df.plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
    # ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel('Angle / degree')
    # ax.set_yticks([0,1])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('plots/wind_angle_tacking/Figure_'+str(i+1), dpi = 150)
    

#!!!
"""auto-correlation"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# plot_acf(train_df_fillna['AvgSoS']);

plot_pacf(train_df_fillna['AWS']);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.2,12.5)
plt.ylim(-.1,1.1)
# plt.savefig('plots/AWS_pacf.png', dpi = 150)

plot_pacf(train_df_fillna['AWA']);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.2,12.5)
plt.ylim(-.3,1.1)
# plt.savefig('plots/AWA_pacf.png', dpi = 150)



#%% Headings

# not very important

fig = plt.figure(dpi = 150)
train_df[['HeadingTrue']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('True heading / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([0,90,180, 270,360])
# plt.xlim(-180,180)
# plt.savefig('plots/HeadingTrue.png', dpi = 150)

fig = plt.figure(dpi = 150)
train_df[['HeadingMag']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Magnetic heading / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([0,90,180, 270,360])
# plt.xlim(-180,180)
# plt.savefig('plots/HeadingMag.png', dpi = 150)

fig = plt.figure(dpi = 150)
train_df[['HoG']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Heading over ground / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([0,90,180, 270,360])
# plt.savefig('plots/HoG.png', dpi = 150)

#%% Pitch, roll, yaw, leeway

fig = plt.figure(dpi = 150)
train_df[['Pitch']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Pitch / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
# plt.savefig('plots/Pitch.png', dpi = 150)

fig = plt.figure(dpi = 150)
train_df[['Roll']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Roll / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
# plt.savefig('plots/Roll.png', dpi = 150)

# !!! roll is quite correlated with apparent wind speed and boat speed over surface
fig = plt.figure(dpi = 150)
sns.scatterplot(x=train_df['Roll'], y=train_df['AWS'])
ax = plt.gca()
ax.set_xlabel('Roll / degree')
ax.set_ylabel('Apparent wind speed / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.ylim(0,35)
# plt.savefig('plots/Roll_AWS.png', dpi = 150)

fig = plt.figure(dpi = 150)
sns.scatterplot(x=train_df['Roll'], y=train_df['SoS'])
ax = plt.gca()
ax.set_xlabel('Roll / degree')
ax.set_ylabel('Speed over surface / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.ylim(0,35)
# plt.savefig('plots/Roll_SoS.png', dpi = 150)

# !!! group by target

train_df[['Roll', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
#               Roll
# Tacking           
# 0.0     -16.800121
# 1.0      -6.791818


fig = plt.figure(dpi = 150)
train_df[['Yaw']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Yaw / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
# plt.savefig('plots/Yaw.png', dpi = 150)

fig = plt.figure(dpi = 150)
train_df[['Leeway']].hist(bins = 20, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Leeway / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
# plt.savefig('plots/Leeway.png', dpi = 150)

fig = plt.figure(dpi = 150)
train_df[['RudderAng']].hist(bins = 20, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Rudder angle / degree')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
plt.xlim(-40,40)
# plt.savefig('plots/RudderAng.png', dpi = 150)

train_df[['AirTemp']].describe()
# Out[306]: 
#              AirTemp
# count  219835.000000
# mean       27.628387
# std         1.821622
# min        21.107229
# 25%        26.328928
# 50%        27.217728
# 75%        29.050879
# max        32.439430

#!!! times series plots

train_df_tack = [train_df.iloc[tack_start[i]-2*(tack_end[i]-tack_start[i]):tack_end[i]+2*(tack_end[i]-tack_start[i])] for i in range(len(tack_start))]
# train_df_tack = pd.concat(train_df_tack, axis = 0)
for i in range(len(train_df_tack)):
    
    train_df_tack[i]['Yaw_principal'] = np.arctan2(np.sin(train_df_tack[i]['Yaw']*np.pi/180),np.cos(train_df_tack[i]['Yaw']*np.pi/180))*180/np.pi
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['Yaw_principal'].max()-train_df_tack[i]['Yaw_principal'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['Yaw_principal'].min()
    


for i in range(len(train_df_tack)):
# for i in range(1):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="Yaw_principal", ax=ax, label = 'Yaw', color='skyblue', linewidth=0.5)
    # train_df_tack[i].plot(x='DateTime', y="TWS", ax=ax, label = 'True wind speed', color='blue', linewidth=0.5)
    # train_df.plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=0.5)
    # train_df.plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
    ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel('Angle / degree')
    # ax.set_yticks([0,1])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('plots/Yaw_tacking/Figure_'+str(i+1), dpi = 150)



for i in range(len(train_df_tack)):
    
    train_df_tack[i]['Roll_principal'] = np.arctan2(np.sin(train_df_tack[i]['Roll']*np.pi/180),np.cos(train_df_tack[i]['Roll']*np.pi/180))*180/np.pi
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['Roll_principal'].max()-train_df_tack[i]['Roll_principal'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['Roll_principal'].min()
    

for i in range(len(train_df_tack)):
# for i in range(1):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="Roll_principal", ax=ax, label = 'Roll', color='skyblue', linewidth=0.5)
    # train_df_tack[i].plot(x='DateTime', y="TWS", ax=ax, label = 'True wind speed', color='blue', linewidth=0.5)
    # train_df.plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=0.5)
    # train_df.plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
    ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel('Angle / degree')
    # ax.set_yticks([0,1])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('plots/Roll_tacking/Figure_'+str(i+1), dpi = 150)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/AWS_TWS_Tacking_time.png', dpi = 150)

#!!!
"""auto-correlation"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# plot_acf(train_df_fillna['AvgSoS']);

plot_pacf(train_df_fillna['Roll']);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.2,20.5)
plt.ylim(-.3,1.1)
# plt.savefig('plots/Roll_pacf.png', dpi = 150)


plot_pacf(train_df_fillna['Yaw']);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.2,20.5)
plt.ylim(-.3,1.1)
# plt.savefig('plots/Yaw_pacf.png', dpi = 150)


#%% Speeds

fig = plt.figure(dpi = 150)
train_df[['SoS']].hist(bins = 20, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Speed over surface / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
# plt.savefig('plots/SoS.png', dpi = 150)

# !!! group by target
train_SoS_Tacking = train_df[['SoS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
#               SoS
# Tacking          
# 0.0      8.749035
# 1.0      3.281336

fig = plt.figure(dpi = 150)
train_df[['AvgSoS']].hist(bins = 20, grid = False, ax = plt.gca(), color = 'skyblue')
# plt.yscale('log')
ax = plt.gca()
ax.set_xlabel('Average speed over surface / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
# plt.savefig('plots/AvgSoS.png', dpi = 150)

# group by target
train_AvgSoS_Tacking = train_df[['AvgSoS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
#            AvgSoS
# Tacking          
# 0.0      7.521353
# 1.0      2.064962

fig = plt.figure(dpi = 150)
sns.barplot(x = ['No','Yes'], y = train_df[['SoS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Tacking')
ax.set_ylabel('Mean speed over surface / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(25,)
# plt.savefig('plots/SoS_target.png', dpi = 150)

fig = plt.figure(dpi = 150)
sns.barplot(x = ['No','Yes'], y = train_df[['AvgSoS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Tacking')
ax.set_ylabel('Mean Average speed over surface / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.ylim(25,)
# plt.savefig('plots/AvgSoS_target.png', dpi = 150)

train_df[['SoG', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
#               SoG
# Tacking          
# 0.0      8.798975
# 1.0      3.334825

fig = plt.figure(dpi = 150)
sns.barplot(x = ['No','Yes'], y = train_df[['SoG', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Tacking')
ax.set_ylabel('Mean speed over ground / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(25,)
# plt.savefig('plots/SoG_target.png', dpi = 150)

train_df[['VMG', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
#               VMG
# Tacking          
# 0.0      4.773949
# 1.0      1.589161

fig = plt.figure(dpi = 150)
sns.barplot(x = ['No','Yes'], y = train_df[['VMG', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
ax = plt.gca()
ax.set_xlabel('Tacking')
ax.set_ylabel('Mean velocity made good / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(25,)
# plt.savefig('plots/VMG_target.png', dpi = 150)

#!!! correlations

cor['AvgSoS'].sort_values(ascending=False)
# SoG             0.806154
# SoS             0.804244
# Latitude        0.785522
# Longitude       0.737277
# VMG             0.596955
# ...
# AirTemp        -0.649621
# Tacking        -0.763796
# ModePilote     -0.786978

cor['SoS'].sort_values(ascending=False)
# SoG             0.999162
# AvgSoS          0.804244
# VMG             0.789141
# ...
# Tacking        -0.720361
# Roll           -0.801109
# ModePilote     -0.880642

cor['SoG'].sort_values(ascending=False)
# SoS             0.999162
# AvgSoS          0.806154
# VMG             0.783535
# AWS             0.749820
# ...
# Tacking        -0.722064
# Roll           -0.800788
# ModePilote     -0.882801

cor['VMG'].sort_values(ascending=False)
# SoS             0.789141
# SoG             0.783535
# ...
# Tacking        -0.626334
# Roll           -0.654115
# ModePilote     -0.709826

#!!!
""" time series plots """

train_df0 = train_df[['DateTime', 'Tacking']]
train_df0['Tacking'] = 10*train_df0['Tacking']

fig = plt.figure(dpi = 150)
ax = plt.gca()
train_df.plot(x='DateTime', y="AvgSoS", ax=ax, label = 'Average Speed over surface', color='royalblue', linewidth=1)
# train_df.plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=0.5)
# train_df.plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
train_df0.plot(x='DateTime', y="Tacking", ax=ax, label = 'Tacking', color='black')
ax = plt.gca()
ax.set_xlabel('Date-Hour')
ax.set_ylabel('Speed / knot')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/AvgSoS_Tacking_time.png', dpi = 150)



train_df_tack = [train_df.iloc[tack_start[i]-2*(tack_end[i]-tack_start[i]):tack_end[i]+2*(tack_end[i]-tack_start[i])] for i in range(len(tack_start))]
# train_df_tack = pd.concat(train_df_tack, axis = 0)
for i in range(len(train_df_tack)):
    
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['AvgSoS'].max()-train_df_tack[i]['AvgSoS'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['AvgSoS'].min()

for i in range(len(train_df_tack)):
# for i in range(1):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    # train_df_tack[i].plot(x='DateTime', y="SoS", ax=ax, label = 'Speed over surface', color='skyblue', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="AvgSoS", ax=ax, label = 'Average speed over surface', color='royalblue', linewidth=1)
    # train_df_tack[i].plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
    ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel('Speed / knot')
    # ax.set_yticks([0,1])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('plots/AvgSoS_tacking/Figure_'+str(i+1), dpi = 150)



train_df_tack = [train_df.iloc[tack_start[i]-2*(tack_end[i]-tack_start[i]):tack_end[i]+2*(tack_end[i]-tack_start[i])] for i in range(len(tack_start))]
# train_df_tack = pd.concat(train_df_tack, axis = 0)
for i in range(len(train_df_tack)):
    
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['SoG'].max()-train_df_tack[i]['SoG'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['SoG'].min()

for i in range(len(train_df_tack)):
# for i in range(1):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    # train_df_tack[i].plot(x='DateTime', y="SoS", ax=ax, label = 'Speed over surface', color='skyblue', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=0.5)
    # train_df_tack[i].plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
    ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel('Speed / knot')
    # ax.set_yticks([0,1])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('plots/SoG_tacking/Figure_'+str(i+1), dpi = 150)


for i in range(len(train_df_tack)):
    
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['SoS'].max()-train_df_tack[i]['SoS'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['SoS'].min()

for i in range(len(train_df_tack)):
# for i in range(1):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="SoS", ax=ax, label = 'Speed over surface', color='skyblue', linewidth=0.5)
    # train_df_tack[i].plot(x='DateTime', y="TWS", ax=ax, label = 'True wind speed', color='blue', linewidth=0.5)
    # train_df_tack[i].plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=0.5)
    # train_df_tack[i].plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=0.5)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
    ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel('Speed / knot')
    # ax.set_yticks([0,1])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('plots/SoS_tacking/Figure_'+str(i+1), dpi = 150)

#!!!
"""auto-correlation"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# plot_acf(train_df_fillna['AvgSoS']);

plot_pacf(train_df_fillna['AvgSoS']);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,10.5)
plt.ylim(-0.1,1.1)
# plt.savefig('plots/AvgSoS_pacf.png', dpi = 150)

plot_pacf(train_df_fillna['SoS'], lags=70);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,12.5)
plt.ylim(-1.1,1.1)
# plt.savefig('plots/SoS_pacf.png', dpi = 150)

plot_pacf(train_df_fillna['SoG'], lags=20);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,12.5)
plt.ylim(-1.1,1.1)
# plt.savefig('plots/SoG_pacf.png', dpi = 150)

from statsmodels.tsa.stattools import ccf

ccf_AvgSoSTacking = ccf(train_df_fillna['AvgSoS'], train_df_fillna['Tacking'])

plt.plot(np.arange(len(ccf_AvgSoSTacking)), ccf_AvgSoSTacking)

plot_pacf(train_df['Tacking']);
ax = plt.gca()
ax.set_xlabel('Lags / second')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,10.5)
# plt.savefig('plots/Tacking_pacf.png', dpi = 150)






#!!!
"""

Summary:
No notable outliers
use ffill for missing values
angles (yaw, AWA, etc.): convert to principal branch (-180,180)
other angles: compute sines and cosines

Our feature selection is informed by the correlation matrix. We DROP:
cols_to_drop = ['Latitude', 'Longitude', 'TWS', 'TWA', 'HeadingTrue', 'HeadingMag', 'RudderAng', 'SoG', 'DateTime', 'VoltageDrawn', 'ModePilote']
Here we used that 'WSoG', 'TWS' and 'AWS' are highly correlated; we choose 'WSoG' and 'AWS'
'SoG' and 'SoS' are almost perfectly correlated; drop 'SoG'

columns = train_df.columns.to_list()
for col in cols_to_drop:
    columns.remove(col)

cols_to_keep = ['CurrentSpeed', 'CurrentDir', 'AWS', 'AWA', 'Roll', 'Pitch', 'HoG', 'AirTemp', 'SoS', 'AvgSoS', 'VMG', 'Leeway', 'TWD', 'WSoG', 'Yaw', 'Tacking']

** It turns out this choice of features is pretty bad =[. See rolling_forecast_feature_selection.py

We imagine that our tacking prediction model may be used in sailing races where time is of the essence. Moreover, the boat may not have Internet (cloud) access. Our models need to be lightweight, so they run fast (low response time) on the local machine on the boat with limited computing reousrces.

ML models

Use a few lags (test it) for features, lag 1 for target (NO!), sliding window rolling forecast.

The training and validation windows should be large enough to include some tacking events; severe class imbalance; tacking occurs too rarely, so our sliding training window may not see any tacking events at all

Models need to be re-trained regularly to keep errors from significantly increasing, but not too frequently (e.g. every second) as it can hog valuable computational resources.

Plan: 
Use 18-hour windows for both training and validation windows
Multi-step prediction-- a model makes predictions for the future 12 hour
At the end of each 12 hour, slide the training and validation windows and train a new model


Evaluation metric: F_beta? AUC?

"""












