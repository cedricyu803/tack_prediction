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

# VoltageDrawn: Volts. Voltage drawn from the boat batteries that power the navigation system and (at night) lights
# ModePilote: unclear. whether the autopilot is active or not

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
train_df = train_df_raw.copy()

# missing values
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
# Tacking           5
# dtype: int64


# fillna with preceding value
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

train_df = train_df[['day', 'hour', 'minute', 'Latitude', 'Longitude', 'CurrentSpeed', 'CurrentDir','WSoG','TWD','TWS', 'TWA', 'AWS', 'AWA', 'HeadingTrue', 'HeadingMag', 'HoG','Pitch', 'Roll','Yaw','Leeway','RudderAng','AirTemp', 'SoG', 'SoS', 'AvgSoS', 'VMG' , 'VoltageDrawn', 'ModePilote', 'Tacking']]


train_df_num = train_df[['day', 'hour', 'minute', 'Latitude', 'Longitude', 'CurrentSpeed', 'CurrentDir','WSoG','TWD','TWS', 'TWA', 'AWS', 'AWA', 'HeadingTrue', 'HeadingMag', 'HoG','Pitch', 'Roll','Yaw','Leeway','RudderAng','AirTemp', 'SoG', 'SoS', 'AvgSoS', 'VMG' , 'VoltageDrawn']].groupby(['day', 'hour', 'minute']).agg(np.nanmean)

train_df_cat = train_df[['day', 'hour', 'minute', 'ModePilote', 'Tacking']].groupby(['day', 'hour', 'minute']).agg(lambda x:x.value_counts().index[0])


train_df = pd.concat([train_df_num, train_df_cat, train_df_DateTime], axis = 1)
train_df = train_df.reset_index()
train_df = train_df.drop(['day', 'hour', 'minute'], axis = 1)

#%% label = 'Tacking'


train_df['Tacking'].describe()
# count    3334.000000
# mean        0.229754
# std         0.420738
# min         0.000000
# 25%         0.000000
# 50%         0.000000
# 75%         0.000000
# max         1.000000
# Name: Tacking, dtype: float64

# !!! imbalanced class


# time series plot

train_df.plot(x='DateTime', y="Tacking", color = 'black')
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 20)
ax.set_ylabel('Tacking', fontsize = 20, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.set_yticks([0,1])
ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 20)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/Tacking_time.png', dpi = 150)


#!!!  test stationarity with augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

ADFresult = adfuller(train_df['Tacking'])
print('ADF Statistic: %f' % ADFresult[0])
print('p-value: %f' % ADFresult[1])
# ADF Statistic: -3.208549
# p-value: 0.019495
# stationary


#!!! tacking duration, except for the first one (which lasted for hours)
df_diff=train_df['Tacking'].diff()
tack_start = list(df_diff[df_diff!=0].dropna().iloc[1::2].index)
# [605, 1015, 1070, 1191, 1967, 2573]
tack_end = list(df_diff[df_diff!=0].dropna().iloc[2::2].index)
# [615, 1019, 1074, 1234, 1993, 2665]
tack_duration = [tack_end[i]-tack_start[i] for i in range(len(tack_end))]
tack_duration # in minutes
# [10, 4, 4, 43, 26, 92]


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
# ModePilote      0.716084
# Roll            0.673679
# HoG             0.634108
# Leeway          0.622034
# Pitch           0.597062
# TWD             0.530878
# AirTemp         0.474234
# CurrentDir      0.474042
# AWA             0.383677
# RudderAng       0.277983
# HeadingMag      0.207515
# TWA             0.129444
# HeadingTrue     0.073261
# VoltageDrawn   -0.184076
# Longitude      -0.379388
# CurrentSpeed   -0.393883
# TWS            -0.400057
# WSoG           -0.403834
# Latitude       -0.429777
# Yaw            -0.526225
# AWS            -0.641126
# VMG            -0.727575
# SoG            -0.732479
# SoS            -0.732490
# AvgSoS         -0.756665
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
ax.set_xticklabels(['Latitude', 'Longitude', 'CurrentSpeed', 'CurrentDir', 'WSoG', 'TWD',
       'TWS', 'TWA', 'AWS', 'AWA', 'HeadingTrue', 'HeadingMag', 'HoG', 'Pitch',
       'Roll', 'Yaw', 'Leeway', 'RudderAng', 'AirTemp', 'SoG', 'SoS', 'AvgSoS',
       'VMG', 'VoltageDrawn', 'ModePilote', 'Tacking'],rotation=60)
# plt.savefig('plots/correlation_matrix_abs.png', dpi = 300)

# !!! small absolute correlation: TWA (except with AWA), magnetic heading, true heading except with yaw, voltage drawn, lat/long except with speeds and yaw
# large absolute correlations between wind speeds, between boat speeds, between boat speeds and wind speeds

"""auto-correlation"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# plot_acf(train_df['Tacking'], lags = 200);

plot_pacf(train_df['Tacking']);
ax = plt.gca()
ax.set_xlabel('Lags / minutes')
ax.set_ylabel('Partial auto-correlation')
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.1,15.)
plt.ylim(-0.25,)
# plt.savefig('plots/Tacking_pacf.png', dpi = 150)


#%% coordinates: Latitude, Longitude

# # Coordinate heatmap using Google Maps API
# # https://stackoverflow.com/questions/28952112/python-have-gps-coordinates-and-corrsponding-values-generate-a-2d-heat-map
# import gmaps
# import gmaps.datasets
# from ipywidgets.embed import embed_minimal_html

# gmaps.configure(api_key="YOUR_GOOGLE_MAPS_API_KEY_HERE")

# # Heat map
# fig = gmaps.figure()
# heatmap_layer = gmaps.heatmap_layer(
#     train_df[["Latitude","Longitude"]].dropna() ,point_radius=15
# )
# fig.add_layer(heatmap_layer)

# embed_minimal_html('export.html', views=[fig])

# # !!! training set coordinates are localised in the Caribbeans; discard because it does not generalise

#%% AirTemp


train_df.plot(x='DateTime', y="AirTemp", color = 'red')
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 20)
ax.set_ylabel('Air temperature / Celsius', fontsize = 20, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(0.05, 1.)
# ax.set_yticks([0,1])
ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 20)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().remove()
plt.ylim(0,)
# plt.savefig('plots/AirTemp_time.png', dpi = 150)

#%% VoltageDrawn, ModePilote

train_df['ModePilote'].unique()
# array([ 5.,  2.])
# binary; will become (0,1) after rescaling by MinMaxScaler

train_df['VoltageDrawn'].describe()
# count    3334.000000
# mean       12.426303
# std         0.560525
# min        11.550000
# 25%        12.100000
# 50%        12.261667
# 75%        12.481667
# max        14.076667
# Name: VoltageDrawn, dtype: float64


#!!! time series plots

train_df0 = train_df[['DateTime', 'VoltageDrawn', 'ModePilote', 'Tacking']]


train_df0['Tacking_rescaled'] = (train_df0['ModePilote'].max()-train_df0['ModePilote'].min())*train_df0['Tacking'] + train_df0['ModePilote'].min()

fig = plt.figure(dpi = 150)
ax = plt.gca()
train_df.plot(x='DateTime', y="ModePilote", ax=ax, label = 'Autopilot Mode', color='green', linewidth=0.8)
train_df0.plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black', linewidth=0.8)
ax.set_xlabel('Date-Hour', fontsize = 12)
# ax.set_ylabel('Tacking', fontsize = 12, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.set_yticks([])
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(1.99,)
plt.legend(loc='lower right', fontsize = 16)
# plt.savefig('plots/ModePilote_tacking', dpi = 150)


train_df0['Tacking_rescaled'] = (train_df0['VoltageDrawn'].max()-train_df0['VoltageDrawn'].min())*train_df0['Tacking'] + train_df0['VoltageDrawn'].min()

fig = plt.figure(dpi = 150)
ax = plt.gca()
train_df.plot(x='DateTime', y="VoltageDrawn", ax=ax, label = 'Voltage Drawn from batteries', color='gold', linewidth=0.8)
train_df0.plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black', linewidth=0.8)
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 12)
ax.set_ylabel('Volt', fontsize = 12, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
# ax.set_ylabel(None)
# ax.set_yticks([train_df0['VoltageDrawn'].min(),train_df0['VoltageDrawn'].max()])
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(train_df0['VoltageDrawn'].min(),)
plt.legend(loc='lower right')
# plt.savefig('plots/VoltageDrawn_tacking', dpi = 150)


train_df_tack = [train_df.iloc[tack_start[i]-2*(tack_end[i]-tack_start[i]):tack_end[i]+2*(tack_end[i]-tack_start[i])] for i in range(len(tack_start))]

for i in range(len(train_df_tack)):
    train_df_tack[i]['Tacking_rescaled'] = 3*train_df_tack[i]['Tacking']+2

# for i in range(1):
for i in range(len(train_df_tack)):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="ModePilote", ax=ax, label = 'Autopilot Mode', color='green', linewidth=0.8)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black', linewidth=0.8)
    ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    ax.set_ylabel(None)
    # ax.set_ylabel('Direction / degree')
    ax.set_yticks([2,5])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(1.99,)
    plt.legend(loc='lower right')
    plt.savefig('plots/ModePilote_tacking/Figure_'+str(i+1), dpi = 150)


for i in range(len(train_df_tack)):
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['VoltageDrawn'].max()-train_df_tack[i]['VoltageDrawn'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['VoltageDrawn'].min()

for i in range(len(train_df_tack)):
# for i in range(1):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="VoltageDrawn", ax=ax, label = 'Voltage Drawn from batteries', color='gold', linewidth=0.8)
    train_df_tack[i].plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black', linewidth=0.8)
    ax = plt.gca()
    ax.set_xlabel('Date-Hour')
    # ax.set_ylabel(None)
    ax.set_ylabel('Volts')
    # ax.set_yticks([train_df0['VoltageDrawn'].min(),train_df0['VoltageDrawn'].max()])
    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(train_df_tack[i]['VoltageDrawn'].min(),)
    plt.legend(loc='lower right')
    plt.savefig('plots/VoltageDrawn_tacking/Figure_'+str(i+1), dpi = 150)




#%% CurrentSpeed, CurrentDir


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

train_df0['Tacking_rescaled'] = (train_df['CurrentSpeed'].max()-train_df['CurrentSpeed'].min())*train_df['Tacking'] + train_df['CurrentSpeed'].min()

ax = plt.gca()
train_df.plot(x='DateTime', y="CurrentSpeed", ax=ax, label = 'Current Speed', color='blue', linewidth=1)
train_df0.plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black', linewidth=1)
ax.set_xlabel('Date-Hour', fontsize = 12)
ax.set_ylabel('Water current speed / knots', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(0.05, 1.)
# ax.set_yticks([0,1])
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0.,)
plt.legend(loc='lower right', fontsize = 16)
# plt.legend().remove()
# plt.savefig('plots/CurrentSpeed', dpi = 150)


train_df0['Tacking_rescaled'] = (train_df['CurrentDir'].max()-train_df['CurrentDir'].min())*train_df['Tacking'] + train_df['CurrentDir'].min()

ax = plt.gca()
train_df.plot(x='DateTime', y="CurrentDir", ax=ax, label = 'Current Direction', color='blue', linewidth=1)
# train_df0.plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black', linewidth=1)
ax.set_xlabel('Date-Hour', fontsize = 12)
ax.set_ylabel('Water current direction / degree', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(0.05, 1.)
# ax.set_yticks([0,1])
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.ylim(1.99,)
plt.legend(loc='lower right', fontsize = 16)
plt.legend().remove()
# plt.savefig('plots/CurrentDir', dpi = 150)



train_df_tack = [train_df.iloc[tack_start[i]-2*(tack_end[i]-tack_start[i]):tack_end[i]+2*(tack_end[i]-tack_start[i])] for i in range(len(tack_start))]

for i in range(len(train_df_tack)):
    
    train_df_tack[i]['CurrDir_principal'] = np.arctan2(np.sin(train_df_tack[i]['CurrentDir']*np.pi/180),np.cos(train_df_tack[i]['CurrentDir']*np.pi/180))*180/np.pi
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['CurrentDir'].max()-train_df_tack[i]['CurrentDir'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['CurrentDir'].min()


for i in range(len(train_df_tack)):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="CurrentDir", ax=ax, label = 'Current Direction', color='blue', linewidth=1)
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
    train_df_tack[i].plot(x='DateTime', y="CurrentSpeed", ax=ax, label = 'Current speed', color='blue', linewidth=1)
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


# !!! correlations

cor['WSoG'].sort_values(ascending=False)
# TWS             0.998421
# AWS             0.906181
# ...
# Pitch          -0.769964
# Roll           -0.781817
# Name: WSoG, dtype: float64
cor['TWS'].sort_values(ascending=False)
# WSoG            0.998421
# AWS             0.906042
# VMG             0.5
# ...
# Pitch          -0.768815
# Roll           -0.781013

cor['AWS'].sort_values(ascending=False)
# WSoG            0.906181
# TWS             0.906042
# SoS             0.789019
# SoG             0.789009
# VMG             0.779557
# AvgSoS          0.650959
# ...
# Pitch          -0.890351
# Roll           -0.931184
# Name: AWS, dtype: float64

cor['TWA'].sort_values(ascending=False)
# AWA             0.895458
# ...
# Name: TWA, dtype: float64

fig = plt.figure(dpi = 150)
sns.scatterplot(x=train_df['WSoG'], y=train_df['TWS'])
ax = plt.gca()
ax.set_xlabel('Wind speed over ground / knot')
ax.set_ylabel('True wind speed / knot')
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0,30)
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

# # !!! group by target

# train_df[['AWS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
# #                AWS
# # Tacking           
# # 0.0      21.296804
# # 1.0      13.933718

# fig = plt.figure(dpi = 150)
# sns.barplot(x = ['No','Yes'], y = train_df[['AWS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
# ax = plt.gca()
# ax.set_xlabel('Tacking')
# ax.set_ylabel('Mean apparent wind speed / knot')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.ylim(12,)
# # plt.savefig('plots/AWS_target.png', dpi = 150)

# train_df[['AWA', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
# #                AWA
# # Tacking           
# # 0.0      35.328167
# # 1.0      53.792533

# fig = plt.figure(dpi = 150)
# sns.barplot(x = ['No','Yes'], y = train_df[['AWA', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
# ax = plt.gca()
# ax.set_xlabel('Tacking')
# ax.set_ylabel('Mean apparent wind angle / degree')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.ylim(25,)
# # plt.savefig('plots/AWA_target.png', dpi = 150)

#!!!
""" time series plots """


ax = plt.gca()
train_df.plot(x='DateTime', y="AWS", ax=ax, label = 'Apparent wind speed', color='skyblue', linewidth=1)
train_df.plot(x='DateTime', y="TWS", ax=ax, label = 'True wind speed', color='royalblue', linewidth=1)
# train_df0.plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black', linewidth=1)
ax.set_xlabel('Date-Hour', fontsize = 12)
ax.set_ylabel('knot', fontsize = 16)
ax.xaxis.set_label_coords(1., -0.05)
# ax.yaxis.set_label_coords(0.05, 1.)
# ax.set_yticks([0,1])
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.ylim(1.99,)
plt.legend(loc='lower right', fontsize = 16)
# plt.legend().remove()
# plt.savefig('plots/Windspeeds', dpi = 150)


ax = plt.gca()
train_df.plot(x='DateTime', y="AWA", ax=ax, label = 'Apparent wind angle', color='skyblue', linewidth=1)
train_df.plot(x='DateTime', y="TWA", ax=ax, label = 'True wind angle', color='royalblue', linewidth=1)
# train_df0.plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black', linewidth=1)
ax.set_xlabel('Date-Hour', fontsize = 12)
ax.set_ylabel('degree', fontsize = 16)
ax.xaxis.set_label_coords(1., -0.05)
# ax.yaxis.set_label_coords(0.05, 1.)
# ax.set_yticks([0,1])
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.ylim(1.99,)
plt.legend(loc='lower right', fontsize = 16)
# plt.legend().remove()
# plt.savefig('plots/Windangles', dpi = 150)



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

plot_pacf(train_df['AWS']);
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

plot_pacf(train_df['AWA']);
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


# fig = plt.figure(dpi = 150)
# train_df[['HeadingTrue']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# # plt.yscale('log')
# ax = plt.gca()
# ax.set_xlabel('True heading / degree')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_yticks([])
# ax.set_xticks([0,90,180, 270,360])
# # plt.xlim(-180,180)
# # plt.savefig('plots/HeadingTrue.png', dpi = 150)

# fig = plt.figure(dpi = 150)
# train_df[['HeadingMag']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# # plt.yscale('log')
# ax = plt.gca()
# ax.set_xlabel('Magnetic heading / degree')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_yticks([])
# ax.set_xticks([0,90,180, 270,360])
# # plt.xlim(-180,180)
# # plt.savefig('plots/HeadingMag.png', dpi = 150)

# fig = plt.figure(dpi = 150)
# train_df[['HoG']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# # plt.yscale('log')
# ax = plt.gca()
# ax.set_xlabel('Heading over ground / degree')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_yticks([])
# ax.set_xticks([0,90,180, 270,360])
# # plt.savefig('plots/HoG.png', dpi = 150)

#%% Pitch, roll, yaw, leeway, rudder

# fig = plt.figure(dpi = 150)
# train_df[['Pitch']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# # plt.yscale('log')
# ax = plt.gca()
# ax.set_xlabel('Pitch / degree')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_yticks([])
# # plt.savefig('plots/Pitch.png', dpi = 150)

# fig = plt.figure(dpi = 150)
# train_df[['Roll']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# # plt.yscale('log')
# ax = plt.gca()
# ax.set_xlabel('Roll / degree')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_yticks([])
# # plt.savefig('plots/Roll.png', dpi = 150)

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

# # !!! group by target

# train_df[['Roll', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
# #               Roll
# # Tacking           
# # 0.0     -16.800121
# # 1.0      -6.791818


# fig = plt.figure(dpi = 150)
# train_df[['Yaw']].hist(bins = 75, grid = False, ax = plt.gca(), color = 'skyblue')
# # plt.yscale('log')
# ax = plt.gca()
# ax.set_xlabel('Yaw / degree')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_yticks([])
# # plt.savefig('plots/Yaw.png', dpi = 150)

# fig = plt.figure(dpi = 150)
# train_df[['Leeway']].hist(bins = 20, grid = False, ax = plt.gca(), color = 'skyblue')
# # plt.yscale('log')
# ax = plt.gca()
# ax.set_xlabel('Leeway / degree')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_yticks([])
# # plt.savefig('plots/Leeway.png', dpi = 150)

# fig = plt.figure(dpi = 150)
# train_df[['RudderAng']].hist(bins = 20, grid = False, ax = plt.gca(), color = 'skyblue')
# # plt.yscale('log')
# ax = plt.gca()
# ax.set_xlabel('Rudder angle / degree')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_yticks([])
# plt.xlim(-40,40)
# # plt.savefig('plots/RudderAng.png', dpi = 150)

# train_df[['AirTemp']].describe()
# # Out[306]: 
# #              AirTemp
# # count  219835.000000
# # mean       27.628387
# # std         1.821622
# # min        21.107229
# # 25%        26.328928
# # 50%        27.217728
# # 75%        29.050879
# # max        32.439430

#!!! times series plots


train_df0 = train_df[['DateTime', 'Tacking', 'Pitch', 'Yaw', 'Roll', 'Leeway', 'RudderAng']]
# train_df0['Tacking'] = 10*train_df0['Tacking']
train_df0['Yaw_principal'] = np.arctan2(np.sin(train_df0['Yaw']*np.pi/180),np.cos(train_df0['Yaw']*np.pi/180))*180/np.pi
train_df0['Pitch_principal'] = np.arctan2(np.sin(train_df0['Pitch']*np.pi/180),np.cos(train_df0['Pitch']*np.pi/180))*180/np.pi
train_df0['Roll_principal'] = np.arctan2(np.sin(train_df0['Roll']*np.pi/180),np.cos(train_df0['Roll']*np.pi/180))*180/np.pi
train_df0['Leeway_principal'] = np.arctan2(np.sin(train_df0['Leeway']*np.pi/180),np.cos(train_df0['Leeway']*np.pi/180))*180/np.pi
train_df0['RudderAng_principal'] = np.arctan2(np.sin(train_df0['RudderAng']*np.pi/180),np.cos(train_df0['RudderAng']*np.pi/180))*180/np.pi
train_df0['Tacking_rescaled'] = (train_df0['RudderAng_principal'].max()-train_df0['RudderAng_principal'].min())*train_df0['Tacking'] + train_df0['RudderAng_principal'].min()

fig = plt.figure(dpi = 150)
ax = plt.gca()
# train_df0.plot(x='DateTime', y="Pitch_principal", ax=ax, label = 'Pitch', color='tomato', linewidth=1)
# train_df.plot(x='DateTime', y="Yaw", ax=ax, label = 'Yaw', color='royalblue', linewidth=1)
train_df0.plot(x='DateTime', y="RudderAng_principal", ax=ax, label = 'Rudder Angle / degree', color='skyblue', linewidth=1)
# train_df.plot(x='DateTime', y="Roll", ax=ax, label = 'Roll', color='gold', linewidth=1)
# train_df.plot(x='DateTime', y="Leeway", ax=ax, label = 'Leeway', color='brown', linewidth=1)
train_df0.plot(x='DateTime', y="Tacking_rescaled", ax=ax, label = 'Tacking', color='black')
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 16)
# ax.set_ylabel('degree', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='lower right', fontsize=16)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/RudderAng_Tacking_time.png', dpi = 150)


train_df_tack = [train_df.iloc[tack_start[i]-2*(tack_end[i]-tack_start[i]):tack_end[i]+2*(tack_end[i]-tack_start[i])] for i in range(len(tack_start))]



for i in range(len(train_df_tack)):
    
    train_df_tack[i]['Pitch_principal'] = np.arctan2(np.sin(train_df_tack[i]['Pitch']*np.pi/180),np.cos(train_df_tack[i]['Pitch']*np.pi/180))*180/np.pi
    train_df_tack[i]['Tacking_rescaled'] = (train_df_tack[i]['Pitch_principal'].max()-train_df_tack[i]['Pitch_principal'].min())*train_df_tack[i]['Tacking'] + train_df_tack[i]['Pitch_principal'].min()
    

for i in range(len(train_df_tack)):
# for i in range(1):
    fig = plt.figure(dpi = 150)
    ax = plt.gca()
    train_df_tack[i].plot(x='DateTime', y="Pitch_principal", ax=ax, label = 'Pitch', color='tomato', linewidth=1)
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
    plt.savefig('plots/Pitch_tacking/Figure_'+str(i+1), dpi = 150)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/AWS_TWS_Tacking_time.png', dpi = 150)



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
    train_df_tack[i].plot(x='DateTime', y="Roll_principal", ax=ax, label = 'Roll', color='gold', linewidth=1)
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


# pitch/yaw... VS other quantities such as speeds and wind
train_df0 = train_df[['DateTime', 'Tacking', 'Pitch', 'Yaw', 'Roll', 'Leeway', 'RudderAng']]
# train_df0['Tacking'] = 10*train_df0['Tacking']
train_df0['Yaw_principal'] = np.arctan2(np.sin(train_df0['Yaw']*np.pi/180),np.cos(train_df0['Yaw']*np.pi/180))*180/np.pi
train_df0['Pitch_principal'] = np.arctan2(np.sin(train_df0['Pitch']*np.pi/180),np.cos(train_df0['Pitch']*np.pi/180))*180/np.pi
train_df0['Roll_principal'] = np.arctan2(np.sin(train_df0['Roll']*np.pi/180),np.cos(train_df0['Roll']*np.pi/180))*180/np.pi
train_df0['Leeway_principal'] = np.arctan2(np.sin(train_df0['Leeway']*np.pi/180),np.cos(train_df0['Leeway']*np.pi/180))*180/np.pi
train_df0['RudderAng_principal'] = np.arctan2(np.sin(train_df0['RudderAng']*np.pi/180),np.cos(train_df0['RudderAng']*np.pi/180))*180/np.pi
train_df0['Tacking_rescaled'] = (train_df0['RudderAng_principal'].max()-train_df0['RudderAng_principal'].min())*train_df0['Tacking'] + train_df0['RudderAng_principal'].min()

train_df0['VMG_rescaled'] = 2*train_df['VMG']-15
train_df0['SoS_rescaled'] = 2*train_df['SoS']-15

fig = plt.figure(dpi = 150)
ax = plt.gca()
# train_df0.plot(x='DateTime', y="Pitch_principal", ax=ax, label = 'Pitch', color='tomato', linewidth=1)
# train_df.plot(x='DateTime', y="Yaw", ax=ax, label = 'Yaw', color='royalblue', linewidth=1)
# train_df0.plot(x='DateTime', y="RudderAng_principal", ax=ax, label = 'Rudder Angle', color='skyblue', linewidth=1)
train_df0.plot(x='DateTime', y="Roll_principal", ax=ax, label = 'Roll', color='gold', linewidth=1)
# train_df.plot(x='DateTime', y="Leeway", ax=ax, label = 'Leeway', color='brown', linewidth=1)
train_df0.plot(x='DateTime', y="SoS_rescaled", ax=ax, label = 'Speed over surface', color='black')
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 16)
# ax.set_ylabel('degree', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='lower right', fontsize=16)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/Roll_SoS_time.png', dpi = 150)



#!!!
"""auto-correlation"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# plot_acf(train_df_fillna['AvgSoS']);

plot_pacf(train_df['Roll']);
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


plot_pacf(train_df['Yaw']);
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

# # !!! group by target
# train_SoS_Tacking = train_df[['SoS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
# #               SoS
# # Tacking          
# # 0.0      8.749035
# # 1.0      3.281336

# fig = plt.figure(dpi = 150)
# train_df[['AvgSoS']].hist(bins = 20, grid = False, ax = plt.gca(), color = 'skyblue')
# # plt.yscale('log')
# ax = plt.gca()
# ax.set_xlabel('Average speed over surface / knot')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_yticks([])
# # plt.savefig('plots/AvgSoS.png', dpi = 150)

# # group by target
# train_AvgSoS_Tacking = train_df[['AvgSoS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
# #            AvgSoS
# # Tacking          
# # 0.0      7.521353
# # 1.0      2.064962

# fig = plt.figure(dpi = 150)
# sns.barplot(x = ['No','Yes'], y = train_df[['SoS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
# ax = plt.gca()
# ax.set_xlabel('Tacking')
# ax.set_ylabel('Mean speed over surface / knot')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.ylim(25,)
# # plt.savefig('plots/SoS_target.png', dpi = 150)

# fig = plt.figure(dpi = 150)
# sns.barplot(x = ['No','Yes'], y = train_df[['AvgSoS', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
# ax = plt.gca()
# ax.set_xlabel('Tacking')
# ax.set_ylabel('Mean Average speed over surface / knot')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # plt.ylim(25,)
# # plt.savefig('plots/AvgSoS_target.png', dpi = 150)

# train_df[['SoG', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
# #               SoG
# # Tacking          
# # 0.0      8.798975
# # 1.0      3.334825

# fig = plt.figure(dpi = 150)
# sns.barplot(x = ['No','Yes'], y = train_df[['SoG', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
# ax = plt.gca()
# ax.set_xlabel('Tacking')
# ax.set_ylabel('Mean speed over ground / knot')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.ylim(25,)
# # plt.savefig('plots/SoG_target.png', dpi = 150)

# train_df[['VMG', 'Tacking']].groupby(['Tacking']).agg(np.nanmean)
# #               VMG
# # Tacking          
# # 0.0      4.773949
# # 1.0      1.589161

# fig = plt.figure(dpi = 150)
# sns.barplot(x = ['No','Yes'], y = train_df[['VMG', 'Tacking']].groupby(['Tacking']).agg(np.nanmean).squeeze().to_list(), color = 'skyblue')
# ax = plt.gca()
# ax.set_xlabel('Tacking')
# ax.set_ylabel('Mean velocity made good / knot')
# ax.set_title(None)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.ylim(25,)
# # plt.savefig('plots/VMG_target.png', dpi = 150)

#!!! correlations

cor['AvgSoS'].sort_values(ascending=False)
# SoG             0.827487
# SoS             0.827461
# Latitude        0.777662
# ...
# Roll           -0.726571
# Tacking        -0.756665
# ModePilote     -0.790835

cor['SoS'].sort_values(ascending=False)
# SoG             0.999989
# VMG             0.883337
# AvgSoS          0.827461
# AWS          
# ...
# Roll           -0.859123
# ModePilote     -0.902530

cor['SoG'].sort_values(ascending=False)
# SoS             0.999989
# VMG             0.883220
# AvgSoS          0.827487
# ...
# Tacking        -0.732479
# Pitch          -0.771597
# Roll           -0.858996
# ModePilote     -0.902574

cor['VMG'].sort_values(ascending=False)
# SoS             0.883337
# SoG             0.883220
# AWS             0.779557
# AvgSoS          0.706333
# ...
# Tacking        -0.727575
# Pitch          -0.763349
# Roll           -0.811532
# ModePilote     -0.824172

#!!!
""" time series plots """

train_df0 = train_df[['DateTime', 'Tacking']]
train_df0['Tacking'] = 10*train_df0['Tacking']

fig = plt.figure(dpi = 150)
ax = plt.gca()
train_df.plot(x='DateTime', y="AvgSoS", ax=ax, label = 'Average Speed over surface', color='royalblue', linewidth=1)
train_df0.plot(x='DateTime', y="Tacking", ax=ax, label = 'Tacking', color='black')
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 16)
ax.set_ylabel('Speed / knot', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='lower right', fontsize=16)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/AvgSoS_Tacking_time.png', dpi = 150)


train_df0 = train_df[['DateTime', 'Tacking']]
train_df0['Tacking'] = 10*train_df0['Tacking']

fig = plt.figure(dpi = 150)
ax = plt.gca()
train_df.plot(x='DateTime', y="SoS", ax=ax, label = 'Speed over surface', color='royalblue', linewidth=1)
train_df0.plot(x='DateTime', y="Tacking", ax=ax, label = 'Tacking', color='black')
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 16)
ax.set_ylabel('Speed / knot', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='lower right', fontsize=16)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/SoS_Tacking_time.png', dpi = 150)

fig = plt.figure(dpi = 150)
ax = plt.gca()
train_df.plot(x='DateTime', y="AvgSoS", ax=ax, label = 'Average Speed over surface', color='royalblue', linewidth=1)
train_df.plot(x='DateTime', y="SoS", ax=ax, label = 'Speed over surface', color='skyblue', linewidth=1)
train_df.plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=1)
train_df.plot(x='DateTime', y="VMG", ax=ax, label = 'Velocity made good', color='pink', linewidth=1)
train_df0.plot(x='DateTime', y="Tacking", ax=ax, label = 'Tacking', color='black')
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 16)
ax.set_ylabel('Speed / knot', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='lower right', fontsize=12)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/speeds_Tacking_time.png', dpi = 150)


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
    train_df_tack[i].plot(x='DateTime', y="SoG", ax=ax, label = 'Speed over ground', color='tomato', linewidth=1)
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


# speeds VS other quantities such as wind
train_df0 = train_df[['DateTime', 'Tacking']]
train_df0['Tacking'] = 10*train_df0['Tacking']
train_df0['ModePilote'] = (train_df['ModePilote'] - 2)*4

fig = plt.figure(dpi = 150)
ax = plt.gca()
train_df.plot(x='DateTime', y="SoS", ax=ax, label = 'Speed over surface', color='royalblue', linewidth=1)
train_df0.plot(x='DateTime', y="ModePilote", ax=ax, label = 'Autopilot', color='red', linewidth=1)
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 16)
ax.set_ylabel('Speed / knot', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='lower right', fontsize=16)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/SoS_ModePilote_time.png', dpi = 150)

train_df0['ModePilote'] = (train_df['ModePilote'] - 6)*6
fig = plt.figure(dpi = 150)
ax = plt.gca()
train_df.plot(x='DateTime', y="Roll", ax=ax, label = 'Roll', color='royalblue', linewidth=1)
train_df0.plot(x='DateTime', y="ModePilote", ax=ax, label = 'Autopilot', color='red', linewidth=1)
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 16)
ax.set_ylabel('Speed / knot', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='lower right', fontsize=16)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/Roll_ModePilote_time.png', dpi = 150)

train_df0['ModePilote'] = (train_df['ModePilote']*8) - 10
fig = plt.figure(dpi = 150)
ax = plt.gca()
train_df.plot(x='DateTime', y="AWS", ax=ax, label = 'Apparent wind speed', color='royalblue', linewidth=1)
train_df0.plot(x='DateTime', y="ModePilote", ax=ax, label = 'Autopilot', color='red', linewidth=1)
ax = plt.gca()
ax.set_xlabel('Date-Hour', fontsize = 16)
ax.set_ylabel('Speed / knot', fontsize = 16, rotation=0)
ax.xaxis.set_label_coords(1., -0.05)
ax.yaxis.set_label_coords(-0.05, 1.)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
# ax.set_yticks([0,1])
ax.set_title(None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='lower right', fontsize=16)
# ax.get_legend().remove()
# plt.xlim(0,)
# plt.savefig('plots/AWS_ModePilote_time.png', dpi = 150)


#!!!
"""auto-correlation"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# plot_acf(train_df_fillna['AvgSoS']);

plot_pacf(train_df['AvgSoS']);
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

plot_pacf(train_df['SoS'], lags=70);
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

plot_pacf(train_df['SoG'], lags=20);
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

ccf_AvgSoSTacking = ccf(train_df['AvgSoS'], train_df['Tacking'])

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

Our correlation plots may inform us to drop certain columns, but actually it is better for model performance to keep the various speeds and angles. Though they are  quite correlated, the small differences may have been spotted by our ML models. See rolling_forecast_feature_selection.py

We imagine that our tacking prediction model is used on a boat that may not have Internet (cloud) access. Our models need to be lightweight, so they run fast (low response time) on the local machine on the boat with limited computing reousrces.

ML models

Use a few lags (test it) for features, lag 1 for target (NO! DO NOT!), sliding window rolling forecast.

The training window should be large enough to include some tacking events; severe class imbalance.We can only predict one time step ahead since we predict tacking based on current and past knowledge one all other variables including weather. Aggregate by minute to allow time for the model to fit and predict.


Plan: 
Use 18-hour windows for training window
One-step prediction-- a model makes predictions for the future 1 minute
data collection: aggregate 60 seconds worth of data and pass it to the model


Evaluation metric: F_beta for beta = 1, 2, 0.5

"""












