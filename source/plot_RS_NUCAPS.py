#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 08:48:58 2020

@author: nal

Plot data from surface measurement, radiosonde, satellite and lidar measurements
"""
import pandas as pd
import numpy as np
import math
import os
from scipy import spatial
import geopy.distance
import scipy.ndimage as ndimage
import xarray as xr
import time
import datetime as dt

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, NullFormatter, ScalarFormatter

from satpy import Scene, find_files_and_readers

from metpy import calc as cc
import metpy.calc as mpcalc
from metpy.units import units
from metpy.interpolate import interpolate_to_grid
import matplotlib.pyplot as plt
from metpy.cbook import get_test_data
from metpy.plots import Hodograph, SkewT
from metpy.interpolate import interpolate_1d

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

######################################## Load data and define numpy arrays ######################################## 
Year = "2020"
Month = "04"
Day = "27"
Hour = "00" # only 00 and 12
Minute="00" # only 00 and 30
Seconds='00'

dynfmt = "%Y%m%d%H%M%S"

##### Surface Measurement ##### -> to be downloaded
data_surf = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/surf_stat_20200427000000.txt')
lowest_pres_SMN = data_surf['90'][0]

pressure_hPa = data_surf['90'].values * units.hPa
temperature_degC = data_surf['91'].values * units.degC
relative_humidity_percent = data_surf['98'].values * units('percent')
dewpoint_degC = cc.dewpoint_from_relative_humidity(temperature_degC, relative_humidity_percent)

###### Radiosondes ######
### read Radiosonde file and save as dataframe ###
RS_data = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/Payerne/RS_concat.nc').to_dataframe()

### choose desired time ###
RS_data = RS_data[RS_data.time_YMDHMS == int(Year+Month+Day+Hour+Minute+Seconds)]
RS_data = RS_data[RS_data.pressure_hPa != 1000] # delete first row (undefined values)
RS_data = RS_data.reset_index(drop=True)

### find pressure at 600 m (for altitude - pressure conversion of Lidar)
nearest_value = find_nearest(RS_data.geopotential_altitude_m,600)
lowest_pres_RS = float(RS_data.pressure_hPa[RS_data.geopotential_altitude_m==nearest_value])

### define variables ###
p_RS = RS_data.pressure_hPa
p_RS_original = RS_data.pressure_hPa.to_frame()

T_RS = RS_data.temperature_degC.to_frame()
T_RS_original = RS_data.temperature_degC.to_frame()
#T_RS_K = RS_data.temperature_degC.to_frame() + 273.15

T_d_RS = RS_data.dew_point_degC.to_frame()
T_d_RS_original = RS_data.dew_point_degC.to_frame()

z_RS = RS_data.geopotential_altitude_m
RH_RS = RS_data.relative_humidity_percent.to_frame()

delta_hPa = 100 # define hPa window
T_RS_list = []
T_d_RS_list = []
p_RS_list = []

### smooth curve with window of hPa ###
for i in range(0,len(p_RS)):
    window_p_min = p_RS.iloc[i] + delta_hPa
    window_p_max = p_RS.iloc[i] - delta_hPa
    min_val = p_RS[0]
    if min_val > window_p_min:
        index_list = np.where(np.logical_and(p_RS>=window_p_min, p_RS<=window_p_max))
        T_index = T_RS[(p_RS > window_p_max) & (p_RS < window_p_min)]
        T_d_index = T_d_RS[(p_RS > window_p_max) & (p_RS < window_p_min)]
        p_index = p_RS[(p_RS > window_p_max) & (p_RS < window_p_min)]
   
        mean_T_RS = np.mean(T_index)
        T_RS_list.append(mean_T_RS)
        
        mean_T_d_RS = np.mean(T_d_index)
        T_d_RS_list.append(mean_T_d_RS)
    else:
        T_index = T_RS[(p_RS > window_p_max) & (p_RS < p_RS.iloc[i])]
        T_d_index = T_d_RS[(p_RS > window_p_max) & (p_RS < p_RS.iloc[i])]
        p_index = p_RS[(p_RS > window_p_max) & (p_RS < p_RS.iloc[i])]
   
        mean_T_RS = np.mean(T_index)
        T_RS_list.append(mean_T_RS)
        
        mean_T_d_RS = np.mean(T_d_index)
        T_d_RS_list.append(mean_T_d_RS)
    
    
T_RS_smoothed = pd.DataFrame(T_RS_list)

T_d_RS_smoothed = pd.DataFrame(T_d_RS_list) 

##############################  NUCAPS ##############################
### read file ###
file_index = 1 # define file
date_NUCAPS = Year + Month + Day
date_NUCAPS = date_NUCAPS +'.txt'
filtered_files = pd.read_csv(os.path.join('/data/COALITION2/PicturesSatellite/results_NAL/NUCAPS', date_NUCAPS))

index_CP = filtered_files.iloc[file_index,3]
min_dist = filtered_files.iloc[file_index,2]
filenames = [filtered_files.iloc[file_index,1]]
global_scene = Scene(reader="nucaps", filenames=filenames)    
NUCAPS_time = filenames = filtered_files.iloc[file_index,1]
NUCAPS_time = NUCAPS_time[65:80]
NUCAPS_time = dt.datetime.strptime(NUCAPS_time,"%Y%m%d%H%M%S%f")
time_dif = np.abs(dt.datetime.strptime(Year+Month+Day+Hour+Minute+Seconds,dynfmt) - NUCAPS_time)

# read variables
var_pres="H2O_MR"
global_scene.load([var_pres], pressure_levels=True)

var_temp = "Temperature"
global_scene.load([var_temp], pressure_levels=True)

# Pressure 
p = global_scene[var_pres].coords['Pressure'].values
df_pres = pd.DataFrame(p)
df_pres = df_pres.loc[[index_CP],:].T
df_pres.columns = ['Pressure']
df_pres_orig = df_pres
index_min_pressure = np.where(df_pres > lowest_pres_RS)[0][0]
df_pres = df_pres[0:int(index_min_pressure)]
p_NUCAPS = (df_pres['Pressure'].values * units.hPa)
p_NUCAPS_orig = (df_pres_orig['Pressure'].values * units.hPa)

# Temperature variables
## Temperature
T = global_scene[var_temp].values
df_Temp = pd.DataFrame(T)
df_Temp = df_Temp.loc[[index_CP],:].T
df_Temp.columns = ['Temperature']
df_Temp = df_Temp[0:int(index_min_pressure)]
T_NUCAPS_1 = df_Temp['Temperature'].values - 273.15
T_NUCAPS = (df_Temp['Temperature'].values * units.kelvin).to(units.degC)

## MIT Temperature
var_temp_MIT = "MIT_Temperature"
global_scene.load([var_temp_MIT], pressure_levels=True)

T = global_scene[var_temp_MIT].values
df_Temp = pd.DataFrame(T)
df_Temp = df_Temp.loc[[index_CP],:].T
df_Temp.columns = ['MIT_Temperature']
df_Temp = df_Temp[0:int(index_min_pressure)]
MIT_T_NUCAPS_1 = df_Temp['MIT_Temperature'].values -273.15
MIT_T_NUCAPS = (df_Temp['MIT_Temperature'].values * units.kelvin).to(units.degC)

## FG Temperature
var_temp_FG = "FG_Temperature"
global_scene.load([var_temp_FG], pressure_levels=True)

T = global_scene[var_temp_FG].values
df_Temp = pd.DataFrame(T)
df_Temp = df_Temp.loc[[index_CP],:].T
df_Temp.columns = ['FG_Temperature']
df_Temp = df_Temp[0:int(index_min_pressure)]
FG_T_NUCAPS_1 = df_Temp['FG_Temperature'].values -273.15
FG_T_NUCAPS = (df_Temp['FG_Temperature'].values * units.kelvin).to(units.degC)

# moisture variables
## H2O MR
var = "H2O_MR"
global_scene.load([var], pressure_levels=True)

WVMR = global_scene[var].values # mass mixing ratio (mWV / mDA) kg/kg
WVMR = WVMR * 1000 # convert to grams
WVMR = WVMR * units('g/kg')
e_1 = mpcalc.vapor_pressure(p_NUCAPS_orig, WVMR)
T_d = mpcalc.dewpoint(e_1) 

df_Temp_D = pd.DataFrame(T_d)
df_Temp_D = df_Temp_D.loc[[index_CP],:].T
df_Temp_D.columns = ['Temperature_D']
df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
T_d_NUCAPS = (df_Temp_D['Temperature_D'].values)

df_Temp_D = pd.DataFrame(T_d.magnitude)
df_Temp_D = df_Temp_D.loc[[index_CP],:].T
df_Temp_D.columns = ['Temperature_D']
df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
T_d_NUCAPS_1 = (df_Temp_D['Temperature_D'].values)

## MIT H2O MR
var = "MIT_H2O_MR"
global_scene.load([var], pressure_levels=True)

WVMR = global_scene[var].values # mass mixing ratio (mWV / mDA) kg/kg
WVMR = WVMR * 1000 # convert to grams
WVMR = WVMR * units('g/kg')
e_1 = mpcalc.vapor_pressure(p_NUCAPS_orig, WVMR)
T_d = mpcalc.dewpoint(e_1)

df_Temp_D = pd.DataFrame(T_d)
df_Temp_D = df_Temp_D.loc[[index_CP],:].T
df_Temp_D.columns = ['MIT_Temperature_D']
df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
MIT_T_d_NUCAPS = (df_Temp_D['MIT_Temperature_D'].values)

df_Temp_D = pd.DataFrame(T_d.magnitude)
df_Temp_D = df_Temp_D.loc[[index_CP],:].T
df_Temp_D.columns = ['MIT_Temperature_D']
df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
MIT_T_d_NUCAPS_1 = (df_Temp_D['MIT_Temperature_D'].values)

# FG H2O MR
var_FG = "FG_H2O_MR"
global_scene.load([var_FG], pressure_levels=True)

WVMR = global_scene[var_FG].values # mass mixing ratio (mWV / mDA) kg/kg
WVMR = WVMR * 1000 # convert to grams
WVMR = WVMR * units('g/kg')
e_1 = mpcalc.vapor_pressure(p_NUCAPS_orig, WVMR)
T_d = mpcalc.dewpoint(e_1)

df_Temp_D = pd.DataFrame(T_d)
df_Temp_D = df_Temp_D.loc[[index_CP],:].T
df_Temp_D.columns = ['FG_Temperature_D']
df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
FG_T_d_NUCAPS = (df_Temp_D['FG_Temperature_D'].values)

df_Temp_D = pd.DataFrame(T_d.magnitude)
df_Temp_D = df_Temp_D.loc[[index_CP],:].T
df_Temp_D.columns = ['FG_Temperature_D']
df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
FG_T_d_NUCAPS_1 = (df_Temp_D['FG_Temperature_D'].values)

### Interpolate RS grid to Satellite grid ###
# Variante 1) Find closest value
#closest_p_list = []
#closest_p_index = []
#for i in range(0,len(df_pres_array)):
#    print(i)
#    nearest = find_nearest(p_RS_array, df_pres_array[i])
#    nearest_index = np.where(p_RS_array == nearest)[0]
#    nearest_index = int(nearest_index[0])
#    closest_p_list.append(nearest)
#    closest_p_index.append(nearest_index)

# select values variables at pressure levels
# smoothed points
#p_RS_1 = p_RS[closest_p_index] 
#T_RS_1 = T_RS[0][closest_p_index] 
#T_d_RS_1 = T_d_RS[0][closest_p_index] 

### add units ###
#p_RS = p_RS * units.hPa
#T_RS = T_RS * units.degC
#T_d_RS = T_d_RS * units.degC

# Variante 2) Interpolate T and Td to pressure levels
p_RS = p_RS.values* units.hPa 
T_RS = T_RS.values * units.degC
T_d_RS = T_d_RS.values * units.degC

p_RS_original = p_RS_original.values * units.hPa
T_RS_original = T_RS_original.values * units.degC
T_d_RS_original = T_d_RS_original.values * units.degC

T_RS = interpolate_1d(p_NUCAPS, p_RS, T_RS, axis=0)
T_d_RS = interpolate_1d(p_NUCAPS, p_RS, T_d_RS, axis = 0)
p_RS = p_NUCAPS

#p_RS_original = p_NUCAPS
#T_RS_original = interpolate_1d(p_NUCAPS, p_RS_original, T_RS_original, axis=0)
#T_d_RS_original = interpolate_1d(p_NUCAPS, p_RS_original, T_d_RS_original, axis = 0)

##############################  RALMO ##############################
RA_data = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/RALMO/Payerne/RA_06610_concat.nc').to_dataframe()

Time_RA=str(Year+Month+Day+Hour+Minute+Seconds)
Time_RA_1= dt.datetime.strptime(Time_RA,dynfmt) + dt.timedelta(minutes=30)
Time_RA=dt.datetime.strftime(Time_RA_1,dynfmt)
RA_data = RA_data[RA_data.time_YMDHMS == int(Time_RA)]

data_comma_temp = RA_data[RA_data['temperature_K']!= 1e+07]

# variables
g = 9.81
m_mol_air = 28.965*10.0**(-3.0)
R = 8.31446
z_RA = data_comma_temp['altitude_m']

temp = data_comma_temp['temperature_K']
T_RA = temp.values
temp_degC = temp - 273.15
temp_degC = temp_degC.values * units.degC

# calculate Lidar pressure levels with RS as lowest level 
p_1 = np.zeros(len(z_RA))
p_1[0] = lowest_pres_RS 
integrant = g*m_mol_air/(R*T_RA)

for i in range(1, len(z_RA)):
    p_1[i] = lowest_pres_RS*math.exp(-np.trapz(integrant[0:i], z_RA[0:i]))

p_1 = p_1 * units.hPa

spez_hum = data_comma_temp['specific_humidity_gkg-1']
spez_hum = spez_hum.values * units('g/kg')
temp_d_degC = cc.dewpoint_from_specific_humidity(spez_hum, temp_degC, p_1)
RH_RA = cc.relative_humidity_from_specific_humidity(spez_hum, temp_degC, p_1) 

####################################### Plot data and save figure ######################################## 
### compare Lidar and RS data 
# Pressure coordinates
fig, ax = plt.subplots(figsize = (5,12))
ax.plot(RH_RA * 100, p_1, color = 'red')
ax.plot(RH_RS, p_RS_original, color = 'black')
ax.set_title(Time_RA_1, fontsize = 16)
ax.set_ylim(1050,400)
ax.set_ylabel('Pressure [hPa]', fontsize = 16)
ax.set_xlabel('RH [%]', fontsize = 16)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.tick_params(labelsize = 16)

# Altitude coordinates
fig, ax = plt.subplots(figsize = (5,12))
ax.plot(RH_RA * 100, z_RA, color = 'red', zorder = 5)
ax.plot(RH_RS, z_RS, color = 'black')
ax.set_title(Time_RA_1, fontsize = 16)
ax.set_ylim(0,10000)
ax.set_ylabel('Altitude [m]', fontsize = 16)
ax.set_xlabel('RH [%]', fontsize = 16)
ax.tick_params(labelsize = 16)

### Skew t log p diagram ###
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig)

# original RS data
skew.plot(p_RS_original, T_RS_original, color = 'red', linewidth = 2, label = 'RS T')
skew.plot(p_RS_original, T_d_RS_original, color = 'red', linewidth=2, label = 'RS Td')

# smoothed RS data
skew.plot(p_RS_original, T_RS_smoothed, color = 'pink', linewidth = 2, label = 'RS T smoothed')
skew.plot(p_RS_original, T_d_RS_smoothed, color = 'pink', linewidth=2, label = 'RS Td smoothed')

# lidar data
skew.plot(p_1, temp_degC, color = 'black', linewidth = 2, zorder=0 ,label = 'Lidar T')
skew.plot(p_1, temp_d_degC, color = 'black', linewidth = 2, zorder = 0, label = 'Lidar Td')

# satellite datad
## temperature
skew.plot(p_NUCAPS, T_NUCAPS,color = 'navy', linewidth=2, label = 'NUCAPS T')
skew.plot(p_NUCAPS, T_d_NUCAPS, color = 'navy', linewidth=2, label = 'NUCAPS Td')

## MIT (only microwave retrieval) temperature
skew.plot(p_NUCAPS, MIT_T_NUCAPS, color = 'steelblue', linewidth=2, label = 'NUCAPS MIT T')
skew.plot(p_NUCAPS, MIT_T_d_NUCAPS, color = 'steelblue', linewidth=2, label = 'NUCAPS MIT Td')

## FG (first guess) temperature
skew.plot(p_NUCAPS, FG_T_NUCAPS, color = 'lightskyblue', linewidth=2, label = 'NUCAPS FG T')
skew.plot(p_NUCAPS, FG_T_d_NUCAPS, color = 'lightskyblue', linewidth=2, label = 'NUCAPS FG Td')

# surface measurement
skew.plot(pressure_hPa, temperature_degC, 'ro', color = 'orange', label = 'surf T')
skew.plot(pressure_hPa, dewpoint_degC, 'ro', color = 'orange', label = 'surf Td')

plt.ylabel('Pressure [hPa]', fontsize = 14)
plt.xlabel('Temperature [°C]', fontsize = 14)
skew.ax.tick_params(labelsize = 14)
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-60, 60)

# textbox
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
textstr = '\n'.join((
    r'Date: %s'% (dt.datetime.strptime(Year+Month+Day+Hour,"%Y%m%d%H"),),
    r'Distance RS - NUCAPS (in km): %s' % (min_dist, ),
    r'Time difference NUCAPS: %s' % (time_dif, )))
    
skew.ax.text(0.02, 0.1, textstr, transform=skew.ax.transAxes,
             fontsize=12, verticalalignment='top', bbox=props)

plt.legend()
#Figure_name = Year + Month + Day + Hour + '_' + str(file_index) + '.png'
#plt.savefig('/data/COALITION2/PicturesSatellite/results_NAL/Plots/' + Figure_name)
plt.show()