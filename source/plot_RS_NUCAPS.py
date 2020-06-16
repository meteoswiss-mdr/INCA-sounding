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
import sys

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def convert_H2O_MR_to_Td(H2O_MR, p):
    # input water vapour mass mixing ration in (mWV / mDA) kg/kg

    WVMR = H2O_MR * 1000 # convert to grams
    WVMR = WVMR * units('g/kg')
    e_1 = mpcalc.vapor_pressure(p_NUCAPS_orig, WVMR)
    T_d = mpcalc.dewpoint(e_1) 
    return T_d

def extract_temperature_profile(T, index_CP, index_min_pressure, var_name='Temperature'):

    df_Temp = pd.DataFrame(T)
    df_Temp = df_Temp.loc[[index_CP],:].T
    df_Temp.columns = [var_name]      
    df_Temp = df_Temp[0:int(index_min_pressure)]
    T_profile = (df_Temp[var_name].values * units.kelvin).to(units.degC)

    return T_profile

def extract_dewpoit_temperature_profile(T_d, index_CP, index_min_pressure, var_name='Temperature_D'):

    df_Temp_D = pd.DataFrame(T_d)
    df_Temp_D = df_Temp_D.loc[[index_CP],:].T
    df_Temp_D.columns = [var_name]
    df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
    T_d_profile = (df_Temp_D[var_name].values)

    return T_d_profile


######################################## Load data and define numpy arrays ######################################## 

SURF_archive = '/data/COALITION2/PicturesSatellite/results_NAL/'
RS_archive   = '/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/Payerne/'
NUCAPS_list_dir = '/data/COALITION2/PicturesSatellite/results_NAL/NUCAPS/'
LIDAR_archive = '/data/COALITION2/PicturesSatellite/results_NAL/RALMO/Payerne/'
OUTPUT_dir    = '/data/COALITION2/PicturesSatellite/results_NAL/Plots/'

######################################## Load data and define numpy arrays ######################################## 
if len(sys.argv) == 1:
    #use default date
    RS_time=dt.datetime(2020,4,27,0,0,0)
elif len(sys.argv) == 6:
    year   = int(sys.argv[1])
    month  = int(sys.argv[2])
    day    = int(sys.argv[3])
    hour   = int(sys.argv[4])
    minute = int(sys.argv[5])
    second = 0
    RS_time = dt.datetime(year,month,day,hour,minute,0)
else:
    print("*** ERROR, unknown number of command line arguemnts")
    quit()


##### Surface Measurement ##### -> to be downloaded
data_surf = pd.read_csv(RS_time.strftime(SURF_archive+'/surf_stat_%Y%m%d%H%M00.txt'))
lowest_pres_SMN = data_surf['90'][0]

pressure_hPa = data_surf['90'].values * units.hPa
temperature_degC = data_surf['91'].values * units.degC
relative_humidity_percent = data_surf['98'].values * units('percent')
dewpoint_degC = cc.dewpoint_from_relative_humidity(temperature_degC, relative_humidity_percent)

###### Radiosondes ######
### read Radiosonde file and save as dataframe ###
RS_data = xr.open_dataset(RS_archive+'/RS_concat.nc').to_dataframe()

### choose desired time ###
RS_data = RS_data[RS_data.time_YMDHMS == int(RS_time.strftime('%Y%m%d%H%M%S'))]
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
filtered_files = pd.read_csv(os.path.join(NUCAPS_list_dir, RS_time.strftime('%Y%m%d.txt')))

min_dist   = min(filtered_files.min_dist)
#min_dist   = 25
i_min_dist = np.argmin(filtered_files.min_dist)
#i_min_dist = np.where(filtered_files.min_dist == min_dist)[0][0]
NUCAPS_file = filtered_files.file[i_min_dist]
index_CP   = filtered_files['index'][i_min_dist]

global_scene = Scene(reader="nucaps", filenames=[NUCAPS_file])
NUCAPS_time = dt.datetime.strptime(NUCAPS_file[65:80],"%Y%m%d%H%M%S%f")
time_dif = np.abs(RS_time - NUCAPS_time)


# read moisture variables H2O_MR
global_scene.load(["H2O_MR"], pressure_levels=True)

# Pressure 
p = global_scene["H2O_MR"].coords['Pressure'].values
df_pres = pd.DataFrame(p)
df_pres = df_pres.loc[[index_CP],:].T
df_pres.columns = ['Pressure']
df_pres_orig = df_pres
index_min_pressure = np.where(df_pres > lowest_pres_RS)[0][0]
df_pres = df_pres[0:int(index_min_pressure)]
p_NUCAPS = (df_pres['Pressure'].values * units.hPa)
p_NUCAPS_orig = (df_pres_orig['Pressure'].values * units.hPa)

# convert H2O MR to pressure 
T_d = convert_H2O_MR_to_Td(global_scene["H2O_MR"].values, p_NUCAPS_orig)
T_d_NUCAPS   = extract_dewpoit_temperature_profile(T_d,           index_CP, index_min_pressure, var_name='Temperature_D')
#T_d_NUCAPS_1 = extract_dewpoit_temperature_profile(T_d.magnitude, index_CP, index_min_pressure, var_name='Temperature_D')

### temperature
# read NUCAPS temperature 
global_scene.load(["Temperature"], pressure_levels=True)
T_NUCAPS = extract_temperature_profile(global_scene["Temperature"].values, index_CP, index_min_pressure, var_name="Temperature")

## read MIT Temperature
global_scene.load(["MIT_Temperature"], pressure_levels=True)
MIT_T_NUCAPS = extract_temperature_profile(global_scene["MIT_Temperature"].values, index_CP, index_min_pressure, var_name="MIT_Temperature")

## read FG Temperature
global_scene.load(["FG_Temperature"], pressure_levels=True)
T_FG_NUCAPS = extract_temperature_profile(global_scene["FG_Temperature"].values, index_CP, index_min_pressure, var_name="FG_Temperature")

### water vapour mass mixing ration and dewpoint temperature 
## MIT H2O MR
global_scene.load(["MIT_H2O_MR"], pressure_levels=True)
T_d_MIT = convert_H2O_MR_to_Td(global_scene["MIT_H2O_MR"].values, p_NUCAPS_orig)
T_d_MIT_NUCAPS = extract_dewpoit_temperature_profile(T_d_MIT, index_CP, index_min_pressure, var_name='MIT_Temperature_D')

# FG H2O MR
global_scene.load(["FG_H2O_MR"], pressure_levels=True)
T_d_FG = convert_H2O_MR_to_Td(global_scene["FG_H2O_MR"].values, p_NUCAPS_orig)
T_d_FG_NUCAPS = extract_dewpoit_temperature_profile(T_d_FG, index_CP, index_min_pressure, var_name='FG_Temperature_D')
#T_d_FG_NUCAPS_1 = extract_dewpoit_temperature_profile(T_d_FG.magnitude, index_CP, index_min_pressure, var_name='FG_Temperature_D')


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
# what does "06610" stand for? 
RA_data = xr.open_dataset(LIDAR_archive+'/RA_06610_concat.nc').to_dataframe()

Time_RA = RS_time + dt.timedelta(minutes=30)
RA_data = RA_data[RA_data.time_YMDHMS == int(dt.datetime.strftime(Time_RA,"%Y%m%d%H%M%S"))]

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
ax.set_title(Time_RA, fontsize = 16)
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
ax.set_title(Time_RA, fontsize = 16)
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
skew.plot(p_RS_original, T_d_RS_smoothed, color = 'pink', linewidth=2, linestyle='dashed', label = 'RS Td smoothed')

# lidar data
skew.plot(p_1, temp_degC, color = 'black', linewidth = 2, zorder=0 ,label = 'Lidar T')
skew.plot(p_1, temp_d_degC, color = 'black', linewidth = 2, linestyle='dashed', zorder = 0, label = 'Lidar Td')

# satellite data
## temperature
skew.plot(p_NUCAPS, T_NUCAPS,color = 'navy', linewidth=2, label = 'NUCAPS T')
skew.plot(p_NUCAPS, T_d_NUCAPS, color = 'navy', linewidth=2, linestyle='dashed', label = 'NUCAPS Td')

## MIT (only microwave retrieval) temperature
skew.plot(p_NUCAPS, MIT_T_NUCAPS, color = 'steelblue', linewidth=1.5, label = 'NUCAPS MIT T')
skew.plot(p_NUCAPS, T_d_MIT_NUCAPS, color = 'steelblue', linewidth=1.5, linestyle='dashed', label = 'NUCAPS MIT Td')

# FG (first guess) temperature
skew.plot(p_NUCAPS, T_FG_NUCAPS, color = 'lightskyblue', linewidth=1, label = 'NUCAPS FG T')
skew.plot(p_NUCAPS, T_d_FG_NUCAPS, color = 'lightskyblue', linewidth=1, linestyle='dashed', label = 'NUCAPS FG Td')

# surface measurement
skew.plot(pressure_hPa, temperature_degC, 'ro', color = 'orange', label = 'surf T')
skew.plot(pressure_hPa, dewpoint_degC, 'bo', color = 'orange', label = 'surf Td')

plt.ylabel('Pressure [hPa]', fontsize = 14)
plt.xlabel('Temperature [°C]', fontsize = 14)
skew.ax.tick_params(labelsize = 14)
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-60, 60)

# textbox
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
textstr = '\n'.join((
    r'Date: %s'% (RS_time.strftime("%Y-%m-%d %H:%M"), ),
    r'Distance RS - NUCAPS (in km): %s' % (min_dist, ),
    r'Time difference NUCAPS: %s' % (time_dif, )))
    
skew.ax.text(0.02, 0.1, textstr, transform=skew.ax.transAxes,
             fontsize=12, verticalalignment='top', bbox=props)

plt.legend()
#output_filename = RS_time.strftime("NUCAPS_skewT_%y%m%d%H%M.png") 
#plt.savefig(OUTPUT_dir+'/'+ output_filename)
plt.show()
