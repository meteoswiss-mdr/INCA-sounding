﻿"""
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
import metpy

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

def mr2rh(T,r,p):
    
    r = r * 10**-3
    f = 1.0016 + 3.15* 10**-6 * p - 0.074/p
    ew = 6.112*np.exp(17.62*T/(243.12 + T))
    eww = f * ew
    ei = 6.112*np.exp(22.46*T/(272.62 + T))
    eii = f * ei
    e = r/(0.62198+r)*p
    rhw = e/eww*100
    rhi = e/eii*100
    
    return rhw, rhi

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

######################################## define paths ######################################## 

SMN_archive = '/data/COALITION2/PicturesSatellite/results_NAL/SwissMetNet/Payerne'
RS_archive   = '/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/Payerne/'
NUCAPS_list_dir = '/data/COALITION2/PicturesSatellite/results_NAL/NUCAPS/txtfiles'
LIDAR_archive = '/data/COALITION2/PicturesSatellite/results_NAL/RALMO/Payerne/'
OUTPUT_dir    = '/data/COALITION2/PicturesSatellite/results_NAL/Plots/'
RM_archive = '/data/COALITION2/PicturesSatellite/results_NAL/Radiometer/'
INCA_archive = '/data/COALITION2/PicturesSatellite/results_NAL'

COSMO_archive = '/data/COALITION2/database/cosmo/T-TD_3D/'
COSMO_coordinate = '/data/COALITION2/PicturesSatellite/results_NAL/COSMO/'
INCA_grid = pd.read_csv(INCA_archive+'/INCA_grid.csv') 

######################################## Load data and define numpy arrays ######################################## 
if len(sys.argv) == 1:
    #use default date
    RS_time=dt.datetime(2020,4,27,0,0,0)
    SMN_time = dt.datetime(2020,4,27,0,0,0)
    COSMO_time = dt.datetime(2020,7,22,0,0,0)
elif len(sys.argv) == 6:
    year   = int(sys.argv[1])
    month  = int(sys.argv[2])
    day    = int(sys.argv[3])
    hour   = int(sys.argv[4])
    minute = int(sys.argv[5])
    second = 0
    RS_time = dt.datetime(year,month,day,hour,minute,0)
    SMN_time = dt.datetime(year,month,day,hour,minute,0)
else:
    print("*** ERROR, unknown number of command line arguemnts")
    quit()

#########################################
##### A) SMN: Surface measurement
#########################################
SMN_data = xr.open_dataset(SMN_archive+'/SMN_concat1.nc').to_dataframe()

SMN_data = SMN_data[SMN_data.time_YMDHMS == int(SMN_time.strftime('%Y%m%d%H%M%S'))]
SMN_data = SMN_data[SMN_data.pressure_hPa != 1000] # delete first row (undefined values)
SMN_data = SMN_data.reset_index(drop=True)

pressure_SMN = SMN_data.pressure_hPa.values * units.hPa
temperature_SMN = SMN_data.temperature_degC.values * units.degC
RH_SMN = SMN_data.relative_humidity_percent.values * units.percent
temperature_d_SMN = cc.dewpoint_from_relative_humidity(temperature_SMN, RH_SMN)
specific_humidity_SMN = cc.specific_humidity_from_dewpoint(temperature_d_SMN, pressure_SMN)

##########################################
##### B) RS: RADIOSONDE 
##########################################
### read Radiosonde file and save as dataframe ###
RS_data = xr.open_dataset(RS_archive+'/RS_concat.nc').to_dataframe()
#RS_data = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/RS_20200722000000')
RS_data = RS_data.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent', '742':'geopotential_altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })

### choose desired time ###
RS_data = RS_data[RS_data.time_YMDHMS == int(RS_time.strftime('%Y%m%d%H%M%S'))]
RS_data = RS_data[RS_data.pressure_hPa != 1000] # delete first row (undefined values)
RS_data = RS_data.reset_index(drop=True)

lowest_pres_RS = RS_data.pressure_hPa.iloc[0]

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

 

lon_Payerne = 6.93608
lat_Payerne = 46.82201

INCA_grid = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/COSMO//inca_topo_levels_hsurf_ccs4.nc') 
lon_1 = INCA_grid.lon_1.values.flatten()
lat_1 = INCA_grid.lat_1.values.flatten()
INCA_grid_1 = pd.DataFrame({'lat' : lat_1, 'lon':lon_1})
tree = spatial.KDTree(INCA_grid_1.values)
coordinates = tree.query([([lat_Payerne, lon_Payerne])])
coordinates = coordinates[1]
Coordinates_1 = INCA_grid_1.loc[coordinates[0]]
INCA_grid_all = INCA_grid.where((INCA_grid.lon_1 == Coordinates_1.lon) & (INCA_grid.lat_1 == Coordinates_1.lat), drop=True)
INCA_grid = pd.DataFrame({'altitude_m' : INCA_grid_all.HFL.values.flatten()})
INCA_grid_boundaries = pd.DataFrame({'altitude_m' : INCA_grid_all.HHL.values.flatten()})
    
T_RS = T_RS.iloc[::-1]
T_d_RS = T_d_RS[::-1]
for i in range(0,len(INCA_grid)):
    print(i)
    if (i == 0):
        window_h_max = INCA_grid.iloc[i] + (INCA_grid.iloc[i] - INCA_grid.iloc[i+1]) / 2
        window_h_min = INCA_grid.iloc[i] - (INCA_grid.iloc[i] - INCA_grid.iloc[i+1]) / 2
    elif (i==49):
        window_h_min = INCA_grid.iloc[i] - (INCA_grid.iloc[(i-1)] - INCA_grid.iloc[i])  / 2
        window_h_max = INCA_grid.iloc[i] + (INCA_grid.iloc[(i-1)] - INCA_grid.iloc[i])  / 2
    else: 
        window_h_min = INCA_grid.iloc[i] - (INCA_grid.iloc[(i-1)] - INCA_grid.iloc[i])  / 2
        window_h_max = INCA_grid.iloc[i] + (INCA_grid.iloc[i] - INCA_grid.iloc[i+1]) / 2
    print('min' + str(window_h_min.values))
    print('max' + str(window_h_max.values))
    
    index_list = np.where(np.logical_and(z_RS>=float(window_h_min.values), z_RS<=float(window_h_max.values)))
    T_index = T_RS[(z_RS < float(window_h_max.values)) & (z_RS > float(window_h_min.values))]
    T_d_index = T_d_RS[(z_RS < float(window_h_max.values)) & (z_RS > float(window_h_min.values))]
    #p_index = z_RS[(z_RS > window_h_max) & (z_RS < window_h_min)]
   
    mean_T_RS = np.mean(T_index)
    T_RS_list.append(mean_T_RS)
        
    mean_T_d_RS = np.mean(T_d_index)
    T_d_RS_list.append(mean_T_d_RS)
 

T_RS_smoothed = pd.DataFrame(T_RS_list)

T_d_RS_smoothed = pd.DataFrame(T_d_RS_list) 

T_d_RS_smoothed = pd.DataFrame(T_d_RS_list) 




INCA_grid = metpy.calc.height_to_pressure_std(INCA_grid.values * units.meters)
##########################################
##### C) NUCAPS: Satellite data 
##########################################
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

##########################################
##### D) RALMO: Raman lidar 
###########################################
RA_data = xr.open_dataset(LIDAR_archive+'/RA_concat_wp').to_dataframe()
Time_RA = RS_time + dt.timedelta(minutes=30)
RA_data = RA_data[RA_data.time_YMDHMS == int(dt.datetime.strftime(Time_RA,"%Y%m%d%H%M%S"))]

RA_data = RA_data[RA_data['temperature_K']!= 1e+07]
RA_data = RA_data[['time_YMDHMS', 'altitude_m', 'specific_humidity_gkg-1','temperature_K', 'pressure_hPa']]

# variables
g = 9.81
m_mol_air = 28.965*10.0**(-3.0)
R = 8.31446
z_RA = RA_data['altitude_m']

temp = RA_data['temperature_K']
T_RA = temp.values
temp_degC = temp - 273.15

p_1 = RA_data.pressure_hPa
specific_humidity_RA = RA_data['specific_humidity_gkg-1']

# relative humidity, differentiate between water and ice
rhw, rhi = mr2rh(temp_degC, specific_humidity_RA, p_1)
RH_RA = np.zeros(len(rhw))
ind_tresh = RA_data.index[RA_data.altitude_m > 5000][0]
RH_RA[0:ind_tresh] = rhw[0:ind_tresh]
RH_RA[ind_tresh:-1] = rhi[ind_tresh:-1]
RH_RA = RH_RA[0:len(RH_RA)]

p_1 = p_1.values * units.hPa
temp_degC = temp_degC.values * units.degC

# calculate dew point temperature 
specific_humidity_RA = specific_humidity_RA.values * units('g/kg')
temp_d_degC = cc.dewpoint_from_specific_humidity(specific_humidity_RA, temp_degC, p_1)
RA_data.dew_point_degC = temp_d_degC
RA_data.temperature_degC = RA_data.temperature_K - 273.15

########################################## 
## E) RM: Radiometer 
##########################################
RM_data = xr.open_dataset(RM_archive+'/radiometer_06610_concat_filtered_0.nc').to_dataframe()
RM_data = RM_data[RM_data.time_YMDHMS_1.dt.month == RS_time.month]
RM_data = RM_data[RM_data.time_YMDHMS_1.dt.day == RS_time.day]
date = nearest(RM_data.time_YMDHMS_1, RS_time)
RM_data = RM_data[RM_data.time_YMDHMS_1 == date]

RM_data["time_YMDHMS"] = RM_data.time_YMDHMS_1.dt.round('30min').values
#RM_data['pressure_hPa'] = '0' 
RM_data = RM_data.reset_index(drop=True)

## add temperature in degC
temp_K = RM_data.temperature_K - 273.15
RM_data.insert(value=temp_K,column = "temperature_degC", loc=7)

## add altitude_m
RM_data.altitude_m = RM_data.altitude_m + 492

## add dewpoint temperature
dewpoint_degC = cc.dewpoint_from_relative_humidity(RM_data.temperature_degC.values * units.degC, RM_data.relative_humidity_percent.values * units.percent) 
RM_data.insert(value=dewpoint_degC,column = "dew_point_degC", loc=6)

# convert altitude into pressure coordinates
g = 9.81
m_mol_air = 28.965*10.0**(-3.0)
R = 8.31446

z_RM = RM_data['altitude_m']
T_RM = RM_data['temperature_K']

# calculate Lidar pressure levels with RS as lowest level 
p_1 = np.zeros(len(z_RM))
p_1[0] = SMN_data.pressure_hPa[0]
integrant = g*m_mol_air/(R*T_RM)

   
for i in range(1, len(z_RM)):
    p_1[i] = RS_data.pressure_hPa[0]*math.exp(-np.trapz(integrant[0:i], z_RM[0:i]))

#RM_data.pressure_hPa = p_1[:]

RM_data.pressure_hPa = metpy.calc.height_to_pressure_std(RM_data.altitude_m.values * units.meters)

########################################## 
## F) COSMO: Model data 
##########################################
#COSMO_coordinate = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/COSMO//COSMO_coordindate_payerne.csv')
#COSMO_data = xr.open_dataset(COSMO_archive+'cosmo1_inca_2020072006_01.nc').to_dataframe()
#COSMO_data =    COSMO_data[COSMO_data.lat_1 == float(COSMO_coordinate.lat_1.values)] 
#COSMO_data =    COSMO_data[COSMO_data.lon_1 == float(COSMO_coordinate.lon_1.values)] 
#COSMO_data = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/COSMO/cosmo1_inca_merge_cut.nc').to_dataframe()
#COSMO_data = COSMO_data.reset_index().rename(columns = {'time' : 'time_YMDHMS', 'p_inca': 'pressure_hPa', 't_inca' : 'temperature_degC'})
#COSMO_data = COSMO_data[COSMO_data.time_YMDHMS == COSMO_time]
# remove duplicate pressures
#COSMO_data = COSMO_data.loc[~COSMO_data.pressure_hPa.duplicated(keep='first')]
#COSMO_data.pressure_hPa  = COSMO_data.pressure_hPa / 100
#COSMO_data.temperature_degC  = COSMO_data.temperature_degC - 273.15

# add altitude 
## read INCA grid
#INCA_altitudpd.read_csv('/data/COALITION2/database/NUCAPS/inca_topo_levels_hsurf_ccs4.nc')
## extract altitudes at Payerne
#COSMO_data.insert(value=INCA_grid.HHL.values, column = "altitude_m", loc = 0)


# add dew point temperature 
#pressure_COSMO = COSMO_data.pressure_hPa.values * units.hPa
#temp_COSMO = (COSMO_data.temperature_degC.values -273.15) * units.degC
#specific_humidity_COSMO = COSMO_data.qv_inca.values * units('g/kg')
#temp_d_COSMO = cc.dewpoint_from_specific_humidity(specific_humidity_COSMO, temp_COSMO, pressure_COSMO)
#COSMO_data.insert(value=temp_d_COSMO, column = "dew_point_degC", loc = 36)


#plt.plot(RM_data.temperature_degC, RM_data.altitude_m, color = 'black', linewidth = 2, zorder=0 ,label = 'Lidar T')
#plt.plot(RM_data.dew_point_degC,RM_data.altitude_m, color = 'black', linewidth = 2, linestyle='dashed', zorder = 0, label = 'Lidar Td')

#plt.plot(RM_data.temperature_degC, RM_data.pressure_hPa, color = 'black', linewidth = 2, zorder=0 ,label = 'Lidar T')
#plt.plot(RM_data.dew_point_degC,RM_data.pressure_hPa, color = 'black', linewidth = 2, linestyle='dashed', zorder = 0, label = 'Lidar Td')
#plt.gca().invert_yaxis()
####################################### Plot data and save figure ######################################## 
### compare Lidar and RS data 
# Pressure coordinates
#fig, ax = plt.subplots(figsize = (5,12))
#ax.plot(RH_RA[2:len(RH_RA)-1], p_1[2:len(p_1)-1], color = 'red')
#ax.plot(RH_RS, p_RS_original, color = 'black')
#ax.set_title(Time_RA, fontsize = 16)
#ax.set_ylim(1050,300)
#ax.set_ylabel('Pressure [hPa]', fontsize = 16)
#ax.set_xlabel('RH [%]', fontsize = 16)
#ax.set_yscale('log')
#ax.yaxis.set_major_formatter(ScalarFormatter())
#ax.yaxis.set_major_locator(MultipleLocator(100))
#ax.yaxis.set_minor_formatter(NullFormatter())
#ax.tick_params(labelsize = 16)

# Altitude coordinates
#fig, ax = plt.subplots(figsize = (5,12))
#ax.plot(RH_RA[1:len(RH_RA)-1], z_RA[1:len(z_RA)-1], color = 'red', zorder = 5)
#ax.plot(RH_RS, z_RS, color = 'black')
#ax.set_title(Time_RA, fontsize = 16)
#ax.set_ylim(0,10000)
#ax.set_ylabel('Altitude [m]', fontsize = 16)
#ax.set_xlabel('RH [%]', fontsize = 16)
#ax.tick_params(labelsize = 16)

fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig)

#########################################
##### A) SMN: Surface measurement
#########################################
skew.plot(pressure_SMN, temperature_SMN, 'ro', color = 'orange', label = 'surf T')
skew.plot(pressure_SMN, temperature_d_SMN, 'bo', color = 'orange', label = 'surf Td')

##########################################
##### B) RS: RADIOSONDE 
##########################################
# original RS data
skew.plot(p_RS_original, T_RS_original, color = 'red', linewidth = 2, label = 'RS T')
skew.plot(p_RS_original, T_d_RS_original, color = 'red', linewidth=2, linestyle = 'dashed', label = 'RS Td')

# smoothed RS data
skew.plot(INCA_grid, T_RS_smoothed, color = 'pink', linewidth = 2, label = 'RS T smoothed')
skew.plot(INCA_grid, T_d_RS_smoothed, color = 'pink', linewidth=2, linestyle='dashed', label = 'RS Td smoothed')

##########################################
##### C) NUCAPS: Satellite data 
##########################################
## temperature
skew.plot(p_NUCAPS, T_NUCAPS,color = 'navy', linewidth=2, label = 'NUCAPS T')
skew.plot(p_NUCAPS, T_d_NUCAPS, color = 'navy', linewidth=2, linestyle='dashed', label = 'NUCAPS Td')

## MIT (only microwave retrieval) temperature
skew.plot(p_NUCAPS, MIT_T_NUCAPS, color = 'steelblue', linewidth=1.5, label = 'NUCAPS MIT T')
skew.plot(p_NUCAPS, T_d_MIT_NUCAPS, color = 'steelblue', linewidth=1.5, linestyle='dashed', label = 'NUCAPS MIT Td')

# FG (first guess) temperature
skew.plot(p_NUCAPS, T_FG_NUCAPS, color = 'lightskyblue', linewidth=1, label = 'NUCAPS FG T')
skew.plot(p_NUCAPS, T_d_FG_NUCAPS, color = 'lightskyblue', linewidth=1, linestyle='dashed', label = 'NUCAPS FG Td')

##########################################
##### D) RALMO: Raman lidar 
###########################################
skew.plot(RA_data.pressure_hPa, RA_data.temperature_degC, color = 'black', linewidth = 2, zorder=0 ,label = 'Lidar T')
skew.plot(RA_data.pressure_hPa, RA_data.dew_point_degC, color = 'black', linewidth = 2, linestyle='dashed', zorder = 0, label = 'Lidar Td')

########################################## 
##### E) RM: Radiometer 
##########################################
skew.plot(RM_data.pressure_hPa, RM_data.temperature_degC, color = 'green', linewidth = 2, zorder=0 ,label = 'Radiometer T')
skew.plot(RM_data.pressure_hPa, RM_data.dew_point_degC, color = 'green', linewidth = 2, linestyle='dashed', zorder = 0, label = 'Radiometer Td')

########################################## 
## F) COSMO: Model data 
##########################################
#skew.plot(COSMO_data.pressure_hPa, COSMO_data.temperature_degC, color = 'orangered', linewidth = 2, zorder=0 ,label = 'COSMO T')
#skew.plot(COSMO_data.pressure_hPa, COSMO_data.dew_point_degC, color = 'orangered', linewidth = 2, linestyle='dashed', zorder = 0, label = 'COSMO Td')


plt.ylabel('Pressure [hPa]', fontsize = 14)
plt.xlabel('Temperature [°C]', fontsize = 14)
skew.ax.tick_params(labelsize = 14)
#skew.ax.set_ylim(1000, 100)
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
