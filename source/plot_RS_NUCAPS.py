#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np

from satpy import Scene, find_files_and_readers
import numpy as np
from datetime import datetime
from metpy import calc as cc
import pandas as pd
import metpy.calc as mpcalc
from metpy.units import units
from metpy.interpolate import interpolate_to_grid
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import Hodograph, SkewT
from metpy.units import units
import datetime as dt

from datetime import datetime
import math 

import os

from scipy import spatial
import numpy as np
import geopy.distance
import scipy.ndimage as ndimage
import xarray as xr
import time

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

dynfmt = "%Y%m%d%H%M%S"
year="%Y"
month="%m"
day="%d"
seconds="%S"

######################################## Load data and define numpy arrays ######################################## 
Year = "2020"
Month = "04"
Day = "27"
Hour = "00" # only 00 and 12
Minute="00" # only 00 and 30
Seconds='00'


##### Surface Measurement #####
#data_surf = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/surf_stat_20190427000000.txt')
#lowest_pres_SMN = data_surf['90'][0]


#Luftdruck_Stationshoehe = data_surf['90']
#Luftdruck_Stationshoehe = Luftdruck_Stationshoehe.values * units.hPa
#Lufttemperatur_2muBoden = data_surf['91']
#Lufttemperatur_2muBoden = Lufttemperatur_2muBoden.values * units.degC
#RelativeHumidity = data_surf['98']
#RelativeHumidity = RelativeHumidity.values * units('percent')
#Lufttemperatur_d_2muBoden = cc.dewpoint_from_relative_humidity(Lufttemperatur_2muBoden, RelativeHumidity)

###### Radiosondes ######
### read Radiosonde file and save as dataframe ###
xr_array = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/Payerne/RS_concat.nc')
df = pd.DataFrame({'746': xr_array['746'], '744' : xr_array['744'], '745' : xr_array['745'], '747' : xr_array['747'], '742' : xr_array['742'], 'Time' : xr_array['termin']})

### choose desired time ###
data_comma = df[df.Time == int(Year+Month+Day+Hour+Minute+Seconds)]
#df = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/Milano/RS_16080_'+Year+Month+Day+Hour+'.txt')
data_comma = data_comma[data_comma['744'] != 1000]
data_comma = data_comma.reset_index(drop=True)

nearest_value = find_nearest(data_comma['742'],600)
lowest_pres_RS = float(data_comma['744'][data_comma['742']==nearest_value])

### define variables ###
RH_RS = data_comma['746']
RH_RS = RH_RS.to_frame()
p_RS = data_comma['744']
p_RS_original = data_comma['744']
p_RS_original = p_RS_original.to_frame()
T_RS = data_comma['745']
T_RS_original = data_comma['745']
T_RS_p = data_comma['745'] + 273.15
T_RS_original = T_RS_original.to_frame()
T_d_RS = data_comma['747']
T_d_RS_original = data_comma['747']
T_d_RS_original = T_d_RS_original.to_frame()
z_RS = data_comma['742']

delta_hPa = 100
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
    
    
T_RS = pd.DataFrame(T_RS_list)

T_d_RS = pd.DataFrame(T_d_RS_list) 


##############################  NUCAPS ##############################
### read file ###
file_index = 0
date_NUCAPS = Year + Month + Day
date_NUCAPS = date_NUCAPS +'.txt'
filtered_files = pd.read_csv(os.path.join('/data/COALITION2/PicturesSatellite/results_NAL/NUCAPS', date_NUCAPS))

index_CP = filtered_files.iloc[file_index,3]

min_dist = filtered_files.iloc[file_index,2]

filenames = [filtered_files.iloc[file_index,1]]
global_scene = Scene(reader="nucaps", filenames=filenames)    

NUCAPS_time = filenames = filtered_files.iloc[file_index,1]
NUCAPS_time = NUCAPS_time[65:80]
NUCAPS_time = datetime.strptime(NUCAPS_time,"%Y%m%d%H%M%S%f")

time_dif = np.abs(datetime.strptime(Year+Month+Day+Hour+Minute+Seconds,dynfmt) - NUCAPS_time)

var_pres="H2O_MR"
global_scene.load([var_pres], pressure_levels=True)

var_temp = "Temperature"
global_scene.load([var_temp], pressure_levels=True)

### define variables ###
# PRESSURE 
p = global_scene[var_pres].coords['Pressure'].values
df_pres = pd.DataFrame(p)
df_pres = df_pres.loc[[index_CP],:].T
df_pres.columns = ['Pressure']
df_pres_orig = df_pres
index_min_pressure = np.where(df_pres > lowest_pres_RS)[0][0]
df_pres = df_pres[0:int(index_min_pressure)]
p_NUCAPS = (df_pres['Pressure'].values * units.hPa)
p_NUCAPS_orig = (df_pres_orig['Pressure'].values * units.hPa)

# TEMPERATURE
# Temperature
var_temp = "Temperature"
global_scene.load([var_temp], pressure_levels=True)

T = global_scene[var_temp].values
df_Temp = pd.DataFrame(T)
df_Temp = df_Temp.loc[[index_CP],:].T
df_Temp.columns = ['Temperature']
df_Temp = df_Temp[0:int(index_min_pressure)]
T_NUCAPS_1 = df_Temp['Temperature'].values - 273.15
T_NUCAPS = (df_Temp['Temperature'].values * units.kelvin).to(units.degC)

# MIT Temperature
var_temp_MIT = "MIT_Temperature"
global_scene.load([var_temp_MIT], pressure_levels=True)

T = global_scene[var_temp_MIT].values
df_Temp = pd.DataFrame(T)
df_Temp = df_Temp.loc[[index_CP],:].T
df_Temp.columns = ['MIT_Temperature']
df_Temp = df_Temp[0:int(index_min_pressure)]
MIT_T_NUCAPS_1 = df_Temp['MIT_Temperature'].values -273.15
MIT_T_NUCAPS = (df_Temp['MIT_Temperature'].values * units.kelvin).to(units.degC)

# FG Temperature
var_temp_FG = "FG_Temperature"
global_scene.load([var_temp_FG], pressure_levels=True)

T = global_scene[var_temp_FG].values
df_Temp = pd.DataFrame(T)
df_Temp = df_Temp.loc[[index_CP],:].T
df_Temp.columns = ['FG_Temperature']
df_Temp = df_Temp[0:int(index_min_pressure)]
FG_T_NUCAPS_1 = df_Temp['FG_Temperature'].values -273.15
FG_T_NUCAPS = (df_Temp['FG_Temperature'].values * units.kelvin).to(units.degC)

### MOISTURE
# H2O MR
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
#T_d_NUCAPS_1 = df_Temp_D['Temperature_D'].values -273.15
T_d_NUCAPS = (df_Temp_D['Temperature_D'].values)

df_Temp_D = pd.DataFrame(T_d.magnitude)
df_Temp_D = df_Temp_D.loc[[index_CP],:].T
df_Temp_D.columns = ['Temperature_D']
df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
#T_d_NUCAPS_1 = df_Temp_D['Temperature_D'].values -273.15
T_d_NUCAPS_1 = (df_Temp_D['Temperature_D'].values)

# MIT H2O MR
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
#MIT_T_d_NUCAPS_1 = df_Temp_D['MIT_Temperature_D'].values -273.15
MIT_T_d_NUCAPS = (df_Temp_D['MIT_Temperature_D'].values)

df_Temp_D = pd.DataFrame(T_d.magnitude)
df_Temp_D = df_Temp_D.loc[[index_CP],:].T
df_Temp_D.columns = ['MIT_Temperature_D']
df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
#MIT_T_d_NUCAPS_1 = df_Temp_D['MIT_Temperature_D'].values -273.15
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
#FG_T_d_NUCAPS_1 = df_Temp_D['FG_Temperature_D'].values -273.15
FG_T_d_NUCAPS = (df_Temp_D['FG_Temperature_D'].values)

df_Temp_D = pd.DataFrame(T_d.magnitude)
df_Temp_D = df_Temp_D.loc[[index_CP],:].T
df_Temp_D.columns = ['FG_Temperature_D']
df_Temp_D = df_Temp_D[0:int(index_min_pressure)]
#FG_T_d_NUCAPS_1 = df_Temp_D['FG_Temperature_D'].values -273.15
FG_T_d_NUCAPS_1 = (df_Temp_D['FG_Temperature_D'].values)

### Interpolate RS grid to Satellite grid ###
p_RS_array = p_RS.values
df_pres_array = df_pres.values
import numpy as np
def find_nearest(array,value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

closest_p_list = []
closest_p_index = []
for i in range(0,len(df_pres_array)):
    print(i)
    nearest = find_nearest(p_RS_array, df_pres_array[i])
    nearest_index = np.where(p_RS_array == nearest)[0]
    nearest_index = int(nearest_index[0])
    closest_p_list.append(nearest)
    closest_p_index.append(nearest_index)

# Variante 2) Find   
### select values variables at pressure levels ###
# smoothed points
p_RS_1 = p_RS[closest_p_index] 
T_RS_1 = T_RS[0][closest_p_index] 
T_d_RS_1 = T_d_RS[0][closest_p_index] 

### add units ###
p_RS = p_RS * units.hPa
T_RS = T_RS * units.degC
T_d_RS = T_d_RS * units.degC
# Variante 2) Interpolate T and Td to pressure levels
p_RS = p_RS.values * units.hPa
T_RS = T_RS.values * units.degC
T_d_RS = T_d_RS.values * units.degC

p_RS_original = p_RS_original.values * units.hPa
T_RS_original = T_RS_original.values * units.degC
T_d_RS_original = T_d_RS_original.values * units.degC

from metpy.interpolate import interpolate_1d
T_RS = interpolate_1d(p_NUCAPS, p_RS, T_RS, axis=0)
T_d_RS = interpolate_1d(p_NUCAPS, p_RS, T_d_RS, axis = 0)
p_RS = p_NUCAPS

#p_RS_original = p_NUCAPS
#T_RS_original = interpolate_1d(p_NUCAPS, p_RS_original, T_RS_original, axis=0)
#T_d_RS_original = interpolate_1d(p_NUCAPS, p_RS_original, T_d_RS_original, axis = 0)

##############################  RALMO ##############################
xr_array = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/RALMO/Payerne/RA_06610_concat.nc')
df = pd.DataFrame({'Altitude' : xr_array['level'], 'Temperature' : xr_array['3147'], 'Specific_humidity' : xr_array['4919'], 'Time' : xr_array['termin']})

Time_RA=str(Year+Month+Day+Hour+Minute+Seconds)
Time_RA= dt.datetime.strptime(Time_RA,dynfmt) + dt.timedelta(minutes=30)
Time_RA=dt.datetime.strftime(Time_RA,dynfmt)

data_comma = df[df.Time == int(Time_RA)]
data_comma_temp = data_comma[data_comma['Temperature']!= 1e+07]


# variables
g = 9.81
m_mol_air = 28.965*10.0**(-3.0)
R = 8.31446

# height 
z_RA = data_comma_temp['Altitude']
# temperature 
temp = data_comma_temp['Temperature']
T_RA = temp.values
temp_degC = temp - 273.15
temp_degC = temp_degC.values * units.degC

# calculate RAMAN pressure levels with SMN as lowest level 
#p = np.zeros(len(z_RA))
#p[0] = lowest_pres_SMN
#integrant = g*m_mol_air/(R*T_RA)

#for i in range(1, len(z_RA)):
#    p[i] = lowest_pres_SMN*math.exp(-np.trapz(integrant[0:i], z_RA[0:i]))

#p = p * units.hPa

# calculate RAMAN pressure levels with RS as lowest level 
p_1 = np.zeros(len(z_RA))
p_1[0] = lowest_pres_RS
integrant = g*m_mol_air/(R*T_RA)

for i in range(1, len(z_RA)):
    p_1[i] = lowest_pres_RS*math.exp(-np.trapz(integrant[0:i], z_RA[0:i]))

p_1 = p_1 * units.hPa

# calculate pressure levels RS
#lowest_pres_RS = float(data_comma['744'][1])
#p_calc_pres = np.zeros(len(z_RS))
#p_calc_pres[0] = lowest_pres_RS
#integrant = g*m_mol_air/(R*T_RS_p)

#for i in range(1, len(z_RS)):
#    p_calc_pres[i] = lowest_pres_RS*math.exp(-np.trapz(integrant[0:i], z_RS[0:i]))

#p_calc_pres = p_calc_pres * units.hPa

#####

spez_hum = data_comma_temp['Specific_humidity']
spez_hum = spez_hum.values * units('g/kg')
temp_d_degC = cc.dewpoint_from_specific_humidity(spez_hum, temp_degC, p_1)

RH_RA = cc.relative_humidity_from_specific_humidity(spez_hum, temp_degC, p_1) 

plt.figure(figsize = (5,12))
plt.plot(RH_RA * 100, p_1, color = 'red', zorder = 5)
plt.plot(RH_RS, p_RS_original, color = 'black')
plt.gca().invert_yaxis()
plt.ylabel('Pressure [hPa]')
plt.xlabel('RH [%]')

plt.figure(figsize = (5,12))
plt.plot(RH_RA * 100, z_RA, color = 'red', zorder = 5)
plt.plot(RH_RS, z_RS, color = 'black')
plt.ylim(0,10000)
plt.ylabel('Height [m]')
plt.xlabel('RH [%]')





####################################### Plot data and save figure ######################################## 
### Skew t log p diagram ###
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig)

RS1 = skew.plot(p_RS_original, T_RS_original, color = 'red', linewidth = 2)
RS2 = skew.plot(p_RS_original, T_d_RS_original, color = 'red', linewidth=2)

#RS1 = skew.plot(p_calc_pres, T_RS_original, color = 'orange', linewidth = 2)
#RS2 = skew.plot(p_calc_pres, T_d_RS_original, color = 'orange', linewidth=2)

#RS1 = skew.plot(p_RS, T_RS, color = 'orange', linewidth = 2)
#RS2 = skew.plot(p_RS, T_d_RS, color = 'orange', linewidth=2)

#RS1 = skew.plot(p_RS_1, T_RS_1, color = 'black', linewidth = 2)
#RS2 = skew.plot(p_RS_1, T_d_RS_1, color = 'black', linewidth=2)

#NC1 = skew.plot(p_NUCAPS, T_NUCAPS,color = 'grey', linewidth=2)
#NC2 = skew.plot(p_NUCAPS, T_d_NUCAPS, color = 'grey', linewidth=2)

#skew.plot(p, temp_degC, color = 'black', linewidth = 2, zorder = 1)
#skew.plot(p, temp_d_degC, color = 'black', linewidth = 2, zorder = 1)

skew.plot(p_1, temp_degC, color = 'green', linewidth = 2, zorder=0)
skew.plot(p_1, temp_d_degC, color = 'green', linewidth = 2, zorder = 0)

#skew.plot(p_RS_original, RH_RS, color = 'orange', linewidth = 2, zorder=0)
#skew.plot(p_1, RH_RA, color = 'orange', linewidth = 2, zorder = 0)

#MIT_T = skew.plot(p_NUCAPS, MIT_T_NUCAPS, color = 'blue', linewidth=2)
#MIT_T_d = skew.plot(p_NUCAPS, MIT_T_d_NUCAPS, color = 'blue', linewidth=2)

#FG_T = skew.plot(p_NUCAPS, FG_T_NUCAPS, color = 'green', linewidth=2)
#FG_T_d = skew.plot(p_NUCAPS, FG_T_d_NUCAPS, color = 'green', linewidth=2)

#Surface_meas = skew.plot(Luftdruck_Stationshoehe, Lufttemperatur_2muBoden, 'ro', color = 'black')
#Surface_meas_d = skew.plot(Luftdruck_Stationshoehe, Lufttemperatur_d_2muBoden, 'ro', color = 'black')

plt.ylabel('Pressure [hPa]', fontsize = 14)
plt.xlabel('Temperature [°C]', fontsize = 14)
skew.ax.tick_params(labelsize = 14)
skew.ax.set_ylim(1000, 400)
skew.ax.set_xlim(-60, 100)

plt.plot(RH_RA, p_1)
plt.plot(RH_RS, p_RS_original)
plt.gca
#props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#textstr = '\n'.join((
#    r'Date: %s'% (datetime.strptime(Year+Month+Day+Hour,"%Y%m%d%H"),),
#    r'Distance RS - NUCAPS (in km): %s' % (min_dist, ),
    #r'Time difference NUCAPS: %s' % (time_dif, )))
    
#skew.ax.text(0.02, 0.1, textstr, transform=skew.ax.transAxes,
#             fontsize=12, verticalalignment='top', bbox=props)

#L = fig.legend(labels= ['RS T','RS Td','RS T smoothed (100 hPa)', 'RS Td smoothed (100 hPa)', 'NUCAPS T','NUCAPS Td', 'Surf T', 'Surf Td'],  loc = 'upper right',bbox_to_anchor=(0.859, 0.82), fontsize = 12)
#Figure_name = Year + Month + Day + Hour + '_' + str(file_index) + '.png'
#plt.savefig('/data/COALITION2/PicturesSatellite/results_NAL/Plots/' + Figure_name)
plt.show()

### Calculate difference between curves ###
# smoothed curve 
#Diff_profile = T_NUCAPS_1 -T_RS
#Diff_profile_MIT = MIT_T_NUCAPS_1 -T_RS
#Diff_profile_FG = FG_T_NUCAPS_1 -T_RS
# original curve
#Diff_profile_original = T_NUCAPS_1 -T_RS_original
#Diff_profile_MIT_original = MIT_T_NUCAPS_1 -T_RS_original
#Diff_profile_FG_original = FG_T_NUCAPS_1 -T_RS_original

#plt.plot(Diff_profile, p_RS, color = 'orange')
#plt.plot(Diff_profile_MIT,p_RS, color = 'blue')
#plt.plot(Diff_profile_FG, p_RS, color = 'green')

#plt.plot(Diff_profile_original, p_RS, color = 'coral')
#plt.plot(Diff_profile_MIT_original,p_RS, color = 'cornflowerblue')
#plt.plot(Diff_profile_FG_original, p_RS, color = 'lime')

#plt.axvline(x=0, color = 'grey', linestyle = '--')
#plt.gca().invert_yaxis()
#plt.ylim(100,1000)
#plt.xlim(-10,15)
#Figure_name = Year + Month + Day + Hour + '_' + str(file_index) + '_Diff_T' +'.png'
#plt.savefig('/data/COALITION2/PicturesSatellite/results_NAL/Plots/' + Figure_name)


# Diff Temperature Td
# smoothed curve 
#Diff_profile = T_d_NUCAPS_1 -T_d_RS
#Diff_profile_MIT = MIT_T_d_NUCAPS_1 -T_d_RS
#Diff_profile_FG = FG_T_d_NUCAPS_1 -T_d_RS
# original curve
#Diff_profile_original = T_d_NUCAPS_1 -T_d_RS_original
#Diff_profile_MIT_original = MIT_T_d_NUCAPS_1 -T_d_RS_original
#Diff_profile_FG_original = FG_T_d_NUCAPS_1 -T_d_RS_original

#plt.plot(Diff_profile, p_RS, color = 'orange')
#plt.plot(Diff_profile_MIT,p_RS, color = 'blue')
#plt.plot(Diff_profile_FG, p_RS, color = 'green')

#plt.plot(Diff_profile_original, p_RS, color = 'coral')
#plt.plot(Diff_profile_MIT_original,p_RS, color = 'cornflowerblue')
#plt.plot(Diff_profile_FG_original, p_RS, color = 'lime')

#plt.axvline(x=0, color = 'grey', linestyle = '--')
#plt.gca().invert_yaxis()
#plt.ylim(100,1000)
#plt.xlim(-10,15)
#Figure_name = Year + Month + Day + Hour + '_' + str(file_index) + '_Diff_Td' + '.png'
#plt.savefig('/data/COALITION2/PicturesSatellite/results_NAL/Plots/' + Figure_name)

