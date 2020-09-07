#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 06:36:07 2020

@author: nal

combine COSMO with Raman Lidar and Microwave Radiometer

std von COSMO: http://intranet.meteoswiss.ch/modelle/Verification/Operational/Seasonal/2019s3/Vertical-profiles/COSMO-1.php
"""
import numpy as np
import matplotlib.pyplot as plt
import urllib
import datetime
import pandas as pd
import xarray as xr

import pandas as pd
import xarray as xr
import numpy as np 
from numpy.lib.scimath import logn

import math
from math import e
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

import datetime as dt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from metpy.interpolate import interpolate_1d
from metpy import calc as cc
from metpy.units import units
import metpy.calc as mpcalc
import metpy 
from metpy.calc import pressure_to_height_std
import os
os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
import xarray
### PARAMETERS FOR MATPLOTLIB :
import matplotlib as mpl
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.cm as cm
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import spatial 

def average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid_1, input_data_filtered, comparison_grid):
    INCA_grid = INCA_grid_1[::-1].reset_index(drop=True)
    input_grid_smoothed_all = pd.DataFrame()
    while firstobj != lastobj:
        nowdate = firstobj.strftime('%Y%m%d')
        #print(nowdate) 
        input_data_time = input_data_filtered[input_data_filtered.time_YMDHMS == firstobj] 
        input_data_time = input_data_time.iloc[::-1].reset_index(drop=True)
        comparison_grid_time = comparison_grid[comparison_grid.time_YMDHMS == firstobj]
        comparison_grid_time = comparison_grid_time.reset_index(drop=True)   
  
        if comparison_grid_time.empty:
            firstobj = firstobj + dt.timedelta(days=1)
            print('now')
        else:  
            input_interp = pd.DataFrame()
            for i in range(0,len(INCA_grid)):
                if (i == 0):
                    window_h_max = INCA_grid.iloc[i] + (INCA_grid.iloc[i+1] - INCA_grid.iloc[i]) / 2
                    window_h_min = INCA_grid.iloc[i] - (INCA_grid.iloc[i+1] - INCA_grid.iloc[i]) / 2
                elif (i==len(INCA_grid)-1):
                    window_h_min = INCA_grid.iloc[i] - (INCA_grid.iloc[i]-INCA_grid.iloc[(i-1)])  / 2
                    window_h_max = INCA_grid.iloc[i] + (INCA_grid.iloc[i]-INCA_grid.iloc[(i-1)])  / 2
                else: 
                    window_h_min = INCA_grid.iloc[i] - (INCA_grid.iloc[i]-INCA_grid.iloc[(i-1)] )  / 2
                    window_h_max = INCA_grid.iloc[i] + (INCA_grid.iloc[i+1]-INCA_grid.iloc[i]) / 2
                        
                input_data_within_bound = input_data_time[(input_data_time.altitude_m <= float(window_h_max)) & (input_data_time.altitude_m >= float(window_h_min))] 
                if window_h_min < np.min(input_data_time.altitude_m):
                    aver_mean = pd.DataFrame({'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan, 'altitude_m' : INCA_grid.loc[i]}, index = [i])
                    #print('small')
                elif input_data_within_bound.altitude_m.count() == 0:
                     aver_mean = pd.DataFrame({'temperature_mean' : griddata(input_data_time.altitude_m.values, input_data_time.temperature_degC.values, INCA_grid.loc[i]), 'temperature_d_mean' : griddata(input_data_time.altitude_m.values, input_data_time.dew_point_degC.values, INCA_grid.loc[i]),'altitude_m' : INCA_grid.loc[i]}, index = [i]).reset_index(drop=True)
                     #print('interpolate')
                else: 
                    aver_mean = pd.DataFrame({'temperature_mean': np.mean(input_data_within_bound.temperature_degC), 'temperature_d_mean' : np.mean(input_data_within_bound.dew_point_degC), 'altitude_m' : (INCA_grid.iloc[i])}, index = [i])
                    #print('average')
                input_interp = input_interp.append(aver_mean)
            input_grid_smoothed_all = input_grid_smoothed_all.append(input_interp)
            input_grid_smoothed_all = input_grid_smoothed_all.reset_index(drop=True)
            firstobj= firstobj + dt.timedelta(days=1) 
    return input_grid_smoothed_all

def open_NUCAPS_file(NUCAPS_file):       
    ds = xr.open_dataset(NUCAPS_file, decode_times=False)  # time units are non-standard, so we dont decode them here 
    units, reference_date = ds.Time.attrs['units'].split(' since ')
    if units=='msec':
        ref_date = dt.datetime.strptime(reference_date,"%Y-%m-%dT%H:%M:%SZ") # usually '1970-01-01T00:00:00Z'
        ds['datetime'] = [ -1 if np.isnan(t) else ref_date + timedelta(milliseconds=t) for t in ds.Time.data]
    return ds

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

######################################## !!! ########################################
firstdate = '20200721000000' # !! define start date + midnight/noon
lastdate = '20200728000000' # !! define end date + midnight/noon
firstobj=dt.datetime.strptime(firstdate,'%Y%m%d%H%M%S')
lastobj=dt.datetime.strptime(lastdate,'%Y%m%d%H%M%S')
######################################## !!! ########################################
lon_payerne = 6.93608#, 9.12, 11.33, 9.17
lat_payerne = 46.8220#1, 48.50, 48.15, 45.26

DT = firstobj.hour

DIFF_COMBINED_MD = pd.DataFrame()
DIFF_RM_MD = pd.DataFrame()
DIFF_COSMO_MD = pd.DataFrame()
  
STD_COMBINED_MD = pd.DataFrame()
STD_RM_MD = pd.DataFrame()
STD_COSMO_MD = pd.DataFrame()

exp = 1
sigma = 5
factor = 1

#fig, ax = plt.subplots(figsize = (6, 12))
#ax.plot(filter_array_2[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, linewidth = 3, label = 'x=2')
#ax.plot(filter_array_3[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, linewidth = 3, label = 'x=3')
#ax.plot(filter_array_4[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, linewidth = 3, label = 'x=4')
#ax.plot(STD_RM_temp_total[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, linewidth = 3, label = 'HATPRO', color = 'navy')
#ax.plot(STD_COSMO_temp_total[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, linewidth = 3, label = 'COSMO', color = 'green')
#ax.set_xlabel('Allowed difference COSMO - HATPRO [K]', size = 16)
#ax.set_ylabel('Altitude [km]', size = 20)
#plt.xticks(fontsize = 20)
#plt.yticks(fontsize = 20)
#ax.set_xticks(np.arange(0,10,1))
#ax.set_title('(x/sigma) * sigma', size = 20)
#x.legend(fontsize = 20)

############################################################ LOAD DATA ############################################################
##### INCA grid
##########################################
INCA_grid = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/COSMO/inca_topo_levels_hsurf_ccs4.nc')
    
### coordinate at Payerne
lon = INCA_grid.lon_1.values
lat = INCA_grid.lat_1.values
lonlat = np.dstack([lat.ravel(), lon.ravel()])[0,:,:]
tree = spatial.KDTree(lonlat)
coordinates = tree.query([([lat_payerne  , lon_payerne ])])
coords_close = lonlat[coordinates[1]]
indexes = np.array(np.where(INCA_grid.lon_1 == coords_close[0,1]))
INCA_grid_payerne = pd.DataFrame({'altitude_m' : INCA_grid.HFL[:, indexes[0], indexes[1]][:,0,0].values})[::-1]
INCA_grid_payerne = INCA_grid_payerne.iloc[:,0].reset_index(drop=True)
    

##########################################
##### COSMO
##########################################
COSMO_data = xr.open_dataset('/data/COALITION2/database/cosmo/T-TD_3D/cosmo1_inca_'+str(dt.datetime.strftime(firstobj, '%Y%m%d'))+'06_06.nc')
      
## define dimensions
n_z = COSMO_data.t_inca.values.shape[1]
n_y = COSMO_data.t_inca.values.shape[2]
n_x = COSMO_data.t_inca.values.shape[3]

            
 
############################################################ A) define data array ############################################################
# COSMO
T_COSMO = COSMO_data.t_inca.values[0,:,:,:][::-1] - 273.15 
T_d_COSMO = metpy.calc.dewpoint_from_specific_humidity(COSMO_data.qv_inca, COSMO_data.t_inca, COSMO_data.p_inca)[0,:,:,:][::-1].magnitude



############################################################ STD SPACE ############################################################
##### SPACE #####
# calculate error with distance
 ##### SPACE #####
# calculate error with distance
T = T_COSMO
points = 345
STD_temp_space=np.zeros((n_z,points+1))
for j in range(0,points-1):
    for k in range(0, n_z-1):
        std_x = np.sqrt(((T[k,0:(n_y-j),:] - T[k,j:(n_y),:])**2)/2)
        std_y = np.sqrt(((T[k,:,0:(n_x-j)] - T[k,:,j:(n_x)])**2)/2)
        STD_temp_space[k,j] = np.mean(0.5 * (std_x[:, j:(n_x)] + std_y[j:(n_y),:]))

                      
T_d = T_d_COSMO
points = 345
STD_temp_d_space=np.zeros((n_z,points+1))
for j in range(0,points-1):
    for k in range(0, n_z-1):
        std_x = np.sqrt(((T_d[k,0:(n_y-j),:] - T_d[k,j:(n_y),:])**2)/2)
        std_y = np.sqrt(((T_d[k,:,0:(n_x-j)] - T_d[k,:,j:(n_x)])**2)/2)
        STD_temp_d_space[k,j] = np.mean(0.5 * (std_x[:, j:(n_x)] + std_y[j:(n_y),:]))
            
          

    # calculate distance from payerne
distance_array = np.zeros((n_y,n_x))
for i in range(n_y):
    for j in range(n_x):
        distance_array[i,j] = np.sqrt((i-indexes[0,0])**2 + (j-indexes[1,0])**2)
        
 
############################################################
# Std in SPACE
############################################################  
# < temperature >
STD_temp_space_Payerne = np.zeros((n_z, n_y,n_x))
STD_temp_space = STD_temp_space
for i in range(0, n_y):
    for j in range(0, n_x):
        dist = distance_array[i,j]
        dist_max = np.ceil(dist)
        dist_min = np.floor(dist)
        diff_max = dist_max - dist
        diff_min = 1 - diff_max
        if (dist_max >= 345) or (dist_min >= 345):
            STD_temp_space_Payerne[:, i, j] = np.full((50,), np.nan)
        else: 
            STD_temp_space_1 = (diff_min / (diff_min + diff_max)  * STD_temp_space[:, int(dist_max)]) + (diff_max / (diff_min + diff_max) * STD_temp_space[:, int(dist_min)]) 
            STD_temp_space_Payerne[:, i, j] = STD_temp_space_1
            
# < temperature >
STD_temp_d_space_Payerne = np.zeros((n_z, n_y,n_x))
STD_temp_d_space = STD_temp_d_space
for i in range(0, n_y):
    for j in range(0, n_x):
        dist = distance_array[i,j]
        dist_max = np.ceil(dist)
        dist_min = np.floor(dist)
        diff_max = dist_max - dist
        diff_min = 1 - diff_max
        if (dist_max >= 345) or (dist_min >= 345):
            STD_temp_d_space_Payerne[:, i, j] = np.full((50,), np.nan)
        else: 
            STD_temp_d_space_1 = (diff_min / (diff_min + diff_max)  * STD_temp_d_space[:, int(dist_max)]) + (diff_max / (diff_min + diff_max) * STD_temp_d_space[:, int(dist_min)]) 
            STD_temp_d_space_Payerne[:, i, j] = STD_temp_d_space_1
    
## plot Std in space and distance array
fig, ax = plt.subplots(figsize = (12, 12))
im = ax.contourf(np.arange(0,346), INCA_grid_payerne, STD_temp_space, cmap = cm.Spectral_r, levels = np.arange(0,9,0.2)) # in y direction
#im = ax.contourf(np.arange(0,points), INCA_grid_payerne, STD_temp_d_space[::-1], cmap = cm.Spectral_r, levels = np.arange(0,9,0.2)) # in y direction
#im = ax.contourf(np.arange(0,710),np.arange(0,640), distance_array, cmap =  cm.YlGnBu_r, levels = np.arange(0,len(distance_array), 30)) # in y direction
ax.set_xlabel('Distance [# grid points]', fontsize = 20)
ax.set_ylabel('Altitude [m]', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
cbar.set_label(label='STD [K]', size = 20)
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_xticklabels([np.arange(0,4.1,0.2)])
ax.scatter(indexes[1,0], indexes[0,0] , color = 'black')

## plot std in space from Payerne
fig, ax = plt.subplots(figsize = (12, 12))
im = ax.contourf(np.arange(0, 640), INCA_grid_payerne, STD_temp_space_Payerne[:, :, indexes[1,0]], cmap = cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
#im = ax.contourf(np.arange(0,710) , np.arange(0,640), distance_array, cmap =  cm.YlGnBu_r, levels  = np.arange(0,640,20)) # in y direction
ax.set_xlabel('grid point in lon direction', fontsize = 20)
ax.set_ylabel('grid points in lat direction', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
cbar.set_label(label='Distance [# grid points]', size = 20)
cbar.ax.tick_params(labelsize=20)
ax.set_xlim(50,640)

############################################################ STD ABSOLUT and TOTAL ############################################################
# COSMO
############################################################
#:::::::::::ABSOLUTE:::::::::::
### < temperature >
COSMO_std_temp = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/COSMO/JJA_2019/scratch/owm/verify/upper-air/JJA_2019/COSMO-1/output_all_stations_6610/allscores.dat', ';')
    
COSMO_std_temp['altitude_m'] = metpy.calc.pressure_to_height_std(COSMO_std_temp.plevel.values/100 * units.hPa) * 1000
COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.varno == 2]
COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.scorename == 'SD']
#COSMO_std_temp.scores = COSMO_std_temp.scores**exp
COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.leadtime == 6][0:20]
COSMO_std_temp['plevel'] = COSMO_std_temp['plevel']
    
COSMO_std_temp = griddata(COSMO_std_temp.altitude_m.values, COSMO_std_temp.scores.values, (INCA_grid_payerne.values))
COSMO_std_temp_absolute = np.zeros((n_z, n_y, n_x))
for i in range(n_y):
    for j in range(n_x):
        COSMO_std_temp_absolute[:, i,j] = COSMO_std_temp
        
### < temperature d>
COSMO_std_temp_d = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/COSMO/JJA_2019/scratch/owm/verify/upper-air/JJA_2019/COSMO-1/output_all_stations_6610/allscores.dat', ';')
    
COSMO_std_temp_d['altitude_m'] = metpy.calc.pressure_to_height_std(COSMO_std_temp_d.plevel.values/100 * units.hPa) * 1000
COSMO_std_temp_d = COSMO_std_temp_d[COSMO_std_temp_d.varno == 29]
COSMO_std_temp_d = COSMO_std_temp_d[COSMO_std_temp_d.scorename == 'SD']
COSMO_std_temp_d = COSMO_std_temp_d[COSMO_std_temp_d.leadtime == 6][0:20]
COSMO_std_temp_d['plevel'] = COSMO_std_temp_d['plevel']
    
COSMO_std_temp_d = griddata(COSMO_std_temp_d.altitude_m.values, COSMO_std_temp_d.scores.values, (INCA_grid_payerne.values))
COSMO_std_temp_d_absolute = np.zeros((n_z, n_y, n_x))
for i in range(n_y):
    for j in range(n_x):
        COSMO_std_temp_d_absolute[:, i,j] = COSMO_std_temp_d
    

#:::::::::::TOTAL:::::::::::
STD_COSMO_temp_total = COSMO_std_temp_absolute 
STD_COSMO_temp_d_total = COSMO_std_temp_d_absolute  
STD_COSMO_temp_total_1 = STD_COSMO_temp_total
    
############################################################
# Radiometer
############################################################
#:::::::::::ABSOLUTE:::::::::::
# < temperature >
RM_std_temp_1 = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RM_temp_'+str(DT)+'.csv')
### exponent
RM_std_temp_1 = RM_std_temp_1
RM_std_temp_1 = factor * RM_std_temp_1.std_temp.values
RM_std_temp_1[RM_std_temp_1 >= 7] = np.nan
STD_RM_temp_absolute = np.zeros((n_z, n_y, n_x))
for i in range(n_y):
    for j in range(n_x):
       STD_RM_temp_absolute[:, i,j] = RM_std_temp_1
    
# < temperature d >
RM_std_temp_d_1 = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RM_temp_d_'+str(DT)+'.csv')
RM_std_temp_d_1 = factor * RM_std_temp_d_1.std_temp_d.values
RM_std_temp_d_1[RM_std_temp_d_1 >= 5] = np.nan
STD_RM_temp_d_absolute = np.zeros((n_z, n_y, n_x))
for i in range(n_y):
    for j in range(n_x):
       STD_RM_temp_d_absolute[:, i,j] = RM_std_temp_d_1   
             

    #:::::::::::TOTAL:::::::::::
STD_RM_temp_total = (STD_RM_temp_absolute) + (STD_temp_space_Payerne)
STD_RM_temp_d_total = (STD_RM_temp_d_absolute) + (STD_temp_d_space_Payerne)

fig, ax = plt.subplots(figsize = (12, 12))
cmap = cm.Spectral_r
#cmap = cmap.set_under('w'a_COSMO_temp
#im = ax.contourf(np.arange(0, 640), INCA_grid_payerne, a_RALMO_temp[:,  :, indexes[1,0]], cmap = cm.Spectral_r, extend = 'max', levels = np.arange(0,1,0.03)) # in y directio
im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  STD_COSMO_temp_total[:, :, indexes[1,0]], cmap = cm.Spectral_r,   levels = np.arange(0,9,0.2)) # in y directio
#im = ax.contourf(np.arange(0,710) , np.arange(0,640),  WEIGHT_RA_temp[15, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,1)) # in y direction
#im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,WEIGHT_RM_temp[:, :, indexes[1,0]], cmap = cmap, levels = [0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # in y direction
#im = ax.contourf(np.arange(0,710) , np.arange(0,640),  WEIGHT_RM_temp[25, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
#im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  WEIGHT_COSMO_temp[:, :, indexes[1,0]], cmap = cm.Spectral_r, levels  = np.arange(0,1)) # in y direction
#im = ax.contourf(np.arange(0,710) , np.arange(0,640),  WEIGHT_COSMO_temp[25, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y directionax.scatter(indexes[0,0], 0, color = 'black')
ax.set_xlabel('grid point in lat direction', fontsize = 20)
ax.set_ylabel('Altitude [m]', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
cbar.set_label(label='Std [K]', size = 20)
cbar.ax.tick_params(labelsize=20)
    
while firstobj != lastobj:              
    print(firstobj)
    lastobj_now = firstobj + dt.timedelta(days=1)
   ############################################################ LOAD DATA ############################################################
    ##### INCA grid
    ##########################################
    INCA_grid = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/COSMO/inca_topo_levels_hsurf_ccs4.nc')
    
    ### coordinate at Payerne
    lon = INCA_grid.lon_1.values
    lat = INCA_grid.lat_1.values
    lonlat = np.dstack([lat.ravel(), lon.ravel()])[0,:,:]
    tree = spatial.KDTree(lonlat)
    coordinates = tree.query([([lat_payerne  , lon_payerne ])])
    coords_close = lonlat[coordinates[1]]
    indexes = np.array(np.where(INCA_grid.lon_1 == coords_close[0,1]))
    INCA_grid_payerne = pd.DataFrame({'altitude_m' : INCA_grid.HFL[:, indexes[0], indexes[1]][:,0,0].values})[::-1]
    INCA_grid_payerne = INCA_grid_payerne.iloc[:,0].reset_index(drop=True)
    
    ##########################################
    ##### RADIOSONDE
    ##########################################l
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&dataSourceId=34&verbose=position&delimiter=comma&parameterIds=744,745,746,742,748,743,747&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=22'
    RS_data = pd.read_csv(url, skiprows = [1], sep=',')
    RS_data = RS_data.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent', '742':'altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })
    RS_data = RS_data[RS_data['temperature_degC'] != 1e+07]
    RS_data['time_YMDHMS'] = pd.to_datetime(RS_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    
    RS_averaged = average_RS_to_INCA_grid(firstobj, lastobj_now, INCA_grid_payerne, RS_data, RS_data)
    RS_averaged = RS_averaged[::-1].reset_index(drop=True)
 
    ##########################################
    ##### COSMO
    ##########################################
    COSMO_data = xr.open_dataset('/data/COALITION2/database/cosmo/T-TD_3D/cosmo1_inca_'+str(dt.datetime.strftime(firstobj, '%Y%m%d'))+'06_06.nc')
      
    ## define dimensions
    n_z = COSMO_data.t_inca.values.shape[1]
    n_y = COSMO_data.t_inca.values.shape[2]
    n_x = COSMO_data.t_inca.values.shape[3]
    ############################################################
    # RADIOMETER
    ############################################################
    #:::::::::::ABSOLUTE:::::::::::
    # < temperature >
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&delimiter=comma&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=31'
    RM = pd.read_csv(url, skiprows = [1], sep=',')
    RM = RM.rename(columns = {'termin' : 'time_YMDHMS' , '3147' : 'temperature_K', '3148' : 'absolute_humidity_gm3', 'level' : 'altitude_m'})
    RM['temperature_degC'] = RM.temperature_K - 273.15
    
    p_w = ((RM.temperature_K * RM.absolute_humidity_gm3) / 2.16679)
    p_s = metpy.calc.saturation_vapor_pressure(RM.temperature_K.values * units.kelvin)
    
    RH = p_w /p_s
    RM['dew_point_degC'] = metpy.calc.dewpoint_from_relative_humidity(RM.temperature_K.values * units.kelvin, RH.values * units.percent)
    
    RM['time_YMDHMS'] = pd.to_datetime(RM.time_YMDHMS, format = '%Y%m%d%H%M%S')
    RM = average_RS_to_INCA_grid(firstobj, lastobj_now, INCA_grid_payerne, RM, RM)[::-1]
    
    # < temperature >
    T_RM_1 = RM.temperature_mean
    T_RM = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            T_RM[:, i,j] = T_RM_1
            
            
    # < temperature d >
    T_d_RM_1 = RM.temperature_d_mean
    T_d_RM = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            T_d_RM[:, i,j] = T_d_RM_1
            
 
    ############################################################ A) define data array ############################################################
    # COSMO
    T_COSMO = COSMO_data.t_inca.values[0,:,:,:][::-1] - 273.15 
    T_d_COSMO = metpy.calc.dewpoint_from_specific_humidity(COSMO_data.qv_inca, COSMO_data.t_inca, COSMO_data.p_inca)[0,:,:,:][::-1].magnitude
   
    
   
    ############################################################ STD SPACE ############################################################
    ##### SPACE #####
    # calculate error with distance
     ##### SPACE #####
    # calculate error with distance
    #T = T_COSMO
    #points = 345
    #STD_temp_space=np.zeros((n_z,points+1))
    #for j in range(0,points-1):
    #    for k in range(0, n_z-1):
    #        std_x = np.sqrt(((T[k,0:(n_y-j),:] - T[k,j:(n_y),:])**2)/2)
    #        std_y = np.sqrt(((T[k,:,0:(n_x-j)] - T[k,:,j:(n_x)])**2)/2)
    #        STD_temp_space[k,j] = np.mean(0.5 * (std_x[:, j:(n_x)] + std_y[j:(n_y),:]))
    
                          
    #T_d = T_d_COSMO
    #points = 345
    #STD_temp_d_space=np.zeros((n_z,points+1))
    #for j in range(0,points-1):
    #    for k in range(0, n_z-1):
    #        std_x = np.sqrt(((T_d[k,0:(n_y-j),:] - T_d[k,j:(n_y),:])**2)/2)
    #        std_y = np.sqrt(((T_d[k,:,0:(n_x-j)] - T_d[k,:,j:(n_x)])**2)/2)
    #        STD_temp_d_space[k,j] = np.mean(0.5 * (std_x[:, j:(n_x)] + std_y[j:(n_y),:]))
                
     
    ############################################################
    # Std in SPACE
    ############################################################  
    # < temperature >
    ##STD_temp_space_Payerne = np.zeros((n_z, n_y,n_x))
    #STD_temp_space = STD_temp_space
    #for i in range(0, n_y):
    #    for j in range(0, n_x):
    #        dist = distance_array[i,j]
    #        dist_max = np.ceil(dist)
    #        dist_min = np.floor(dist)
    #        diff_max = dist_max - dist
    #        diff_min = 1 - diff_max
    #        if (dist_max >= 345) or (dist_min >= 345):
    #            STD_temp_space_Payerne[:, i, j] = np.full((50,), np.nan)
    #        else: 
    #            STD_temp_space_1 = (diff_min / (diff_min + diff_max)  * STD_temp_space[:, int(dist_max)]) + (diff_max / (diff_min + diff_max) * STD_temp_space[:, int(dist_min)]) 
    #            STD_temp_space_Payerne[:, i, j] = STD_temp_space_1
                
    # < temperature >
    #STD_temp_d_space_Payerne = np.zeros((n_z, n_y,n_x))
    #STD_temp_d_space = STD_temp_d_space
    #for i in range(0, n_y):
    #    for j in range(0, n_x):
    #        dist = distance_array[i,j]
    #        dist_max = np.ceil(dist)
    #        dist_min = np.floor(dist)
    #        diff_max = dist_max - dist
    #        diff_min = 1 - diff_max
    #        if (dist_max >= 345) or (dist_min >= 345):
    #            STD_temp_d_space_Payerne[:, i, j] = np.full((50,), np.nan)
    #        else: 
    #            STD_temp_d_space_1 = (diff_min / (diff_min + diff_max)  * STD_temp_d_space[:, int(dist_max)]) + (diff_max / (diff_min + diff_max) * STD_temp_d_space[:, int(dist_min)]) 
    #            STD_temp_d_space_Payerne[:, i, j] = STD_temp_d_space_1 
    
        ## plot data
    #fig, ax = plt.subplots(figsize = (12, 12))
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne, T_COSMO[:,:, indexes[0,0]], cmap = cm.coolwarm, levels = np.arange(-50,35,5)) # in y direction
    #ax.scatter(indexes[1,0], indexes[0,0], color = 'black')
    #ax.set_xlabel('y [grid points]', fontsize = 20)
    #ax.set_ylabel('altitude [m]', fontsize = 20)
    #plt.xticks(fontsize = 20)
    #plt.yticks(fontsize = 20)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.2)
    #cbar = fig.colorbar(im, ticks = np.arange(-50, 50, 5), cax = cax, orientation= 'vertical')
    #cbar.set_label(label='temperature [°C]', size = 20)
    #cbar.ax.tick_params(labelsize=20) 


                
    #fig, ax = plt.subplots(figsize = (12, 12))
    #m = ax.imshow(STD_space[0:50,0:100], cmap = cm.Spectral)
    #m = ax.contourf(np.arange(0,points), np.arange(0,50), STD_space[::-1], cmap = cm.Spectral_r, levels = np.arange(0,4.1,0.2)) # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  STD_RM_temp_absolute[:, :, indexes[1,0]], cmap = cm.Spectral_r, levels  = np.arange(0,12, 0.2)) # in y direction
    #im = ax.contourf(np.arange(0,710) , np.arange(0,640),   STD_RA_temp_total[25, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
    #ax.scatter(indexes[1,0], indexes[0,0], color = 'black')
    #ax.set_xlabel('Distance [# grid points]', fontsize = 20)
    #ax.set_ylabel('Altitude [m]', fontsize = 20)
    #plt.xticks(fontsize = 20)
    #plt.yticks(fontsize = 20)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.2)
    #cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
    #cbar.set_label(label='STD [K]', size = 20)
    #cbar.ax.tick_params(labelsize=20)
    
    
    
    
    #### FILTER 
    ### interval
    #### filter
    #filter_array = np.abs((sigma / STD_RM_temp_total) * STD_COSMO_temp_total)
    #DIFF_RM = np.abs(T_COSMO - T_RM)
    #T_RM[DIFF_RM >= filter_array] = np.nan

    
    
    

    
    # < temperature >
    STD_RM_temp_total_1 = STD_RM_temp_total
    STD_RM_temp_total_1[np.isnan(T_RM)] = np.nan
    sigma_RADIOMETER = STD_RM_temp_total_1**2
    sigma_COSMO = STD_COSMO_temp_total**2

    STD_RADIOMETER_COSMO = np.nansum(np.stack((sigma_COSMO, sigma_RADIOMETER)), axis = 0)

    a_COSMO_temp = sigma_RADIOMETER/ STD_RADIOMETER_COSMO
    a_COSMO_temp[np.isnan(STD_RM_temp_total)] = 1
    
    a_RADIOMETER_temp = sigma_COSMO / STD_RADIOMETER_COSMO
    a_RADIOMETER_temp[np.isnan(STD_RM_temp_total)] = 0
    

    # < temperature d >
    STD_RM_temp_d_total_1 = STD_RM_temp_d_total
    STD_RM_temp_d_total_1[np.isnan(T_d_RM)] = np.nan
    sigma_RADIOMETER_temp_d = STD_RM_temp_d_total_1**2
    sigma_COSMO_temp_d = STD_COSMO_temp_d_total**2

    STD_RADIOMETER_COSMO_temp_d = np.nansum(np.stack((sigma_COSMO_temp_d, sigma_RADIOMETER_temp_d)), axis = 0)

    a_COSMO_temp_d = sigma_RADIOMETER_temp_d/ STD_RADIOMETER_COSMO_temp_d
    a_COSMO_temp_d[np.isnan(STD_RM_temp_d_total)] = 1
    
    a_RADIOMETER_temp_d = sigma_COSMO_temp_d / STD_RADIOMETER_COSMO_temp_d
    a_RADIOMETER_temp_d[np.isnan(STD_RM_temp_d_total)] = 0
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors
    ## plot std total and weights
    fig, ax = plt.subplots(figsize = (12, 12))
    cmap = cm.Spectral_r
    #cmap = cmap.set_under('w'a_COSMO_temp
    im = ax.contourf(np.arange(0, 640), INCA_grid_payerne, a_COSMO_temp[:,  :, indexes[1,0]], cmap = cm.Spectral_r, extend = 'max', levels = np.arange(0,1,0.03)) # in y directio
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  STD_RM_temp_total[:, :, indexes[1,0]], cmap = cm.Spectral_r,   levels = np.arange(0,9,0.2)) # in y directio
    #im = ax.contourf(np.arange(0,710) , np.arange(0,640),  WEIGHT_RA_temp[15, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,1)) # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,WEIGHT_RM_temp[:, :, indexes[1,0]], cmap = cmap, levels = [0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # in y direction
    #im = ax.contourf(np.arange(0,710) , np.arange(0,640),  WEIGHT_RM_temp[25, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  WEIGHT_COSMO_temp[:, :, indexes[1,0]], cmap = cm.Spectral_r, levels  = np.arange(0,1)) # in y direction
    #im = ax.contourf(np.arange(0,710) , np.arange(0,640),  WEIGHT_COSMO_temp[25, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
    ax.scatter(indexes[0,0], 0, color = 'black')
    ax.set_xlabel('grid point in lat direction', fontsize = 20)
    ax.set_ylabel('Altitude [m]', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
    cbar.set_label(label='Std [K]', size = 20)
    cbar.ax.tick_params(labelsize=20)
    
    
    ############################################################ COMBINE DATASETS ###########A################################################
    T_COMBINED = np.nansum(np.stack(((a_COSMO_temp * T_COSMO) ,  (a_RADIOMETER_temp * T_RM))), axis = 0)
    T_d_COMBINED = np.nansum(np.stack(((a_COSMO_temp_d * T_d_COSMO) ,  (a_RADIOMETER_temp_d * T_d_RM))), axis = 0)
 
    fig, ax = plt.subplots(figsize = (5, 12))
    ax.plot(T_RM[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, color = 'black', label = 'T RALMO', linewidth = 3)
    ax.plot(T_d_RM[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, color = 'black', label = 'Td RALMO', linestyle = '--', linewidth = 3)
    ax.plot(RS_averaged.temperature_mean, INCA_grid_payerne, color = 'navy', label = 'T Radiometer', linewidth = 3)
    ax.plot(RS_averaged.temperature_d_mean, INCA_grid_payerne, color = 'navy', label = 'Td Radiometer', linestyle = '--',linewidth = 3)
    ax.plot(T_COSMO[:,indexes[0,0],indexes[1,0]], INCA_grid_payerne, color = 'green', label = 'T COSMO', linewidth = 3)
    ax.plot(T_d_COSMO[:,indexes[0,0],indexes[1,0]], INCA_grid_payerne, color = 'green', label = 'Td COSMO', linestyle = '--',linewidth = 3)
    ax.plot(T_COMBINED[:,indexes[0,0],indexes[1,0]], INCA_grid_payerne, color = 'red', label = 'T COMBINED', linewidth = 3)
    ax.plot(T_d_COMBINED[:,indexes[0,0],indexes[1,0]], INCA_grid_payerne, color = 'red', label = 'Td COMBINED', linestyle = '--',linewidth = 3)
   
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.legend()
     
    
    
    
    
    
    
    DIFF_COSMO_temp = pd.DataFrame({'DIFF' : (RS_averaged.temperature_mean -  T_COSMO[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})
    DIFF_COMBINED_temp = pd.DataFrame({'DIFF' : (RS_averaged.temperature_mean - T_COMBINED[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})
    DIFF_RM_temp = pd.DataFrame({'DIFF' : (RS_averaged.temperature_mean - T_RM[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})

    
    fig, ax = plt.subplots(figsize = (5, 12))  
    ax.plot(DIFF_COMBINED_temp.DIFF, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
    ax.plot(DIFF_COSMO_temp.DIFF, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
    ax.plot(DIFF_RM_temp.DIFF, INCA_grid_payerne, color = 'navy', label = 'HATPRO', linewidth = 2)
    ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
    ax.set_xlabel('Difference [K]', size = 20)
    ax.set_ylabel('Altitude [m]', size = 20)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1500)
    ax.set_xticks(np.arange(-6,7, 2))
    ax.tick_params(labelsize = 20)
    ax.set_yticks(np.arange(0,15000,1000))
    ax.grid()
    ax.legend(fontsize = 20)
  
    DIFF_COMBINED_MD = DIFF_COMBINED_MD.append(DIFF_COMBINED_temp)
    DIFF_RM_MD = DIFF_RM_MD.append(DIFF_RM_temp)
    DIFF_COSMO_MD = DIFF_COSMO_MD.append(DIFF_COSMO_temp)
    
    
    
    
    
    
    
    STD_COSMO_temp = pd.DataFrame({'STD' : np.nanstd((RS_averaged.temperature_mean.reset_index(drop=True),  T_COSMO[:, indexes[0,0], indexes[1,0]]), axis = 0), 'altitude_m' : INCA_grid_payerne})
    STD_COSMO_temp[np.isnan(T_COSMO)] = np.nan
    STD_COMBINED_temp = pd.DataFrame({'STD' :np.nanstd( (RS_averaged.temperature_mean.reset_index(drop=True),  T_COMBINED[:, indexes[0,0], indexes[1,0]]), axis = 0), 'altitude_m' : INCA_grid_payerne})
    STD_COMBINED_temp[np.isnan(T_COMBINED)] = np.nan
    STD_RM_temp = pd.DataFrame({'STD' :np.nanstd( (RS_averaged.temperature_mean.reset_index(drop=True),  T_RM[:, indexes[0,0], indexes[1,0]]), axis = 0), 'altitude_m' : INCA_grid_payerne})
    STD_RM_temp[np.isnan(T_RM)] = np.nan
    
    fig, ax = plt.subplots(figsize = (5, 12))  
    ax.plot(STD_COMBINED_temp.STD, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
    ax.plot(STD_COSMO_temp.STD, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
    ax.plot(STD_RM_temp.STD, INCA_grid_payerne, color = 'navy', label = 'HATPRO', linewidth = 2)

    ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
    ax.set_xlabel('RMSE [K]', size = 20)
    ax.set_ylabel('Altitude [m]', size = 20)
    ax.set_xlim(0,3)
    #ax.set_ylim(0,1500)
    ax.set_xticks(np.arange(0,3))
    ax.tick_params(labelsize = 20)
    ax.set_yticks(np.arange(0,15000,1000))
    ax.grid()
    ax.legend(fontsize = 20)
    ax.fill_betweenx( INCA_grid_payerne, STD_COSMO_temp.STD, STD_COMBINED_temp.STD, color= 'gold', where = STD_COSMO_temp.STD > STD_COMBINED_temp.STD, interpolate = True, alpha = 1)
    ax.fill_betweenx( INCA_grid_payerne, STD_COSMO_temp.STD, STD_COMBINED_temp.STD, color= 'grey', where = STD_COSMO_temp.STD < STD_COMBINED_temp.STD,interpolate = True, alpha= 1)

    #fig, ax = plt.subplots(figsize = (5, 12))  
    #ax.plot(STD_COMBINED_temp_d.STD, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
    #ax.plot(STD_COSMO_temp_d.STD, INCA_grid_payerne, color = 'green', label = 'COSMO', linestyle = '--',linewidth = 2)
    #ax.plot(STD_RM_temp_d.STD, INCA_grid_payerne, color = 'navy', label = 'HATPRO', linestyle = '--',linewidth = 2)
    #ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
    #ax.set_xlabel('RMSE [K]', size = 20)
    #ax.set_ylabel('Altitude [m]', size = 20)
    #ax.set_xlim(0,5)
    #ax.set_ylim(0,1500)
    #ax.set_xticks(np.arange(0,3))
    #ax.tick_params(labelsize = 20)
    #ax.set_yticks(np.arange(0,15000,1000))
    #ax.grid()
    #ax.legend(fontsize = 20)
    #ax.fill_betweenx( INCA_grid_payerne, STD_COSMO_temp_d.STD, STD_COMBINED_temp_d.STD, color= 'gold', where = STD_COSMO_temp_d.STD > STD_COMBINED_temp_d.STD, interpolate = True, alpha = 1)
    #ax.fill_betweenx( INCA_grid_payerne, STD_COSMO_temp_d.STD, STD_COMBINED_temp_d.STD, color= 'grey', where = STD_COSMO_temp_d.STD < STD_COMBINED_temp_d.STD,interpolate = True, alpha= 1) 

    
    STD_COMBINED_MD  = STD_COMBINED_MD.append(STD_COMBINED_temp)
    STD_RM_MD = STD_RM_MD.append(STD_RM_temp)
    STD_COSMO_MD = STD_COSMO_MD.append(STD_COSMO_temp)

 
    
    
    
    firstobj = firstobj + dt.timedelta(days=1)
    




    

    





#:::::::::::DIFF:::::::::::
DIFF_COMBINED = DIFF_COMBINED_MD.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()
DIFF_COSMO = DIFF_COSMO_MD.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()
DIFF_RM = DIFF_RM_MD.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()

fig, ax = plt.subplots(figsize = (5, 12))  
ax.plot(DIFF_COMBINED.mean_all, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
ax.plot(DIFF_COSMO.mean_all, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
ax.plot(DIFF_RM.mean_all, INCA_grid_payerne, color = 'navy', label = 'HATPRO', linewidth = 2)
ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
ax.set_xlabel('Difference [K]', size = 20)
ax.set_ylabel('Altitude [m]', size = 20)
ax.set_xlim(0,1)
ax.set_ylim(0,1500)
ax.set_xticks(np.arange(-6,7, 2))
ax.tick_params(labelsize = 20)
ax.set_yticks(np.arange(0,15000,1000))
ax.grid()
ax.legend(fontsize = 20)


#:::::::::::STD:::::::::::
DIFF_COMBINED_MD.insert(column = 'number' , value =  DIFF_COMBINED_MD.DIFF.values, loc = 2)
DIFF_COMBINED_MD.number[~np.isnan(DIFF_COMBINED_MD.number)] = 1
total_number = DIFF_COMBINED_MD.groupby('altitude_m')['number'].count()

DIFF_COMBINED_MD.DIFF = DIFF_COMBINED_MD.DIFF**2
SUM_COMBINED_MD = DIFF_COMBINED_MD.groupby('altitude_m')['DIFF'].sum().to_frame(name='STD').reset_index()
STD_COMBINED = (np.sqrt(SUM_COMBINED_MD.STD)) / total_number.values



DIFF_COSMO_MD.insert(column = 'number' , value =  DIFF_COSMO_MD.DIFF.values, loc = 2)
DIFF_COSMO_MD.number[~np.isnan(DIFF_COSMO_MD.number)] = 1
total_number = DIFF_COSMO_MD.groupby('altitude_m')['number'].count()

DIFF_COSMO_MD.DIFF = DIFF_COSMO_MD.DIFF**2
SUM_COSMO_MD = DIFF_COSMO_MD.groupby('altitude_m')['DIFF'].sum().to_frame(name='STD').reset_index()
STD_COSMO = np.sqrt(SUM_COSMO_MD.STD) / total_number.values



DIFF_RM_MD.insert(column = 'number' , value =  DIFF_RM_MD.DIFF.values, loc = 2)
DIFF_RM_MD.number[~np.isnan(DIFF_RM_MD.number)] = 1
total_number = DIFF_RM_MD.groupby('altitude_m')['number'].count()

DIFF_RM_MD.DIFF = DIFF_RM_MD.DIFF**2
SUM_RM_MD = DIFF_RM_MD.groupby('altitude_m')['DIFF'].sum().to_frame(name='STD').reset_index()
STD_RM = np.sqrt(SUM_RM_MD.STD) / total_number.values

    


fig, ax = plt.subplots(figsize = (5, 12))  
ax.plot(STD_COMBINED, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
ax.plot(STD_COSMO, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
ax.plot(STD_RM, INCA_grid_payerne, color = 'navy', label = 'HATPRO', linewidth = 2)
ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
ax.set_xlabel('RSME [K]', size = 20)
ax.set_ylabel('Altitude [m]', size = 20)
ax.set_xlim(0,3)
#ax.set_ylim(0,1500)
ax.set_xticks(np.arange(0,3))
ax.tick_params(labelsize = 20)
ax.set_yticks(np.arange(0,15000,1000))
ax.grid()
ax.legend(fontsize = 20)
ax.fill_betweenx( INCA_grid_payerne, STD_COSMO, STD_COMBINED, color= 'gold', where = STD_COSMO > STD_COMBINED, interpolate = True, alpha = 1)
ax.fill_betweenx( INCA_grid_payerne, STD_COSMO, STD_COMBINED, color= 'grey', where = STD_COSMO < STD_COMBINED,interpolate = True, alpha= 1)

   













                           
