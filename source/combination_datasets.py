#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 06:36:07 2020

@author: nal

Create simple 
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


def average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid_1, input_data_filtered, comparison_grid):
    INCA_grid = INCA_grid_1[::-1].reset_index(drop=True)
    input_grid_smoothed_all = pd.DataFrame()
    while firstobj != lastobj:
        nowdate = firstobj.strftime('%Y%m%d')
        print(nowdate) 
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
                    print('small')
                elif input_data_within_bound.altitude_m.count() == 0:
                     aver_mean = pd.DataFrame({'temperature_mean' : griddata(input_data_time.altitude_m.values, input_data_time.temperature_degC.values, INCA_grid.loc[i]), 'temperature_d_mean' : griddata(input_data_time.altitude_m.values, input_data_time.dew_point_degC.values, INCA_grid.loc[i]),'altitude_m' : INCA_grid.loc[i]}, index = [i]).reset_index(drop=True)
                     print('interpolate')
                else: 
                    aver_mean = pd.DataFrame({'temperature_mean': np.mean(input_data_within_bound.temperature_degC), 'temperature_d_mean' : np.mean(input_data_within_bound.dew_point_degC), 'altitude_m' : (INCA_grid.iloc[i])}, index = [i])
                    print('average')
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

lon_payerne = 6.93608#, 9.12, 11.33, 9.17
lat_payerne = 46.8220#1, 48.50, 48.15, 45.26
from scipy import spatial 

# one date
date = '20200721'
firstdate = date + '000000'
firstobj=dt.datetime.strptime(firstdate,'%Y%m%d%H%M%S')
lastobj= firstobj + dt.timedelta(days=1)

# time period
firstdate = '20200721000000'
lastdate = '20200722000000'
firstobj=dt.datetime.strptime(firstdate,'%Y%m%d%H%M%S')
lastobj=dt.datetime.strptime(lastdate,'%Y%m%d%H%M%S')

DIFF_COMBINED_MD = pd.DataFrame()
DIFF_RA_MD = pd.DataFrame()
DIFF_COSMO_MD = pd.DataFrame()
  
STD_COMBINED_MD = pd.DataFrame()
STD_RA_MD = pd.DataFrame()
STD_COSMO_MD = pd.DataFrame()
  
while firstobj != lastobj: 
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
    ##########################################
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&dataSourceId=34&verbose=position&delimiter=comma&parameterIds=744,745,746,742,748,743,747&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=22'
    RS_data = pd.read_csv(url, skiprows = [1], sep=',')
    RS_data = RS_data.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent', '742':'altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })
    RS_data = RS_data[RS_data['temperature_degC'] != 1e+07]
    RS_data['time_YMDHMS'] = pd.to_datetime(RS_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    
    RS_averaged = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid_payerne, RS_data, RS_data)
    RS_averaged = RS_averaged[::-1].reset_index(drop=True)
 
    ##########################################
    ##### COSMO
    ##########################################
    COSMO_data = xr.open_dataset('/data/COALITION2/database/cosmo/T-TD_3D/cosmo1_inca_'+str(dt.datetime.strftime(firstobj, '%Y%m%d'))+'06_06.nc')
        
    ### variation in space
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
    RM = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid_payerne, RM, RM)
    
    # < temperature >
    T_RM_1 = RM.temperature_mean[::-1]
    T_RM = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            T_RM[:, i,j] = T_RM_1
            
    # < temperature d >
    T_d_RM_1 = RM.temperature_d_mean[::-1]
    T_d_RM = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            T_d_RM[:, i,j] = T_d_RM_1
            
    ##########################################
    ##### RALMO
    ##########################################
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&measCatNr=2&delimiter=comma&dataSourceId=38&parameterIds=4919,4906,4907,3147,4908,4909,4910,4911,4912,4913,4914,4915&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&profTypeIds=1104&obsTypeIds=30+'
    RA = pd.read_csv(url, skiprows = [1], sep=',')

    RA = RA.rename(columns = {'termin':'time_YMDHMS', 'level':'altitude_m', '4919': 'specific_humidity_gkg-1', '4906':'uncertainty_specific_humidity_gkg-1', '4907':'vertical_resolution_specific_humidity_m', '3147':'temperature_K', '4908':'uncertainty_temperature_K', '4909': 'vertical_resolution_temperature', '4910':'normalised_backscatter', '4911':'uncertainty_backscatter', '4912': 'vert_resolution_backscatter', '4913': 'aerosol_dispersion_rate', '4914': 'uncertainty_dispersion_rate', '4915' : 'vertical_resolution_aerosol_dispersion_rate'})
    RA['temperature_K'][RA['temperature_K']== int(10000000)] = np.nan
    RA['temperature_degC'] = RA.temperature_K - 273.15
    
    ## add dewpoint temperature
    pressure = metpy.calc.height_to_pressure_std(RA.altitude_m.values * units.meters)
    dewpoint_degC = cc.dewpoint_from_specific_humidity(RA['specific_humidity_gkg-1'].values * units('g/kg'), (RA.temperature_K.values) * units.kelvin, pressure)
    RA.insert(value=dewpoint_degC,column = "dew_point_degC", loc=11)
    RA['dew_point_degC'][RA['specific_humidity_gkg-1']== int(10000000)] = np.nan
    
    #RA = RA_1
    RA['time_YMDHMS'] = pd.to_datetime(RA.time_YMDHMS, format = '%Y%m%d%H%M%S')
    RA = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid_payerne, RA, RA)
    
    ############################################################ A) define data array ############################################################
    # COSMO
    T_COSMO = COSMO_data.t_inca.values[0,:,:,:][::-1] - 273.15 
    T_d_COSMO = metpy.calc.dewpoint_from_specific_humidity(COSMO_data.qv_inca, COSMO_data.t_inca, COSMO_data.p_inca)[0,:,:,:][::-1]
   
    # RALMO
    # < temperature >
    T_RA_1 = RA.temperature_mean[::-1]
    T_RA = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            T_RA[:, i,j] = T_RA_1
            
    # < temperature d >
    T_d_RA_1 = RA.temperature_d_mean[::-1]
    T_d_RA = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            T_d_RA[:, i,j] = T_d_RA_1
    
            

    #fig, ax = plt.subplots(figsize = (12, 12))
    #im = ax.contourf(np.arange(0,710), np.arange(0,640), T_d_COSMO[12,:,:], cmap = cm.coolwarm, levels = np.arange(-50, 35,5),extend = 'max') # in y direction
    #im = ax.contourf(np.arange(0,710), np.arange(0,640), T_RA[13,:,:], cmap = cm.coolwarm, levels = np.arange(-50, 50,5), extend = 'max') # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne, T_RA[:,:, indexes[0,0]], cmap = cm.coolwarm, levels = np.arange(-50,35,5)) # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne, T_d_COSMO[:,:, indexes[0,0]], cmap = cm.coolwarm, levels = np.arange(-50,35,5)) # in y direction
    #im = ax.contourf(np.arange(0,710), np.arange(0,640), T_d_RM[13,:,:], cmap = cm.coolwarm, levels = np.arange(-50, 50,5), extend = 'max') # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne, T_d_RM[:,:, indexes[0,0]], cmap = cm.coolwarm, levels = np.arange(-50,35,5)) # in y direction
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

    ############################################################ STD SPACE ############################################################
    ##### SPACE #####
    # calculate error with distance
    T = T_COSMO[::-1]
    points = 640-345
    STD_temp_space=np.zeros((n_z,points))
    for j in range(0,points-1):
        for k in range(0, n_z-1):
            std_x = np.sqrt(((T[k,0:(n_y-j),:] - T[k,j:(n_y),:])**2)/2)
            std_y = np.sqrt(((T[k,:,0:(n_x-j)] - T[k,:,j:(n_x)])**2)/2)
            STD_temp_space[k,j] = np.mean(0.5 * (std_x[:, j:(n_x)] + std_y[j:(n_y),:]))
            
    T_d = T_d_COSMO[::-1]
    points = 640-345
    STD_temp_d_space=np.zeros((n_z,points))
    for j in range(0,points-1):
        for k in range(0, n_z-1):
            std_x = np.sqrt(((T_d[k,0:(n_y-j),:] - T_d[k,j:(n_y),:])**2)/2)
            std_y = np.sqrt(((T_d[k,:,0:(n_x-j)] - T_d[k,:,j:(n_x)])**2)/2)
            STD_temp_d_space[k,j] = np.mean(0.5 * (std_x[:, j:(n_x)] + std_y[j:(n_y),:])).magnitude
            
    # calculate distance from payerne
    distance_array = np.zeros((n_y,n_x))
    for i in range(n_y):
        for j in range(n_x):
            distance_array[i,j] = np.sqrt((i-indexes[0,0])**2 + (j-indexes[1,0])**2)
    
    fig, ax = plt.subplots(figsize = (12, 12))
    #im = ax.imshow(STD_space[0:50,0:100], cmap = cm.Spectral)
    #im = ax.contourf(np.arange(0,points), np.arange(0,50), STD_space[::-1], cmap = cm.Spectral_r, levels = np.arange(0,4.1,0.2)) # in y direction
    im = ax.contourf(np.arange(0,points), INCA_grid_payerne, STD_temp_space[::-1], cmap = cm.Spectral_r, levels = np.arange(0,9,0.2)) # in y direction
    #im = ax.contourf(np.arange(0,710),np.arange(0,640), distance_array, cmap =  cm.Spectral_r, levels = np.arange(0, points, 10)) # in y direction
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
    ############################################################ STD ABSOLUT and TOTAL ############################################################
    # COSMO
    ############################################################
    #:::::::::::ABSOLUTE:::::::::::
    ### < temperature >
    COSMO_std_temp = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/COSMO/JJA_2019/scratch/owm/verify/upper-air/JJA_2019/COSMO-1/output_all_stations_6610/allscores.dat', ';')
    
    COSMO_std_temp['altitude_m'] = metpy.calc.pressure_to_height_std(COSMO_std_temp.plevel.values/100 * units.hPa) * 1000
    COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.varno == 2]
    COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.scorename == 'SD']
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
            COSMO_std_temp_d_absolute[:, i,j] = COSMO_std_temp
    

    #:::::::::::TOTAL:::::::::::
    STD_COSMO_temp_total = COSMO_std_temp_absolute 
    STD_COSMO_temp_d_total = COSMO_std_temp_d_absolute
    
    STD_COSMO_temp_total_1 = STD_COSMO_temp_total
    
    
    ############################################################
    # SPACE
    ############################################################  
    # < temperature >
    STD_temp_space_Payerne = np.zeros((n_z, n_y,n_x))
    STD_temp_space = STD_temp_space[::-1]
    for i in range(0, n_y):
        for j in range(0, n_x):
            dist = distance_array[i,j]
            dist_max = np.ceil(dist)
            dist_min = np.floor(dist)
            diff_max = dist_max - dist
            diff_min = 1 - diff_max
            if (dist_max >= 295) or (dist_min >= 295):
                STD_temp_space_Payerne[:, i, j] = np.full((50,), np.nan)
            else: 
                STD_temp_space_1 = (diff_min / (diff_min + diff_max)  * STD_temp_space[:, int(dist_max)]) + (diff_max / (diff_min + diff_max) * STD_temp_space[:, int(dist_min)]) 
                STD_temp_space_Payerne[:, i, j] = STD_temp_space_1
    
    # < temperature d >
    STD_temp_d_space_Payerne = np.zeros((n_z, n_y,n_x))
    STD_temp_d_space = STD_temp_d_space[::-1]
    for i in range(0, n_y):
        for j in range(0, n_x):
            dist = distance_array[i,j]
            dist_max = np.ceil(dist)
            dist_min = np.floor(dist)
            diff_max = dist_max - dist
            diff_min = 1 - diff_max
            if (dist_max >= 295) or (dist_min >= 295):
                STD_temp_d_space_Payerne[:, i, j] = np.full((50,), np.nan)
            else: 
                STD_temp_d_space_1 = (diff_min / (diff_min + diff_max)  * STD_temp_d_space[:, int(dist_max)]) + (diff_max / (diff_min + diff_max) * STD_temp_d_space[:, int(dist_min)]) 
                STD_temp_d_space_Payerne[:, i, j] = STD_temp_d_space_1
                   
    #fig, ax = plt.subplots(figsize = (12, 12))
    #im = ax.imshow(STD_space[0:50,0:100], cmap = cm.Spectral)
    #im = ax.contourf(np.arange(0,points), np.arange(0,50), STD_space[::-1], cmap = cm.Spectral_r, levels = np.arange(0,4.1,0.2)) # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne, STD_temp_d_space_Payerne[:, :, indexes[1,0]], cmap = cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
    #im = ax.contourf(np.arange(0,710) , np.arange(0,640),  STD_temp_d_space_Payerne[35, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
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
    
    ############################################################
    # RALMO
    ############################################################
    #:::::::::::ABSOLUTE:::::::::::
    # < temperature >
    RA_std_temp_1 = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RA_temp_12.csv')
    RA_std_temp_1[RA_std_temp_1 >= 10] = np.nan
    RA_std_temp_1 = RA_std_temp_1.std_temp.values
    STD_RA_temp_absolute = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
           STD_RA_temp_absolute[:, i,j] = RA_std_temp_1
              
    # < temperature d >
    RA_std_temp_d_1 = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RA_temp_d_12.csv')
    RA_std_temp_d_1 = RA_std_temp_d_1.std_temp_d.values
    STD_RA_temp_d_absolute = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
           STD_RA_temp_d_absolute[:, i,j] = RA_std_temp_d_1
                       
    #:::::::::::TOTAL:::::::::::
    STD_RA_temp_total = np.sqrt((STD_RA_temp_absolute)**2 + (STD_temp_space_Payerne)**2)
    STD_RA_temp_d_total = np.sqrt((STD_RA_temp_d_absolute)**2 + (STD_temp_d_space_Payerne)**2)
    STD_RA_temp_total_1 = STD_RA_temp_total
    
    #fig, ax = plt.subplots(figsize = (12, 12))
    #m = ax.imshow(STD_space[0:50,0:100], cmap = cm.Spectral)
    #m = ax.contourf(np.arange(0,points), np.arange(0,50), STD_space[::-1], cmap = cm.Spectral_r, levels = np.arange(0,4.1,0.2)) # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  STD_RA_temp_d_absolute[:, :, indexes[1,0]], cmap = cm.Spectral_r, levels  = np.arange(0,11, 0.2)) # in y direction
    #im = ax.contourf(np.arange(0,710) , np.arange(0,640),   STD_RA_temp_d_total[25, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
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
    ############################################################
    # Radiometer
    ############################################################
    #:::::::::::ABSOLUTE:::::::::::
    # < temperature >
    RM_std_temp_1 = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RM_temp_12.csv')
    RM_std_temp_1 = RM_std_temp_1.std_temp.values
    STD_RM_temp_absolute = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
           STD_RM_temp_absolute[:, i,j] = RM_std_temp_1
    
    # < temperature d >
    RM_std_temp_d_1 = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RM_temp_d_12.csv')
    RM_std_temp_d_1 = RM_std_temp_d_1.std_temp_d.values
    STD_RM_temp_d_absolute = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
           STD_RM_temp_d_absolute[:, i,j] = RM_std_temp_d_1                

    #:::::::::::TOTAL:::::::::::
    STD_RM_temp_total = np.sqrt((STD_RM_temp_absolute)**2 + (STD_temp_space_Payerne)**2) 
    STD_RM_temp_d_total = np.sqrt((STD_RM_temp_d_absolute)**2 + (STD_temp_d_space_Payerne)**2)
    STD_RM_temp_total_1 = STD_RM_temp_total
                
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
      
    ############################################################ DEFINE WEIGHTS ############################################################ 
    STD_RA_temp_total_1[np.isnan(STD_RA_temp_total_1)] = 0
    STD_RM_temp_total_1[np.isnan(STD_RM_temp_total_1)] = 0
    STD_COSMO_temp_total_1[np.isnan(STD_COSMO_temp_total_1)] = 0
    STD_total = STD_RA_temp_total_1 + STD_COSMO_temp_total_1 + STD_RM_temp_total_1
    ############################################################
    # RALMO
    ############################################################   
    WEIGHT_RA_temp = STD_RA_temp_total / STD_total
    WEIGHT_RA_temp[WEIGHT_RA_temp == np.nan] = 0
    #WEIGHT_RA_temp = 1/ STD_RA_temp_total
    ############################################################
    # COSMO
    ############################################################    
    WEIGHT_COSMO_temp = STD_COSMO_temp_total / STD_total
    WEIGHT_COSMO_temp[WEIGHT_COSMO_temp == np.nan] = 0
    #WEIGHT_COSMO_temp = 1/ STD_COSMO_temp_total
    ############################################################
    # RADIOMETER
    ############################################################   
    WEIGHT_RM_temp = STD_RM_temp_total / STD_total
    WEIGHT_RM_temp[WEIGHT_RM_temp == np.nan] = 0
    #WEIGHT_RM_temp = 1/ STD_RM_temp_total
    WEIGHT_COSMO_temp[WEIGHT_COSMO_temp == np.inf] = 1

    #fig, ax = plt.subplots(figsize = (12, 12))
    #cmap = cm.Spectral_r
    #cmap.set_bad('red',0)
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  STD_RM_temp_total[:, :, indexes[1,0]], cmap = cm.Spectral_r) # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  STD_COSMO_temp_total[:, :, indexes[1,0]], cmap = cm.YlGn, levels = np.arange(0,8,0.2)) # in y directio
    #im = ax.contourf(np.arange(0,710) , np.arange(0,640),  WEIGHT_RA_temp[15, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,1)) # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,WEIGHT_RM_temp[:, :, indexes[1,0]], cmap = cmap, levels = [0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # in y direction
    #im = ax.contourf(np.arange(0,710) , np.arange(0,640),  WEIGHT_RM_temp[25, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  WEIGHT_COSMO_temp[:, :, indexes[1,0]], cmap = cm.Spectral_r, levels  = np.arange(0,1)) # in y direction
    #im = ax.contourf(np.arange(0,710) , np.arange(0,640),  WEIGHT_COSMO_temp[25, :, :], cmap =  cm.Spectral_r, levels  = np.arange(0,9, 0.2)) # in y direction
    #ax.scatter(indexes[1,0], indexes[0,0], color = 'black')
    #ax.set_xlabel('Distance [# grid points]', fontsize = 20)
    #ax.set_ylabel('Altitude [m]', fontsize = 20)
    #plt.xticks(fontsize = 20)
    #plt.yticks(fontsize = 20)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.2)
    #cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
    #cbar.set_label(label='Weight', size = 20)
    #cbar.ax.tick_params(labelsize=20)

    T_RA[np.isnan(T_RA)] = 0
    T_RM[np.isnan(T_RM)] = 0
    ############################################################ COMBINE DATASETS ############################################################
    temperature_profile = (WEIGHT_COSMO_temp * T_COSMO) + (WEIGHT_RA_temp * T_RA) + (WEIGHT_RM_temp * T_RM)
    #temperature_d_profile = (WEIGHT_COSMO_temp_d * T_d_COSMO) + (WEIGHT_RA_temp_d * T_d_RA) + (WEIGHT_RM_temp_d * T_d_RM)
    temperature_profile_payerne_temp = temperature_profile[:, indexes[0,0], indexes[1,0]]
    #temperature_profile_payerne_temp_d = temperature_d_profile[:, indexes[0,0], indexes[1,0]]
    
    fig, ax = plt.subplots(figsize = (12, 12))
    cmap = cm.Spectral_r
    cmap.set_bad('red',0)
    #im = ax.contourf(np.arange(0, 640), INCA_grid_payerne,  T_COSMO[:, :, indexes[1,0]], cmap = cm.coolwarm, levels = np.arange(-15,35,5)) # in y direction
    im = ax.contourf(np.arange(0,710) , np.arange(0,640),  temperature_profile[15, :, :], cmap =  cm.Spectral_r, levels = np.arange(-15,35,5)) # in y direction   
    ax.scatter(indexes[1,0], indexes[0,0], color = 'black')
    ax.set_xlabel('Distance [# grid points]', fontsize = 20)
    ax.set_ylabel('Altitude [m]', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
    cbar.set_label(label='Weight', size = 20)
    cbar.ax.tick_params(labelsize=20)
    
    T_RA[T_RA == 0] = np.nan
    T_RM[T_RM == 0] = np.nan
    
    T_RM[T_RM == 0] = np.nan
    T_d_RM[T_d_RM == 0] = np.nan
    T_RA[T_RA == 0] = np.nan
    T_d_RA[T_d_RA == 0] = np.nan
    T_COSMO[T_COSMO == 0] = np.nan
    T_d_COSMO[T_d_COSMO == 0] = np.nan

    number = RS_averaged.groupby('altitude_m')['temperature_mean'].count()

    DIFF_COMBINED_temp = RS_averaged.temperature_mean - temperature_profile_payerne_temp
    DIFF_COMBINED_temp_d = RS_averaged.temperature_d_mean - temperature_profile_payerne_temp
    DIFF_RA_temp = RS_averaged.temperature_mean - (T_RA[:, indexes[0,0], indexes[1,0]])
    DIFF_RA_temp_d = RS_averaged.temperature_d_mean - T_d_RA[:, indexes[0,0], indexes[1,0]]
    DIFF_COSMO_temp = RS_averaged.temperature_mean - T_COSMO[:, indexes[0,0], indexes[1,0]]
    #DIFF_COSMO_temp_d = RS_averaged.temperature_d_mean - T_d_COSMO[:, indexes[0,0], indexes[1,0]][::-1]
    DIFF_RM_temp = RS_averaged.temperature_mean - T_RM[:, indexes[0,0], indexes[1,0]]
    DIFF_RM_temp_d = RS_averaged.temperature_d_mean - T_d_RM[:, indexes[0,0], indexes[1,0]]
    
    DIFF_COMBINED_MD_temp = DIFF_COMBINED_temp.append(DIFF_COMBINED_temp)
    DIFF_COMBINED_MD_temp_d = DIFF_COMBINED_temp.append(DIFF_COMBINED_temp_d)
    DIFF_RA_MD_temp = DIFF_RA_temp.append(DIFF_RA_temp)
    DIFF_RA_MD_temp_d = DIFF_RA_temp.append(DIFF_RA_temp_d)
    DIFF_COSMO_MD_temp = DIFF_COSMO_temp.append(DIFF_COSMO_temp)
   #DIFF_COSMO_MD_temp_d = DIFF_COSMO_temp_d.append(DIFF_COSMO_temp_d)
    DIFF_RM_MD_temp = DIFF_RM_temp.append(DIFF_RM_temp)
    DIFF_RM_MD_temp_d = DIFF_RM_temp_d.append(DIFF_RM_temp_d)
    
    STD_COMBINED_temp = np.sqrt((DIFF_COMBINED_temp)**2 / number.values)
    STD_COMBINED_temp_d = np.sqrt((DIFF_COMBINED_temp_d)**2 / number.values)
    STD_RA_temp = np.sqrt((DIFF_RA_temp)**2 / number.values)   
    STD_RA_temp_d = np.sqrt((DIFF_RA_temp_d)**2 / number.values)
    STD_COSMO_temp = np.sqrt((DIFF_COSMO_temp)**2 / number.values)
    #STD_COSMO_temp_d = np.sqrt((DIFF_COSMO_temp_d)**2 / number.values)
    STD_RM_temp = np.sqrt((DIFF_RM_temp)**2 / number.values)
    STD_RM_temp_d = np.sqrt((DIFF_RM_temp_d)**2 / number.values)
    
    STD_COMBINED_MD_temp = STD_COMBINED_temp.append(STD_COMBINED_temp)
    STD_COMBINED_MD_temp_d = STD_COMBINED_temp_d.append(STD_COMBINED_temp_d)
    STD_RA_MD_temp = STD_RA_temp.append(STD_RA_temp)
    STD_RA_MD_temp_d = STD_RA_temp_d.append(STD_RA_temp_d)
    STD_COSMO_MD_temp = STD_COSMO_temp.append(STD_COSMO_temp)
    #STD_COSMO_MD_temp_d = STD_COSMO_temp_d.append(STD_COSMO_temp_d)
    STD_RM_MD_temp = STD_RM_temp.append(STD_RM_temp)
    STD_RM_MD_temp_d = STD_RM_temp_d.append(STD_RM_temp_d)
    
    firstobj = firstobj + dt.timedelta(days=1)
    
############################################################ PLOT ############################################################
fig, ax = plt.subplots(figsize = (5, 12))
ax.plot(T_RA[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, color = 'black', label = 'RALMO', linewidth = 3)
ax.plot(T_RM[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, color = 'orange', label = 'Radiometer', linewidth = 3)
ax.plot(T_COSMO[:,indexes[0,0],indexes[1,0]], INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 3)

ax.plot(RS_data.temperature_degC, RS_data.altitude_m, color = 'blue', label = 'Radiosonde', linewidth = 3)
    # combined
ax.plot(temperature_profile_payerne_temp, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 5, zorder = 0)
ax.set_ylim(0,3000)
ax.set_xlim(-20,30)
ax.legend()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

#:::::::::::HORIZONTAL:::::::::::
fig, ax = plt.subplots(figsize = (12, 12))  
#ax.contourf(np.arange(0,710),np.arange(0,640), temperature_profile[10,:,:], cmap = cm.coolwarm) # COMBINED
#ax.contourf(np.arange(0,710),np.arange(0,640), T_COSMO[10,:,:], cmap = cm.coolwarm) # COSMO
#ax.contourf(np.arange(0,710),np.arange(0,640), T_RA[4,:,:], cmap = cm.coolwarm) # RA
#ax.scatter(indexes[0,0], indexes[1,0], temperature_profile_payerne, color = 'black')
ax.tick_params(labelsize = 20, color = 'darkslategrey')
ax.grid(color = 'darkslategrey', linewidth = 1)
    


#:::::::::::DIFFERENZ::::::
DIFF_COMBINED = DIFF_COMBINED_MD_temp.groupby(DIFF_COMBINED_MD_temp.index).mean()
DIFF_RA = DIFF_RA_MD_temp.groupby(DIFF_RA_MD_temp.index).mean()
DIFF_COSMO = DIFF_COSMO_MD_temp.groupby(DIFF_COSMO_MD_temp.index).mean()
DIFF_RM = DIFF_RM_MD_temp.groupby(DIFF_RM_MD_temp.index).mean()
    
#DIFF_COMBINED = RS_averaged.temperature_mean - temperature_profile_payerne[::-1]
#DIFF_RA = RS_averaged.temperature_mean - T_RA[:, indexes[0,0], indexes[1,0]][::-1]
#DIFF_COSMO = RS_averaged.temperature_mean - T_COSMO[:, indexes[0,0], indexes[1,0]][::-1]

fig, ax = plt.subplots(figsize = (5, 12))  
ax.plot(DIFF_COMBINED, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
ax.plot(DIFF_RA, INCA_grid_payerne, color = 'orange', label = 'RALMO', linewidth = 2)
ax.plot(DIFF_COSMO, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
ax.plot(DIFF_RM, INCA_grid_payerne, color = 'navy', label = 'Radiometer', linewidth = 2)
ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
ax.set_xlabel('bias [K]', size = 20)
ax.set_ylabel('altitude [m]', size = 20)
ax.set_xlim(-10,10)
ax.set_xticks(np.arange(-10,11,2))
ax.tick_params(labelsize = 20)
ax.set_yticks(np.arange(0,13000,1000))
ax.grid()
ax.legend()

#:::::::::::STD:::::::::::
number = RS_averaged.groupby('altitude_m')['temperature_mean'].count()
STD_COMBINED = np.sqrt((DIFF_COMBINED_temp)**2 / number.values)
STD_RA = np.sqrt((DIFF_RA_temp)**2 / number.values)   
STD_COSMO = np.sqrt((DIFF_COSMO_temp)**2 / number.values)
STD_RM = np.sqrt((DIFF_RM_temp)**2 / number.values)

STD_COMBINED = STD_COMBINED_MD_temp.groupby(STD_COMBINED_MD_temp.index).mean()
STD_RA = STD_RA_MD_temp.groupby(STD_RA_MD_temp.index).mean()
STD_COSMO = STD_COSMO_MD_temp.groupby(STD_COSMO_MD_temp.index).mean()
STD_RM = STD_RM_MD_temp.groupby(STD_RM_MD_temp.index).mean()

fig, ax = plt.subplots(figsize = (5, 12))  
ax.plot(STD_COMBINED, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
ax.plot(STD_RA, INCA_grid_payerne, color = 'orange', label = 'RALMO', linewidth = 2)
ax.plot(STD_COSMO, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
ax.plot(STD_RM, INCA_grid_payerne, color = 'navy', label = 'Radiometer', linewidth = 2)
ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
ax.set_xlabel('std [K]', size = 20)
ax.set_ylabel('altitude [m]', size = 20)
ax.set_xlim(0,10)
#ax.set_ylim(0,1500)
ax.set_xticks(np.arange(0,10,1))
ax.tick_params(labelsize = 20)
ax.set_yticks(np.arange(0,13000,1000))
ax.grid()
ax.legend()
ax.fill_betweenx( INCA_grid_payerne, STD_COSMO_temp, STD_COMBINED_temp, color= 'gold', where = STD_COSMO_temp > STD_COMBINED_temp, interpolate = True, alpha = 1)
ax.fill_betweenx( INCA_grid_payerne, STD_COSMO_temp, STD_COMBINED_temp, color= 'grey', where = STD_COSMO_temp < STD_COMBINED_temp,interpolate = True, alpph= 1)























                           
