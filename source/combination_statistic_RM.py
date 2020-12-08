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
#os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

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

def expand_in_space(data, n_z, n_y, n_x):
    data_array = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            data_array[:, i,j] = data
    return data_array

def read_RALMO(firstobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&measCatNr=2&delimiter=comma&dataSourceId=38&parameterIds=4919,4906,4907,3147,4908,4909,4910,4911,4912,4913,4914,4915&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&profTypeIds=1104&obsTypeIds=30+'
    RA = pd.read_csv(url, skiprows = [1], sep=',')
    RA = RA.rename(columns = {'termin':'time_YMDHMS', 'level':'altitude_m', '4919': 'specific_humidity_gkg-1', '4906':'uncertainty_specific_humidity_gkg-1', '4907':'vertical_resolution_specific_humidity_m', '3147':'temperature_K', '4908':'uncertainty_temperature_K', '4909': 'vertical_resolution_temperature', '4910':'normalised_backscatter', '4911':'uncertainty_backscatter', '4912': 'vert_resolution_backscatter', '4913': 'aerosol_dispersion_rate', '4914': 'uncertainty_dispersion_rate', '4915' : 'vertical_resolution_aerosol_dispersion_rate'})

    # < temperature >
    RA['temperature_K'][RA['temperature_K']== int(10000000)] = np.nan
    RA['temperature_degC'] = RA.temperature_K - 273.15
       
    # < temperature d >
    pressure = metpy.calc.height_to_pressure_std(RA.altitude_m.values * units.meters)
    dewpoint_degC = cc.dewpoint_from_specific_humidity(RA['specific_humidity_gkg-1'].values * units('g/kg'), (RA.temperature_K.values) * units.kelvin, pressure)
    RA.insert(value=dewpoint_degC,column = "dew_point_degC", loc=11)
    RA['dew_point_degC'][RA['specific_humidity_gkg-1']== int(10000000)] = np.nan
    
    RA['time_YMDHMS'] = pd.to_datetime(RA.time_YMDHMS, format = '%Y%m%d%H%M%S')
    
    return RA

def read_HATPRO(firstobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&delimiter=comma&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=31'
    RM = pd.read_csv(url, skiprows = [1], sep=',')
    RM = RM.rename(columns = {'termin' : 'time_YMDHMS' , '3147' : 'temperature_K', '3148' : 'absolute_humidity_gm3', 'level' : 'altitude_m'})
    # < temperature >
    RM['temperature_degC'] = RM.temperature_K - 273.15
    
    p_w = ((RM.temperature_K * RM.absolute_humidity_gm3) / 2.16679)
    RM['dew_point_degC'] = metpy.calc.dewpoint((p_w.values * units.Pa))
    
    RM['time_YMDHMS'] = pd.to_datetime(RM.time_YMDHMS, format = '%Y%m%d%H%M%S')
    
    return RM

def read_HATPRO(firstobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&delimiter=comma&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=31'
    RM = pd.read_csv(url, skiprows = [1], sep=',') 
    RM = RM.rename(columns = {'termin' : 'time_YMDHMS' , '3147' : 'temperature_K', '3148' : 'absolute_humidity_gm3', 'level' : 'altitude_m'})
    # < temperature >
    RM['temperature_degC'] = RM.temperature_K - 273.15
        
    p_w = ((RM.temperature_K * RM.absolute_humidity_gm3) / 2.16679)
    RM['dew_point_degC'] = metpy.calc.dewpoint((p_w.values * units.Pa))
        
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/integral/wmo_ind?locationIds=06610&measCatNr=1&dataSourceId=38&parameterIds=3150,5560,5561&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&delimiter=comma&obsTypeIds=31'
    RM_quality_flag = pd.read_csv(url, skiprows = [1], sep=',')
    RM['quality_flag'] = np.nan * len(RM)
    RM['radiometer_quality_flag_temperature_profile'] = np.nan * len(RM)
    RM['radiometer_quality_flag_humidity_profile'] = np.nan * len(RM)

            
    RM['time_YMDHMS'] = pd.to_datetime(RM.time_YMDHMS, format = '%Y%m%d%H%M%S')                                         
    return RM


def calculate_distance_from_onepoint(n_x, n_y, indexes):
    distance_array = np.zeros((n_y,n_x))
    for i in range(n_y):
        for j in range(n_x):
            distance_array[i,j] = np.sqrt((i-indexes[0,0])**2 + (j-indexes[1,0])**2)
    return distance_array

def calculate_STD_with_distance(points, n_x, n_y, n_z, data):
    STD_temp_space=np.zeros((n_z,(points+1)))
    for j in range(0, (points-1)):
        for k in range(0, (n_z-1)):
            std_x = np.sqrt(((data[k,0:(n_y-j),:] - data[k,j:(n_y),:])**2)/2)
            std_y = np.sqrt(((data[k,:,0:(n_x-j)] - data[k,:,j:(n_x)])**2)/2)
            STD_temp_space[k,j] = np.mean(0.5 * (std_x[:, j:(n_x)] + std_y[j:(n_y),:]))
    return STD_temp_space

def read_radiosonde(firstobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&dataSourceId=34&verbose=position&delimiter=comma&parameterIds=744,745,746,742,748,743,747&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=22'
    RS_data = pd.read_csv(url, skiprows = [1], sep=',')
    RS_data = RS_data.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent', '742':'altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })
    RS_data = RS_data[RS_data['temperature_degC'] != 1e+07]
    RS_data['time_YMDHMS'] = pd.to_datetime(RS_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    return RS_data

def read_SMN(firstobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/surface/wmo_ind?locationIds=06610&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&parameterIds=90,91,98&delimiter=comma'
    SMN_data = pd.read_csv(url, skiprows = [1], sep=',')
    SMN_data = SMN_data.rename(columns = {'termin' : 'time_YMDHMS', '90':'pressure_hPa', '91': 'temperature_degC', '98':'relative_humidity_percent'})
    SMN_data['time_YMDHMS'] = pd.to_datetime(SMN_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    return SMN_data

def plot_in_latlon_dir(data, levels, xlabel, ylabel, cbarlabel, cmap):
    fig, ax = plt.subplots(figsize = (12, 12))
    im = ax.contourf(np.arange(0,710),np.arange(0,640), data, cmap =  cmap, levels = levels) # in y direction
    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
    cbar.set_label(label=cbarlabel, size = 20)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_xticklabels([np.arange(0,4.1,0.2)])
    ax.scatter(indexes[1,0], indexes[0,0] , color = 'black')
    
def plot_in_lat_dir(data, levels, xlabel, ylabel, cbarlabel, cmap, points):
    fig, ax = plt.subplots(figsize = (12, 12))
    im = ax.contourf(np.arange(0,points), INCA_grid_payerne.values, data, cmap =  cmap, levels = levels) # in y direction
    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
    cbar.set_label(label=cbarlabel, size = 20)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_xticklabels([np.arange(0,4.1,0.2)])
    ax.axvline(indexes[0,0], color = 'black')
    #ax.axvline(indexes_NUCAPS[0,0],color = 'purple')
    
def plot_in_lon_dir(data, levels, xlabel, ylabel, cbarlabel, cmap, points):
    fig, ax = plt.subplots(figsize = (12, 12))
    im = ax.contourf(np.arange(0,points), INCA_grid_payerne.values, data, cmap =  cmap, levels = levels) # in y direction
    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
    cbar.set_label(label=cbarlabel, size = 20)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_xticklabels([np.arange(0,4.1,0.2)])
    ax.scatter(indexes[1,0], 0 , color = 'black')
    
def std_from_point(data, distance_array):
    STD_temp_space_point = np.zeros((n_z, n_y,n_x))
    STD_temp_space = data[::-1]
    for i in range(0, n_y):
        for j in range(0, n_x):
            dist = distance_array[i,j]
            dist_max = np.ceil(dist)
            dist_min = np.floor(dist)
            diff_max = dist_max - dist
            diff_min = 1 - diff_max
            if (dist_max >= 345) or (dist_min >= 345):
                STD_temp_space_point[:, i, j] = np.full((50,), np.nan)
            else: 
                data_1 = (diff_min / (diff_min + diff_max)  * data[:, int(dist_max)]) + (diff_max / (diff_min + diff_max) * data[:, int(dist_min)]) 
                STD_temp_space_point[:, i, j] = data_1
    return STD_temp_space_point

def plot_profile(T_COMBINED, T_COSMO, T_RM): 
    fig, ax = plt.subplots(figsize = (5, 12))
    ax.plot(T_COMBINED[:,indexes[0,0],indexes[1,0]], INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 5, zorder = 0)
    ax.plot(T_COSMO[:,indexes[0,0],indexes[1,0]], INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 3)
    ax.plot(T_RM[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, color = 'navy', label = 'HATPRO', linewidth = 3)
    #ax.plot(T_NUCAPS[:, indexes[0,0], indexes[1,0]], INCA_grid_payerne, color = 'purple', label = 'NUCPAS', linewidth = 3)
    ax.legend(fontsize = 20)
    plt.xticks(np.arange(-50, 30, 20), fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.set_xlabel('Temperature [K]', fontsize = 20)
    ax.set_ylabel('Altitude [m]', fontsize = 20)

    ax.legend(fontsize = 20)
    
def plot_diff(DIFF_COMBINED, DIFF_COSMO, DIFF_RM,  INCA_grid_payerne):
    fig, ax = plt.subplots(figsize = (5, 12))  
    ax.plot(DIFF_COMBINED, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
    ax.plot(DIFF_COSMO, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
    ax.plot(DIFF_RM, INCA_grid_payerne, color = 'navy', label = 'HATPRO', linewidth = 2)
    #ax.plot(DIFF_NUCAPS_temp, INCA_grid_payerne, color = 'purple', label = 'NUCAPS', linewidth = 2)
    ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
    ax.set_xlabel('Difference [K]', size = 20)
    ax.set_ylabel('Altitude [m]', size = 20)
    ax.set_xlim(0,1)
    #ax.set_ylim(0,1500)
    ax.set_xticks(np.arange(-6,7, 2))
    ax.tick_params(labelsize = 20)
    ax.set_yticks(np.arange(0,15000,1000))
    ax.grid()
    ax.legend(fontsize = 20)
  
def plot_diff_save(DIFF_COMBINED, DIFF_COSMO, DIFF_RM,  INCA_grid_payerne, name):
    fig, ax = plt.subplots(figsize = (5, 12))  
    ax.plot(DIFF_COMBINED, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
    ax.plot(DIFF_COSMO, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
    ax.plot(DIFF_RM, INCA_grid_payerne, color = 'navy', label = 'HATPRO', linewidth = 2)
    #ax.plot(DIFF_NUCAPS_temp, INCA_grid_payerne, color = 'purple', label = 'NUCAPS', linewidth = 2)
    ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
    ax.set_xlabel('Difference [K]', size = 20)
    ax.set_ylabel('Altitude [m]', size = 20)
    ax.set_xlim(0,1)
    #ax.set_ylim(0,1500)
    ax.set_xticks(np.arange(-6,7, 2))
    ax.tick_params(labelsize = 20)
    ax.set_yticks(np.arange(0,15000,1000))
    ax.grid()
    ax.legend(fontsize = 20)
    fig.savefig(name)
    
def plot_std(STD_COMBINED, STD_COSMO, STD_RM):
    fig, ax = plt.subplots(figsize = (5, 12))  
    ax.plot(STD_COMBINED, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
    ax.plot(STD_COSMO, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
    ax.plot(STD_RM, INCA_grid_payerne, color = 'slategrey', label = 'HATPRO', linewidth = 2)
    #ax.plot(STD_NUCAPS, INCA_grid_payerne, color = 'purple', label = 'NUCAPS', linewidth = 2)
    ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
    ax.set_xlabel('Std [K]', size = 20)
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
    
def plot_std_save(STD_COMBINED, STD_COSMO, STD_RM, name):
    fig, ax = plt.subplots(figsize = (5, 12))  
    ax.plot(STD_COMBINED, INCA_grid_payerne, color = 'red', label = 'combined', linewidth = 2)
    ax.plot(STD_COSMO, INCA_grid_payerne, color = 'green', label = 'COSMO', linewidth = 2)
    ax.plot(STD_RM, INCA_grid_payerne, color = 'slategrey', label = 'HATPRO', linewidth = 2)
    #ax.plot(STD_NUCAPS, INCA_grid_payerne, color = 'purple', label = 'NUCAPS', linewidth = 2)
    ax.axvline(x=0, linewidth = 2, color = 'dimgrey', linestyle = '--')
    ax.set_xlabel('Std [K]', size = 20)
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
    fig.savefig(name)

def open_NUCAPS_file(NUCAPS_file):       
    ds = xr.open_dataset(NUCAPS_file, decode_times=False)  # time units are non-standard, so we dont decode them here 
    units, reference_date = ds.Time.attrs['units'].split(' since ')

    if units=='msec':
        ref_date = datetime.strptime(reference_date,"%Y-%m-%dT%H:%M:%SZ") # usually '1970-01-01T00:00:00Z'
        ds['datetime'] = [ -1 if np.isnan(t) else ref_date + timedelta(milliseconds=t) for t in ds.Time.data]
    return ds

def reshape_NUCAPS(NUCAPS_data, distance):
    p_NUCAPS = NUCAPS_data.Pressure.values
    p_NUCAPS = pd.DataFrame({'pressure_hPa': (np.tile(p_NUCAPS, 730))})
    p_NUCAPS_1 = p_NUCAPS['pressure_hPa'].values * units.hPa
    
    T_NUCAPS = NUCAPS_data.Temperature.values -273.15
    T_NUCAPS = pd.DataFrame({'temperature_degC' : np.reshape(T_NUCAPS, (T_NUCAPS.shape[0] * T_NUCAPS.shape[1]))})
    T_NUCAPS_1 = (NUCAPS_data.Temperature.values -273.15) * units.degC

    H2O_MR = np.reshape(NUCAPS_data['H2O_MR'].values,(T_NUCAPS.shape[0] * T_NUCAPS.shape[1])) * units('g/kg')
    H2O_MR_1 = NUCAPS_data['H2O_MR'].values
    H2O_MR = pd.DataFrame({'mixing_ratio' : H2O_MR})
    
    p_NUCAPS_2 = NUCAPS_data.Pressure.values * units.hPa
    WVMR = (H2O_MR_1 * 1000) * units('g/kg')
    e_1 = mpcalc.vapor_pressure(p_NUCAPS_2, WVMR)
    T_d = mpcalc.dewpoint(e_1) 
    T_d_NUCAPS = pd.DataFrame({'dew_point_degC' : np.reshape(T_d, (T_NUCAPS.shape[0] * T_NUCAPS.shape[1]))}) 
    
    datetime_NUCAPS = NUCAPS_data.datetime.values
    datetime_NUCAPS = pd.DataFrame({'time_YMDHMS_1': (np.repeat(datetime_NUCAPS, 100))})
        
    # round datime to next RS time (0000 or 1200)
    datetime_NUCAPS_round = pd.DataFrame(datetime_NUCAPS.time_YMDHMS_1.dt.ceil('60 min').values, columns = ['time_YMDHMS'])
    datetime_NUCAPS_round.time_YMDHMS[datetime_NUCAPS_round.time_YMDHMS.dt.hour == 1] = datetime_NUCAPS_round.time_YMDHMS - dt.timedelta(hours=1)
    datetime_NUCAPS_round.time_YMDHMS[datetime_NUCAPS_round.time_YMDHMS.dt.hour == 13] = datetime_NUCAPS_round.time_YMDHMS - dt.timedelta(hours=1)
    
    quality_flag = NUCAPS_data.Quality_Flag.values
    quality_flag = pd.DataFrame({'quality_flag' : (np.repeat(quality_flag, 100))})
    
    surf_pres = NUCAPS_data.Surface_Pressure.values
    surf_pres = pd.DataFrame({'surf_pres': (np.repeat(surf_pres, 100))})
    
    topography = NUCAPS_data.Topography.values
    topography = pd.DataFrame({'topography' : (np.repeat(topography, 100))})
    
    skin_temp = NUCAPS_data.Skin_Temperature
    skin_temp = pd.DataFrame({'skin_temp' : (np.repeat(skin_temp, 100))})
    
    dist = NUCAPS_data.distance.values
    dist = pd.DataFrame({'dist' : (np.repeat(dist, 100))})
    
    Longitude = NUCAPS_data.Longitude.values
    Longitude = pd.DataFrame({'Longitude' : (np.repeat(Longitude, 100))})
    
    Latitude = NUCAPS_data.Latitude.values
    Latitude = pd.DataFrame({'Latitude' : (np.repeat(Latitude, 100))})
    
    NUCAPS_data = pd.concat([p_NUCAPS, T_NUCAPS, T_d_NUCAPS, dist, datetime_NUCAPS, datetime_NUCAPS_round, skin_temp, quality_flag, H2O_MR, surf_pres, topography, Longitude, Latitude], axis = 1)
    NUCAPS_data = NUCAPS_data.dropna()  
    
    NUCAPS_data = NUCAPS_data[NUCAPS_data.dist <= distance]
    NUCAPS_data = NUCAPS_data[NUCAPS_data.pressure_hPa <= NUCAPS_data.surf_pres]
    return NUCAPS_data

def add_altitude_NUCAPS(NUCAPS_data_filtered, firstobj_NUCAPS, lastobj_month):
    NUCAPS_data_filtered['altitude_m'] = 0
       
    altitude_m  = pd.DataFrame()   
    while firstobj_NUCAPS != lastobj_month:
        nowdate = firstobj_NUCAPS.strftime('%Y%m%d')
        NUCAPS_data_time = NUCAPS_data_filtered[(NUCAPS_data_filtered['time_YMDHMS'] == firstobj_NUCAPS)]
        NUCAPS_data_time = NUCAPS_data_time.reset_index(drop=True)
        NUCAPS_data_time['altitude_m'] = '0'
        
        SMN_data_time = read_SMN(firstobj_NUCAPS)
        
        if NUCAPS_data_time.empty:
            firstobj_NUCAPS = firstobj_NUCAPS + dt.timedelta(days=1)
            print('nope')
            
        else:  
            print(nowdate)
            data_comma_temp = NUCAPS_data_time[['time_YMDHMS', 'pressure_hPa', 'temperature_degC', 'altitude_m']]
            data_comma_temp = data_comma_temp.append({'time_YMDHMS' : SMN_data_time.time_YMDHMS.iloc[0], 'pressure_hPa' : SMN_data_time.pressure_hPa[0],  'temperature_degC': SMN_data_time.temperature_degC[0], 'altitude_m' :491},ignore_index=True)
            data_comma_temp = data_comma_temp[::-1].reset_index(drop=True)
            
            for i in range(1,len(data_comma_temp)):
                p_profile = (data_comma_temp.pressure_hPa.iloc[i-1], data_comma_temp.pressure_hPa.iloc[i]) * units.hPa
                t_profile = (data_comma_temp.temperature_degC.iloc[i-1], data_comma_temp.temperature_degC.iloc[i]) * units.degC
                deltax = metpy.calc.thickness_hydrostatic(p_profile, t_profile)
                data_comma_temp.altitude_m[i] = data_comma_temp.altitude_m.iloc[i-1] + (deltax.magnitude)
            
          
            altitude_m_1 = pd.DataFrame(data_comma_temp.altitude_m[1:])
            altitude_m = altitude_m.append(altitude_m_1[::-1])
            firstobj_NUCAPS = firstobj_NUCAPS + dt.timedelta(days=1)     
                                         
    NUCAPS_data_filtered.altitude_m = altitude_m.values
    return NUCAPS_data_filtered

def calculate_std(DIFF_COMBINED_MD):
    DIFF_COMBINED_MD.insert(column = 'number' , value =  DIFF_COMBINED_MD.DIFF.values, loc = 2)
    DIFF_COMBINED_MD.number[~np.isnan(DIFF_COMBINED_MD.number)] = 1
    total_number = DIFF_COMBINED_MD.groupby('altitude_m')['number'].count()
    
    DIFF_COMBINED_MD.DIFF = DIFF_COMBINED_MD.DIFF**2
    SUM_COMBINED_MD = DIFF_COMBINED_MD.groupby('altitude_m')['DIFF'].sum().to_frame(name='STD').reset_index()
    STD_COMBINED_temp = (np.sqrt(SUM_COMBINED_MD.STD)) / total_number.values
    return STD_COMBINED_temp

def read_SMN(firstobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/surface/wmo_ind?locationIds=06610&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&parameterIds=90,91,98&delimiter=comma'
    SMN_data = pd.read_csv(url, skiprows = [1], sep=',')
    SMN_data = SMN_data.rename(columns = {'termin' : 'time_YMDHMS', '90':'pressure_hPa', '91': 'temperature_degC', '98':'relative_humidity_percent'})
    SMN_data['time_YMDHMS'] = pd.to_datetime(SMN_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    return SMN_data

def convert_altitude_to_pressure(input_data, firstobj, integrant):
    p_1 = np.zeros(len(input_data))
    p_1[0] = read_SMN(firstobj).pressure_hPa[0]
    for i in range(1, len(input_data)):
        p_1[i] = p_1[0]*math.exp(-np.trapz(integrant[0:i], input_data.altitude_m[0:i]))
    
    return p_1
    


######################################## !!! ########################################
firstdate = '20200721120000' # !! define start date + midnight/noon
lastdate = '20200728120000' # !! define end date + midnight/noon
firstobj=dt.datetime.strptime(firstdate,'%Y%m%d%H%M%S')
lastobj=dt.datetime.strptime(lastdate,'%Y%m%d%H%M%S')
######################################## !!! ########################################
lon_payerne = 6.93608#, 9.12, 11.33, 9.17
lat_payerne = 46.8220#1, 48.50, 48.15, 45.26

DT = firstobj.hour

INCA_archive = '/data/COALITION2/internships/nal/data/COSMO/'
INCA_COSMO_archive ='/data/COALITION2/database/cosmo/T-TD_3D/'
NUCAPS_archive = '/data/COALITION2/internships/nal/data/NUCAPS/save_NUCAPS'
season_year = 'JJA_2019' 
COSMO_std_archive   = '/data/COALITION2/internships/nal/std_files/COSMO/'+str(season_year)+'/scratch/owm/verify/upper-air/'+str(season_year)+'/COSMO-1/output_all_stations_6610'
archive_plots = '/data/COALITION2/internships/nal/Plots/'

DIFF_COMBINED_MD_temp = pd.DataFrame()
DIFF_COMBINED_MD_temp_d = pd.DataFrame()
DIFF_RM_MD_temp = pd.DataFrame()
DIFF_RM_MD_temp_d = pd.DataFrame()
DIFF_COSMO_MD_temp = pd.DataFrame()
DIFF_COSMO_MD_temp_d = pd.DataFrame()
  
STD_COMBINED_MD_temp = pd.DataFrame()
STD_COMBINED_MD_temp_d = pd.DataFrame()
STD_RM_MD_temp = pd.DataFrame()
STD_RM_MD_temp_d = pd.DataFrame()
STD_COSMO_MD_temp = pd.DataFrame()
STD_COSMO_MD_temp_d = pd.DataFrame()

exp = 1
sigma = 5
factor = 1
while firstobj != lastobj:              
    print(firstobj)
    lastobj_now = firstobj + dt.timedelta(days=1)
    ############################################################ LOAD DATA ############################################################
    ##### INCA grid
    ##########################################
    INCA_grid = xr.open_dataset(INCA_archive+'inca_topo_levels_hsurf_ccs4.nc')
        
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
    ########################################r##
    RS_data = read_radiosonde(firstobj)
        
    RS_averaged = average_RS_to_INCA_grid(firstobj, lastobj_now, INCA_grid_payerne, RS_data, RS_data)
    RS_averaged = RS_averaged[::-1].reset_index(drop=True)
    
    ##########################################
    ##### COSMO
    ##########################################
    time = dt.datetime.strftime(firstobj- dt.timedelta(hours=6), '%Y%m%d%H')
    COSMO_data = xr.open_dataset(INCA_COSMO_archive+'/'+str(firstobj.year)+'/'+str(firstobj.strftime('%m'))+'/'+str(firstobj.strftime('%d'))+'/cosmo1_inca_'+str(time)+'_06.nc')
    #COSMO_data = xr.open_dataset('/data/COALITION2/database/cosmo/T-TD_3D/cosmo-1e_inca_'+str(time)+'_06_00.nc')
    print(COSMO_data.time.values)
    ## define dimensionsr
    n_z = COSMO_data.t_inca.values.shape[1]
    n_y = COSMO_data.t_inca.values.shape[2]
    n_x = COSMO_data.t_inca.values.shape[3]
    
    T_COSMO = COSMO_data.t_inca.values[0,:,:,:][::-1] - 273.15 
    T_d_COSMO = metpy.calc.dewpoint_from_specific_humidity(COSMO_data.qv_inca, COSMO_data.t_inca, COSMO_data.p_inca)[0,:,:,:][::-1].magnitude
    
    ############################################################
    # RADIOMETER
    ############################################################
    RM = read_HATPRO(firstobj)
    
    RM['temperature_degC'][RM.quality_flag == 1] = np.nan
    RM['dew_point_degC'][RM.quality_flag == 1] = np.nan
    
    # smooth to INCA grid
    RM = average_RS_to_INCA_grid(firstobj, lastobj_now, INCA_grid_payerne, RM, RM)[::-1]
    
    # expand in space
    T_RM = expand_in_space(RM.temperature_mean, n_z, n_y, n_x)
    T_d_RM = expand_in_space(RM.temperature_d_mean, n_z, n_y, n_x) 
                
    ##########################################
    ##### RALMO
    ##########################################
    # read RALMO 
    #RA = read_RALMO(firstobj)
        
    # smooth to INCA grid
    #RA = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid_payerne, RA, RA)[::-1]
    
     # expand in space
    #T_RA = expand_in_space(RA.temperature_mean, n_z, n_y, n_x)
    #T_d_RA = expand_in_space(RA.temperature_d_mean, n_z, n_y, n_x) 
    
    ##########################################
    ##### NUCAPS
    ##########################################
    # read NUCAPS
    NUCAPS = open_NUCAPS_file(NUCAPS_archive+'/NUCAPS_Payerne_-120min_60min_3500km_Aug2020.nc')
    NUCAPS = reshape_NUCAPS(NUCAPS, 50)
    NUCAPS = NUCAPS[NUCAPS.time_YMDHMS == firstobj]
    
    #lon = INCA_grid.lon_1.values
    #lat = INCA_grid.lat_1.values
    #lonlat = np.dstack([lat.ravel(), lon.ravel()])[0,:,:]
    #tree = spatial.KDTree(lonlat)
    #coordinates = tree.query([([NUCAPS.Latitude.iloc[0], NUCAPS.Longitude.iloc[0] ])])
    #coords_close = lonlat[coordinates[1]]
    #indexes_NUCAPS = np.array(np.where(INCA_grid.lon_1 == coords_close[0,1]))
    
    # add altitude
    #SMN_data = read_SMN(firstobj)
    #NUCAPS = add_altitude_NUCAPS(NUCAPS, firstobj, lastobj_now) 
    
    # smooth to INCA grid
    #NUCAPS = average_RS_to_INCA_grid(firstobj, lastobj_now, INCA_grid_payerne, NUCAPS, NUCAPS)[::-1]
    
    # expand in space
    #T_NUCAPS = expand_in_space(NUCAPS.temperature_mean, n_z, n_y, n_x)
    #T_d_NUCAPS = expand_in_space(NUCAPS.temperature_d_mean, n_z, n_y, n_x) 
    
    
    ############################################################ STD SPACE ############################################################
    # calculate error with distance
    ## from Payerne
    STD_temp_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_COSMO)
    STD_temp_d_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_d_COSMO)
    # plot
    #plot_in_lat_dir(STD_temp_space, np.arange(0,9,0.2), 'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 345)
    #plot_in_lat_dir(STD_temp_d_space, np.arange(0,9,0.2), 'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 345)
            
    ############################################################ STD ABSOLUT and TOTAL ############################################################
    # COSMO
    ############################################################
    ### < temperature >
    COSMO_std_temp = pd.read_csv(COSMO_std_archive+'/allscores.dat', ';')   
    COSMO_std_temp['altitude_m'] = metpy.calc.pressure_to_height_std(COSMO_std_temp.plevel.values/100 * units.hPa) * 1000
    COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.varno == 2]
    COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.scorename == 'SD']
    COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.leadtime == 6][0:20]
    COSMO_std_temp['plevel'] = COSMO_std_temp['plevel']
    COSMO_std_temp = griddata(COSMO_std_temp.altitude_m.values, COSMO_std_temp.scores.values, (INCA_grid_payerne.values))
    
    COSMO_std_temp_absolute = expand_in_space(COSMO_std_temp, n_z, n_y, n_x)   
 
    ### < temperature d>
    COSMO_std_temp_d = pd.read_csv(COSMO_std_archive+'/allscores.dat', ';')  
    COSMO_std_temp_d['altitude_m'] = metpy.calc.pressure_to_height_std(COSMO_std_temp_d.plevel.values/100 * units.hPa) * 1000
    COSMO_std_temp_d = COSMO_std_temp_d[COSMO_std_temp_d.varno == 59]
    COSMO_std_temp_d = COSMO_std_temp_d[COSMO_std_temp_d.scorename == 'SD']
    COSMO_std_temp_d = COSMO_std_temp_d[COSMO_std_temp_d.leadtime == 6][0:20]
    COSMO_std_temp_d['plevel'] = COSMO_std_temp_d['plevel']
    COSMO_std_temp_d = griddata(COSMO_std_temp_d.altitude_m.values, COSMO_std_temp_d.scores.values, (INCA_grid_payerne.values))

    COSMO_std_temp_d_absolute = expand_in_space(COSMO_std_temp_d, n_z, n_y, n_x)    
    
    #:::::::::::TOTAL:::::::::::
    STD_COSMO_temp_total = COSMO_std_temp_absolute 
    STD_COSMO_temp_d_total = COSMO_std_temp_d_absolute      
           
    ############################################################
    # Std in SPACE
    ############################################################
    # calculate std from Payerne
    distance_array_Payerne = calculate_distance_from_onepoint(n_x, n_y, indexes)
    # plot
    #plot_in_latlon_dir(distance_array_Payerne,np.arange(len(distance_array_Payerne), 30), 'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.YlGnBu_r)
    
    STD_temp_space_Payerne = std_from_point(STD_temp_space, distance_array_Payerne)
    STD_temp_d_space_Payerne = std_from_point(STD_temp_d_space, distance_array_Payerne)      
    # plot
    #plot_in_lat_dir(STD_temp_space_Payerne[:, :, indexes[0,0]],  np.arange(0,9,0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    #plot_in_lat_dir(STD_temp_d_space_Payerne[:, :, indexes[0,0]],  np.arange(0,9,0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    
    #plot_in_lon_dir(STD_temp_space_Payerne[:, :, indexes[1,0]],  np.arange(0,9,0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    #plot_in_lon_dir(STD_temp_d_space_Payerne[:, :, indexes[1,0]],  np.arange(0,9,0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    
    # calculate std from NUCAPS point
    #distance_array_Payerne_NUCAPS = calculate_distance_from_onepoint(n_x, n_y, indexes_NUCAPS)
    
    #STD_temp_space_Payerne_NUCAPS = std_from_point(STD_temp_space, distance_array_Payerne_NUCAPS)
    #STD_temp_d_space_Payerne_NUCAPS = std_from_point(STD_temp_d_space, distance_array_Payerne_NUCAPS)    
    # plot
    #plot_in_lat_dir(STD_temp_space_Payerne_NUCAPS[:, :, indexes[0,0]],  np.arange(0,9,0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    #plot_in_lat_dir(STD_temp_d_space_Payerne_NUCAPS[:, :, indexes[0,0]],  np.arange(0,9,0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    
    #plot_in_lon_dir(STD_temp_space_Payerne_NUCAPS[:, :, indexes[1,0]],  np.arange(0,9,0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    #plot_in_lon_dir(STD_temp_d_space_Payerne_NUCAPS[:, :, indexes[1,0]],  np.arange(0,9,0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    
    ############################################################
    # RALMO
    ############################################################
    #:::::::::::ABSOLUTE:::::::::::
    # read data
    RA_std_temp = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RA_temp_'+str(DT)+'.csv')
    RA_std_temp_d = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RA_temp_d_'+str(DT)+'.csv')
    # expand in space 
    STD_RA_temp_absolute = expand_in_space(RA_std_temp.std_temp.values, n_z, n_y, n_x)
    STD_RA_temp_d_absolute = expand_in_space(RA_std_temp_d.std_temp_d.values, n_z, n_y, n_x)
       
    #:::::::::::TOTAL:::::::::::
    STD_RA_temp_total = STD_RA_temp_absolute + STD_temp_space_Payerne
    STD_RA_temp_d_total = STD_RA_temp_d_absolute + STD_temp_d_space_Payerne  
    ## plot
    #plot_in_lat_dir(STD_RA_temp_absolute[:, :, indexes[1,0]], np.arange(0,11, 0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    #plot_in_lat_dir(STD_RA_temp_d_absolute[:, :, indexes[1,0]],  np.arange(0,11, 0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    
    ############################################################
    # HATPRO
    ############################################################
    #:::::::::::ABSOLUTE:::::::::::
    # read data
    RM_std_temp = factor * pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RM_temp_'+str(DT)+'.csv')
    RM_std_temp_d = factor * pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RM_temp_d_'+str(DT)+'.csv')
     # expand in space 
    STD_RM_temp_absolute = expand_in_space(RM_std_temp.std_temp.values, n_z, n_y, n_x)
    STD_RM_temp_d_absolute = expand_in_space(RM_std_temp_d.std_temp_d.values, n_z, n_y, n_x)                  
    
    #:::::::::::TOTAL:::::::::::
    STD_RM_temp_total = STD_RM_temp_absolute + STD_temp_space_Payerne
    STD_RM_temp_d_total = STD_RM_temp_d_absolute + STD_temp_d_space_Payerne 
    ## plot
    #plot_in_lat_dir(STD_RM_temp_absolute[:, :, indexes[1,0]], np.arange(0,11, 0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    #plot_in_lat_dir(STD_RM_temp_d_absolute[:, :, indexes[1,0]],  np.arange(0,11, 0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    
    ############################################################
    # NUCAPS
    ############################################################
    #:::::::::::ABSOLUTE:::::::::::
    # read data
    #NUCAPS_std_temp = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_NUCAPS_temp_'+str(DT)+'.csv')
    #NUCAPS_std_temp_d = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_NUCAPS_temp_d_'+str(DT)+'.csv')
     # expand in space 
    #STD_NUCAPS_temp_absolute = expand_in_space(NUCAPS_std_temp.std_temp.values, n_z, n_y, n_x)
    #STD_NUCAPS_temp_d_absolute = expand_in_space(NUCAPS_std_temp_d.std_temp_d.values, n_z, n_y, n_x)  
    
    #STD_temp_space_Payerne = std_from_point(STD_temp_space, distance_array_Payerne)
    #STD_temp_d_space_Payerne = std_from_point(STD_temp_d_space, distance_array_Payerne)                 
    
    #:::::::::::TOTAL:::::::::::
    #STD_NUCAPS_temp_total = STD_NUCAPS_temp_absolute + STD_temp_space_Payerne_NUCAPS
    #STD_NUCAPS_temp_d_total = STD_NUCAPS_temp_d_absolute + STD_temp_d_space_Payerne_NUCAPS
    ## plot
    #plot_in_lat_dir(STD_NUCAPS_temp_total[:, :, indexes[1,0]], np.arange(0,11, 0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)
    #plot_in_lat_dir(STD_NUCAPS_temp_total[:, :, indexes[1,0]],  np.arange(0,11, 0.2),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)


    ############################################################ DEFINE WEIGHTS ############################################################ 
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
    
    plot_in_lat_dir(a_RADIOMETER_temp[:, :, indexes[1,0]], np.arange(0,1.1, 0.05),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)

    # < temperature d >
    STD_RM_temp_d_total_1 = STD_RM_temp_d_total
    STD_RM_temp_d_total_1[np.isnan(T_d_RM)] = np.nan
    sigma_RADIOMETER = STD_RM_temp_d_total_1**2
    sigma_COSMO = STD_COSMO_temp_d_total**2

    STD_RADIOMETER_COSMO = np.nansum(np.stack((sigma_COSMO, sigma_RADIOMETER)), axis = 0)

    a_COSMO_temp_d = sigma_RADIOMETER/ STD_RADIOMETER_COSMO
    a_COSMO_temp_d[np.isnan(STD_RM_temp_d_total)] = 1
    
    a_RADIOMETER_temp_d = sigma_COSMO / STD_RADIOMETER_COSMO
    a_RADIOMETER_temp_d[np.isnan(STD_RM_temp_d_total)] = 0
    
    plot_in_lat_dir(a_RADIOMETER_temp_d[:, :, indexes[1,0]], np.arange(0,1.1, 0.05),'Distance [# grid points]', 'Altitude [m]', 'STD [K]', cm.Spectral_r, 640)

    
    ############################################################ COMBINE DATASETS ###########A################################################
    T_COMBINED = np.nansum(np.stack(((a_COSMO_temp * T_COSMO) ,  (a_RADIOMETER_temp * T_RM))), axis = 0)
    plot_profile(T_COMBINED,  T_COSMO, T_RM)
     
    T_d_COMBINED = np.nansum(np.stack(((a_COSMO_temp_d * T_d_COSMO) ,  (a_RADIOMETER_temp_d * T_d_RM))), axis = 0)
    plot_profile(T_d_COMBINED,  T_d_COSMO, T_d_RM)
       
    ############################################################ BIAS AND STD ############################################################ 
    # std
    DIFF_COSMO_temp = pd.DataFrame({'DIFF' : (RS_averaged.temperature_mean -  T_COSMO[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})
    DIFF_COSMO_temp_d = pd.DataFrame({'DIFF' : (RS_averaged.temperature_d_mean -  T_d_COSMO[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})
    DIFF_COMBINED_temp = pd.DataFrame({'DIFF' : (RS_averaged.temperature_mean - T_COMBINED[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})
    DIFF_COMBINED_temp_d = pd.DataFrame({'DIFF' : (RS_averaged.temperature_d_mean - T_d_COMBINED[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})
    DIFF_RM_temp = pd.DataFrame({'DIFF' : (RS_averaged.temperature_mean - T_RM[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})
    DIFF_RM_temp_d = pd.DataFrame({'DIFF' : (RS_averaged.temperature_d_mean - T_d_RM[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})

    #DIFF_NUCAPS_temp = pd.DataFrame({'DIFF' : (RS_averaged.temperature_mean - T_NUCAPS[:, indexes[0,0], indexes[1,0]]), 'altitude_m' :INCA_grid_payerne})
    
    plot_diff_save(DIFF_COMBINED_temp.DIFF, DIFF_COSMO_temp.DIFF, DIFF_RM_temp.DIFF, INCA_grid_payerne, archive_plots+'/Plots/DIFF_temp_'+dt.datetime.strftime(firstobj, '%Y%m%d%H')+'.png')
    plot_diff_save(DIFF_COMBINED_temp_d.DIFF, DIFF_COSMO_temp_d.DIFF, DIFF_RM_temp_d.DIFF, INCA_grid_payerne, archive_plots+'/Plots/DIFF_temp_d_'+dt.datetime.strftime(firstobj, '%Y%m%d%H')+'.png')
 
    DIFF_COMBINED_MD_temp = DIFF_COMBINED_MD_temp.append(DIFF_COMBINED_temp)
    DIFF_COMBINED_MD_temp_d = DIFF_COMBINED_MD_temp_d.append(DIFF_COMBINED_temp_d)
    DIFF_RM_MD_temp = DIFF_RM_MD_temp.append(DIFF_RM_temp)
    DIFF_RM_MD_temp_d = DIFF_RM_MD_temp_d.append(DIFF_RM_temp_d)
    DIFF_COSMO_MD_temp = DIFF_COSMO_MD_temp.append(DIFF_COSMO_temp)
    DIFF_COSMO_MD_temp_d = DIFF_COSMO_MD_temp_d.append(DIFF_COSMO_temp_d)
    #DIFF_NUCAPS_MD = DIFF_NUCAPS_MD.append(DIFF_NUCAPS_temp)
    
    # std
    STD_COSMO_temp = pd.DataFrame({'STD' : np.nanstd((RS_averaged.temperature_mean.reset_index(drop=True),  T_COSMO[:, indexes[0,0], indexes[1,0]]), axis = 0), 'altitude_m' : INCA_grid_payerne})
    STD_COSMO_temp_d = pd.DataFrame({'STD' : np.nanstd((RS_averaged.temperature_d_mean.reset_index(drop=True),  T_d_COSMO[:, indexes[0,0], indexes[1,0]]), axis = 0), 'altitude_m' : INCA_grid_payerne})
    STD_COMBINED_temp = pd.DataFrame({'STD' :np.nanstd( (RS_averaged.temperature_mean.reset_index(drop=True),  T_COMBINED[:, indexes[0,0], indexes[1,0]]), axis = 0), 'altitude_m' : INCA_grid_payerne})
    STD_COMBINED_temp_d = pd.DataFrame({'STD' :np.nanstd( (RS_averaged.temperature_d_mean.reset_index(drop=True),  T_d_COMBINED[:, indexes[0,0], indexes[1,0]]), axis = 0), 'altitude_m' : INCA_grid_payerne})
    STD_RM_temp = pd.DataFrame({'STD' :np.nanstd( (RS_averaged.temperature_mean.reset_index(drop=True),  T_RM[:, indexes[0,0], indexes[1,0]]), axis = 0), 'altitude_m' : INCA_grid_payerne})
    STD_RM_temp_d = pd.DataFrame({'STD' :np.nanstd( (RS_averaged.temperature_d_mean.reset_index(drop=True),  T_d_RM[:, indexes[0,0], indexes[1,0]]), axis = 0), 'altitude_m' : INCA_grid_payerne})
    
    plot_std_save(STD_COMBINED_temp.STD.values, STD_COSMO_temp.STD.values,  STD_RM_temp.STD.values, archive_plots+'/Plots/STD_temp_'+str(dt.datetime.strftime(firstobj, '%Y%m%d%H'))+'.png')
    plot_std_save(STD_COMBINED_temp_d.STD.values, STD_COSMO_temp_d.STD.values,  STD_RM_temp_d.STD.values, archive_plots+'/Plots/STD_temp_d'+str(dt.datetime.strftime(firstobj, '%Y%m%d%H'))+'.png')
 
    firstobj = firstobj + dt.timedelta(days=1)
    
#:::::::::::DIFF:::::::::::
DIFF_COMBINED_temp = DIFF_COMBINED_MD_temp.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()
DIFF_COMBINED_temp_d = DIFF_COMBINED_MD_temp_d.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()
DIFF_COSMO_temp = DIFF_COSMO_MD_temp.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()
DIFF_COSMO_temp_d = DIFF_COSMO_MD_temp_d.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()
DIFF_RM_temp = DIFF_RM_MD_temp.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()
DIFF_RM_temp_d = DIFF_RM_MD_temp_d.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()
#DIFF_NUCAPS = DIFF_NUCAPS_MD.groupby('altitude_m')['DIFF'].mean().to_frame(name='mean_all').reset_index()

plot_diff_save(DIFF_COMBINED_temp.mean_all, DIFF_COSMO_temp.mean_all, DIFF_RM_temp.mean_all,  INCA_grid_payerne, archive_plots+'/Plots/DIFF_temp_'+str(firstdate)+'_'+str(lastdate)+'.png')
plot_diff_save(DIFF_COMBINED_temp_d.mean_all, DIFF_COSMO_temp_d.mean_all, DIFF_RM_temp_d.mean_all,  INCA_grid_payerne, archive_plots+'/Plots/DIFF_temp_d'+str(firstdate)+'_'+str(lastdate)+'.png')

#:::::::::::STD:::::::::::
STD_COMBINED_temp = calculate_std(DIFF_COMBINED_MD_temp)
STD_COMBINED_temp_d = calculate_std(DIFF_COMBINED_MD_temp_d)
STD_COSMO_temp = calculate_std(DIFF_COSMO_MD_temp)
STD_COSMO_temp_d = calculate_std(DIFF_COSMO_MD_temp_d)
STD_RM_temp = calculate_std(DIFF_RM_MD_temp)
STD_RM_temp_d = calculate_std(DIFF_RM_MD_temp_d)


plot_std_save(STD_COMBINED_temp, STD_COSMO_temp, STD_RM_temp,archive_plots+'STD_temp_'+str(firstdate)+'_'+str(lastdate)+'.png')
plot_std_save(STD_COMBINED_temp_d, STD_COSMO_temp_d, STD_RM_temp_d, archive_plots+'/Plots/STD_temp_d_'+str(firstdate)+'_'+str(lastdate)+'.png')












                           
