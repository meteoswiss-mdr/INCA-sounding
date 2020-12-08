#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:28:02 2020

@author: nal

####
calculate standard deviation of HATPRO, RALMO and NUCAPS  over a 1-yr interval 
####
"""
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
from scipy import spatial

############################################################################# FUNCTIONS ############################################################################# 
def open_NUCAPS_file(NUCAPS_file):       
    ds = xr.open_dataset(NUCAPS_file, decode_times=False)  # time units are non-standard, so we dont decode them here 
    units, reference_date = ds.Time.attrs['units'].split(' since ')
    if units=='msec':
        ref_date = datetime.strptime(reference_date,"%Y-%m-%dT%H:%M:%SZ") # usually '1970-01-01T00:00:00Z'
        ds['datetime'] = [ -1 if np.isnan(t) else ref_date + timedelta(milliseconds=t) for t in ds.Time.data]
    return ds

def convert_H2O_MR_to_Td(H2O_MR, p): # H2O_MR -> input water vapour mass mixing ration in (mWV / mDA) kg/kg
    WVMR = H2O_MR * 1000 # convert to grams
    WVMR = WVMR * units('g/kg')
    e_1 = mpcalc.vapor_pressure(p_NUCAPS_orig, WVMR)
    T_d = mpcalc.dewpoint(e_1) 
    return T_d

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid_1, input_data_filtered, comparison_grid):
    INCA_grid = INCA_grid_1[::-1].reset_index(drop=True)
    input_grid_smoothed_all = pd.DataFrame()
    while firstobj != lastobj:
        nowdate = firstobj.strftime('%Y%m%d')
        print(nowdate) 
        input_data_time = input_data_filtered[input_data_filtered.time_YMDHMS == firstobj] 
        input_data_time = input_data_time.iloc[::-1].reset_index(drop=True)
        if input_data_time.empty:
            input_data_time = input_data_filtered[input_data_filtered.time_YMDHMS == firstobj - dt.timedelta(days=1)]
            input_data_time = input_data_time.reset_index(drop=True)
            print('now!!')
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
                if window_h_min.values < np.min(input_data_time.altitude_m):
                    aver_mean = pd.DataFrame({'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan, 'altitude_m' : INCA_grid.loc[i], 'distance': np.nan})
                    print('small')
                elif window_h_max.values > np.max(input_data_time.altitude_m):
                    aver_mean = pd.DataFrame({'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan, 'altitude_m' : INCA_grid.loc[i], 'distance': np.nan})
                    print('big')
                elif input_data_within_bound.altitude_m.count() <= 1:
                     aver_mean = pd.DataFrame({'temperature_mean' : griddata(input_data_time.altitude_m.values, input_data_time.temperature_degC.values, INCA_grid.loc[i].values), 'temperature_d_mean' : griddata(input_data_time.altitude_m.values, input_data_time.dew_point_degC.values, INCA_grid.loc[i]),'altitude_m' : INCA_grid.loc[i].values}).reset_index(drop=True)
                     print('interpolate')
                else: 
                    aver_mean = pd.DataFrame({'temperature_mean': np.mean(input_data_within_bound.temperature_degC), 'temperature_d_mean' : np.mean(input_data_within_bound.dew_point_degC), 'altitude_m' : (INCA_grid.iloc[i].values)}, index = [i])
                    print('average')
                input_interp = input_interp.append(aver_mean)
            input_grid_smoothed_all = input_grid_smoothed_all.append(input_interp)
            input_grid_smoothed_all = input_grid_smoothed_all.reset_index(drop=True)
            firstobj= firstobj + dt.timedelta(days=1) 
    return input_grid_smoothed_all   

def subtract_running_window(firstobj, lastobj, NUCAPS_smoothed_INCA_all, NUCAPS_smoothed_INCA_all_time, RS_smoothed_INCA, window_size_days):
    for i in range(len(NUCAPS_smoothed_INCA_all)):
        print(i)
        NUCAPS_smoothed_INCA_all = NUCAPS_smoothed_INCA_all.reset_index(drop=True)
        lastobj_window = NUCAPS_smoothed_INCA_all_time.time_YMDHMS[i] - timedelta(days=1)
        firstobj_window = lastobj_window - timedelta(days=window_size_days)
        altitude_i  =  NUCAPS_smoothed_INCA_all.altitude_m[i]

        NUCAPS_smoothed_INCA_all_window = NUCAPS_smoothed_INCA_all_time[(NUCAPS_smoothed_INCA_all_time.time_YMDHMS.dt.strftime('%m%d%H%M%S') >= firstobj_window.strftime('%m%d%H%M%S')) & (NUCAPS_smoothed_INCA_all_time.time_YMDHMS.dt.strftime('%m%d%H%M%S') <= lastobj_window.strftime('%m%d%H%M%S')) & (NUCAPS_smoothed_INCA_all_time.altitude_m == altitude_i)]
        RS_smoothed_INCA_window = RS_smoothed_INCA[(RS_smoothed_INCA.time_YMDHMS.dt.strftime('%m%d%H%M%S').isin(NUCAPS_smoothed_INCA_all_window.time_YMDHMS.dt.strftime('%m%d%H%M%S')))& (RS_smoothed_INCA.altitude_m == altitude_i)]
        NUCAPS_smoothed_INCA_all_window = NUCAPS_smoothed_INCA_all_window[['altitude_m', 'temperature_mean', 'temperature_d_mean']]
        
        if NUCAPS_smoothed_INCA_all_window.empty: 
            print('yes')
            bias_t = 0
            bias_t_d = 0
        else:
            bias_t = np.subtract(NUCAPS_smoothed_INCA_all_window.temperature_mean.reset_index(drop=True), RS_smoothed_INCA_window.temperature_mean.reset_index(drop=True))
            bias_t = pd.DataFrame({'diff_temp':bias_t.reset_index(drop=True), 'altitude_m':NUCAPS_smoothed_INCA_all_window.altitude_m.reset_index(drop=True)})  
            bias_t = bias_t.astype(float)
            bias_t = bias_t.groupby('altitude_m')['diff_temp'].mean().to_frame(name='mean_all').reset_index(drop=True)

            bias_t_d = np.subtract(NUCAPS_smoothed_INCA_all_window.temperature_d_mean.reset_index(drop=True), RS_smoothed_INCA_window.temperature_d_mean.reset_index(drop=True))
            bias_t_d = pd.DataFrame({'diff_temp_d':bias_t_d.reset_index(drop=True), 'altitude_m':NUCAPS_smoothed_INCA_all_window.altitude_m.reset_index(drop=True)})  
            bias_t_d = bias_t_d.astype(float)
            bias_t_d = bias_t_d.groupby('altitude_m')['diff_temp_d'].mean().to_frame(name='mean_all').reset_index(drop=True)

            NUCAPS_smoothed_INCA_all.temperature_mean[i] = NUCAPS_smoothed_INCA_all.temperature_mean[i]- float(bias_t.mean_all.values)
            NUCAPS_smoothed_INCA_all.temperature_d_mean[i] = NUCAPS_smoothed_INCA_all.temperature_d_mean[i] - float(bias_t_d.mean_all.values)    

    return NUCAPS_smoothed_INCA_all

def filter_uncertainty_temp(RA_mean_temp_uncertainty, RA_smoothed_INCA_1):        
    altitude_m = RA_mean_temp_uncertainty.altitude_m
    index_list = pd.DataFrame()
    index_list_no = pd.DataFrame()
    for i in range(len(altitude_m)):
       print(altitude_m[i]) 
       value_unc = RA_mean_temp_uncertainty.mean_all[i] 
       RA_smoothed_INCA_altitude = RA_smoothed_INCA_1[RA_smoothed_INCA_1.altitude_m == altitude_m.iloc[i]]
       indexes = pd.DataFrame((np.where(RA_smoothed_INCA_1.altitude_m == altitude_m.iloc[i]))).T
       for j in range(len(RA_smoothed_INCA_altitude)):
           print(j)
           if (np.abs(RA_smoothed_INCA_altitude.temperature_mean_unc.iloc[j]) <= np.abs(value_unc)):    
               print('yes')
               index_yes = pd.DataFrame({'index_list' :indexes.iloc[j].values}, index = [0]) 
               index_list = index_list.append(index_yes)
           else:
               print('no')
               index_no = pd.DataFrame({'index_list' :indexes.iloc[j].values}, index = [0]) 
               index_list_no = index_list_no.append(index_no)
                        
    RA_smoothed_INCA_1.temperature_mean.iloc[index_list_no.values] = np.nan 
    return RA_smoothed_INCA_1 , index_list, index_list_no    
    
def filter_uncertainty_temp_d(RA_mean_temp_uncertainty, RA_smoothed_INCA_1):        
    altitude_m = RA_mean_temp_uncertainty.altitude_m
    index_list = pd.DataFrame()
    index_list_no = pd.DataFrame()
    for i in range(len(altitude_m)):
       print(altitude_m[i]) 
       value_unc = RA_mean_temp_d_uncertainty.mean_all[i] 
       RA_smoothed_INCA_altitude = RA_smoothed_INCA_1[RA_smoothed_INCA_1.altitude_m == altitude_m.iloc[i]]
       indexes = pd.DataFrame((np.where(RA_smoothed_INCA_1.altitude_m == altitude_m.iloc[i]))).T
       for j in range(len(RA_smoothed_INCA_altitude)):
           print(j)
           if (np.abs(RA_smoothed_INCA_altitude['uncertainty_specific_humidity_gkg-1'].iloc[j]) <= np.abs(value_unc)):    
               print('yes')
               index_yes = pd.DataFrame({'index_list' :indexes.iloc[j].values}, index = [0]) 
               index_list = index_list.append(index_yes)
           else:
               print('no')
               index_no = pd.DataFrame({'index_list' :indexes.iloc[j].values}, index = [0]) 
               index_list_no = index_list_no.append(index_no)
                        
    RA_smoothed_INCA_1.temperature_d_mean.iloc[index_list_no.values] = np.nan 
    return RA_smoothed_INCA_1, index_list, index_list_no  
     
def calc_bias_temp(input_smoothed_INCA, RS_smoothed_INCA): 
     diff_temp = np.subtract(input_smoothed_INCA.temperature_mean.reset_index(drop=True), RS_smoothed_INCA.temperature_mean.reset_index(drop=True), axis = 1)
     diff_temp = pd.DataFrame({'diff_temp':diff_temp.values, 'altitude_m': input_smoothed_INCA.altitude_m.values})
     diff_temp_mean = diff_temp.groupby('altitude_m')['diff_temp'].mean().to_frame(name='mean_all').reset_index() 
     diff_temp_mean = pd.DataFrame({'diff_temp':diff_temp_mean.mean_all, 'altitude_m': diff_temp_mean.altitude_m})
     return diff_temp_mean  

def calc_bias_temp_d(input_smoothed_INCA, RS_smoothed_INCA):       
     diff_temp_d = np.subtract(input_smoothed_INCA.temperature_d_mean.reset_index(drop=True), RS_smoothed_INCA.temperature_d_mean.reset_index(drop=True), axis = 1)
     diff_temp_d = pd.DataFrame({'diff_temp_d':diff_temp_d.values, 'altitude_m': input_smoothed_INCA.altitude_m})
     diff_temp_d_mean = diff_temp_d.groupby('altitude_m')['diff_temp_d'].mean().to_frame(name='mean_all').reset_index()
     diff_temp_d_mean = pd.DataFrame({'diff_temp_d_mean':diff_temp_d_mean.mean_all, 'altitude_m': diff_temp_d_mean.altitude_m})  
     return diff_temp_d_mean
 
def calc_std_temp(input_data_smoothed_INCA, RS_data_smoothed_INCA):   
    diff_temp = np.subtract(input_data_smoothed_INCA.temperature_mean.reset_index(drop=True), RS_data_smoothed_INCA.temperature_mean.reset_index(drop=True), axis = 1)
    diff_temp = pd.DataFrame({'diff_temp':diff_temp, 'altitude_mean': RS_data_smoothed_INCA.altitude_m.values})
        
    diff_temp_d = np.subtract(input_data_smoothed_INCA.temperature_d_mean.reset_index(drop=True), RS_data_smoothed_INCA.temperature_d_mean.reset_index(drop=True), axis = 1)
    diff_temp_d = pd.DataFrame({'diff_temp_d':diff_temp_d, 'altitude_m': RS_data_smoothed_INCA.altitude_m.values})
        
    diff_temp_ee = diff_temp.diff_temp
    diff_temp_ee_sqr = diff_temp_ee **2
        
    diff_temp_d_ee = diff_temp_d.diff_temp_d
    diff_temp_d_ee_sqr = diff_temp_d_ee **2
         
    altitude_diff = pd.DataFrame({'altitude_m' : input_data_smoothed_INCA.altitude_m.values, 'diff_temp_ee_sqr': diff_temp_ee_sqr, 'diff_temp_d_ee_sqr': diff_temp_d_ee_sqr})
        
    number_temp = altitude_diff.groupby(['altitude_m'])['diff_temp_ee_sqr'].count().to_frame(name='number').reset_index()
    number_temp[number_temp == 0] = np.nan
    number_temp_d = altitude_diff.groupby(['altitude_m'])['diff_temp_d_ee_sqr'].count().to_frame(name='number').reset_index()
    number_temp_d[number_temp_d == 0] = np.nan
        
    diff_temp_sqr = altitude_diff.groupby(['altitude_m'])['diff_temp_ee_sqr'].sum()
    diff_temp_sqr = diff_temp_sqr.reset_index(drop=True)
    diff_temp_d_sqr = altitude_diff.groupby(['altitude_m'])['diff_temp_d_ee_sqr'].sum()
    diff_temp_d_sqr = diff_temp_d_sqr.reset_index(drop=True)
        
    std_temp = np.sqrt((diff_temp_sqr / (number_temp.number)))
    std_temp = pd.DataFrame({'std_temp': std_temp, 'altitude_m': number_temp.altitude_m})
    std_temp_d = np.sqrt((diff_temp_d_sqr / ( number_temp_d.number)))
    std_temp_d = pd.DataFrame({'std_temp_d': std_temp_d, 'altitude_m': number_temp.altitude_m})
        
    return std_temp, std_temp_d, number_temp, number_temp_d

############################################################################# define time #############################################################################
### !! time span
firstdate = '20190501000000'
lastdate = '20200501000000'
firstobj=dt.datetime.strptime(firstdate,'%Y%m%d%H%M%S')
lastobj=dt.datetime.strptime(lastdate,'%Y%m%d%H%M%S')

lon_Payerne = 6.93608
lat_Payerne = 46.82201

DT = firstobj.hour

dist = 1000
   
while firstobj != lastobj:
    print(firstobj)
    # !! define time span
    #lastobj_month = firstobj + relativedelta(months=1) # split into month
    lastobj_month = lastobj # the whole year
    ##################################################################### define paths and read data ##################################################################
    ## read data
    RS_archive   = '/data/COALITION2/internships/nal/data/Radiosondes/Payerne/'
    RA_archive   = '/data/COALITION2/internships/nal/data/RALMO/Payerne/'
    SMN_archive = '/data/COALITION2/internships/nal/data/SwissMetNet/Payerne'
    NUCAPS_archive = '/data/COALITION2/internships/nal/data/NUCAPS'
    INCA_archive = '/data/COALITION2/internships/nal/data/COSMO/'
    MEAN_PROFILES_archive = '/data/COALITION2/internships/nal/data/Plots/mean_profiles'
    BIAS_archive = '/data/COALITION2/internships/nal/data/Plots/bias/'
    STD_archive = '/data/COALITION2/internships/nal/data/Plots/std/'
    RM_archive = '/data/COALITION2/internships/nal/data/Radiometer/Payerne'
    
    ## read data
    RS_data = xr.open_dataset(RS_archive+'/RS_concat.nc').to_dataframe()
    RA_data = xr.open_dataset(RA_archive+'/RA_concat_wp').to_dataframe()
    SMN_data = xr.open_dataset(SMN_archive+'/SMN_concat1.nc').to_dataframe()
    #SMN_data = xr.open_dataset(SMN_archive+/'SMN_concat1.nc').to_dataframe()
    #RM_data = xr.open_dataset(RM_archive+'/RM_concat.nc').to_dataframe()
    #RM_data_quality_flag = xr.open_dataset(RM_archive+'/RM_concat_quality_flag.nc').to_dataframe()
    #RM_data_quality_flag = RM_data_quality_flag.reset_index(drop=True)
    #x = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/Radiometer/radiometer_payerne/temperature/C00-V859_190501.CMP.TPC.NC')
    #RM_data = xr.open_dataset(RM_archive+'/radiometer_06610_concat_filtered_'+str(DT)+'.nc').to_dataframe()
    RM_data = xr.open_dataset(RM_archive+'/RM_concat_new1.nc').to_dataframe()
    #RM_data_quality_flag = xr.open_dataset(RM_archive+'/RM_concat_quality_flag.nc').to_dataframe()
    NUCAPS_data = open_NUCAPS_file(NUCAPS_archive+'/NUCAPS_Payerne_-120min_60min_3500km.nc')
    
    INCA_grid = xr.open_dataset(INCA_archive+'/inca_topo_levels_hsurf_ccs4.nc') 
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
    
    ####################################################### PREPARE DATASETS: ADD RELEVANT AVARIABLES, DELTE NAN VALUES ########################################################
    ##### A) SMN: Surface measurement
    ##########################################
    # convert time to datetime format
    SMN_data['time_YMDHMS'] = pd.to_datetime(SMN_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    
    SMN_times = SMN_data.time_YMDHMS[SMN_data.precipitation_mm == 0]
    
    ##########################################
    ##### B) RS: Radiosonde
    ##########################################
    missing_date = '20200121000000'
    missing_date=dt.datetime.strptime(missing_date,'%Y%m%d%H%M%S')
    RS_data = RS_data[RS_data.temperature_degC != 10000000.0]
    RS_data = RS_data.rename(columns={'geopotential_altitude_m' : 'altitude_m'})
    # convert time to datetime formate
    RS_data['time_YMDHMS'] = pd.to_datetime(RS_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    #nodef =  pd.DataFrame({'wmo_id': 7755, 'time_YMDHMS' : missing_date, 'track_type': 8, 'prof_type': 141, 'type': 0, 'level': np.nan, 'pressure_hPa': np.nan, 'temperature_degC' : np.nan, 'relative_humidity_percent': np.nan, 'altitude_m' : np.nan, 'wind_speed_ms-1' : np.nan, 'wind_dir_deg' : np.nan, 'dew_point_degC' : np.nan}, index = [0])
    #RS_data = RS_data.append(nodef)
    #RS_data = RS_data.sort_values(by = ['time_YMDHMS'])
    #RS_data = RS_data.reset_index(drop=True)

    ##########################################
    ##### C) NUCAPS: Satellite data 
    ##########################################
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
    
    NUCAPS_data = pd.concat([p_NUCAPS, T_NUCAPS, T_d_NUCAPS, dist, datetime_NUCAPS, datetime_NUCAPS_round, skin_temp, quality_flag, H2O_MR, surf_pres, topography], axis = 1)
    NUCAPS_data = NUCAPS_data.dropna()  
    
    NUCAPS_data = NUCAPS_data[NUCAPS_data.dist <= dist]
    NUCAPS_data = NUCAPS_data[NUCAPS_data.pressure_hPa <= NUCAPS_data.surf_pres]

    ##########################################
    ##### D) RALMO: Raman lidar 
    ##########################################
    ## add temperature in degC
    RA_data['temperature_degC'] = RA_data.temperature_K - 273.15
    RA_data['temperature_degC'][RA_data['temperature_K']== int(10000000)] = np.nan
    RA_data['uncertainty_temperature_K'][RA_data['temperature_K']==int(1000000)] = np.nan
    
    ## add dewpoint temperature
    dewpoint_degC = cc.dewpoint_from_specific_humidity(RA_data['specific_humidity_gkg-1'].values * units('g/kg'), (RA_data.temperature_K.values) * units.kelvin, RA_data.pressure_hPa.values * units.hPa)
    RA_data.insert(value=dewpoint_degC,column = "dew_point_degC", loc=11)
    RA_data['dew_point_degC'][RA_data['specific_humidity_gkg-1']== int(10000000)] = np.nan
    
    ## add dewpoint temperature uncertainty 
    RA_data['uncertainty_specific_humidity_gkg-1'][RA_data['uncertainty_specific_humidity_gkg-1']== int(10000000)] = np.nan
    #dewpoint_degC_uncertainty = cc.dewpoint_from_specific_humidity((np.abs(RA_data['uncertainty_specific_humidity_gkg-1']).values * units('g/kg')), (RA_data.temperature_K.values) * units.kelvin, RA_data.pressure_hPa.values * units.hPa)    
    #RA_data.insert(value=dewpoint_degC_uncertainty,column = "uncertainty_dew_point_K", loc=11)
    RA_data['uncertainty_specific_humidity_gkg-1_temperature'] = cc.dewpoint_from_specific_humidity((np.abs(RA_data['uncertainty_specific_humidity_gkg-1']).values * units('g/kg')), (RA_data.temperature_K.values) * units.kelvin, RA_data.pressure_hPa.values * units.hPa)   
    
    #!! ## add relative humidity
    #!! #RH_percent = cc.relative_humidity_from_specific_humidity(RA_data['specific_humidity_gkg-1'].values * units('g/kg'), (RA_data.temperature_K.values +273.15) * units.kelvin, RA_data.pressure_hPa.values * units.hPa)
    #!! #RA_data.insert(value=temp_K,column = "relative_humidity_percent", loc=12) 
  
    # convert time to datetime format
    RA_data['time_YMDHMS'] = pd.to_datetime(RA_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    

    
    
    ##########################################
    ## RM: Radiometer 
    ##########################################
    ## add temperature in degC
    RM_data['temperature_degC'] =  RM_data.temperature_K - 273.15
    
    ## add dewpoint temperature
    p_w = ((RM_data.temperature_K * RM_data.absolute_humidity_gm3) / 2.16679)
    p_s = metpy.calc.saturation_vapor_pressure(RM_data.temperature_K.values * units.kelvin)
    
    RH = p_w /p_s
    RM_data['dew_point_degC'] = metpy.calc.dewpoint_from_relative_humidity(RM_data.temperature_K.values * units.kelvin, RH.values * units.percent)   
 
    RM_data['time_YMDHMS'] = pd.to_datetime(RM_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
        
    RM_data['temperature_degC'][RM_data.quality_flag == 1] = np.nan
    RM_data['dew_point_degC'][RM_data.quality_flag == 1] = np.nan
    
    ####################################################### FILTER TIME #################################################################################
    ## filter time span
    RS_data_filtered = RS_data[(RS_data['time_YMDHMS'] >= firstobj) & (RS_data['time_YMDHMS'] < lastobj_month)] 
    RA_data_filtered = RA_data[(RA_data['time_YMDHMS'] >= firstobj) & (RA_data['time_YMDHMS'] < lastobj_month)]
    NUCAPS_data_filtered = NUCAPS_data[(NUCAPS_data['time_YMDHMS'] >= firstobj) & (NUCAPS_data['time_YMDHMS'] < lastobj_month)]
    RM_data_filtered = RM_data[(RM_data['time_YMDHMS'] >= firstobj) & (RM_data['time_YMDHMS'] < lastobj_month)]
    
    ## filter noon or midnight
    RS_data_filtered = RS_data_filtered[RS_data_filtered['time_YMDHMS'].dt.hour == DT]
    RS_data_filtered = RS_data_filtered.reset_index()
    
    RA_data_filtered = RA_data_filtered[RA_data_filtered['time_YMDHMS'].dt.hour == DT]
    RA_data_filtered = RA_data_filtered.reset_index()
    
    NUCAPS_data_filtered = NUCAPS_data_filtered[(NUCAPS_data_filtered['time_YMDHMS'].dt.hour == DT)]
    NUCAPS_data_filtered = NUCAPS_data_filtered.reset_index()
    
    RM_data_filtered = RM_data_filtered[(RM_data_filtered['time_YMDHMS'].dt.hour == DT)]
    RM_data_filtered = RM_data_filtered.reset_index(drop=True)

    ####################################################### ADD ALTITUDE TO NUCAPS ############################################################################################## 
    # !!!!! implement directly into dataset !!!!!
    firstobj_NUCAPS = firstobj
    
    #NUCAPS_data_filtered = RS_data_filtered
    NUCAPS_data_filtered['altitude_m'] = 0
    #NUCAPS_data_filtered = RS_data_filtered
       
    altitude_m  = pd.DataFrame()   
    while firstobj_NUCAPS != lastobj_month:
        nowdate = firstobj_NUCAPS.strftime('%Y%m%d')
        NUCAPS_data_time = NUCAPS_data_filtered[(NUCAPS_data_filtered['time_YMDHMS'] == firstobj_NUCAPS)]
        NUCAPS_data_time = NUCAPS_data_time.reset_index(drop=True)
        NUCAPS_data_time['altitude_m'] = '0'
        
        SMN_data_time = SMN_data[(SMN_data['time_YMDHMS'] == firstobj_NUCAPS)]
        
        RS_data_time = RS_data[(RS_data['time_YMDHMS'] == firstobj_NUCAPS)]
        
        if NUCAPS_data_time.empty:
            firstobj_NUCAPS = firstobj_NUCAPS + dt.timedelta(days=1)
            print('nope')
            
        else:  
            print(nowdate)
            data_comma_temp = NUCAPS_data_time[['time_YMDHMS', 'pressure_hPa', 'temperature_degC', 'altitude_m']]
            #data_comma_temp = data_comma_temp.append({'time_YMDHMS' : NUCAPS_data_time.time_YMDHMS[0], 'pressure_hPa' : NUCAPS_data_time.surf_pres.iloc[0],  'temperature_degC': NUCAPS_data_time.skin_temp.iloc[0] - 273.15, 'altitude_m' : NUCAPS_data_time.topography[0]}, ignore_index=True)
            data_comma_temp = data_comma_temp.append({'time_YMDHMS' : SMN_data_time.time_YMDHMS[0], 'pressure_hPa' : SMN_data_time.pressure_hPa[0],  'temperature_degC': SMN_data_time.temperature_degC[0], 'altitude_m' :491}, ignore_index=True)
            #data_comma_temp = data_comma_temp.append({'time_YMDHMS' : RS_data_time.time_YMDHMS.iloc[0], 'pressure_hPa' : RS_data_time.pressure_hPa[1],  'temperature_degC': RS_data_time.temperature_degC[1], 'altitude_m' : RS_data_time.altitude_m[1]}, ignore_index=True)
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
    
    # !!!!! implement directly into dataset !!!!!
    # filter NUCAPS according to quality flags
    NUCAPS_data_all = NUCAPS_data_filtered
    NUCAPS_data_0 = NUCAPS_data_filtered[NUCAPS_data_filtered.quality_flag == 0] # clear sky: IR and MR retrieval 
    NUCAPS_data_1 = NUCAPS_data_filtered[NUCAPS_data_filtered.quality_flag == 1] # cloudy: MR only retrieval 
    NUCAPS_data_9 = NUCAPS_data_filtered[NUCAPS_data_filtered.quality_flag== 9] # precipitating conditions: failed IR + MW retreival 

    ####################################################### INTERPOLATE DATASETS TO INCA GRID ################################################################################### 
    ##### B) RS: Radiosonde
    ##########################################
    ## original data, no smoothing 
    RS_original_mean_temp = RS_data_filtered.groupby(['altitude_m'])['temperature_degC'].mean().to_frame(name='mean_all').reset_index()
    RS_original_mean_temp_d = RS_data_filtered.groupby(['altitude_m'])['dew_point_degC'].mean().to_frame(name='mean_all').reset_index()
    
    # over all times
    RS_smoothed_all = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, RS_data_filtered)
    RS_smoothed_all_mean_temp =  RS_smoothed_all.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    RS_smoothed_all_mean_temp_d =  RS_smoothed_all.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    ##########################################
    ##### C) RALMO: Raman lidar
    ##########################################
    RA_original_smoothed_INCA_mean_temp = RA_data_filtered.groupby('altitude_m')['temperature_degC'].mean().to_frame(name='mean_all').reset_index()
    RA_original_smoothed_INCA_mean_temp_d = RA_data_filtered.groupby('altitude_m')['dew_point_degC'].mean().to_frame(name='mean_all').reset_index()
    
    RS_smoothed_RA = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, RA_data_filtered)
    RS_smoothed_RA_mean_temp = RS_smoothed_RA.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    RS_smoothed_RA_mean_temp_d = RS_smoothed_RA.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    RA_smoothed_INCA = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid,  RA_data_filtered, RA_data_filtered)
    RA_smoothed_INCA_1 = RA_smoothed_INCA.reset_index(drop=True)
    RA_smoothed_INCA_mean_temp = RA_smoothed_INCA.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    RA_smoothed_INCA_mean_temp_d = RA_smoothed_INCA.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    ## RA uncertainty 
    #RA_mean_temp_uncertainty = RA_smoothed_INCA.groupby(['altitude_m'])['temperature_mean_unc'].mean().to_frame(name='mean_all').reset_index()
    #RA_mean_temp_d_uncertainty = RA_smoothed_INCA.groupby(['altitude_m'])['uncertainty_specific_humidity_gkg-1'].mean().to_frame(name='mean_all').reset_index()
    
    ### filter uncertainty ###      
    # absolute filter 
    #RA_smoothed_INCA_korr_unc_temp_abs =  RA_smoothed_INCA_1
    #RA_smoothed_INCA_korr_unc_temp_abs.temperature_mean[RA_smoothed_INCA_korr_unc_temp_abs.temperature_mean_unc >= float(10)] = np.nan
    #RA_smoothed_INCA_korr_unc_temp_abs.temperature_d_mean[RA_smoothed_INCA_korr_unc_temp_abs['uncertainty_specific_humidity_gkg-1'] >= float(10)] = np.nan
    #RA_smoothed_INCA_korr_unc_temp_abs = RA_smoothed_INCA_korr_unc_temp_abs.reset_index(drop=True)
    
    ##RS_smoothed_RA_korr_unc_abs_temp = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, RA_smoothed_INCA_korr_unc_temp_abs)
    #RS_smoothed_RA_mean_temp_korr_unc_abs = RS_smoothed_RA_korr_unc_abs_temp.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    #RS_smoothed_RA_mean_temp_d_korr_unc_abs = RS_smoothed_RA_korr_unc_abs_temp.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    #RA_smoothed_INCA_mean_temp_korr_unc_abs = RA_smoothed_INCA_korr_unc_temp_abs.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    #RA_smoothed_INCA_mean_temp_d_korr_unc_abs = RA_smoothed_INCA_korr_unc_temp_abs.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()   
    
    # relative filter
    #RA_smoothed_INCA_korr_unc_temp_rel,index_list, index_list_no =  filter_uncertainty_temp(RA_mean_temp_uncertainty, RA_smoothed_INCA_1)                 
    #RA_smoothed_INCA_korr_unc_temp_d_rel,index_list_, index_list_no =  filter_uncertainty_temp_d(RA_mean_temp_d_uncertainty, RA_smoothed_INCA_1)
    
    #RS_smoothed_RA_korr_unc_rel_temp = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, RA_smoothed_INCA_korr_unc_temp_rel)
    #RS_smoothed_RA_korr_unc_rel_temp_d = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, RA_smoothed_INCA_korr_unc_temp_d_rel)
    #RS_smoothed_RA_mean_temp_korr_unc_rel = RS_smoothed_RA_korr_unc_rel_temp.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    #RS_smoothed_RA_mean_temp_d_korr_unc_rel = RS_smoothed_RA_korr_unc_rel_temp_d.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()

    #RA_smoothed_INCA_mean_temp_korr_unc_rel = RA_smoothed_INCA_korr_unc_temp_rel.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    #RA_smoothed_INCA_mean_temp_d_korr_unc_rel = RA_smoothed_INCA_korr_unc_temp_d_rel.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()   
    
 
    


    ###########################################
    ##### D) NUCAPS: Satellite data 
    ##########################################
    ## all times
    RS_smoothed_NUCAPS_all = average_RS_to_INCA_grid(firstobj, lastobj_month, INCA_grid,RS_data_filtered, NUCAPS_data_all)
    RS_smoothed_NUCAPS_mean_temp_all = RS_smoothed_NUCAPS_all.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    RS_smoothed_NUCAPS_mean_temp_d_all = RS_smoothed_NUCAPS_all.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    ### ORIGINAL 
    NUCAPS_smoothed_INCA_all = average_RS_to_INCA_grid(firstobj, lastobj_month, INCA_grid,NUCAPS_data_all, NUCAPS_data_all)
    NUCAPS_smoothed_INCA_all_1 = NUCAPS_smoothed_INCA_all
    NUCAPS_smoothed_INCA_all = NUCAPS_smoothed_INCA_all[['altitude_m', 'temperature_mean', 'temperature_d_mean']]
    
    NUCAPS_smoothed_INCA_all = NUCAPS_smoothed_INCA_all.astype(float)
    NUCAPS_smoothed_INCA_mean_temp_all = NUCAPS_smoothed_INCA_all.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    NUCAPS_smoothed_INCA_mean_temp_d_all = NUCAPS_smoothed_INCA_all.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    
    ### quality flag ###
    ## times with quality flag 0
    RS_smoothed_NUCAPS_0 = average_RS_to_INCA_grid(firstobj, lastobj_month, INCA_grid, RS_data_filtered, NUCAPS_data_0)
    if RS_smoothed_NUCAPS_0.empty:
        RS_smoothed_NUCAPS_0 = pd.DataFrame({'altitude_m' : np.nan, 'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan}, index = [0])
    RS_smoothed_NUCAPS_mean_temp_0 = RS_smoothed_NUCAPS_0.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    RS_smoothed_NUCAPS_mean_temp_d_0 = RS_smoothed_NUCAPS_0.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    NUCAPS_smoothed_INCA_0 = average_RS_to_INCA_grid(firstobj, lastobj_month, INCA_grid, NUCAPS_data_0, NUCAPS_data_0)
    if NUCAPS_smoothed_INCA_0.empty:
        NUCAPS_smoothed_INCA_0 = pd.DataFrame({'altitude_m' : np.nan, 'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan}, index = [0])
    NUCAPS_smoothed_INCA_0 = NUCAPS_smoothed_INCA_0[['altitude_m', 'temperature_mean', 'temperature_d_mean']]
    NUCAPS_smoothed_INCA_0 = NUCAPS_smoothed_INCA_0.astype(float)
    NUCAPS_smoothed_INCA_mean_temp_0 = NUCAPS_smoothed_INCA_0.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    NUCAPS_smoothed_INCA_mean_temp_d_0 = NUCAPS_smoothed_INCA_0.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    ## times with quality flag 1
    RS_smoothed_NUCAPS_1 = average_RS_to_INCA_grid(firstobj, lastobj_month, INCA_grid, RS_data_filtered, NUCAPS_data_1)
    if RS_smoothed_NUCAPS_1.empty:
        RS_smoothed_NUCAPS_1 = pd.DataFrame({'altitude_m' : np.nan, 'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan}, index = [0])
    RS_smoothed_NUCAPS_mean_temp_1 = RS_smoothed_NUCAPS_1.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    RS_smoothed_NUCAPS_mean_temp_d_1 = RS_smoothed_NUCAPS_1.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    NUCAPS_smoothed_INCA_1 = average_RS_to_INCA_grid(firstobj, lastobj_month, INCA_grid, NUCAPS_data_1, NUCAPS_data_1)
    if NUCAPS_smoothed_INCA_1.empty:
        NUCAPS_smoothed_INCA_1 = pd.DataFrame({'altitude_m' : np.nan, 'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan}, index = [0])
    NUCAPS_smoothed_INCA_1 = NUCAPS_smoothed_INCA_1[['altitude_m', 'temperature_mean', 'temperature_d_mean']]
    NUCAPS_smoothed_INCA_1 = NUCAPS_smoothed_INCA_1.astype(float)
    NUCAPS_smoothed_INCA_mean_temp_1 = NUCAPS_smoothed_INCA_1.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    NUCAPS_smoothed_INCA_mean_temp_d_1 = NUCAPS_smoothed_INCA_1.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    ## times with quality flag 9
    RS_smoothed_NUCAPS_9 = average_RS_to_INCA_grid(firstobj, lastobj_month, INCA_grid, RS_data_filtered, NUCAPS_data_9)
    if RS_smoothed_NUCAPS_9.empty:
        RS_smoothed_NUCAPS_9 = pd.DataFrame({'altitude_m' : np.nan, 'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan}, index = [0])      
    RS_smoothed_NUCAPS_mean_temp_9 = RS_smoothed_NUCAPS_9.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    RS_smoothed_NUCAPS_mean_temp_d_9 = RS_smoothed_NUCAPS_9.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
     
    NUCAPS_smoothed_INCA_9 = average_RS_to_INCA_grid(firstobj, lastobj_month, INCA_grid,  NUCAPS_data_filtered, NUCAPS_data_9)
    if NUCAPS_smoothed_INCA_9.empty:
        NUCAPS_smoothed_INCA_9 = pd.DataFrame({'altitude_m' : np.nan, 'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan}, index = [0])
    NUCAPS_smoothed_INCA_9 = NUCAPS_smoothed_INCA_9[['altitude_m', 'temperature_mean', 'temperature_d_mean']]
    NUCAPS_smoothed_INCA_9 = NUCAPS_smoothed_INCA_9.astype(float)
    NUCAPS_smoothed_INCA_mean_temp_9 = NUCAPS_smoothed_INCA_9.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    NUCAPS_smoothed_INCA_mean_temp_d_9 = NUCAPS_smoothed_INCA_9.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
   
    
    
    
    ##########################################
    ##### E) RM: Radiometer
    ##########################################
    RM_smoothed_INCA_mean_temp_original = RM_data_filtered.groupby('altitude_m')['temperature_degC'].mean().to_frame(name='mean_all').reset_index()
    RM_smoothed_INCA_mean_temp_d_original = RM_data_filtered.groupby('altitude_m')['dew_point_degC'].mean().to_frame(name='mean_all').reset_index()
    
    RS_smoothed_RM = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, RM_data_filtered)
    RS_smoothed_RM_mean_temp = RS_smoothed_RM.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    RS_smoothed_RM_mean_temp_d = RS_smoothed_RM.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    RM_smoothed_INCA = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid,  RM_data_filtered, RM_data_filtered)
    RM_smoothed_INCA_mean_temp = RM_smoothed_INCA.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
    RM_smoothed_INCA_mean_temp_d = RM_smoothed_INCA.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    

    ####################################################### CALCULATE BIAS AND STD ############################################################################################## 
    ### BIAS ###
    ##########################################
    ##### C) NUCAPS: Satellite data #####
    ##########################################
    # original data
    diff_temp_mean_NUCAPS_all = calc_bias_temp(NUCAPS_smoothed_INCA_all.reset_index(drop=True), RS_smoothed_NUCAPS_all.reset_index(drop=True))
    diff_temp_d_mean_NUCAPS_all = calc_bias_temp_d(NUCAPS_smoothed_INCA_all.reset_index(drop=True), RS_smoothed_NUCAPS_all.reset_index(drop=True))

    ### quality flags ###
     
    # bewteen Radiosonde and NUCAPS 0
    diff_temp_mean_NUCAPS_0 = calc_bias_temp(NUCAPS_smoothed_INCA_0, RS_smoothed_NUCAPS_0)
    diff_temp_d_mean_NUCAPS_0 = calc_bias_temp_d(NUCAPS_smoothed_INCA_0, RS_smoothed_NUCAPS_0)
    
    # bewteen Radiosonde and NUCAPS 1
    diff_temp_mean_NUCAPS_1 = calc_bias_temp(NUCAPS_smoothed_INCA_1, RS_smoothed_NUCAPS_1)
    diff_temp_d_mean_NUCAPS_1 = calc_bias_temp_d(NUCAPS_smoothed_INCA_1, RS_smoothed_NUCAPS_1)
    
    # bewteen Radiosonde and NUCAPS 9
    diff_temp_mean_NUCAPS_9 = calc_bias_temp(NUCAPS_smoothed_INCA_9, RS_smoothed_NUCAPS_9)
    diff_temp_d_mean_NUCAPS_9 = calc_bias_temp_d(NUCAPS_smoothed_INCA_9, RS_smoothed_NUCAPS_9)
    



    ##########################################
    ##### D) RALMO: Raman lidar 
    #########################################
    # between Radiosonde and RALMO 
    diff_temp_mean_RA = calc_bias_temp(RA_smoothed_INCA.reset_index(drop=True), RS_smoothed_RA.reset_index(drop=True))   
    diff_temp_d_mean_RA = calc_bias_temp_d(RA_smoothed_INCA.reset_index(drop=True), RS_smoothed_RA.reset_index(drop=True))
    
    #diff_temp_mean_RA_korr_unc_abs = calc_bias_temp(RA_smoothed_INCA_korr_unc_temp_abs, RS_smoothed_RA_korr_unc_abs_temp)   
    #diff_temp_d_mean_RA_korr_unc_abs = calc_bias_temp_d( RA_smoothed_INCA_korr_unc_temp_abs, RS_smoothed_RA_korr_unc_abs_temp)
    
    #diff_temp_mean_RA_korr_unc_rel = calc_bias_temp(RA_smoothed_INCA_korr_unc_temp_rel, RS_smoothed_RA_korr_unc_rel_temp)   
    #diff_temp_d_mean_RA_korr_unc_rel = calc_bias_temp_d( RA_smoothed_INCA_korr_unc_temp_d_rel, RS_smoothed_RA_korr_unc_rel_temp_d)
    


     
    
    ##########################################
    ## E) RM: Radiometer 
    ##########################################
    diff_temp_mean_RM = calc_bias_temp(RM_smoothed_INCA.reset_index(drop=True), RS_smoothed_RM.reset_index(drop=True))   
    diff_temp_d_mean_RM = calc_bias_temp_d(RM_smoothed_INCA.reset_index(drop=True), RS_smoothed_RM.reset_index(drop=True))
    

    
    ### STD ###
    ##########################################
    ##### C) NUCAPS: Satellite data #####
    ##########################################  
    # bewteen Radiosonde and all NUCAPS
    std_temp_NUCAPS_all, std_temp_d_NUCAPS_all, number_temp_NUCAPS_all, number_temp_d_NUCAPS_all = calc_std_temp(NUCAPS_smoothed_INCA_all.reset_index(drop=True), RS_smoothed_NUCAPS_all)
    #std_temp_NUCAPS_all.to_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_NUCAPS_temp_12.csv')
    #std_temp_d_NUCAPS_all.to_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_NUCAPS_temp_d_12.csv')
    
    ### quality flag ###
    # minus running bias
    # bewteen Radiosonde and NUCAPS 0
    std_temp_NUCAPS_0, std_temp_d_NUCAPS_0, number_temp_NUCAPS_0, number_temp_d_NUCAPS_0, = calc_std_temp(NUCAPS_smoothed_INCA_0, RS_smoothed_NUCAPS_0)
    
    # bewteen Radiosonde and NUCAPS 1
    std_temp_NUCAPS_1, std_temp_d_NUCAPS_1, number_temp_NUCAPS_1, number_temp_d_NUCAPS_1, = calc_std_temp(NUCAPS_smoothed_INCA_1, RS_smoothed_NUCAPS_1)
    
    # bewteen Radiosonde and NUCAPS 9
    std_temp_NUCAPS_9, std_temp_d_NUCAPS_9, number_temp_NUCAPS_9, number_temp_d_NUCAPS_9, = calc_std_temp(NUCAPS_smoothed_INCA_9, RS_smoothed_NUCAPS_9)
    
    NUCAPS_data_all_1 = NUCAPS_data_all[['dist', 'index', 'time_YMDHMS_1']].round()
    dist_num = NUCAPS_data_all_1.groupby(['dist', 'time_YMDHMS_1'], as_index = False).count()

    ##########################################
    ##### D) RALMO: Raman lidar 
    #########################################
    std_temp_RA, std_temp_d_RA, number_temp_RA, number_temp_d_RA = calc_std_temp(RA_smoothed_INCA.reset_index(drop=True), RS_smoothed_RA)
    #std_temp_RA.to_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RA_temp_00_nov_may.csv')
    #std_temp_d_RA.to_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RA_temp_d_12.csv')
    #std_temp_RA_korr_unc_abs, std_temp_d_RA_korr_unc_abs, number_temp_RA_korr_unc_abs, number_temp_d_RA_korr_unc_abs = calc_std_temp(RA_smoothed_INCA_korr_unc_temp_abs, RS_smoothed_RA_korr_unc_abs_temp)
    #std_temp_RA_korr_unc_rel, std_temp_d_RA_korr_unc_rel, number_temp_RA_korr_unc_rel, number_temp_d_RA_korr_unc_rel = calc_std_temp(RA_smoothed_INCA_korr_unc_temp_rel, RS_smoothed_RA_korr_unc_rel_temp)
  
   
  
    ##########################################
    ## E) RM: Radiometer 
    ##########################################
    std_temp_RM, std_temp_d_RM, number_temp_RM, number_temp_d_RM = calc_std_temp(RM_smoothed_INCA.reset_index(drop=True), RS_smoothed_RM)
    #std_temp_RM.to_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RM_temp_0.csv')
    #std_temp_d_RM.to_csv('/data/COALITION2/PicturesSatellite/results_NAL/Std_files/std_RM_temp_d_0.csv')



    ###################################################### PLOT MEAN PROFILE, BIAS AND STD ###################################################################################### 
    ### mean profile ###
    fig, ax = plt.subplots(figsize = (5,12))
        
    ##########################################
    ##### B) RS: Radiosonde - BLUEISH
    ########################################## 
    # no smoothing 
    ax.plot(RS_original_mean_temp.mean_all, RS_original_mean_temp.altitude_m, color = 'navy', label = 'original RS T', zorder = 1)
    ax.plot(RS_original_mean_temp_d.mean_all, RS_original_mean_temp_d.altitude_m, color = 'navy', label = 'original RS Td', zorder = 1)
    
    # all times
    ax.plot(RS_smoothed_all_mean_temp.mean_all,  RS_smoothed_all_mean_temp.altitude_m, color = 'aqua',linewidth = 2,  label = 'smoothed RS Td all', zorder = 1)
    ax.plot(RS_smoothed_all_mean_temp_d.mean_all,  RS_smoothed_all_mean_temp_d.altitude_m, color = 'aqua',linewidth = 2,  label = 'smoothed RS Td all', zorder = 1)
        
    ##########################################
    ##### C) NUCAPS: Satellite data - GREENISH
    ########################################## 
    # all
    ax.plot(RS_smoothed_NUCAPS_mean_temp_all.mean_all,  RS_smoothed_NUCAPS_mean_temp_all.altitude_m, color = 'aqua',linewidth = 2,  label = 'RS Td, all NUCAPS', zorder = 1)
    ax.plot(RS_smoothed_NUCAPS_mean_temp_d_all.mean_all,  RS_smoothed_NUCAPS_mean_temp_d_all.altitude_m, color = 'aqua',linewidth = 2,  label = 'RS Td, all NUCAPS', zorder = 1)
    
    ax.plot(NUCAPS_smoothed_INCA_mean_temp_all.mean_all, NUCAPS_smoothed_INCA_mean_temp_all.altitude_m, color = 'magenta',linewidth = 2,  label = 'all NUCAPS, T', zorder = 1)
    ax.plot(NUCAPS_smoothed_INCA_mean_temp_d_all.mean_all, NUCAPS_smoothed_INCA_mean_temp_d_all.altitude_m, color = 'magenta',linewidth = 2,  label = 'all NUCAPS, Td', zorder = 1)
    
    #ax.plot(NUCAPS_smoothed_INCA_mean_temp_all_min_bias_run.mean_all, NUCAPS_smoothed_INCA_mean_temp_all_min_bias_run.altitude_m, color = 'gold',linewidth = 2,  label = 'all NUCAPS T, removed bias', zorder = 1)
    #ax.plot(NUCAPS_smoothed_INCA_mean_temp_d_all_min_bias_run.mean_all, NUCAPS_smoothed_INCA_mean_temp_d_all_min_bias_run.altitude_m, color = 'gold',linewidth = 2,  label = 'all NUCAPS Td, removed bias', zorder = 1)
        
    # times with quality flag 0
    ax.plot(RS_smoothed_NUCAPS_mean_temp_0.mean_all,  RS_smoothed_NUCAPS_mean_temp_0.altitude_m, color = 'aqua',linewidth = 2,  label = 'RS T, NUCAPS 0', zorder = 1)
    ax.plot(RS_smoothed_NUCAPS_mean_temp_d_0.mean_all,  RS_smoothed_NUCAPS_mean_temp_d_0.altitude_m, color = 'aqua',linewidth = 2,  label = 'RS Td, NUCAPS 0', zorder = 1)
        
    ax.plot(NUCAPS_smoothed_INCA_mean_temp_0.mean_all, NUCAPS_smoothed_INCA_mean_temp_0.altitude_m, color = 'green',linewidth = 2,  label = 'NUCAPS 0 T', zorder = 1)
    ax.plot(NUCAPS_smoothed_INCA_mean_temp_d_0.mean_all, NUCAPS_smoothed_INCA_mean_temp_0.altitude_m, color = 'green',linewidth = 2,  label = 'NUCAPS 0 Td', zorder = 1)
    
    ## times with quality flag 1
    #ax.plot(RS_smoothed_NUCAPS_mean_temp_1.mean_all,  RS_smoothed_NUCAPS_mean_temp_1.altitude_m, color = 'aqua',linewidth = 2,  label = 'RS T, NUCAPS 1', zorder = 1)
    #ax.plot(RS_smoothed_NUCAPS_mean_temp_d_1.mean_all,  RS_smoothed_NUCAPS_mean_temp_d_1.altitude_m, color = 'aqua',linewidth = 2,  label = 'RS Td, NUCAPS 1', zorder = 1)
        
    #ax.plot(NUCAPS_smoothed_INCA_mean_temp_1.mean_all, NUCAPS_smoothed_INCA_mean_temp_1.altitude_m, color = 'lawngreen',linewidth = 2,  label = 'NUCAPS 0, T', zorder = 1)
    #ax.plot(NUCAPS_smoothed_INCA_mean_temp_d_1.mean_all, NUCAPS_smoothed_INCA_mean_temp_1.altitude_m, color = 'lawngreen',linewidth = 2,  label = 'NUCAPS 0, Td', zorder = 1)
        
    ## times with quality flag 9
    #ax.plot(RS_smoothed_NUCAPS_mean_temp_9.mean_all,  RS_smoothed_NUCAPS_mean_temp_9.altitude_m, color = 'aqua',linewidth = 2,  label = 'RS T, NUCAPS 9', zorder = 1)
    #ax.plot(RS_smoothed_NUCAPS_mean_temp_d_9.mean_all, RS_smoothed_NUCAPS_mean_temp_d_9.altitude_m, color = 'aqua',linewidth = 2,  label = 'RS, Td NUCAPS 9', zorder = 1)
        
    #ax.plot(NUCAPS_smoothed_INCA_mean_temp_9.mean_all, NUCAPS_smoothed_INCA_mean_temp_9.altitude_m, color = 'lightgreen',linewidth = 2,  label = 'NUCAPS 9 T', zorder = 1)
    #ax.plot(NUCAPS_smoothed_INCA_mean_temp_d_9.mean_all, NUCAPS_smoothed_INCA_mean_temp_9.altitude_m, color = 'lightgreen',linewidth = 2,  label = 'NUCAPS 9 Td', zorder = 1)        
       
    #ax.plot(RA_smoothed_INCA_mean_temp_korr_unc_rel.mean_all, RA_smoothed_INCA_mean_temp_korr_unc_rel.altitude_m, color = 'magenta',linewidth = 2,  label = 'RA T rel unc', zorder = 1)
    #ax.plot(RA_smoothed_INCA_mean_temp_d_korr_unc_rel.mean_all, RA_smoothed_INCA_mean_temp_d_korr_unc_rel.altitude_m, color = 'magenta',linewidth = 2,  label = 'RA T rel unc', zorder = 1)
        
    #ax.plot(RA_mean_temp_uncertainty.mean_all, RA_mean_temp_uncertainty.altitude_m, color = 'green',linewidth = 2,  label = 'RA T uncertainty', zorder = 1)
    #ax.plot(RA_mean_temp_d_uncertainty.mean_all, RA_mean_temp_d_uncertainty.altitude_m, color = 'magenta',linewidth = 2,  label = 'RA Td smoothed RA', zorder = 1)
         
    #ax.plot(RA_smoothed_INCA_mean_temp_filunc.mean_all, RA_smoothed_INCA_mean_temp_filunc.altitude_m, color = 'green',linewidth = 2,  label = 'smoothed RA Td', zorder = 1)
    #ax.plot(RA_smoothed_INCA_mean_temp_d_filunc.mean_all, RA_smoothed_INCA_mean_temp_filunc.altitude_m, color = 'green',linewidth = 2,  label = 'smoothed RA Td', zorder = 1)
        
    ## uncertainty
    #ax.fill_betweenx(RA_mean_temp_uncertainty.altitude_m,(RA_smoothed_INCA_mean_temp.mean_all + RA_mean_temp_uncertainty.mean_all), (RA_smoothed_INCA_mean_temp.mean_all - RA_mean_temp_uncertainty.mean_all),  alpha = 0.2, color = 'orangered', label = 'mean RA T', zorder = 2)
    #ax.fill_betweenx(RA_mean_temp_d_uncertainty.altitude_m,(RA_smoothed_INCA_mean_temp_d.mean_all + RA_mean_temp_d_uncertainty.mean_all), (RA_smoothed_INCA_mean_temp_d.mean_all - RA_mean_temp_d_uncertainty.mean_all), alpha = 0.4, color = 'navy', label = 'mean RA Td', linestyle = '--',zorder = 3)
    
    ##########################################
    ## E) RM: Radiometer - orangeish
    ##########################################
    #ax.plot(RM_smoothed_INCA_mean_temp_original.mean_all, RM_smoothed_INCA_mean_temp_original.altitude_m, color = 'orangered',linewidth = 2,  label = 'original RM Td', zorder = 1)
    #ax.plot(RM_smoothed_INCA_mean_temp_d.mean_all, RM_smoothed_INCA_mean_temp.altitude_m, color = 'orangered',linewidth = 2,  label = 'original RM Td', zorder = 1) 
        
    #ax.plot(RS_smoothed_RM_mean_temp.mean_all[:-1],  RS_smoothed_RM_mean_temp.altitude_m[:-1], color = 'aqua',linewidth = 2,  label = 'smoothed RS Td, RA', zorder = 1)
    #ax.plot(RS_smoothed_RM_mean_temp_d.mean_all[:-1],  RS_smoothed_RM_mean_temp_d.altitude_m[:-1], color = 'aqua',linewidth = 2,  label = 'smoothed RS Td, RA', zorder = 1)
        
    #ax.plot(RM_smoothed_INCA_mean_temp.mean_all, RM_smoothed_INCA_mean_temp.altitude_m, color = 'bisque',linewidth = 2,  label = 'smoothed RM Td', zorder = 1)
    #ax.plot(RM_smoothed_INCA_mean_temp_d.mean_all, RM_smoothed_INCA_mean_temp.altitude_m, color = 'bisque',linewidth = 2,  label = 'smoothed RM Td', zorder = 1)
    
   

    ax.set_ylabel('Altitude [m]', fontsize = 20)
    ax.set_xlabel('Temperature [°C]', fontsize = 20)
    ax.tick_params(labelsize = 20)
    ax.legend(fontsize = 15, loc = 'upper right')

    ax.set_ylim(0, 4000)
    ax.set_xlim(-25,20)
    #fig.savefig(MEAN_PROFILES_archive+'/MEANPROFILES_NUCAPS_1000m_0000_'+firstobj.strftime('%Y%m'))
    



    ### BIAS ###
    ##########################################
    fig = plt.figure(figsize = (12,18))
    ax1 = fig.add_axes([0.1,0.1,0.4,0.8])
    ax2 = fig.add_axes([0.5,0.1,0.2,0.8])
       
    ax1.set_ylabel('Altitude [m]', fontsize = 30)
    ax1.set_xlabel('Temperature [°C]', fontsize = 30)
    ax1.tick_params(labelsize = 30)
    ax1.set_title('Bias', fontsize = 30)
    ax1.set_ylim(0, 13000)
    ax1.set_xlim(-4, 4.1)
    ax1.set_yticks(np.arange(0,13000, 1000))
    ax1.set_xticks(np.arange(-4, 4.1, 2))
    ax1.axvspan(-2, -1, alpha=0.5, color='grey', zorder = 0)
    ax1.axvspan(1, 2, alpha=0.5, color='grey', zorder = 0)
    ax1.axvspan(-1, 1, alpha=0.5, color='lightgrey', zorder = 0)
    ax1.vlines(0, 0, 13000, color ='black', linestyle = "--")
    ax1.vlines(-1,0,13000, color = 'black', linestyle = "--", linewidth = 3)
    ax1.vlines(1,0,13000, color = 'black', linestyle = "--", linewidth = 3)
    ax1.vlines(-2,0,13000, color = 'darkslategrey', linestyle = "--", linewidth = 3)
    ax1.vlines(2,0,13000, color = 'darkslategrey', linestyle = "--", linewidth = 3)
    ax1.grid()  
    
    ax2.set_xlabel('Absolute #', fontsize = 30)
    ax2.tick_params(labelsize = 30)
    ax2.set_title('# of measurements', fontsize = 30)
    ax2.set_ylim(0, 13000)
    ax2.set_yticks([])
    ax2.set_xlim(0, 450)
    ax2.grid()  
    
    ##########################################
    ##### C) NUCAPS: Satellite data - GREENISH
    ########################################## 
    # all
    ax1.plot(diff_temp_mean_NUCAPS_all.diff_temp, diff_temp_mean_NUCAPS_all.altitude_m, color = 'darkslategrey', linewidth = 4, label = 'T NUCAPS all', zorder = 0)
    ax1.plot(diff_temp_d_mean_NUCAPS_all.diff_temp_d_mean, diff_temp_d_mean_NUCAPS_all.altitude_m, color = 'darkslategrey', linewidth = 4, label = 'Td NUCAPS all', linestyle = '--', zorder = 1)
     
    ### quality flag ###    
    # times with quality flag 0
    ax1.plot(diff_temp_mean_NUCAPS_0.diff_temp, diff_temp_mean_NUCAPS_0.altitude_m, color = 'green', linewidth = 3, label = 'T NUCAPS 0', zorder = 0)
    ax1.plot(diff_temp_d_mean_NUCAPS_0.diff_temp_d_mean, diff_temp_d_mean_NUCAPS_0.altitude_m, color = 'green', linewidth = 3, label = 'Td NUCAPS 0', linestyle = '--', zorder = 1)

    # times with quality flag 1
    ax1.plot(diff_temp_mean_NUCAPS_1.diff_temp, diff_temp_mean_NUCAPS_1.altitude_m, color = 'lawngreen', linewidth = 3, label = 'T NUCAPS 1', zorder = 0)
    ax1.plot(diff_temp_d_mean_NUCAPS_1.diff_temp_d_mean, diff_temp_d_mean_NUCAPS_1.altitude_m, color = 'lawngreen', linewidth = 3, label = 'Td NUCAPS 1', linestyle = '--', zorder = 1)
    
    # times with quality flag 9
    #ax1.plot(diff_temp_mean_NUCAPS_9.diff_temp, diff_temp_mean_NUCAPS_9.altitude_m, color = 'lightgreen', linewidth = 2, label = 'T NUCAPS 9', zorder = 0)
    #ax1.plot(diff_temp_d_mean_NUCAPS_9.diff_temp_d_mean, diff_temp_d_mean_NUCAPS_9.altitude_m, color = 'lightgreen', linewidth = 2, label = 'Td NUCAPS 9', linestyle = '--', zorder = 1)
    
    #####
    # all
    ax2.plot(number_temp_NUCAPS_all.number, number_temp_NUCAPS_all.altitude_m, color = 'darkslategrey', linewidth = 6 , zorder = 0)
    ax2.plot(number_temp_d_NUCAPS_all.number, number_temp_NUCAPS_all.altitude_m, color = 'darkslategrey', linewidth = 6, linestyle = '--', zorder = 1)
    
    ### quality flag ###    
    # times with quality flag 0
    ax2.plot(number_temp_NUCAPS_0.number, number_temp_NUCAPS_0.altitude_m, color = 'green', linewidth = 2,  zorder = 0)
    ax2.plot(number_temp_d_NUCAPS_0.number, number_temp_NUCAPS_0.altitude_m, color = 'green', linewidth = 2, linestyle = '--', zorder = 1)
    
    # times with quality flag 1
    ax2.plot(number_temp_NUCAPS_1.number, number_temp_NUCAPS_1.altitude_m, color = 'lawngreen', linewidth = 2, zorder = 0)
    ax2.plot(number_temp_d_NUCAPS_1.number, number_temp_NUCAPS_1.altitude_m, color = 'lawngreen', linewidth = 2, linestyle = '--', zorder = 1)
    
    # times with quality flag 9
    #ax2.plot(number_temp_NUCAPS_9.number, number_temp_NUCAPS_9.altitude_m, color = 'lightgreen', linewidth = 2,  zorder = 0)
    #ax2.plot(number_temp_d_NUCAPS_9.number, number_temp_NUCAPS_9.altitude_m, color = 'lightgreen', linewidth = 2, linestyle = '--', zorder = 1)
    
    
    ##########################################
    #### D) RALMO: Raman lidar - REDISH
    #########################################
    ax1.plot(diff_temp_mean_RA.diff_temp, diff_temp_mean_RA.altitude_m, color = 'maroon', linewidth = 3, label = 'T RA', zorder = 0)
    ax1.plot(diff_temp_d_mean_RA.diff_temp_d_mean, diff_temp_d_mean_RA.altitude_m, color = 'maroon', linewidth = 3, linestyle = '--',label = 'Td RA', zorder = 1)
    
    #ax1.plot(diff_temp_mean_RA_korr_unc_abs.diff_temp, diff_temp_mean_RA_korr_unc_abs.altitude_m, color = 'orangered', linewidth = 3, label = 'T korr unc abs', zorder = 1000)
    #ax1.plot(diff_temp_d_mean_RA_korr_unc_abs.diff_temp_d_mean, diff_temp_d_mean_RA_korr_unc_abs.altitude_m, color = 'orangered', linestyle = '--',linewidth = 3, label = 'Td korr unc abs', zorder = 1)
    
     #ax1.plot(diff_temp_mean_RA_korr_unc_rel.diff_temp, diff_temp_mean_RA_korr_unc_rel.altitude_m, color = 'magenta', linewidth = 3, label = 'T korr unc rel', zorder = 0)
    #ax1.plot(diff_temp_d_mean_RA_korr_unc_rel.diff_temp_d_mean, diff_temp_d_mean_RA_korr_unc_rel.altitude_m, color = 'magenta', linestyle = '--',linewidth = 3, label = 'Td korr unc rel', zorder = 1)
    
    # uncertainty
    #ax1.plot(RA_mean_temp_uncertainty.mean_all, RA_mean_temp_uncertainty.altitude_m, color = 'magenta', linewidth = 2, label = 'T uncertainty', zorder = 0)
    #ax1.plot(np.abs(RA_mean_temp_d_uncertainty.mean_all), RA_mean_temp_d_uncertainty.altitude_m, color = 'aqua', linewidth = 2, label = 'Td uncertainty', zorder = 1)
    
    #####
    ax2.plot(number_temp_RA.number, number_temp_RA.altitude_m, color = 'maroon', linewidth = 3, zorder = 0)
    ax2.plot(number_temp_d_RA.number, number_temp_d_RA.altitude_m, color = 'maroon', linewidth = 3,   zorder = 1)
    
    #ax2.plot(number_temp_RA_korr_unc_abs.number, number_temp_RA_korr_unc_abs.altitude_m, color = 'orangered', linewidth = 3,  zorder = 0)
    #ax2.plot(number_temp_d_RA_korr_unc_abs.number, number_temp_d_RA_korr_unc_abs.altitude_m, color = 'orangered', linewidth = 3, linestyle = '--', zorder = 1)
    
    #ax2.plot(number_temp_RA_korr_unc_rel.number, number_temp_RA_korr_unc_rel.altitude_m, color = 'magenta', linewidth = 3,  zorder = 0)
    #ax2.plot(number_temp_d_RA_korr_unc_rel.number, number_temp_d_RA_korr_unc_rel.altitude_m, color = 'magenta', linewidth = 3, linestyle = '--', zorder = 1)
    
    ##########################################
    ## E) RM: Radiometer  - BLACKISH
    ##########################################
    ax1.plot(diff_temp_mean_RM.diff_temp, diff_temp_mean_RM.altitude_m, color = 'navy', linewidth = 3, label = 'T RM', zorder = 0)
    ax1.plot(diff_temp_d_mean_RM.diff_temp_d_mean, diff_temp_d_mean_RM.altitude_m, color = 'navy', linewidth = 3, linestyle = '--',label = 'Td RM', zorder = 1)
    
    #####
    ax2.plot(number_temp_RM.number, number_temp_RM.altitude_m, color = 'navy', linewidth = 2,  zorder = 0)
    ax2.plot(number_temp_d_RM.number, number_temp_RM.altitude_m, color = 'navy', linewidth = 2, linestyle = 'dotted', zorder = 1)
    
    
    #fig.savefig(BIAS_archive + '/BIAS_NUCAPS_1000m_1200_ALL', dpi=300, bbox_inches = "tight")
    ax1.legend(fontsize = 25)

    
    
    
    ### STD ###
    ##########################################
    fig = plt.figure(figsize = (12,18))
    ax1 = fig.add_axes([0.1,0.1,0.4,0.8])
    ax2 = fig.add_axes([0.5,0.1,0.2,0.8])
    
    #ax1.set_ylabel('Altitude [m]', fontsize = 30)
    ax1.set_xlabel('Temperature [°C]', fontsize = 30)
    ax1.tick_params(labelsize = 30)
    ax1.set_title('Std', fontsize = 30)
    ax1.set_ylim(0, 13000)
    ax1.set_xlim(0, 8)
    ax1.set_yticks(np.arange(0,13000, 1000))
    ax1.set_xticks(np.arange(0, 8, 2))
    ax1.axvspan(0, 1, alpha=0.5, color='dimgrey', zorder = 0)
    ax1.axvspan(1, 2, alpha=0.5, color='grey', zorder = 0)
    ax1.axvspan(2, 6, alpha=0.5, color='lightgrey', zorder = 0)
    ax1.grid()  
    
    ax2.set_xlabel('Absolute #', fontsize = 30)
    ax2.tick_params(labelsize = 30)
    ax2.set_title('# of measurements', fontsize = 30)
    ax2.set_yticks(np.arange(0,13000, 1000))
    #ax2.set_yticklabels(ax2.yaxis.get_ticklabels()[::4])
    ax2.yaxis.tick_right()
    ax2.set_ylim(0, 13000)
    ax2.set_xlim(0,450)
    ax2.set_xticks(np.arange(0,450,200))
    ax2.grid()
     
    ##########################################
    ##### C) NUCAPS: Satellite data #####
    ########################################## 
    # all
    ax1.plot(std_temp_NUCAPS_all.std_temp, std_temp_NUCAPS_all.altitude_m, color = 'darkslategrey', linewidth = 6, label = 'T NUCAPS all', zorder = 0)
    ax1.plot(std_temp_d_NUCAPS_all.std_temp_d, std_temp_NUCAPS_all.altitude_m, color = 'darkslategrey', linewidth = 6, label = 'Td NUCAPS all', linestyle = '--',zorder = 1)
    
    ### quality flags ###
    # times with quality flag 0
    ax1.plot(std_temp_NUCAPS_0.std_temp, std_temp_NUCAPS_0.altitude_m, color = 'green', linewidth = 2, label = 'T NUCAPS 0', zorder = 0)
    ax1.plot(std_temp_d_NUCAPS_0.std_temp_d, std_temp_NUCAPS_0.altitude_m, color = 'green', linewidth =2, label = 'Td NUCAPS 0', linestyle = '--',zorder = 1)
    
    # times with quality flag 1
    ax1.plot(std_temp_NUCAPS_1.std_temp, std_temp_NUCAPS_1.altitude_m, color = 'lawngreen', linewidth =2, label = 'T NUCAPS 1', zorder = 0)
    ax1.plot(std_temp_d_NUCAPS_1.std_temp_d, std_temp_NUCAPS_1.altitude_m, color = 'lawngreen', linewidth = 2, label = 'Td NUCAPS 1', linestyle = '--',zorder = 1)
    
    # times with quality flag 9
    #ax1.plot(std_temp_NUCAPS_9.std_temp, std_temp_NUCAPS_9.altitude_m, color = 'lightgreen', linewidth = 2, label = 'T', zorder = 0)
    #ax1.plot(std_temp_d_NUCAPS_9.std_temp_d, std_temp_NUCAPS_9.altitude_m, color = 'lightgreen', linewidth = 2, label = 'Td', zorder = 1)
    
    
    #####
    ### NUCAPS ###
    # all
    ax2.plot(number_temp_NUCAPS_all.number, number_temp_NUCAPS_all.altitude_m, color = 'darkslategrey', linewidth = 6,  zorder = 0)
    ax2.plot(number_temp_d_NUCAPS_all.number, number_temp_d_NUCAPS_all.altitude_m, color = 'darkslategrey', linewidth = 6, linestyle = '--', zorder = 1)
    
    ### quality flag ###    
    # times with quality flag 0
    ax2.plot(number_temp_NUCAPS_0.number, number_temp_NUCAPS_0.altitude_m, color = 'green', linewidth = 2,  zorder = 0)
    ax2.plot(number_temp_d_NUCAPS_0.number, number_temp_d_NUCAPS_0.altitude_m, color = 'green', linewidth = 2, linestyle = '--', zorder = 1)
    
    # times with quality flag 1
    ax2.plot(number_temp_NUCAPS_1.number, number_temp_NUCAPS_1.altitude_m, color = 'lawngreen', linewidth = 2,  zorder = 0)
    ax2.plot(number_temp_d_NUCAPS_1.number, number_temp_d_NUCAPS_1.altitude_m, color = 'lawngreen', linewidth = 2, linestyle = '--', zorder = 1)
    
    # times with quality flag 9
    #ax2.plot(number_temp_NUCAPS_9.number, number_temp_NUCAPS_9.altitude_m, color = 'lightgreen', linewidth = 2,  zorder = 0)
    #ax2.plot(number_temp_d_NUCAPS_9.number, number_temp_d_NUCAPS_9.altitude_m, color = 'lightgreen', linewidth = 2, linestyle = '--', zorder = 1)
    
    

    ##########################################
    ##### D) RALMO: Raman lidar 
    #########################################
    ax1.plot(std_temp_RA.std_temp, std_temp_RA.altitude_m, color = 'maroon', linewidth = 3, label = 'T RA', zorder = 0)
    ax1.plot(std_temp_d_RA.std_temp_d, std_temp_RA.altitude_m, color = 'maroon', linewidth = 3, label = 'Td RA', linestyle = '--',zorder = 1)

    #ax1.plot(std_temp_RA_korr_unc_abs.std_temp, std_temp_RA_korr_unc_abs.altitude_m, color = 'orangered', linewidth = 3, label = 'T korr unc abs', zorder = 0)
    #ax1.plot(std_temp_d_RA_korr_unc_abs.std_temp_d, std_temp_RA_korr_unc_abs.altitude_m, color = 'orangered', linewidth = 3, linestyle = '--',label = 'Td korr unc abs', zorder = 1)
    
    #ax1.plot(std_temp_RA_korr_unc_rel.std_temp, std_temp_RA_korr_unc_rel.altitude_m, color = 'magenta', linewidth = 3, label = 'T korr unc rel', zorder = 0)
    #ax1.plot(std_temp_d_RA_korr_unc_rel.std_temp_d, std_temp_RA_korr_unc_rel.altitude_m, color = 'magenta', linewidth = 3, linestyle = '--',label = 'Td korr unc rel', zorder = 1)
    
   
    
    #####
    ax2.plot(number_temp_RA.number, number_temp_RA.altitude_m, color = 'maroon', linewidth = 3, zorder = 0)
    ax2.plot(number_temp_d_RA.number, number_temp_d_RA.altitude_m, color = 'maroon', linewidth = 3,  linestyle = '--', zorder = 1)
    
    #ax2.plot(number_temp_RA_korr_unc_abs.number, number_temp_RA_korr_unc_abs.altitude_m, color = 'orangered', linewidth = 3,  zorder = 0)
    #ax2.plot(number_temp_d_RA_korr_unc_abs.number, number_temp_d_RA_korr_unc_abs.altitude_m, color = 'orangered', linewidth = 3, linestyle = '--', zorder = 1)
    
    #ax2.plot(number_temp_RA_korr_unc_rel.number, number_temp_RA_korr_unc_rel.altitude_m, color = 'magenta', linewidth = 3,  zorder = 0)
    #ax2.plot(number_temp_d_RA_korr_unc_rel.number, number_temp_d_RA_korr_unc_rel.altitude_m, color = 'magenta', linewidth = 3, linestyle = '--', zorder = 1)
   
    ##########################################
    ## E) RM: Radiometer 
    ##########################################
    ax1.plot(std_temp_RM.std_temp, std_temp_RM.altitude_m, color = 'navy', linewidth = 3, label = 'T RM', zorder = 0)
    ax1.plot(std_temp_d_RM.std_temp_d, std_temp_RM.altitude_m, color = 'navy', linewidth = 3, label = 'Td RM', linestyle = '--',zorder = 1)

    #####
    ax2.plot(number_temp_RM.number, number_temp_RM.altitude_m, color = 'navy', linewidth = 3,  zorder = 0)
    ax2.plot(number_temp_d_RM.number, number_temp_RM.altitude_m, color = 'navy', linewidth = 3, linestyle = '--', zorder = 1)
    
    #fig.savefig(STD_archive+'/STD_NUCAPS_1000m_1200_ALL',dpi=300, bbox_inches = "tight")
    ax1.legend(fontsize = 25)
    
    firstobj= lastobj_month 
    

 
    





 