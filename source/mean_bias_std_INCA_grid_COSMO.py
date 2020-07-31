#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:42:49 2020

@author: nal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:49:11 2020

@author: nal
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

############################################################################# FUNCTIONS ############################################################################# 
def interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, input_data_filtered, comparison_grid):
    INCA_grid = INCA_grid.HHL[::-1]
    INCA_grid = INCA_grid.reset_index(drop=True)
    input_grid_smoothed_all = pd.DataFrame()
    while firstobj != lastobj:
        nowdate = firstobj.strftime('%Y%m%d')
        print(nowdate) 
        input_data_time = input_data_filtered[input_data_filtered.time_YMDHMS == firstobj]
        input_data_time = input_data_time.reset_index(drop=True)
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
            altitude_min = min(comparison_grid_time.altitude_m)
            altitude_max = max(comparison_grid_time.altitude_m)
                
            INCA_grid_min = find_nearest(INCA_grid, altitude_min)
            INCA_grid_max = find_nearest(INCA_grid, altitude_max)
                
            INCA_grid_lim = INCA_grid[(INCA_grid <= INCA_grid_max) & (INCA_grid >= INCA_grid_min)]
            INCA_grid_lim = INCA_grid_lim.reset_index(drop=True)
                
            input_data = input_data_time.altitude_m.reset_index(drop=True)
            input_temp = input_data_time.temperature_degC.reset_index(drop=True)
            input_temp_d = input_data_time.dew_point_degC.reset_index(drop=True)
            
            # for datasets with uncertainty indication
            if 'uncertainty_temperature_K' in input_data_time.columns: 
                print('yes')
                input_temp_unc = input_data_time.uncertainty_temperature_K.reset_index(drop=True)
                input_temp_d_unc = input_data_time['uncertainty_specific_humidity_gkg-1'].reset_index(drop=True)
                input_temperature_interp = pd.DataFrame({'temperature_mean' : griddata(input_data.values, input_temp.values, INCA_grid_lim.values)})
                input_temperature_d_interp = pd.DataFrame({'temperature_d_mean' : griddata(input_data.values, input_temp_d.values, INCA_grid_lim.values)})
                input_temperature_uncertainty = pd.DataFrame({'temperature_mean_unc' : griddata(input_data.values, input_temp_unc.values, INCA_grid_lim.values)})
                input_temperature_d_uncertainty = pd.DataFrame({'uncertainty_specific_humidity_gkg-1' : griddata(input_data.values, input_temp_d_unc.values, INCA_grid_lim.values)})
                
                input_interp = pd.DataFrame({'altitude_m':INCA_grid_lim, 'temperature_mean': input_temperature_interp.temperature_mean, 'temperature_d_mean' : input_temperature_d_interp.temperature_d_mean, 'temperature_mean_unc': input_temperature_uncertainty.temperature_mean_unc , 'uncertainty_specific_humidity_gkg-1': input_temperature_d_uncertainty['uncertainty_specific_humidity_gkg-1'], 'time_YMDHMS': firstobj})
                input_grid_smoothed_all = input_grid_smoothed_all.append(input_interp)
                firstobj= firstobj + dt.timedelta(days=1) 
                
            # for datasets with no uncertainty indication
            else: 
                input_temperature_interp = pd.DataFrame({'temperature_mean' : griddata(input_data.values, input_temp.values, INCA_grid_lim.values)})
                input_temperature_d_interp = pd.DataFrame({'temperature_d_mean' : griddata(input_data.values, input_temp_d.values, INCA_grid_lim.values)})
                input_interp = pd.DataFrame({'altitude_m':INCA_grid_lim, 'temperature_mean': input_temperature_interp.temperature_mean, 'temperature_d_mean' : input_temperature_d_interp.temperature_d_mean, 'time_YMDHMS': firstobj})              
                input_grid_smoothed_all = input_grid_smoothed_all.append(input_interp)
            
                firstobj= firstobj + dt.timedelta(days=1) 
    return input_grid_smoothed_all
     
# to calculate bias of temperature
def calc_bias_temp(input_smoothed_INCA, RS_smoothed_INCA): 
     diff_temp = np.subtract(input_smoothed_INCA.temperature_mean.reset_index(drop=True), RS_smoothed_INCA.temperature_mean.reset_index(drop=True), axis = 1)
     diff_temp = pd.DataFrame({'diff_temp':diff_temp.values, 'altitude_m': input_smoothed_INCA.altitude_m.values})
     diff_temp_mean = diff_temp.groupby('altitude_m')['diff_temp'].mean().to_frame(name='mean_all').reset_index() 
     diff_temp_mean = pd.DataFrame({'diff_temp':diff_temp_mean.mean_all, 'altitude_m': diff_temp_mean.altitude_m})
     return diff_temp_mean  

# to calculate bias of dew point temperature
def calc_bias_temp_d(input_smoothed_INCA, RS_smoothed_INCA):       
     diff_temp_d = np.subtract(input_smoothed_INCA.temperature_d_mean.reset_index(drop=True), RS_smoothed_INCA.temperature_d_mean.reset_index(drop=True), axis = 1)
     diff_temp_d = pd.DataFrame({'diff_temp_d':diff_temp_d.values, 'altitude_m': input_smoothed_INCA.altitude_m})
     diff_temp_d_mean = diff_temp_d.groupby('altitude_m')['diff_temp_d'].mean().to_frame(name='mean_all').reset_index()
     diff_temp_d_mean = pd.DataFrame({'diff_temp_d_mean':diff_temp_d_mean.mean_all, 'altitude_m': diff_temp_d_mean.altitude_m})  
     return diff_temp_d_mean
 
    
# to calculate std of temperature and dew point temperature
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
  
def subtract_running_window(firstobj, lastobj, NUCAPS_smoothed_INCA_all, NUCAPS_smoothed_INCA_all_time, RS_smoothed_INCA, window_size_days):
    for i in range(len(NUCAPS_smoothed_INCA_all)):
        print(i)
        lastobj_window = NUCAPS_smoothed_INCA_all_time.time_YMDHMS.iloc[i]
        firstobj_window = lastobj_window- timedelta(days=window_size_days)
        altitude_i  =  NUCAPS_smoothed_INCA_all.altitude_m.iloc[i]
         
        NUCAPS_smoothed_INCA_all_window = NUCAPS_smoothed_INCA_all_time[(NUCAPS_smoothed_INCA_all_time.time_YMDHMS >= firstobj_window) & (NUCAPS_smoothed_INCA_all_time.time_YMDHMS <= lastobj_window) & (NUCAPS_smoothed_INCA_all_time.altitude_m == altitude_i)]
        NUCAPS_smoothed_INCA_all_window = NUCAPS_smoothed_INCA_all_window[['altitude_m', 'temperature_mean', 'temperature_d_mean']]
        RS_smoothed_INCA_window = RS_smoothed_INCA[(RS_smoothed_INCA.time_YMDHMS >= firstobj_window) & (RS_smoothed_INCA.time_YMDHMS <= lastobj_window)]
        
       
        
        bias_t = np.subtract(NUCAPS_smoothed_INCA_all_window.temperature_mean.reset_index(drop=True), RS_smoothed_INCA_window.temperature_mean.reset_index(drop=True))
        bias_t = pd.DataFrame({'diff_temp':bias_t.reset_index(drop=True), 'altitude_m':NUCAPS_smoothed_INCA_all_window.altitude_m.reset_index(drop=True)})  
        bias_t = bias_t.astype(float)
        bias_t = bias_t.groupby('altitude_m')['diff_temp'].mean().to_frame(name='mean_all').reset_index()
        
        bias_t_d = np.subtract(NUCAPS_smoothed_INCA_all_window.temperature_d_mean.reset_index(drop=True), RS_smoothed_INCA_window.temperature_d_mean.reset_index(drop=True))
        bias_t_d = pd.DataFrame({'diff_temp_d':bias_t_d.reset_index(drop=True), 'altitude_m':NUCAPS_smoothed_INCA_all_window.altitude_m.reset_index(drop=True)})  
        bias_t_d = bias_t_d.astype(float)
        bias_t_d = bias_t_d.groupby('altitude_m')['diff_temp_d'].mean().to_frame(name='mean_all').reset_index()
        
        
        NUCAPS_smoothed_INCA_all.temperature_mean[i] = float(NUCAPS_smoothed_INCA_all.temperature_mean.iloc[i]) - float(bias_t.mean_all.values)
        NUCAPS_smoothed_INCA_all.temperature_d_mean[i] = float(NUCAPS_smoothed_INCA_all.temperature_d_mean.iloc[i]) - float(bias_t_d.mean_all.values)    
        
    return NUCAPS_smoothed_INCA_all

# to search an array for the nearest value 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


############################################################################# define time #############################################################################
### !! time span
firstdate = '2020072000000'
lastdate = '2020072200000'
firstobj=dt.datetime.strptime(firstdate,'%Y%m%d%H%M%S')
lastobj=dt.datetime.strptime(lastdate,'%Y%m%d%H%M%S')

### !! daytime ### (midnight: 0000, noon: 1200)
daytime = 'midnight' # alternatively 'noon' possible
if daytime == 'midnight':
   DT = 0
   DT_str = 2 * str(0)
   DT_NUCAPS = 23
else:
   DT = 12
   DT_NUCAPS = 11
   
RS_archive   = '/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/Payerne/'

##################################################################### define paths and read data ###################################################################       
##########################################
## COSMO 
##########################################
COSMO_coordinate = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/COSMO/COSMO_coordindate_payerne.csv')
INCA_coordinate = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/COSMO//INCA_coordindate_payerne.csv')
INCA_grid_1 = xr.open_dataset('//data/COALITION2/PicturesSatellite/results_NAL/COSMO///inca_topo_levels_hsurf_ccs4.nc')
INCA_grid = pd.read_csv('/data/COALITION2/PicturesSatellite/results_NAL/INCA_grid.csv') 
INCA_data = INCA_grid_1.where((INCA_grid_1.lon_1 == float(INCA_coordinate.lon_1.values)) & (INCA_grid_1.lat_1 == float(INCA_coordinate.lat_1.values)), drop = True)
h_COSMO = INCA_data.HHL[:,0,0].to_pandas().rename('altitude_m').reset_index(drop=True)
h_COSMO_mean = pd.DataFrame()
for i in range(0, len(h_COSMO)-1):
    print(i)
    h_COSMO_i = pd.DataFrame({'altitude_m' : np.mean([h_COSMO[i], h_COSMO[i+1]])}, index = [0])
    h_COSMO_mean = h_COSMO_mean.append(h_COSMO_i)
        
h_COSMO = h_COSMO_mean.reset_index(drop=True)   

# 12_00
COSMO_data_12_00 = xr.open_mfdataset('/data/COALITION2/database/cosmo/T-TD_3D/*00_00.nc')
COSMO_data_12_00 =  COSMO_data_12_00.where(( COSMO_data_12_00.lon_1 == float(COSMO_coordinate.lon_1.values)) & ( COSMO_data_12_00.lat_1 == float(COSMO_coordinate.lat_1.values)), drop=True)
    
p_COSMO = pd.DataFrame({'pressure_hPa': (COSMO_data_12_00['p_inca'][:,:,0,0].values.flatten()/100)}).reset_index(drop=True)
t_COSMO = pd.DataFrame({'temperature_degC' : (COSMO_data_12_00['t_inca'][:,:,0,0].values.flatten()-273.15)}).reset_index(drop=True)
spec_hum = pd.DataFrame({'specific_humidity_gkg-1' : (COSMO_data_12_00['qv_inca'][:,:,0,0].values.flatten()) * 1000}).reset_index(drop=True)
temp_d_COSMO = pd.DataFrame({'dew_point_degC' : (cc.dewpoint_from_specific_humidity(spec_hum.values * units('g/kg'), t_COSMO.values * units.degC, p_COSMO.values * units.hPa).magnitude).flatten()}).reset_index(drop=True)
time_COSMO = pd.DataFrame({'time_YMDHMS' : np.repeat(COSMO_data_12_00.time, 50)})
h_data_12_00 = h_COSMO.append([h_COSMO] * int((len(p_COSMO)/50)-1)).reset_index(drop=True)
    
COSMO_data_12_00 = pd.concat([p_COSMO, t_COSMO,  temp_d_COSMO, time_COSMO,  h_data_12_00], axis = 1)


# 03_09
COSMO_data_03_09 = xr.open_mfdataset('/data/COALITION2/database/cosmo/T-TD_3D/*15_09.nc')
COSMO_data_03_09 =  COSMO_data_03_09.where(( COSMO_data_03_09.lon_1 == float(COSMO_coordinate.lon_1.values)) & ( COSMO_data_03_09.lat_1 == float(COSMO_coordinate.lat_1.values)), drop=True)
    
p_COSMO = pd.DataFrame({'pressure_hPa': (COSMO_data_03_09['p_inca'][:,:,0,0].values.flatten()/100)}).reset_index(drop=True)
t_COSMO = pd.DataFrame({'temperature_degC' : (COSMO_data_03_09['t_inca'][:,:,0,0].values.flatten()-273.15)}).reset_index(drop=True)
spec_hum = pd.DataFrame({'specific_humidity_gkg-1' : (COSMO_data_03_09['qv_inca'][:,:,0,0].values.flatten()) * 1000}).reset_index(drop=True)
temp_d_COSMO = pd.DataFrame({'dew_point_degC' : (cc.dewpoint_from_specific_humidity(spec_hum.values * units('g/kg'), t_COSMO.values * units.degC, p_COSMO.values * units.hPa).magnitude).flatten()}).reset_index(drop=True)
time_COSMO = pd.DataFrame({'time_YMDHMS' : np.repeat(COSMO_data_03_09.time, 50)})
h_data_03_09 = h_COSMO.append([h_COSMO] * int((len(p_COSMO)/50)-1)).reset_index(drop=True)
    
COSMO_data_03_09 = pd.concat([p_COSMO, t_COSMO,  temp_d_COSMO, time_COSMO,  h_data_03_09], axis = 1)


# 06_06
COSMO_data_06_06 = xr.open_mfdataset('/data/COALITION2/database/cosmo/T-TD_3D/*18_06.nc')
COSMO_data_06_06 =  COSMO_data_06_06.where(( COSMO_data_06_06.lon_1 == float(COSMO_coordinate.lon_1.values)) & ( COSMO_data_06_06.lat_1 == float(COSMO_coordinate.lat_1.values)), drop=True)
    
p_COSMO = pd.DataFrame({'pressure_hPa': (COSMO_data_06_06['p_inca'][:,:,0,0].values.flatten()/100)}).reset_index(drop=True)
t_COSMO = pd.DataFrame({'temperature_degC' : (COSMO_data_06_06['t_inca'][:,:,0,0].values.flatten()-273.15)}).reset_index(drop=True)
spec_hum = pd.DataFrame({'specific_humidity_gkg-1' : (COSMO_data_06_06['qv_inca'][:,:,0,0].values.flatten()) * 1000}).reset_index(drop=True)
temp_d_COSMO = pd.DataFrame({'dew_point_degC' : (cc.dewpoint_from_specific_humidity(spec_hum.values * units('g/kg'), t_COSMO.values * units.degC, p_COSMO.values * units.hPa).magnitude).flatten()}).reset_index(drop=True)
time_COSMO = pd.DataFrame({'time_YMDHMS' : np.repeat(COSMO_data_06_06.time, 50)})
h_data_06_06 = h_COSMO.append([h_COSMO] * int((len(p_COSMO)/50)-1)).reset_index(drop=True)
    
COSMO_data_06_06 = pd.concat([p_COSMO, t_COSMO,  temp_d_COSMO, time_COSMO,  h_data_06_06], axis = 1)
    
# 00_12
COSMO_data_00_12 = xr.open_mfdataset('/data/COALITION2/database/cosmo/T-TD_3D/*12_12.nc')
COSMO_data_00_12 =  COSMO_data_00_12.where(( COSMO_data_00_12.lon_1 == float(COSMO_coordinate.lon_1.values)) & ( COSMO_data_00_12.lat_1 == float(COSMO_coordinate.lat_1.values)), drop=True)

p_COSMO = pd.DataFrame({'pressure_hPa': (COSMO_data_00_12['p_inca'][:,:,0,0].values.flatten()/100)}).reset_index(drop=True)
t_COSMO = pd.DataFrame({'temperature_degC' : (COSMO_data_00_12['t_inca'][:,:,0,0].values.flatten()-273.15)}).reset_index(drop=True)
spec_hum = pd.DataFrame({'specific_humidity_gkg-1' : (COSMO_data_00_12['qv_inca'][:,:,0,0].values.flatten()) * 1000}).reset_index(drop=True)
temp_d_COSMO = pd.DataFrame({'dew_point_degC' : (cc.dewpoint_from_specific_humidity(spec_hum.values * units('g/kg'), t_COSMO.values * units.degC, p_COSMO.values * units.hPa).magnitude).flatten()}).reset_index(drop=True)
time_COSMO = pd.DataFrame({'time_YMDHMS' : np.repeat(COSMO_data_00_12.time, 50)})
h_data_00_12 = h_COSMO.append([h_COSMO] * int((len(p_COSMO)/50)-1)).reset_index(drop=True)
    
COSMO_data_00_12 = pd.concat([p_COSMO, t_COSMO,  temp_d_COSMO, time_COSMO,  h_data_00_12], axis = 1)
    
    
# 09_03
COSMO_data_09_03 = xr.open_mfdataset('/data/COALITION2/database/cosmo/T-TD_3D/*21_03.nc')
COSMO_data_09_03 =  COSMO_data_09_03.where(( COSMO_data_09_03.lon_1 == float(COSMO_coordinate.lon_1.values)) & ( COSMO_data_09_03.lat_1 == float(COSMO_coordinate.lat_1.values)), drop=True)

p_COSMO = pd.DataFrame({'pressure_hPa': (COSMO_data_09_03['p_inca'][:,:,0,0].values.flatten()/100)}).reset_index(drop=True)
t_COSMO = pd.DataFrame({'temperature_degC' : (COSMO_data_09_03['t_inca'][:,:,0,0].values.flatten()-273.15)}).reset_index(drop=True)
spec_hum = pd.DataFrame({'specific_humidity_gkg-1' : (COSMO_data_09_03['qv_inca'][:,:,0,0].values.flatten()) * 1000}).reset_index(drop=True)
temp_d_COSMO = pd.DataFrame({'dew_point_degC' : (cc.dewpoint_from_specific_humidity(spec_hum.values * units('g/kg'), t_COSMO.values * units.degC, p_COSMO.values * units.hPa).magnitude).flatten()}).reset_index(drop=True)
time_COSMO = pd.DataFrame({'time_YMDHMS' : np.repeat(COSMO_data_09_03.time, 50)})
h_data_09_03 = h_COSMO.append([h_COSMO] * int((len(p_COSMO)/50)-1)).reset_index(drop=True)
    
COSMO_data_09_03 = pd.concat([p_COSMO, t_COSMO,  temp_d_COSMO, time_COSMO,  h_data_09_03], axis = 1)

##########################################
##### RS: Radiosonde
##########################################
Times = COSMO_data_00_12.time_YMDHMS.unique()
RS_data = pd.DataFrame()
for i in range(len(Times)): 
    print(i)
    Time_x = pd.to_datetime(Times[i])
    RS_data = RS_data.append(pd.read_csv(RS_archive+Time_x.strftime("%Y")+'/'+Time_x.strftime("%m")+'/'+Time_x.strftime("%d")+'/RS_'+'06610'+'_'+Time_x.strftime('%Y%m%d%H')+'.txt'))
    
RS_data = RS_data.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent', '742':'altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })

RS_data = RS_data[RS_data.temperature_degC != 10000000.0]
RS_data['time_YMDHMS'] = pd.to_datetime(RS_data.time_YMDHMS, format = '%Y%m%d%H%M%S')


RS_data_filtered = RS_data.reset_index()
firstobj = np.min(RS_data.time_YMDHMS)
lastobj = np.max(RS_data.time_YMDHMS)
    
####################################################### interpolate to INCA grid and calculate mean profile #######################################################
RS_smoothed_COSMO_12_00 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, COSMO_data_12_00)
RS_smoothed_COSMO_mean_temp_12_00 = RS_smoothed_COSMO_12_00.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
RS_smoothed_COSMO_mean_temp_d_12_00 = RS_smoothed_COSMO_12_00.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()

COSMO_smoothed_COSMO_12_00 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, COSMO_data_12_00, COSMO_data_12_00)
COSMO_smoothed_COSMO_mean_temp_12_00 = COSMO_smoothed_COSMO_12_00.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
COSMO_smoothed_COSMO_mean_temp_d_12_00 = COSMO_smoothed_COSMO_12_00.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()
    
    

RS_smoothed_COSMO_03_09 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, COSMO_data_03_09)
RS_smoothed_COSMO_mean_temp_03_09 = RS_smoothed_COSMO_03_09.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
RS_smoothed_COSMO_mean_temp_d_03_09 = RS_smoothed_COSMO_03_09.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()

COSMO_smoothed_COSMO_03_09 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, COSMO_data_03_09, COSMO_data_03_09)
COSMO_smoothed_COSMO_mean_temp_03_09 = COSMO_smoothed_COSMO_03_09.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
COSMO_smoothed_COSMO_mean_temp_d_03_09 = COSMO_smoothed_COSMO_03_09.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()

   

RS_smoothed_COSMO_06_06 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, COSMO_data_06_06)
RS_smoothed_COSMO_mean_temp_06_06 = RS_smoothed_COSMO_06_06.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
RS_smoothed_COSMO_mean_temp_d_06_06 = RS_smoothed_COSMO_06_06.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()

COSMO_smoothed_COSMO_06_06 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, COSMO_data_06_06, COSMO_data_06_06)
COSMO_smoothed_COSMO_mean_temp_06_06 = COSMO_smoothed_COSMO_06_06.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
COSMO_smoothed_COSMO_mean_temp_d_06_06 = COSMO_smoothed_COSMO_06_06.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()


RS_smoothed_COSMO_00_12 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, COSMO_data_00_12)
RS_smoothed_COSMO_mean_temp_00_12 = RS_smoothed_COSMO_00_12.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
RS_smoothed_COSMO_mean_temp_d_00_12= RS_smoothed_COSMO_00_12.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()

COSMO_smoothed_COSMO_00_12 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, COSMO_data_00_12, COSMO_data_00_12)
COSMO_smoothed_COSMO_mean_temp_00_12 = COSMO_smoothed_COSMO_00_12.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
COSMO_smoothed_COSMO_mean_temp_d_00_12 = COSMO_smoothed_COSMO_00_12.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()


RS_smoothed_COSMO_09_03 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, RS_data_filtered, COSMO_data_09_03)
RS_smoothed_COSMO_mean_temp_09_03 = RS_smoothed_COSMO_09_03.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
RS_smoothed_COSMO_mean_temp_d_09_03= RS_smoothed_COSMO_09_03.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()

COSMO_smoothed_COSMO_09_03 = interpolate_RS_to_INCA_grid(firstobj, lastobj, INCA_grid, COSMO_data_09_03, COSMO_data_09_03)
COSMO_smoothed_COSMO_mean_temp_09_03 = COSMO_smoothed_COSMO_09_03.groupby('altitude_m')['temperature_mean'].mean().to_frame(name='mean_all').reset_index()
COSMO_smoothed_COSMO_mean_temp_d_09_03 = COSMO_smoothed_COSMO_09_03.groupby('altitude_m')['temperature_d_mean'].mean().to_frame(name='mean_all').reset_index()


####################################################### CALCULATE BIAS AND STD ############################################################################################## 
### BIAS ###
##########################################
##### COSMO #####
##########################################
# bewteen Radiosonde and all NUCAPS
diff_temp_mean_COSMO_12_00 = calc_bias_temp( COSMO_smoothed_COSMO_12_00, RS_smoothed_COSMO_12_00)
diff_temp_d_mean_COSMO_12_00 = calc_bias_temp_d(COSMO_smoothed_COSMO_12_00, RS_smoothed_COSMO_12_00)

diff_temp_mean_COSMO_09_03 = calc_bias_temp( COSMO_smoothed_COSMO_09_03, RS_smoothed_COSMO_09_03)
diff_temp_d_mean_COSMO_09_03 = calc_bias_temp_d(COSMO_smoothed_COSMO_09_03, RS_smoothed_COSMO_09_03)

diff_temp_mean_COSMO_06_06 = calc_bias_temp( COSMO_smoothed_COSMO_06_06, RS_smoothed_COSMO_06_06)
diff_temp_d_mean_COSMO_06_06 = calc_bias_temp_d(COSMO_smoothed_COSMO_06_06, RS_smoothed_COSMO_06_06)

diff_temp_mean_COSMO_03_09 = calc_bias_temp( COSMO_smoothed_COSMO_03_09, RS_smoothed_COSMO_03_09)
diff_temp_d_mean_COSMO_03_09 = calc_bias_temp_d(COSMO_smoothed_COSMO_03_09, RS_smoothed_COSMO_03_09)

diff_temp_mean_COSMO_00_12 = calc_bias_temp( COSMO_smoothed_COSMO_00_12, RS_smoothed_COSMO_00_12)
diff_temp_d_mean_COSMO_00_12 = calc_bias_temp_d(COSMO_smoothed_COSMO_00_12, RS_smoothed_COSMO_00_12)
 

   
### STD ###
##########################################
##### COSMO #####
##########################################  
# bewteen Radiosonde and all NUCAPS
std_temp_COSMO_12_00, std_temp_d_COSMO_12_00, number_temp_COSMO_12_00, number_temp_d_COSMO_12_00 = calc_std_temp(COSMO_smoothed_COSMO_12_00, RS_smoothed_COSMO_12_00)

std_temp_COSMO_09_03, std_temp_d_COSMO_09_03, number_temp_COSMO_09_03, number_temp_d_COSMO_09_03 = calc_std_temp(COSMO_smoothed_COSMO_09_03, RS_smoothed_COSMO_09_03)

std_temp_COSMO_06_06, std_temp_d_COSMO_06_06, number_temp_COSMO_06_06, number_temp_d_COSMO_06_06 = calc_std_temp(COSMO_smoothed_COSMO_06_06, RS_smoothed_COSMO_06_06)

std_temp_COSMO_03_09, std_temp_d_COSMO_03_09, number_temp_COSMO_03_09, number_temp_d_COSMO_03_09 = calc_std_temp(COSMO_smoothed_COSMO_03_09, RS_smoothed_COSMO_03_09)

std_temp_COSMO_00_12, std_temp_d_COSMO_00_12, number_temp_COSMO_00_12, number_temp_d_COSMO_00_12 = calc_std_temp(COSMO_smoothed_COSMO_00_12, RS_smoothed_COSMO_00_12)



###################################################### PLOT MEAN PROFILE, BIAS AND STD ###################################################################################### 




### BIAS ###
##########################################
fig = plt.figure(figsize = (18,23))
ax1 = fig.add_axes([0.1,0.1,0.4,0.8])
ax2 = fig.add_axes([0.5,0.1,0.2,0.8])
       
ax1.set_ylabel('Altitude [m]', fontsize = 30)
ax1.set_xlabel('Temperature [°C]', fontsize = 30)
ax1.tick_params(labelsize = 30)
ax1.set_title('Bias', fontsize = 30)
ax1.set_ylim(0, 13000)
ax1.set_xlim(-2, 2)
ax1.set_yticks(np.arange(0,13000, 1000))
ax1.set_xticks(np.arange(-2, 2, 1))
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
ax2.set_yticks(np.arange(0,13000, 1000))
#ax2.set_xticks(np.arange(0, 26, 5))
ax2.set_yticklabels(ax2.yaxis.get_ticklabels()[::4])
ax2.yaxis.tick_right()
ax2.set_ylim(0, 13000)
#ax2.set_xlim(0, 30, 5)
ax2.grid()  
    
    
##########################################
##### C) NUCAPS: Satellite data - GREENISH
########################################## 
ax1.plot( diff_temp_mean_COSMO_12_00.diff_temp,  diff_temp_mean_COSMO_12_00.altitude_m, color = 'midnightblue', linewidth = 5, label = 'T COSMO, 12_00', zorder = 0)
ax1.plot( diff_temp_d_mean_COSMO_12_00.diff_temp_d_mean,  diff_temp_d_mean_COSMO_12_00.altitude_m, color = 'midnightblue', linewidth = 5, linestyle = '--',label = 'Td COSMO, 12_00', zorder = 1)

ax1.plot( diff_temp_mean_COSMO_09_03.diff_temp,  diff_temp_mean_COSMO_09_03.altitude_m, color = 'darkmagenta', linewidth = 3, label = 'T COSMO, 09_03', zorder = 0)
ax1.plot( diff_temp_d_mean_COSMO_09_03.diff_temp_d_mean,  diff_temp_d_mean_COSMO_09_03.altitude_m, color = 'darkmagenta', linewidth = 3, linestyle = '--',label = 'Td COSMO, 03_09', zorder = 1)

ax1.plot( diff_temp_mean_COSMO_06_06.diff_temp,  diff_temp_mean_COSMO_06_06.altitude_m, color = 'crimson', linewidth = 3, label = 'T COSMO, 06_06', zorder = 0)
ax1.plot( diff_temp_d_mean_COSMO_06_06.diff_temp_d_mean,  diff_temp_d_mean_COSMO_06_06.altitude_m, color = 'crimson', linewidth = 3, linestyle = '--',label = 'Td COSMO, 06_06', zorder = 1)

ax1.plot( diff_temp_mean_COSMO_03_09.diff_temp,  diff_temp_mean_COSMO_03_09.altitude_m, color = 'magenta', linewidth = 3, label = 'T COSMO, 03_09', zorder = 0)
ax1.plot( diff_temp_d_mean_COSMO_03_09.diff_temp_d_mean,  diff_temp_d_mean_COSMO_03_09.altitude_m, color = 'magenta', linewidth = 3, linestyle = '--',label = 'Td COSMO, 03_09', zorder = 1)

ax1.plot( diff_temp_mean_COSMO_00_12.diff_temp,  diff_temp_mean_COSMO_00_12.altitude_m, color = 'pink', linewidth = 3, label = 'T COSMO, 00_12', zorder = 0)
ax1.plot( diff_temp_d_mean_COSMO_00_12.diff_temp_d_mean,  diff_temp_d_mean_COSMO_00_12.altitude_m, color = 'pink', linewidth = 3, linestyle = '--',label = 'Td COSMO, 00_12', zorder = 1)
    
#####
ax2.plot(number_temp_COSMO_12_00.number, number_temp_COSMO_12_00.altitude_m, color = 'midnightblue', linewidth = 5,  zorder = 0)
ax2.plot(number_temp_d_COSMO_12_00.number, number_temp_COSMO_12_00.altitude_m, color = 'midnightblue', linewidth = 5, linestyle = 'dotted', zorder = 1)

ax2.plot(number_temp_COSMO_09_03.number, number_temp_COSMO_09_03.altitude_m, color = 'darkmagenta', linewidth = 2,  zorder = 0)
ax2.plot(number_temp_d_COSMO_09_03.number, number_temp_COSMO_09_03.altitude_m, color = 'darkmagenta', linewidth = 2, linestyle = 'dotted', zorder = 1)

ax2.plot(number_temp_COSMO_06_06.number, number_temp_COSMO_06_06.altitude_m, color = 'crimson', linewidth = 2,  zorder = 0)
ax2.plot(number_temp_d_COSMO_06_06.number, number_temp_COSMO_06_06.altitude_m, color = 'crimson', linewidth = 2, linestyle = 'dotted', zorder = 1)

ax2.plot(number_temp_COSMO_03_09.number, number_temp_COSMO_03_09.altitude_m, color = 'magenta', linewidth = 2,  zorder = 0)
ax2.plot(number_temp_d_COSMO_03_09.number, number_temp_COSMO_03_09.altitude_m, color = 'magenta', linewidth = 2, linestyle = 'dotted', zorder = 1)

ax2.plot(number_temp_COSMO_00_12.number, number_temp_COSMO_00_12.altitude_m, color = 'pink', linewidth = 2,  zorder = 0)
ax2.plot(number_temp_d_COSMO_00_12.number, number_temp_COSMO_00_12.altitude_m, color = 'pink', linewidth = 2, linestyle = 'dotted', zorder = 1)
    
#fig.savefig(BIAS_archive + '/BIAS_NUCAPS_1000m_1200_ALL', dpi=300, bbox_inches = "tight")
ax1.legend(fontsize = 25)

    
    
    
### STD ###
##########################################
fig = plt.figure(figsize = (18,23))
ax1 = fig.add_axes([0.1,0.1,0.4,0.8])
ax2 = fig.add_axes([0.5,0.1,0.2,0.8])
    
#ax1.set_ylabel('Altitude [m]', fontsize = 30)
ax1.set_xlabel('Temperature [°C]', fontsize = 30)
ax1.tick_params(labelsize = 30)
ax1.set_title('Std', fontsize = 30)
ax1.set_ylim(0, 13000)
ax1.set_xlim(0, 6)
ax1.set_yticks(np.arange(0,13000, 1000))
ax1.set_xticks(np.arange(0, 6, 2))
ax1.axvspan(0, 1, alpha=0.5, color='dimgrey', zorder = 0)
ax1.axvspan(1, 2, alpha=0.5, color='grey', zorder = 0)
ax1.axvspan(2, 6, alpha=0.5, color='lightgrey', zorder = 0)
ax1.grid()  
    
ax2.set_xlabel('Absolute #', fontsize = 30)
ax2.tick_params(labelsize = 30)
ax2.set_title('# of measurements', fontsize = 30)
ax2.set_yticks(np.arange(0,13000, 1000))
ax2.set_yticklabels(ax2.yaxis.get_ticklabels()[::4])
ax2.yaxis.tick_right()
ax2.set_ylim(0, 13000)
ax2.grid()

ax1.plot(std_temp_COSMO_12_00.std_temp, std_temp_COSMO_12_00.altitude_m, color = 'midnightblue', linewidth = 5, label = 'T COSMO, 12_00', zorder = 0)
ax1.plot(std_temp_d_COSMO_12_00.std_temp_d, std_temp_COSMO_12_00.altitude_m, color = 'midnightblue', linewidth = 5, label = 'Td COSMO, 12_00', zorder = 1)

ax1.plot(std_temp_COSMO_09_03.std_temp, std_temp_COSMO_09_03.altitude_m, color = 'darkmagenta', linewidth = 3, label = 'T COSMO, 09_03', zorder = 0)
ax1.plot(std_temp_d_COSMO_09_03.std_temp_d, std_temp_COSMO_09_03.altitude_m, color = 'darkmagenta', linewidth = 3, label = 'Td COSMO, 09_03', zorder = 1)

ax1.plot(std_temp_COSMO_06_06.std_temp, std_temp_COSMO_06_06.altitude_m, color = 'crimson', linewidth = 3, label = 'T COSMO, 06_06', zorder = 0)
ax1.plot(std_temp_d_COSMO_06_06.std_temp_d, std_temp_COSMO_06_06.altitude_m, color = 'crimson', linewidth = 3, label = 'Td COSMO, 06_06', zorder = 1)

ax1.plot(std_temp_COSMO_03_09.std_temp, std_temp_COSMO_03_09.altitude_m, color = 'magenta', linewidth = 3, label = 'T COSMO, 03_09', zorder = 0)
ax1.plot(std_temp_d_COSMO_03_09.std_temp_d, std_temp_COSMO_03_09.altitude_m, color = 'magenta', linewidth = 3, label = 'Td COSMO, 03_09', zorder = 1)

ax1.plot(std_temp_COSMO_00_12.std_temp, std_temp_COSMO_00_12.altitude_m, color = 'pink', linewidth = 3, label = 'T COSMO, 00_12', zorder = 0)
ax1.plot(std_temp_d_COSMO_00_12.std_temp_d, std_temp_COSMO_00_12.altitude_m, color = 'pink', linewidth = 3, label = 'Td COSMO, 00_12', zorder = 1)

#####
ax2.plot(number_temp_COSMO_12_00.number, number_temp_COSMO_12_00.altitude_m, color = 'midnightblue', linewidth = 5,  zorder = 0)
ax2.plot(number_temp_d_COSMO_12_00.number, number_temp_d_COSMO_12_00.altitude_m, color = 'midnightblue', linewidth = 5, linestyle = 'dotted', zorder = 1)

ax2.plot(number_temp_COSMO_09_03.number, number_temp_COSMO_09_03.altitude_m, color = 'darkmagenta', linewidth = 3,  zorder = 0)
ax2.plot(number_temp_d_COSMO_09_03.number, number_temp_d_COSMO_09_03.altitude_m, color = 'darkmagenta', linewidth = 3, linestyle = 'dotted', zorder = 1)

ax2.plot(number_temp_COSMO_06_06.number, number_temp_COSMO_06_06.altitude_m, color = 'crimson', linewidth = 3,  zorder = 0)
ax2.plot(number_temp_d_COSMO_06_06.number, number_temp_d_COSMO_06_06.altitude_m, color = 'crimson', linewidth = 3, linestyle = 'dotted', zorder = 1)

ax2.plot(number_temp_COSMO_03_09.number, number_temp_COSMO_03_09.altitude_m, color = 'magenta', linewidth = 3,  zorder = 0)
ax2.plot(number_temp_d_COSMO_03_09.number, number_temp_d_COSMO_03_09.altitude_m, color = 'magenta', linewidth = 3, linestyle = 'dotted', zorder = 1)

ax2.plot(number_temp_COSMO_00_12.number, number_temp_COSMO_00_12.altitude_m, color = 'pink', linewidth = 3,  zorder = 0)
ax2.plot(number_temp_d_COSMO_00_12.number, number_temp_d_COSMO_00_12.altitude_m, color = 'pink', linewidth = 3, linestyle = 'dotted', zorder = 1)
  
#fig.savefig(STD_archive+'/STD_NUCAPS_1000m_1200_ALL',dpi=300, bbox_inches = "tight")
#ax1.legend(fontsize = 25)

ax1.legend(fontsize = 25,bbox_to_anchor=(2.05, 1.0, 0.3, 0.2), loc='upper left')











 