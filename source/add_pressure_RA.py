#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:21:01 2020

@author: nal

add pressure to Raman Lidar and save as nc file
"""
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import math
import metpy.calc as cc
from metpy.units import units

######################################## define paths ######################################## 
SMN_archive = '/data/COALITION2/PicturesSatellite/results_NAL/SwissMetNet/Payerne'
RA_archive   = '/data/COALITION2/PicturesSatellite/results_NAL/RALMO/Payerne/'

######################################## read data ######################################## 
SMN_data = xr.open_dataset(SMN_archive+'/SMN_concat1.nc').to_dataframe()
RA_data = xr.open_dataset(RA_archive+'/RA_06610_concat.nc').to_dataframe()
RA_data['pressure_hPa'] = '0'
RA_data = RA_data.reset_index(drop=True)

######################################## calculate pressure values and add to dataframe ########################################  
firstdate_RA = '20190501000000' # define time span
lastdate_RA = '20200501000000'
step_RA = 30 # define time step in minutes

nc_name_RA = 'RA_concat_wp'

firstobj=dt.datetime.strptime(firstdate_RA,'%Y%m%d%H%M%S')
lastobj=dt.datetime.strptime(lastdate_RA,'%Y%m%d%H%M%S')

g = 9.81
m_mol_air = 28.965*10.0**(-3.0)
R = 8.31446
    
while firstobj != lastobj: # loop over days 
    print(firstobj.strftime('%Y%m%d%H%M%S')) 
    # extract time SMN
    SMN_data_time = SMN_data[SMN_data.time_YMDHMS == int(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))]
    dewpoint_SMN = cc.dewpoint_from_relative_humidity((SMN_data_time.temperature_degC.values + 273.15) * units.kelvin, SMN_data_time.relative_humidity_percent.values * units.percent)
    specific_humidity_SMN = cc.specific_humidity_from_dewpoint(dewpoint_SMN, SMN_data_time.pressure_hPa.values * units.hPa)
  
    # extract time raman lidar and delte temperature nan values
    indexNames = RA_data[(RA_data.time_YMDHMS == int(dt.datetime.strftime(firstobj,"%Y%m%d%H%M%S"))) & (RA_data.temperature_K == 1e+07)].index
    RA_data.drop(indexNames, inplace = True)
    RA_data_time = RA_data[RA_data.time_YMDHMS == int(dt.datetime.strftime(firstobj,"%Y%m%d%H%M%S"))]
     
    if RA_data_time.empty:
        firstobj= firstobj + dt.timedelta(minutes=30)
        
    else:
        # add SMN to RA
        data_comma_temp = RA_data_time[RA_data_time['temperature_K']!= 1e+07]
        data_comma_temp = data_comma_temp[['time_YMDHMS', 'altitude_m', 'temperature_K']]
        data_comma_temp = data_comma_temp.reset_index(drop=True)
        data_comma_temp.loc[-1] = [SMN_data_time.time_YMDHMS.iloc[0], 491, SMN_data_time.temperature_degC[0] + 273.15]
        data_comma_temp.index = data_comma_temp.index + 1
        data_comma_temp = data_comma_temp.sort_index()
        data_comma_temp = data_comma_temp.reset_index(drop=True)
    
        z_RA = data_comma_temp['altitude_m']
        T_RA = data_comma_temp['temperature_K']

        # calculate Lidar pressure levels with RS as lowest level 
        p_1 = np.zeros(len(z_RA))
        p_1[0] = SMN_data_time.pressure_hPa[0]
        integrant = g*m_mol_air/(R*T_RA)
        
        for i in range(1, len(z_RA)):
            p_1[i] = SMN_data_time.pressure_hPa[0]*math.exp(-np.trapz(integrant[0:i], z_RA[0:i]))
        
        RA_data.pressure_hPa[RA_data.time_YMDHMS == int(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))] = p_1[1:]

        firstobj= firstobj + dt.timedelta(minutes=30)
        
ds = xr.Dataset(RA_data)
ds.to_netcdf(RA_archive+'/'+nc_name_RA) # save dataframe to nc file
