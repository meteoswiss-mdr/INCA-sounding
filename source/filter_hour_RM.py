#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 08:11:42 2020

@author: nal
"""

import xarray as xr
import pandas as pd
RM_archive = '/data/COALITION2/PicturesSatellite/results_NAL/Radiometer/radiometer_payerne'
RM_data = xr.open_dataset(RM_archive+'/radiometer_06610_concat.nc').to_dataframe()
        
DT = 12
del RM_data['altitude_layers_2']
    
RM_data_1 = RM_data[0:72808488]
RM_data_2 = RM_data[72808488:145616976]
RM_data_3 = RM_data[145616976: 218425464]
RM_data_4 = RM_data[ 218425464:-1]
    
    
RM_data_1 = RM_data_1.reset_index().rename(columns = {'time': 'time_YMDHMS'})
RM_data_1 = RM_data_1[(RM_data_1['time_YMDHMS'].dt.hour == DT)]
RM_data_1 = RM_data_1.rename(columns = {'temperature_profiles' : 'temperature_degC', 'altitude_layers' : 'altitude_m', 'x': 'level'})
    
RM_data_2 = RM_data_2.reset_index().rename(columns = {'time': 'time_YMDHMS'})
RM_data_2 = RM_data_2[(RM_data_2['time_YMDHMS'].dt.hour == DT)]
RM_data_2 = RM_data_2.rename(columns = {'temperature_profiles' : 'temperature_degC', 'altitude_layers' : 'altitude_m', 'x': 'level'})
    
RM_data_3 = RM_data_3.reset_index().rename(columns = {'time': 'time_YMDHMS'})
RM_data_3 = RM_data_3[(RM_data_3['time_YMDHMS'].dt.hour == DT)]
RM_data_3 = RM_data_3.rename(columns = {'temperature_profiles' : 'temperature_degC', 'altitude_layers' : 'altitude_m', 'x': 'level'})
    
RM_data_4 = RM_data_4.reset_index().rename(columns = {'time': 'time_YMDHMS'})
RM_data_4 = RM_data_4[(RM_data_4['time_YMDHMS'].dt.hour == DT)]
RM_data_4 = RM_data_4.rename(columns = {'temperature_profiles' : 'temperature_degC', 'altitude_layers' : 'altitude_m', 'x': 'level'})


RM_data = pd.concat([RM_data_1, RM_data_2, RM_data_3, RM_data_4])
ds = xr.Dataset(RM_data)
ds.to_netcdf(RM_archive+'/radiometer_06610_concat_filtered_'+str(DT)+'.nc') # save dataframe to nc file 
