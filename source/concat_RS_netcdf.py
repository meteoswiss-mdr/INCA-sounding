#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 08:48:58 2020

@author: nal
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pandas as pd
import numpy as np
import urllib3
import datetime as dt
import fnmatch
import collections
import xarray as xr
import netCDF4 as nc4

location_name = 'Payerne' # define location name
location_nr = '06610' # define location nr
# Payerne -> 06610, Munich -> 10868, Milano -> 16080, Stuttgart -> 10739
dynfmt = "%Y%m%d%H"
firstdate = '2019050100'
lastdate = '2020050100'
step = 12
firstobj=dt.datetime.strptime(firstdate,dynfmt)
lastobj=dt.datetime.strptime(lastdate,dynfmt)

pd_concat = pd.DataFrame()
while firstobj != lastobj: # loop over days
    nowdate=dt.datetime.strftime(firstobj,dynfmt)
    print(nowdate)     
    year = nowdate[0:4]
    month = nowdate[4:6]
    day = nowdate[6:8]
    path_file_txt = '/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/'+location_name+'/'+year+'/'+month+'/'+day+'/RS_'+location_nr+'_'+nowdate+'.txt'
    data_comma = pd.read_csv(path_file_txt)
    pd_concat = pd_concat.append(data_comma)
    firstobj= firstobj + dt.timedelta(hours=step)
 
ds = pd_concat.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent', '742':'geopotential_altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })
ds = xr.Dataset(ds)
ds.to_netcdf('/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/'+location_name+'/RS_concat.nc') # replace filename
    

