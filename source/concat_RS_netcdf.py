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

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import Hodograph, SkewT
from metpy.units import units

import xarray as xr

import urllib3

#startdate = 20200228000000
#enddate = 20200228000000
#startdate1 = 20200228120000
#enddate1 = 20200228120000
#url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&measCatNr=1&dataSourceId=34&delimiter=COMMA&parameterIds=744,745,746,742,748,743,747&date='+str(startdate)+'-'+str(enddate)+'&obsTypeIds=22'
#url1 = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&measCatNr=1&dataSourceId=34&delimiter=COMMA&parameterIds=744,745,746,742,748,743,747&date='+str(startdate1)+'-'+str(enddate1)+'&obsTypeIds=22'

import datetime as dt
import fnmatch
import collections
import xarray as xr
import netCDF4 as nc4
dynfmt = "%Y%m%d%H"
firstdate = '2019050100'
lastdate = '2020050100'
step = 12
firstobj=dt.datetime.strptime(firstdate,dynfmt)
lastobj=dt.datetime.strptime(lastdate,dynfmt)
#netcdf = nc4.Dataset('collocated_files_1.nc', 'w', format = 'NETCDF4')
# Time filter -> choose only those files between hour 11 and 13 and 23 and 01
pd_concat = pd.DataFrame()
while firstobj != lastobj: # loop over days
    nowdate=dt.datetime.strftime(firstobj,dynfmt)
    print(nowdate)     
    path_file_txt = '/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/Stuttgart/RS_10739_'+nowdate+'.txt'
    data_comma = pd.read_csv(path_file_txt)

    pd_concat = pd_concat.append(data_comma)

    firstobj= firstobj + dt.timedelta(hours=step)
    
ds = xr.Dataset(pd_concat)
ds.to_netcdf('/data/COALITION2/PicturesSatellite/results_NAL/Radiosondes/Stuttgart/RS_10739_concat.nc')
    
#xr_array = xr.open_dataset('RS_concat.nc')
#Pressure = xr_array.Pressure.values
#Pressure = pd.DataFrame(Pressure, columns = ['Pressure'])
#Temperature = xr_array['745'].values
#Temperature = pd.DataFrame(Temperature, columns = ['Temperature'])
#Temperature_d = xr_array['747'].values
#Temperature_d = pd.DataFrame(Temperature_d, columns = ['Temperature_d'])
#Time = xr_array['termin'].values
#Time = pd.DataFrame(Time, columns = ['Time'])
#df = pd.concat([Pressure,Temperature,Temperature_d, Time], axis = 1)

#des_Time = 20190928000000
#df_time = df[df.Time == des_Time]

