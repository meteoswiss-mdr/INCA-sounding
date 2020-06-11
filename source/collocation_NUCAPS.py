#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:56:36 2020

@author: nal
"""
import numpy as np
from datetime import datetime
import pandas as pd
import os
from scipy import spatial
import geopy.distance

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from satpy import Scene, find_files_and_readers

from metpy import calc as cc
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.pyplot as plt
from metpy.plots import Hodograph, SkewT

##############################  FILTER TIME ##############################
# Collocate data: save only those values that are within 100 km distance and +-1 Stunde 
import datetime as dt
import fnmatch
import collections
import xarray as xr
import netCDF4 as nc4
dynfmt = "%Y%m%d"
firstdate = '20200102'
lastdate = '20200103'
step = 1
datasets = []
firstobj=dt.datetime.strptime(firstdate,dynfmt)
lastobj=dt.datetime.strptime(lastdate,dynfmt)
# Time filter -> choose only those files between hour 11 and 13 and 23 and 01
while firstobj != lastobj: # loop over days
    result = []
    nowdate=dt.datetime.strftime(firstobj,dynfmt)
    print(nowdate)
    # list all files of a day
    sYDM_00 = str(nowdate + str('00'))
    sYDM_11 = str(nowdate + str('11'))
    sYDM_12 = str(nowdate + str('12'))
    sYDM_23 = str(nowdate + str('23'))
    
    try: 
        files = os.listdir(os.path.join('/data/COALITION2/database/NUCAPS',firstobj.strftime('%Y/%m/%d')))
        directory = os.path.join('/data/COALITION2/database/NUCAPS',firstobj.strftime('%Y/%m/%d'))
        #keywords = ['s'+sYDM_00, 's'+sYDM_01, 's'+sYDM_11, 's'+sYDM_12, 's'+sYDM_13, 's'+sYDM_23]
        keywords = ['s'+sYDM_00, 's'+sYDM_11, 's'+sYDM_12, 's'+sYDM_23]
        for filename in files:
            for keyword in keywords:
                if keyword in filename:
                    result.append(os.path.join(directory,filename))      
        print(len(result))        
##############################  FILTER AREA ##############################
        distance_array = np.zeros(120)
        distance = pd.DataFrame(distance_array, columns = ['distance'])
        files_small = pd.DataFrame(columns = ['file','min_dist','index'])
        files_large = pd.DataFrame(columns = ['file','min_dist','index'])
        i = 0
        #start_time = time.time()
        for i in range(0,len(result)):
            print(i)
            filenames = [result[i]]
            global_scene = Scene(reader="nucaps", filenames=filenames) 
            global_scene.load(["Temperature", "H2O_MR"], pressure_levels=True)

            lon = global_scene["Temperature"].coords['Longitude'].values
            lat = global_scene["Temperature"].coords['Latitude'].values
              
            for j in range(0,lon.size):
                #print(j)
                if  lon[j] == -9999 or lat[j] == -9999:
                    distance.distance[j] = 'nan'
                    print('nan value')
                else:
                    coords_1 = (lat[j],lon[j])
                    coords_2 = (46.812,6.943)
                    dist = geopy.distance.geodesic(coords_1, coords_2).km
                    distance.distance[j] = dist
            min_dist = np.min(np.abs(distance))
            ixd = distance.loc[distance['distance'] == min_dist[0]]
            index_CP = int(ixd.index.values[0])
            CP = int(ixd.distance.iloc[0])
                           
            if CP <= 100:
                files_small = files_small.append({'date': nowdate, 'file':result[i],'min_dist':CP,'index':index_CP}, ignore_index=True)
                profile_1 = global_scene["Temperature"][index_CP,:]
                datasets.append(profile_1)
            else:
                files_large = files_large.append({'file':result[i], 'min_dist':CP, 'index':index_CP}, ignore_index=True)
            
    except FileNotFoundError:
        print('loop skipped')
            
    firstobj= firstobj + dt.timedelta(days=step)

combined = xr.concat(datasets, dim='Time')


### why can't I save the file??
#combined.save_datasets(writer='cf', datasets=["Temperature"], filename='test.nc', exclude_attrs=['wavelength', 'resolution', 'polarization', 'calibration', 'level', 'modifier'])
#combined.save_datasets(writer='cf')

del(combined.attrs['wavelength'])
del(combined.attrs['resolution'])
del(combined.attrs['polarization'])
del(combined.attrs['calibration'])
del(combined.attrs['level'])
del(combined.attrs['start_time'])
del(combined.attrs['end_time'])

from satpy import Scene, find_files_and_readers
from datetime import datetime

start_time    = datetime.strptime("202002031333190", "%Y%m%d%H%M%S%f")
end_time      = datetime.strptime("202002031333190", "%Y%m%d%H%M%S%f")

files_sat = find_files_and_readers(start_time=start_time, end_time=end_time, 
                           base_dir="/data/COALITION2/database/NUCAPS/2020/02/03",
                           reader='nucaps',prerequisites=[DatasetID('hej')])  
global_scene = Scene(reader="nucaps", filenames=files_sat)
global_scene.load(["Temperature"],pressure_levels=True)



del(global_scene[var].attrs['wavelength'])
del(global_scene[var].attrs['resolution'])
del(global_scene[var].attrs['polarization'])
del(global_scene[var].attrs['calibration'])
del(global_scene[var].attrs['level'])
del(global_scene[var].attrs['modifiers'])
global_scene[var].attrs.shape = (120,100)
global_scene[var].attrs['start_time'] = str(global_scene[var].attrs['start_time'])
global_scene[var].attrs['end_time'] = str(global_scene[var].attrs['end_time'])

global_scene[var].to_netcdf('test.nc')


combined = combined.to_dataset()

combined.to_netcdf('/data/COALITION2/PicturesSatellite/results_NAL/NUCAPS/collocated_file.nc')
profile_1.to_netcdf()

#remote_data = xr.open_dataset('http://iridl.ldeo.columbia.edu/SOURCES/.OSU/.PRISM/.monthly/dods',decode_times=False)
#remote_data.to_netcdf('test')


### read nc file ###
# read
combined_1 = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/NUCAPS/collocated_file.nc')
# convert to dataframe
time = combined['Time'].values
Time = pd.DataFrame(columns = ['Time_since', 'start_time', 'end_time'])
Time['Time_since'] = time
dynfmt = "%Y%m%d%H%M%S%f"
start_1 = str(19700101000000)
start_1 = dt.datetime.strptime(start_1,dynfmt)
Time['start_time'] = start_1

for i in range(0, len(Time)):
    Time['end_time'][i] = Time['start_time'][i] + pd.Timedelta(milliseconds=Time['Time_since'][i])

time = Time['end_time']
Pressure = combined.Pressure.values 
Longitude = combined.Longitude.values
Latitudes = combined.Latitude.values
Temperature = combined.values

dynfmt = "%Y%m%d%H"
des_time = str(2020010211)
des_time = dt.datetime.strptime(des_time, dynfmt)
Index = Time.index[time == des_time]






# create a netcdf file: https://pyhogs.github.io/intro_netcdf4.html
#import netCDF4 as nc4

#netcdf = nc4.Dataset('collocated_files_16.nc', 'w', format = 'NETCDF4')
#nowdate = netcdf.createGroup('Coll_Files_' + str(nowdate))
#nowdate.createDimension('date_1', len(files_small))
#nowdate.createDimension('time', len(files_small))
#nowdate.createDimension('min_dist', len(files_small))
#nowdate.createDimension('index', len(files_small))
#nowdate.createDimension('filename_1', len(files_small))

#Starttime = nowdate.createVariable('starttime', 'i4', 'time')
#Distance = nowdate.createVariable('distance','i4', 'min_dist')
#Index = nowdate.createVariable('index', 'i4', 'index')
#Date = nowdate.createVariable('date', 'i4', 'date_1')
#Filename = nowdate.createVariable('filename', 'S117', 'filename_1')

#Starttime[:] = files_small.start_time
#Distance[:] = files_small.min_dist
#Index[:] = files_small.index
#Date[:] = files_small.date
#Filename[:] = files_small.file.values


# read a netCDF file
#netcdf_read = nc4.Dataset('collocated_files.nc', 'r')
#tempgrp = netcdf_read.groups["Coll_Files_20200201"]
#tempgrp.variables.keys()
#starttime = tempgrp.variables['starttime'][:]
#index_min_distance = tempgrp.variables['index'][:]
#min_distance = tempgrp.variables['distance'][:]
#date = tempgrp.variables['date'][:]
#filename= tempgrp.variables['filename'][:]


#data = np.random.rand(4, 3)

#locs = ['IA', 'IL', 'IN']

#times = pd.date_range('2000-01-01', periods=4)

#foo = xr.DataArray(data, coords=[times, locs], dims=['time', 'space'])

#data = files_small.values
#data = np.array(files_small.date.values,files_small.file.values, files_small.min_dist.values, files_small.index.values)
#data = files_small.to_numpy()
#firstobj = [files_small.date.values]
#filenames = [files_small.file.values]
#distance = [files_small.min_dist]
#index = [files_small.index]
#l = xr.DataArray(files_small, coords = [firstobj, filenames, distance, index], dims = ["date", "filenames", "distance", "index"] )

#l = xr.DataArray(files_small, coords = [files_small.min_dist, files_small.file])
#l.to_netcdf()

#l = xr.DataArray(files_small)



#files_small_dd = dd.from_pandas(files_small, npartitions = 4)
#files_small_da = files_small_dd.to_dask_array(lengths = True)
#file_collocated = files_small_da
#data = [files_small_da,file_collocated]
#file_collocated = da.concatenate(data, axis=0)



#file_collocated.to_netCDF('collocated_data.nc')
#cm = da.concatenate(x,l)
