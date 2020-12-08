#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import ntpath
from datetime import datetime, timedelta
import sys
import numpy as np
import math
import pandas as pd
import geopy.distance
import xarray as xr
from os import path
from copy import deepcopy
#from satpy import Scene

# see position of radiosondes in Europe http://radiosonde.eu/RS00-D/RS02C-D.html
position_names={}
position_names[tuple([46.82201, 6.93608])]='Payerne'
position_names[tuple([45.4614,   9.2831])]='Milan'      # milan-lineate
position_names[tuple([48.2442,  11.5525])]='Munich'     # munich-oberschleissheim
position_names[tuple([48.8333,   9.2000])]='Stuttgart'  #
position_names[tuple([47.2603,  11.3439])]='Innsbruck'  # flughafen
position_names[tuple([45.7264,   5.0778])]='Lyon'       # Exupery
position_names[tuple([43.8569,   4.4064])]='Nimes'
position_names[tuple([45.9806,  13.0592])]='Udine'      # Udine/RIVOLTO RDS

##################################################################
a=1
vars3D=["MIT_Temperature","Temperature","FG_Temperature","H2O","MIT_H2O","FG_H2O","H2O_MR","MIT_H2O_MR","FG_H2O_MR"]

vars2D=["CrIS_FORs","View_Angle","Satellite_Height","FG_Mean_CO2","Mean_CO2","Solar_Zenith",
        "Ascending_Descending","Topography","Land_Fraction","Surface_Pressure","Skin_Temperature","MIT_Skin_Temperature",
        "FG_Skin_Temperature","MW_Surface_Class","MW_Surface_Emis","N_Smw_Per_FOV","nemis_Per_FOV","ncemis_Per_FOV",
        "ncemis_Per_FOV","ncld_Per_FOV","Quality_Flag"]   #,"Time","Latitude","Longitude"

vars1D=["quality_information"]


##################################################################

def find_NUCAPS_files(time_ref, dt1, dt2, NUCAPS_archive = "/data/COALITION2/database/NUCAPS/%Y/%m/%d/", verbose=False):

    # input: 
    # time_ref  datetime   * reference date around you like to find NUCAPS files
    # dt1       delta time * dt to define start time (with respect to time_ref) when the search will start
    # dt2       delta time * dt to define end   time (with respect to time_ref) when the search will end

    datetime_start = time_ref + timedelta(minutes=dt1)
    datetime_end   = time_ref + timedelta(minutes=dt2)
    print("*** find NUCAPS files between ", str(datetime_start), " and ", str(datetime_end))
    
    # find files from the current date 
    NUCAPS_files_wildcard = time_ref.strftime(NUCAPS_archive+"NUCAPS-EDR_v2r0_j01_s*.nc")
    NUCAPS_files = np.array(glob.glob(NUCAPS_files_wildcard))   # change from list to np.array to make use of np.where 
    #if verbose:
    #    print("files from today", NUCAPS_files)
    
    # if the time span includes times before midnight, add files from yesterday 
    if ( datetime_start.strftime("%Y%m%d") != time_ref.strftime("%Y%m%d")):
        NUCAPS_files_wildcard = datetime_start.strftime(NUCAPS_archive+"NUCAPS-EDR_v2r0_j01_s*.nc")
        NUCAPS_files = np.concatenate((np.array(glob.glob(NUCAPS_files_wildcard)), NUCAPS_files), axis=None)
        #if verbose:
        #    print("files from today and yesterday", NUCAPS_files)
        
    # if the time span includes times after midnight, add files from tomorrow    
    if ( time_ref.strftime("%Y%m%d") != datetime_end.strftime("%Y%m%d") ):
        NUCAPS_files_wildcard = datetime_end.strftime(NUCAPS_archive+"NUCAPS-EDR_v2r0_j01_s*.nc")
        NUCAPS_files = np.concatenate( (NUCAPS_files, np.array(glob.glob(NUCAPS_files_wildcard))), axis=None)
        #if verbose:
        #    print("files from today and tomorrow", NUCAPS_files)
    
    NUCAPS_file_dates = np.empty(len(NUCAPS_files), dtype=object)
    #NUCAPS_file_dates = np.empty(len(NUCAPS_files), dtype='datetime64')
    for i, NUCAPS_file in enumerate(NUCAPS_files):
        NUCAPS_file_dates[i] = datetime.strptime(ntpath.basename(NUCAPS_file)[21:21+15], "%Y%m%d%H%M%S%f")
        #if verbose:
        #    print(NUCAPS_file_dates[i])

    # search for files between dt1 minutes after radiosonde launch timestamp (before timestamp if dt1 is negative)
    # to dt2 minutes radiosonde after launch time stamp (before timestamp if dt2 is negative)
    NUCAPS_files = NUCAPS_files[np.where( np.logical_and(time_ref + timedelta(minutes=dt1)<NUCAPS_file_dates, NUCAPS_file_dates<time_ref + timedelta(minutes=dt2)))]

    if verbose:
        print("    found following files: ", NUCAPS_files)
    
    return NUCAPS_files

##################################################################

# remark: calculates almost the same value as geopy.distance.great_circle
def distance(origin, destination):
    # Haversine formula example in Python
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

##################################################################

def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)

##################################################################

def open_NUCAPS_file(NUCAPS_file, remove_data_below_surface=True, verbose=False):
    ## read variable with lat/lon coordinates        
    ds = xr.open_dataset(NUCAPS_file, decode_times=False)  # time units are non-standard, so we dont decode them here

    ## remove data for pressures higher than surface pressures
    if remove_data_below_surface:
        for var in ds.keys():
            if "Pressure" in ds[var].coords:
                if verbose:
                    print ("    remove values below surface pressure for "+var)
                ds[var] =  ds[var].where(ds[var].coords["Pressure"] < ds["Surface_Pressure"])
        
    # convert Time into datetime objects 
    units, reference_date = ds.Time.attrs['units'].split(' since ')
    if units=='msec':
        ref_date = datetime.strptime(reference_date,"%Y-%m-%dT%H:%M:%SZ") # usually '1970-01-01T00:00:00Z'
        NUCAPS_datetimes = [ np.nan if np.isnan(t) else ref_date + timedelta(milliseconds=t) for t in ds.Time.data]
        NUCAPS_datetimes_xr = xr.DataArray(NUCAPS_datetimes, coords=ds.Time.coords, dims=ds.Time.dims)
        NUCAPS_datetimes_xr.name = "datetime"
        # datetime is saved as np.datetime64, when searching for a certain time convert to this data type before
        # time_ref = datetime.strptime('2020-05-02T00:04:07.983000', '%Y-%m-%dT%H:%M:%S.%f')
        # ds.Temperature[np.where(ds.datetimes == np.datetime64(time_ref))].values
        ds["datetime"] = NUCAPS_datetimes_xr
    else:
        print("*** ERROR in open_NUCAPS_file (collocation_NUCPAS.py)")
        print("    unknown time units "+units)
        quit()

    # round ds.datetime by 6 hours = 360min -> add 180min and trancate by 360min
    NUCAPS_datetimes_rounded = np.array(ds.datetime+np.timedelta64(180,'m'), dtype='datetime64[360m]')
    NUCAPS_datetimes_rounded_xr = xr.DataArray(NUCAPS_datetimes_rounded, coords=ds.Time.coords, dims=ds.Time.dims)
    NUCAPS_datetimes_rounded_xr.name = "datetime_rounded"
    ds["datetime_rounded"] = NUCAPS_datetimes_rounded_xr
        
    #### old version to open NUCAPS with pytroll, but xarray is more flexible and "lazy"
    ##variables=["Temperature"]
    ##global_scene = Scene(reader="nucaps", filenames=[NUCAPS_file])
    ##global_scene.load([variables[0]])
    ##lons = global_scene[variables[0]].coords['Longitude'].values
    ##lats = global_scene[variables[0]].coords['Latitude'].values
    
    # ds needs to be closed in the main function with ds.close()
    return ds

##################################################################

def calc_distance_NUCAPS_to_location(time_ref, dt1, dt2, latlon_ref, cache=True, cacheoverwrite=False, cachedir="/tmp/", verbose=False):
    
    print("... calcualte distances to reference point: ", latlon_ref)

    if cache or cacheoverwrite:
        #cache_file = time_ref.strftime(cachedir+"%Y%m%d%H%M_NUCAPS_"+get_position_name(latlon_ref)+"_"+'{:d}'.format(dt1)+"min_"+'{:d}'.format(dt2)+"min.txt")
        cache_file = time_ref.strftime(cachedir+"%Y%m%d%H%M_NUCAPS_"+get_position_name(latlon_ref)+"_"+'{:d}'.format(dt1)+"min_"+'{:d}'.format(dt2)+"min.pkl")

    if (cache and path.exists(cache_file)) and not cacheoverwrite:
        if verbose:
            print("    read pre-calculated distances from", cache_file)
        if cache_file[-3:]=="pkl":
            files_and_distances = pd.read_pickle(cache_file)
        else:
            files_and_distances = pd.read_csv(cache_file, sep='\t')
    else:
        # read every file and calculate the distances between NUCAPS observation and latlon_ref

        # search NUCAPS files during the radio sonde ascent (specified as dt1 and dt2 in minutes)
        NUCAPS_files = find_NUCAPS_files(time_ref, dt1, dt2, verbose=verbose)
        #print(NUCAPS_files)
        
        files_and_distances = pd.DataFrame(columns = ['file','min_dist','min_index','datetime'])

        for NUCAPS_file in NUCAPS_files:
            if verbose:
                print("read ", NUCAPS_file)

            ds = open_NUCAPS_file(NUCAPS_file)
            lons = ds.Longitude.data
            lats = ds.Latitude.data

            distances = np.full(len(lats),None,np.float)

            for i, [lat,lon] in enumerate(zip(lats, lons)):
                if ( np.isnan(lat) or np.isnan(lon) ):
                    distances[i] = np.nan
                else:
                    distances[i] = geopy.distance.distance([lat,lon], latlon_ref).km
                    #distances[i] = geopy.distance.great_circle([lat,lon], latlon_ref).km
                    #distances[i] = geopy.distance.geodesic([lat,lon], latlon_ref).km
                    #distances[i] = distance([lat,lon], latlon_ref)
                #print(f'{lat:8.3f} {lon:8.3f} {distances[i]:10.3f}')

            # get minimum distance (ignore nan values)
            dist_min  = np.nanmin(distances)
            index_min = np.where(distances == dist_min)[0][0]  # take the first instance, where the distance is minimal (this could be modified later)

            # save result in a pandas data frame  
            files_and_distances = files_and_distances.append({'file':NUCAPS_file,'min_dist':distances[index_min],
                                                              'min_index':index_min, 'datetime':ds.datetime.data[index_min]}, ignore_index=True)
            #print(NUCAPS_file, lats[index_min], lons[index_min], distances[index_min], ds.Time.data[index_min], ds['datetime'].data[index_min])

            #close dataset
            ds.close()

            #print(files_and_distances)
        if cache or cacheoverwrite:
            if verbose:
                print("write calculated distances to ", cache_file)
            if cache_file[-3:]=="pkl":
                files_and_distances.to_pickle(cache_file)
            else:
                files_and_distances.to_csv(cache_file, sep='\t', index=False)
                
    return files_and_distances

##################################################################

def get_position_name(latlon):

    if tuple(latlon) in position_names.keys():
        return position_names[tuple(latlon)]
    else:
        return 'lat'+'{:04d}'.format(int(latlon[0]*100))+'lon''{:05d}'.format(int(latlon[1]*100))

##################################################################


def create_dummy_profile(time_ref, var_names):

    print("... read, ", "dummy_profile.nc")
    dummy_profile = xr.open_dataset("dummy_profile.nc", decode_times=False)

    #print("replace time variable")
    dt64 = np.datetime64(time_ref)
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'ms')
    dummy_profile.coords['Time'].data = ts
    #dummy_profile.coords['Latitude'].data = np.nan
    #dummy_profile.coords['Longitude'].data = np.nan

    #for var in var_names: 
    #    print (dummy_profile[var].data, type(dummy_profile[var].data))
        
    return dummy_profile

##################################################################

def get_closest_NUCAPS_profile(time_ref, dt1, dt2, dx, latlon_ref, var_names=['Temperature'], verbose=False):

    # function to return closest NUCAPS profile in terms of time and space
    # input
    # time_ref datetime_object * reference time
    # dt1      int             * timedelta in minutes before time_ref, where collocations are accepted
    # dt2      int             * timedelta in minutes after time_ref,  where collocations are accepted
    # dx       int             * maximum horizontal distance in km
    # vars     string array    * variables that should be extracted from NUCAPS file 
     
    print("*** get closest profile to: ", latlon_ref, str(time_ref), ", dt1=", dt1,"min , dt2=", dt2," min")
    files_and_distances = calc_distance_NUCAPS_to_location(time_ref, dt1, dt2, latlon_ref, verbose=verbose)

    if files_and_distances.shape[0] > 0:
        min_dist = np.min(files_and_distances.min_dist)
        if min_dist < dx:
            i_file = np.where(files_and_distances.min_dist == np.min(files_and_distances.min_dist))[0][0]

            NUCAPS_file = files_and_distances.file.iloc[i_file]
            if not path.isfile(NUCAPS_file):
                print("... Warning, NUCAPS file "+NUCAPS_file+" does not exist, recalculate distance ")
                files_and_distances = calc_distance_NUCAPS_to_location(time_ref, dt1, dt2, latlon_ref, cacheoverwrite=True, cachedir="/tmp/")
                if files_and_distances.shape[0] > 0:
                    min_dist = np.min(files_and_distances.min_dist)
                    if min_dist < dx:
                        i_file = np.where(files_and_distances.min_dist == np.min(files_and_distances.min_dist))[0][0]
                        NUCAPS_file = files_and_distances.file.iloc[i_file]
                    
            print("... read data from: ", NUCAPS_file, ", distance: ", files_and_distances.min_dist.iloc[i_file], " km")
            
            ds = open_NUCAPS_file(NUCAPS_file)
            
            # index with minimum distance within the file 
            i_min = files_and_distances.min_index[i_file]
            
            # create fake distance xarray 
            dist_xr = xr.DataArray(np.full(ds["Skin_Temperature"].size,np.nan), coords=ds["Skin_Temperature"].coords, dims=ds["Skin_Temperature"].dims)
            dist_xr.name="distance"
            dist_xr[i_min] = min_dist
            ds["distance"]=dist_xr
            #print(ds["distance"],ds["distance"]["Number_of_CrIS_FORs"==i_min],ds["distance"][i_min],i_min)
            
            #return ds[var]["Number_of_CrIS_FORs"==i_min]
            #return xr.merge([ds[var]["Number_of_CrIS_FORs"==i_min] for var in var_names])
            return xr.merge([ds[var][i_min] for var in var_names])
        else:
            print("!!! no matching profile found, min_dist=", min_dist, ", return dummy profile")
            return create_dummy_profile(time_ref, var_names)
    else:
        print("!!! no matching profile found, sufficiently close in time, return dummy profile")
        return create_dummy_profile(time_ref, var_names)
    
####################################################
####################################################


if __name__ == '__main__':

    if len(sys.argv) == 1:
        #use default date
        datetime_RS = datetime(2019,5,1,0,0)
        #datetime_RS = datetime(2020,12,21,0,0)
    elif len(sys.argv) == 6:
        year    = int(sys.argv[1])
        month   = int(sys.argv[2])
        day     = int(sys.argv[3])
        hour    = int(sys.argv[4])
        minute  = int(sys.argv[5])
        datetime_RS = datetime(year,month,day,hour,minute,0)
    else:
        print("*** ERROR, unknown number of command line arguements")
        print("    usage: python concat_NUCAPS.py 2020 02 01 12 00")
        quit()

####################################################
    # this part could be changed by users

    # list of location for radiosondes is here:
    # https://www.ncdc.noaa.gov/data-access/weather-balloon/integrated-global-radiosonde-archive
    #
    #latlon_radiosonde = [46.82201, 6.93608]   # payerne  
    #latlon_radiosonde = [48.2442,  11.5525  ] # munich-oberschleissheim
    #latlon_radiosonde = [45.4614,   9.2831]   # milan-lineate
    #latlon_radiosonde = [48.8333,   9.2000]   # stuttgart
    #latlon_radiosonde = [47.2603,   11.3439]  # innsbruck flughafen
    #latlon_radiosonde = [45.7264,    5.0778]  # Lyon Exupery
    #latlon_radiosonde = [43.8569,    4.4064]  # Nimes-COURBESSAC
    latlon_radiosonde = [45.9806,   13.0592]  # Udine/RIVOLTO RDS
    

    #dt1 = -60 # delta time in min before (negative) time_ref to search for NUCAPS files 
    #dt2 =   0 # delta time in min after  (positive) time_ref to search for NUCAPS files
    #dt1 = -120 # delta time in min before (negative) time_ref to search for NUCAPS files 
    #dt2 =   60 # delta time in min after  (positive) time_ref to search for NUCAPS files
    dt1 = -720 # delta time in min before (negative) time_ref to search for NUCAPS files 
    dt2 =   60 # delta time in min after  (positive) time_ref to search for NUCAPS files
    dx = 3500

    verbose=True
    
    #time_end = datetime_RS + timedelta(days=1)
    #time_end = datetime_RS + timedelta(days=366)
    time_end = datetime(2020,8,26,12,0)
    #var_names = ["Pressure","Effective_Pressure","MIT_Temperature","Temperature","FG_Temperature","H2O","MIT_H2O","FG_H2O","H2O_MR","MIT_H2O_MR","FG_H2O_MR"]
    var_names = ["MIT_Temperature","Temperature","FG_Temperature","H2O","MIT_H2O","FG_H2O","H2O_MR","MIT_H2O_MR","FG_H2O_MR"] + vars2D + ["distance"]
    #var_names = ["MIT_Temperature","Surface_Pressure", "MW_Surface_Class", "distance"]

    # this part could be changed by users
####################################################
    
    print("Collocate NUCAPS with radiosonde station: "+get_position_name(latlon_radiosonde) )
    print("start date: "+str(datetime_RS) )
    print("end   date: "+str(time_end))
    print("delta t: "+str(dt1)+" min, "+str(dt2)+"min")
    print("=============================")
    
    tt = deepcopy(datetime_RS)
    while tt <= time_end:
        
        # get profiles from the closest observation 
        profile = get_closest_NUCAPS_profile(tt, dt1, dt2, dx, latlon_radiosonde, var_names=var_names, verbose=verbose)

        # if variable collocated_profiles already exists 
        
        if 'collocated_profiles' in globals():
            collocated_profiles = xr.concat([ collocated_profiles, profile ], 'Number_of_CrIS_FORs')
        else:
            collocated_profiles = deepcopy(profile)

        #only needed once for a set of variables, only work if there is a collocated NUCAPS measurement 
        save_dummy_profile=False
        if save_dummy_profile:
            print("... create dummpy profile")
            collocated_profiles = deepcopy(profile)
            dummy_profile = deepcopy(profile) 
            for var in var_names:
                if var in vars2D+["distance"]:
                    dummy_profile[var].data = np.nan
                else:    
                    dummy_profile[var].data[:] = np.nan
            dummy_profile.to_netcdf("dummy_profile.nc")

        # close netCDF file
        profile.close()
            
        tt += timedelta(hours=12)

    #print(collocated_profiles)
    ncfile_collocated = "NUCAPS_"+get_position_name(latlon_radiosonde)+"_"+'{:d}'.format(dt1)+"min_"+'{:d}'.format(dt2)+"min_"+'{:d}'.format(dx)+"km.nc"
    #ncfile_collocated = "test_"+get_position_name(latlon_radiosonde)+"_"+'{:d}'.format(dt1)+"min_"+'{:d}'.format(dt2)+"min_"+'{:d}'.format(dx)+"km.nc"
    collocated_profiles.to_netcdf(ncfile_collocated)
    
