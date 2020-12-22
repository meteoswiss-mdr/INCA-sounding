from scipy.interpolate import griddata
import pandas as pd
import numpy as np
import datetime as dt
import netCDF4 as nc4
import xarray as xr
import metpy.calc
from metpy.units import units
import sys
import sounding_uncertainty as sun
import sounding_utils as sut
import sounding_toINCAgrid as sto
import sounding_config as CFG

lastdate = dt.datetime.strptime(sys.argv[1], "%Y%m%d%H%M")+dt.timedelta(days=1)
firstdate = lastdate - dt.timedelta(days = CFG.validation_period) 

firstdate = dt.datetime.strptime('201909270000', "%Y%m%d%H%M")
lastdate = dt.datetime.strptime('202012100000', "%Y%m%d%H%M")



altitude = sto.extract_INCA_grid_onepoint(CFG.dir_INCA, CFG.validation_station, CFG.validation_station.station, "val")
altitude_m = altitude[CFG.validation_station.station[0]][0]
indexes = altitude[CFG.validation_station.station[0]][1]

concat_RS = sun.concat_RS(CFG.validation_station.station_nr[0],firstdate, lastdate, 24, altitude_m)
concat_cosmo = sun.concat_cosmo(CFG.validation_station.station_nr[0], firstdate, lastdate, 24, indexes, CFG.validation_leadtime)
bias_cosmo, std_cosmo, count_cosmo = sun.calculate_bias_std(concat_RS, concat_cosmo)

if CFG.save_test_plots == True:
    sut.plot_bias(bias_cosmo, count_cosmo, altitude_m/1000, -4, 3.9, -0.7,0.7, "cosmo", firstdate.strftime("%Y%m%d%H%M"))
    sut.plot_std(std_cosmo, count_cosmo, altitude_m/1000, 0, 2, 0,3, "cosmo", firstdate.strftime("%Y%m%d%H%M"))
    
bias_std_cosmo = xr.merge([bias_cosmo, std_cosmo])
bias_std_cosmo.attrs["starttime"] = str(firstdate)
bias_std_cosmo.attrs["endtime"] = str(lastdate)
outfn_cosmo = CFG.dir_std_COSMO / f"cosmo_bias_std.nc"
bias_std_cosmo.to_netcdf(outfn_cosmo)

if CFG.RM_data:
    concat_RM = sun.concat_RM(CFG.validation_station.station_nr[0], firstdate, lastdate, 24, altitude_m)
    bias_RM, std_RM, count_RM = sun.calculate_bias_std(concat_RS, concat_RM)
  
    if CFG.save_test_plots == True:
        sut.plot_bias(bias_RM, count_RM, altitude_m/1000, -4, 3.9, -2,2, "RM", firstdate.strftime("%Y%m%d%H%M"))
        sut.plot_std(std_RM, count_RM, altitude_m/1000, 0, 3, 0,2, "RM", firstdate.strftime("%Y%m%d%H%M"))
    
    bias_std_RM = xr.merge([bias_RM, std_RM])
    bias_std_RM.attrs["starttime"] = str(firstdate)
    bias_std_RM.attrs["endtime"] = str(lastdate)
    outfn_RM = CFG.dir_std_COSMO / f"RM_bias_std.nc"
    bias_std_RM.to_netcdf(outfn_RM)
    
if CFG.RA_data:
    concat_RA = sun.concat_RA(CFG.validation_station.station_nr[0], firstdate, lastdate, 24, altitude_m)
    bias_RA, std_RA, count_RA = sun.calculate_bias_std(concat_RS, concat_RA)
    
    if CFG.save_test_plots == True:
        sut.plot_bias(bias_RA, count_RA, altitude_m/1000, -4, 3.9, -2,2, "RA")
        sut.plot_std(std_RA, count_RA, altitude_m/1000, 0, 3, 0,2, "RA")

    bias_std_RA = xr.merge([bias_RA, std_RA])
    bias_std_RA.attrs["starttime"] = str(firstdate)
    bias_std_RA.attrs["endtime"] = str(lastdate)
    outfn_RA = CFG.dir_std_COSMO / f"RA_bias_std.nc"
    bias_std_RA.to_netcdf(outfn_RA)
