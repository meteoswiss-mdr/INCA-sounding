#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:42:18 202

@author: nal

####
plot standard deviation of COSMO
Std values can be found on APN intranet: http://zueux240.meteoswiss.ch/modelle/Verification/Operational/Seasonal/2019s3/Vertical-profiles/COSMO-1.php
####
"""
import pandas as pd 
import metpy
from metpy.units import units
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import xarray as xr
from scipy import spatial

############################################################################# define time and coordinate #############################################################################
season_year = 'JJA_2019' #

lon_Payerne = 6.93608
lat_Payerne = 46.82201
##################################################################### define paths and read data ##################################################################
COSMO_std_archive   = '/data/COALITION2/internships/nal/std_files/COSMO/'+str(season_year)+'/scratch/owm/verify/upper-air/'+str(season_year)+'/COSMO-1/output_all_stations_6610/'
INCA_archive = '/data/COALITION2/internships/nal/data/COSMO/'

COSMO_std_temp = pd.read_csv(COSMO_std_archive+'allscores.dat', ';') 

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
    
COSMO_std_temp['altitude_m'] = metpy.calc.pressure_to_height_std(COSMO_std_temp.plevel.values/100 * units.hPa) * 1000
COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.varno == 59] # 2: T, 59 Td
COSMO_std_temp = COSMO_std_temp[COSMO_std_temp.scorename == 'SD']
COSMO_std_temp['plevel'] = COSMO_std_temp['plevel']
COSMO_std_temp_0 = COSMO_std_temp[COSMO_std_temp.leadtime == 0][0:20]
COSMO_std_temp_6 = COSMO_std_temp[COSMO_std_temp.leadtime == 6][0:20]
COSMO_std_temp_12 = COSMO_std_temp[COSMO_std_temp.leadtime == 12][0:20]
COSMO_std_temp_18 = COSMO_std_temp[COSMO_std_temp.leadtime == 18][0:20]
COSMO_std_temp_24 = COSMO_std_temp[COSMO_std_temp.leadtime == 24][0:20]
COSMO_std_temp_30 = COSMO_std_temp[COSMO_std_temp.leadtime == 30][0:20]
    
COSMO_std_temp_0 = griddata(COSMO_std_temp_0.altitude_m.values, COSMO_std_temp_0.scores.values, (INCA_grid.values))
COSMO_std_temp_6 = griddata(COSMO_std_temp_6.altitude_m.values, COSMO_std_temp_6.scores.values, (INCA_grid.values))
COSMO_std_temp_12 = griddata(COSMO_std_temp_12.altitude_m.values, COSMO_std_temp_12.scores.values, (INCA_grid.values))
COSMO_std_temp_18 = griddata(COSMO_std_temp_18.altitude_m.values, COSMO_std_temp_18.scores.values, (INCA_grid.values))
COSMO_std_temp_24 = griddata(COSMO_std_temp_24.altitude_m.values, COSMO_std_temp_24.scores.values, (INCA_grid.values))
COSMO_std_temp_30 = griddata(COSMO_std_temp_30.altitude_m.values, COSMO_std_temp_30.scores.values, (INCA_grid.values))

##################################################################### plot_data ##################################################################    
fig, ax = plt.subplots(figsize=(4.5,7)) 
ax.plot(COSMO_std_temp_0, INCA_grid, label = 0)
ax.plot(COSMO_std_temp_6, INCA_grid, label = 6, linewidth = 6)
ax.plot(COSMO_std_temp_12, INCA_grid, label = 12)
ax.plot(COSMO_std_temp_18, INCA_grid, label = 18)
ax.plot(COSMO_std_temp_24, INCA_grid, label = 24)
ax.plot(COSMO_std_temp_30, INCA_grid, label = 30)
ax.set_ylabel('Altitude [m]', size = 14)
ax.set_xlabel('STD [K]', size = 14)
ax.set_xlim(-0.5,8.2) 
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)