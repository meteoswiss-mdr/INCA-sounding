#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:18:38 2020

@author: nal
"""


"""
Plot Radiosounding, 11.05.2020
based on:
    Import radiosounding data:
        https://confluence.meteoswiss.ch/display/APP/pymchdwh
    Skew_T_log_P diagram:
        https://unidata.github.io/MetPy/latest/tutorials/upperair_soundings.html
        https://github.com/Unidata/MetPy/issues/1003
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import Hodograph, SkewT
from metpy.units import units

## download from Data Ware House
import urllib3
startdate = 20190928000000
enddate = 20190928000000
url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&measCatNr=1&dataSourceId=34&delimiter=COMMA&parameterIds=744,745,746,742,748,743,747&date=20200309120000&obsTypeIds=22'
data_comma = pd.read_csv(url, skiprows = [1], sep=',')
data_comma = data_comma[data_comma['745'] != 1e+07]
url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/surface/wmo_ind?locationIds=06610&delimiter=comma&parameterIds=90,91,98,196,197,2761,2762,2763,2764,2765,2766,2767,2768,2775,2776,2777,2778,2779,2780,2781,2784,2785,2786,2790,2792,93,94&date=20200309120000'
SMN = pd.read_csv(url, skiprows = [1],sep=',')

p = data_comma['744'].values * units.hPa
T = data_comma['745'].values * units.degC
z = data_comma['742'].values * units.meters
Td = data_comma['747'].values * units.degC
RH = data_comma['746'].values * units.percent
wind_speed = data_comma['748'].values * units('m/s').to('knots')
wind_dir =data_comma['743'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)


import xarray as xr
from scipy.interpolate import griddata
from scipy import spatial
lon_Payerne = 6.93608
lat_Payerne = 46.82201
INCA_grid = xr.open_dataset('/data/COALITION2/PicturesSatellite/results_NAL/COSMO//inca_topo_levels_hsurf_ccs4.nc') 
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
INCA_grid = np.flip(INCA_grid.altitude_m)
T_RS_list = []
T_d_RS_list = []
p_RS_list = []
    
INCA_grid = INCA_grid.reset_index(drop=True)
for i in range(0,len(INCA_grid)):
    print(i)
    if (i == 0):
        window_h_max = INCA_grid.iloc[i] + ((INCA_grid.iloc[i+1]-INCA_grid.iloc[(i)]) / 2)
        window_h_min = INCA_grid.iloc[i] - ((INCA_grid.iloc[i+1]-INCA_grid.iloc[(i)]) / 2)
    elif (i==(len(INCA_grid)-1)):
        window_h_max = INCA_grid.iloc[i] + ((INCA_grid.iloc[i] - INCA_grid.iloc[(i-1)])  / 2)
        window_h_min = INCA_grid.iloc[i] - ((INCA_grid.iloc[i] - INCA_grid.iloc[i-1])  / 2)
    else: 
        window_h_min = INCA_grid.iloc[i] - ((INCA_grid.iloc[i]-INCA_grid.iloc[(i-1)])  / 2)
        window_h_max = INCA_grid.iloc[i] + ((INCA_grid.iloc[i+1]-(INCA_grid.iloc[i])) / 2)
    print('min' + str(window_h_min))
    print('max' + str(window_h_max))
    
    index_list = np.where(np.logical_and(z.magnitude>=float(window_h_min), z.magnitude<=float(window_h_max)))
    T_index = T.magnitude[(z.magnitude <= float(window_h_max)) & (z.magnitude >= float(window_h_min))]
    T_d_index = Td.magnitude[(z.magnitude <= float(window_h_max)) & (z.magnitude >= float(window_h_min))]
    #p_index = pz_RS[(z_RS > window_h_max) & (pz_RS < window_h_min)]
   
    mean_T_RS = np.mean(T_index)
    T_RS_list.append(mean_T_RS)
        
    mean_T_d_RS = np.mean(T_d_index)
    T_d_RS_list.append(mean_T_d_RS)

T_RS_list = np.asarray(T_RS_list)
T = T_RS_list[~np.isnan(T_RS_list)]* units.degC
T_d_RS_list = np.asarray(T_d_RS_list)
Td = T_d_RS_list[~np.isnan(T_d_RS_list)]* units.degC
import metpy

INCA_grid = INCA_grid[2:50].reset_index(drop=True)
g = 9.81
m_mol_air = 28.965*10.0**(-3.0)
R = 8.31446
# calculate Lidar pressure levels with RS as lowest level 
p_1 = np.zeros(len(INCA_grid))
p_1[0] = SMN['90'].values
integrant = g*m_mol_air/(R*(T.magnitude-273.15))
import math
for i in range(1, len(INCA_grid)):
    p_1[i] = p_1[0]*math.exp(-np.trapz(integrant[0:i], INCA_grid[0:i].values))
        
p = p_1 * units.hPa

p = metpy.calc.height_to_pressure_std(INCA_grid.values * units.meters )

# Calculate the LCL
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])

print(lcl_pressure, lcl_temperature)

# Calculate the parcel profile.
parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')


fig = plt.figure(figsize=(9, 9))

# Grid for plots
skew = SkewT(fig, rotation=45)

ax1 = fig.add_axes([0.98,0.2,0.1,0.3])
ax2 = fig.add_axes([1.14,0.2,0.1,0.3])
ax1.axis('off')
ax2.axis('off')
ax_hod = fig.add_axes([0.96, 0.58, 0.3, 0.3])

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
#skew.plot_barbs(p[0:p.size:100],u[0:u.size:100],v[0:v.size:100])
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 60)

# Plot LCL as black dot
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

# Plot the parcel profile as a black line
skew.plot(p, parcel_prof, 'k', linewidth=2)

# Shade areas of CAPE and CIN
skew.shade_cin(p, T, parcel_prof)
skew.shade_cape(p, T, parcel_prof)

# Plot a zero degree isotherm
skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Create a hodograph
h = Hodograph(ax_hod, component_range=80.)
h.add_grid(increment=20)
im = h.plot_colormapped(u, v, wind_speed)
cbpos = fig.add_axes([1.18, 0.48, 0.1, 0.5])
cbpos.axis('off')
fig.colorbar(im)

prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
lcl_p, lcl_t = mpcalc.lcl(p[0], T[0], Td[0])
parcel_t_start = prof.T[0]
parcel_p_start = p[0]

lfc_p, lfc_t = mpcalc.lfc(p,T,Td,prof)
el_p,el_t = mpcalc.el(p,T,Td)
cape,cin = mpcalc.cape_cin(p,T,Td, parcel_prof)
mucape,mucin = mpcalc.most_unstable_cape_cin(p,T,Td)








skew.ax.set_ylim(1000,200)
skew.ax.set_xlim(-40,20)
ax1.axis('off')
ax1.text(0.01,1.0,'Parcel T Start:',size=12)
ax1.text(0.01,0.94,'Parcel P Start:',size=12)
ax1.text(0.01,0.88,'LCL P:',size=12)
ax1.text(0.01,0.82,'LCL T:',size=12)
ax1.text(0.01,0.76,'LFC P:',size=12)
ax1.text(0.01,0.7,'LFC T:',size=12)
ax1.text(0.01,0.64,'CAPE:',size=12)
ax1.text(0.01,0.58,'CIN:',size=12)

ax2.text(0.01,1.0,'{0} degC'.format(np.round(parcel_t_start.magnitude,0),size=14))
ax2.text(0.01,0.94,'{0} hPa'.format(np.round(parcel_p_start.magnitude,0),size=14))
ax2.text(0.01,0.88,'{0} hPa'.format(np.round(lcl_p.magnitude,0), size=14))
ax2.text(0.01,0.82,'{0} degC'.format(np.round(lcl_t.magnitude,0),size=14))
ax2.text(0.01,0.76,'{0} hPa'.format(np.round(lfc_p.magnitude,0),size=14))
ax2.text(0.01,0.7,'{0} degC'.format(np.round(lfc_t.magnitude,0),size=14))
ax2.text(0.01,0.64,'{0} J/kg'.format(np.round(cape.magnitude,0),size=14))
ax2.text(0.01,0.58,'{0} J/kg'.format(np.round(cin.magnitude,0),size=14))



# Show the plot
plt.show()

