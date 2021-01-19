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
import math

import metpy.calc as cc
from metpy.cbook import get_test_data
from metpy.plots import Hodograph, SkewT
from metpy.units import units

import xarray as xr
from scipy.interpolate import griddata
from scipy import spatial
import urllib3
import datetime as dt

def average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid_1, input_data_filtered, comparison_grid):
    INCA_grid = INCA_grid_1[::-1].reset_index(drop=True)
    input_grid_smoothed_all = pd.DataFrame()
    while firstobj != lastobj:
        nowdate = firstobj.strftime('%Y%m%d')
        print(nowdate) 
        input_data_time = input_data_filtered[input_data_filtered.time_YMDHMS == firstobj] 
        input_data_time = input_data_time.iloc[::-1].reset_index(drop=True)
        comparison_grid_time = comparison_grid[comparison_grid.time_YMDHMS == firstobj]
        comparison_grid_time = comparison_grid_time.reset_index(drop=True)   
  
        if comparison_grid_time.empty:
            firstobj = firstobj + dt.timedelta(days=1)
            print('now')
        else:  
            input_interp = pd.DataFrame()
            for i in range(0,len(INCA_grid)):
                if (i == 0):
                    window_h_max = INCA_grid.iloc[i] + (INCA_grid.iloc[i+1] - INCA_grid.iloc[i]) / 2
                    window_h_min = INCA_grid.iloc[i] - (INCA_grid.iloc[i+1] - INCA_grid.iloc[i]) / 2
                elif (i==len(INCA_grid)-1):
                    window_h_min = INCA_grid.iloc[i] - (INCA_grid.iloc[i]-INCA_grid.iloc[(i-1)])  / 2
                    window_h_max = INCA_grid.iloc[i] + (INCA_grid.iloc[i]-INCA_grid.iloc[(i-1)])  / 2
                else: 
                    window_h_min = INCA_grid.iloc[i] - (INCA_grid.iloc[i]-INCA_grid.iloc[(i-1)] )  / 2
                    window_h_max = INCA_grid.iloc[i] + (INCA_grid.iloc[i+1]-INCA_grid.iloc[i]) / 2
                        
                input_data_within_bound = input_data_time[(input_data_time.altitude_m <= float(window_h_max)) & (input_data_time.altitude_m >= float(window_h_min))] 
                if window_h_min < np.min(input_data_time.altitude_m):
                    aver_mean = pd.DataFrame({'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan, 'altitude_m' : INCA_grid.loc[i]}, index = [i])
                    print('small')
                elif input_data_within_bound.altitude_m.count() == 0:
                     aver_mean = pd.DataFrame({'temperature_mean' : griddata(input_data_time.altitude_m.values, input_data_time.temperature_degC.values, INCA_grid.loc[i]), 'temperature_d_mean' : griddata(input_data_time.altitude_m.values, input_data_time.dew_point_degC.values, INCA_grid.loc[i]),'altitude_m' : INCA_grid.loc[i]}, index = [i]).reset_index(drop=True)
                     print('interpolate')
                else: 
                    aver_mean = pd.DataFrame({'temperature_mean': np.mean(input_data_within_bound.temperature_degC), 'temperature_d_mean' : np.mean(input_data_within_bound.dew_point_degC), 'altitude_m' : (INCA_grid.iloc[i])}, index = [i])
                    print('average')
                input_interp = input_interp.append(aver_mean)
            input_grid_smoothed_all = input_grid_smoothed_all.append(input_interp)
            input_grid_smoothed_all = input_grid_smoothed_all.reset_index(drop=True)
            firstobj= firstobj + dt.timedelta(days=1) 
    return input_grid_smoothed_all
  

def read_radiosonde(firstobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&dataSourceId=34&verbose=position&delimiter=comma&parameterIds=744,745,746,742,748,743,747&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=22'
    RS_data = pd.read_csv(url, skiprows = [1], sep=',')
    RS_data = RS_data.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent', '742':'altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })
    RS_data = RS_data[RS_data['temperature_degC'] != 1e+07]
    RS_data['time_YMDHMS'] = pd.to_datetime(RS_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    return RS_data

def read_SMN(firstobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/surface/wmo_ind?locationIds=06610&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&parameterIds=90,91,98&delimiter=comma'
    SMN_data = pd.read_csv(url, skiprows = [1], sep=',')
    SMN_data = SMN_data.rename(columns = {'termin' : 'time_YMDHMS', '90':'pressure_hPa', '91': 'temperature_degC', '98':'relative_humidity_percent'})
    SMN_data['time_YMDHMS'] = pd.to_datetime(SMN_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    return SMN_data

def convert_altitude_to_pressure(z_altitude_m, T_temperature_degC, firstobj, g, m_mol_air, R):
    p_1 = np.zeros(len(z_altitude_m))
    p_1[0] = read_SMN(firstobj).pressure_hPa[0]
    integrant = g*m_mol_air/(R*(T_temperature_degC))
    for i in range(1, len(z_altitude_m)):
        p_1[i] = p_1[0]*math.exp(-np.trapz(integrant[0:i], z_altitude_m[0:i]))
    return p_1

def convert_altitude_to_pressure(z_altitude_m, T_temperature_degC, firstobj):
    p_1 = np.zeros(len(z_altitude_m))
    p_1[0] = read_SMN(firstobj).pressure_hPa[0]
    for i in range(1, len(z_altitude_m)):
        p_1[i] = cc.add_height_to_pressure((p_1[i-1] * units.hPa), (z_altitude_m.iloc[i]-z_altitude_m.iloc[i-1]) * units.meters).magnitude
    return p_1



#p = convert_altitude_to_pressure(RS_data.altitude_m, RS_data.temperature_degC, firstobj)

####################################### Definition ########################################
firstdate = '20200918120000'
lastdate = '20200919120000'
firstobj=dt.datetime.strptime(firstdate,'%Y%m%d%H%M%S')
lastobj=dt.datetime.strptime(lastdate,'%Y%m%d%H%M%S')

lon_payerne = 6.93608
lat_payerne = 46.82201

INCA_archive = '/data/COALITION2/internships/nal/data/COSMO/'
####################################### load data ########################################
########################################## 
## INCA grid  
##########################################
##########################################
INCA_grid = xr.open_dataset(INCA_archive+'/inca_topo_levels_hsurf_ccs4.nc')
        
### coordinate at Payerne
lon = INCA_grid.lon_1.values
lat = INCA_grid.lat_1.values
lonlat = np.dstack([lat.ravel(), lon.ravel()])[0,:,:]
tree = spatial.KDTree(lonlat)
coordinates = tree.query([([lat_payerne  , lon_payerne ])])
coords_close = lonlat[coordinates[1]]
indexes = np.array(np.where(INCA_grid.lon_1 == coords_close[0,1]))
INCA_grid_payerne = pd.DataFrame({'altitude_m' : INCA_grid.HFL[:, indexes[0], indexes[1]][:,0,0].values})[::-1]
INCA_grid_payerne = INCA_grid_payerne.iloc[:,0].reset_index(drop=True)

########################################## 
## Radiosonde data  
##########################################
RS_data = read_radiosonde(firstobj)  
RS_averaged = average_RS_to_INCA_grid(firstobj, lastobj, INCA_grid_payerne, RS_data, RS_data)[::-1].reset_index(drop=True)
g = 9.81
m_mol_air = 28.965*10.0**(-3.0)
R = 8.31446

#### RADIOSONDE
#RS_averaged = pd.DataFrame({'temperature_mean' : RS_data.temperature_degC, 'temperature_d_mean' : RS_data.dew_point_degC, 'pressure_hPa' : RS_data.pressure_hPa, 'altitude_m' : RS_data.altitude_m}).reset_index(drop=True)

g = 9.81
m_mol_air = 28.965*10.0**(-3.0)
R = 8.31446

#### CONVERTED
#### 1) original standard
#RS_averaged = pd.DataFrame({'temperature_mean' : RS_data.temperature_degC, 'temperature_d_mean' : RS_data.dew_point_degC, 'altitude_m' : RS_data.altitude_m}).reset_index(drop=True)
#RS_averaged['pressure_hPa'] = cc.height_to_pressure_std(RS_averaged.altitude_m.values * units.meters)
#RS_averaged = RS_averaged.reset_index(drop=True)
#### 2) original converted
#RS_averaged = pd.DataFrame({'temperature_mean' : RS_data.temperature_degC, 'temperature_d_mean' : RS_data.dew_point_degC, 'altitude_m' : RS_data.altitude_m}).reset_index(drop=True)
#RS_averaged['pressure_hPa'] = cc.height_to_pressure_std(RS_data.altitude_m.values * units.meters) #convert_altitude_to_pressure(RS_data.altitude_m, RS_data.temperature_degC, firstobj)
#RS_averaged = RS_averaged.reset_index(drop=True)
#### 3) smoothed
RS_averaged['pressure_hPa'] = convert_altitude_to_pressure(RS_averaged.altitude_m, RS_averaged.temperature_mean, firstobj)
RS_averaged = RS_averaged.dropna().reset_index(drop=True)

#plt.plot(RS_averaged.temperature_mean, RS_averaged.pressure_hPa, color = 'green')
#plt.plot(RS_data.temperature_degC, RS_data.pressure_hPa, color = 'blue')
#plt.gca().invert_yaxis()

#plt.plot(RS_averaged.temperature_mean, RS_averaged.altitude_m, color = 'green', zorder = 1000)
#plt.plot(RS_data.temperature_degC, RS_data.altitude_m, color = 'blue')

# Calculate the LCL
lcl_pressure, lcl_temperature = cc.lcl(RS_averaged.pressure_hPa[0] * units.hPa, RS_averaged.temperature_mean[0] * units.degC, RS_averaged.temperature_d_mean[0] * units.degC)
print(lcl_pressure, lcl_temperature)

# Calculate the parcel profile.
parcel_prof = cc.parcel_profile(RS_averaged.pressure_hPa.values * units.hPa, RS_averaged.temperature_mean[0] * units.degC, RS_averaged.temperature_d_mean[0] * units.degC).to('degC')


fig = plt.figure(figsize=(9, 9))

# Grid for plots
skew = SkewT(fig, rotation=45)

ax1 = fig.add_axes([0.98,0.2,0.1,0.3])
ax2 = fig.add_axes([1.14,0.2,0.1,0.3])
ax1.axis('off')
ax2.axis('off')
#ax_hod = fig.add_axes([0.96, 0.58, 0.3, 0.3])

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(RS_averaged.pressure_hPa * units.hPa, RS_averaged.temperature_mean * units.degC, 'r')
skew.plot(RS_averaged.pressure_hPa * units.hPa, RS_averaged.temperature_d_mean.values * units.degC, 'g')
#skew.plot_barbs(p[0:p.size:100],u[0:u.size:100],v[0:v.size:100])
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 60)

# Plot LCL as black dot
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

# Plot the parcel profile as a black line
skew.plot(RS_averaged.pressure_hPa, parcel_prof, 'k', linewidth=2)

# Shade areas of CAPE and CIN
skew.shade_cin(RS_averaged.pressure_hPa.values * units.hPa, RS_averaged.temperature_mean.values * units.degC, parcel_prof)
skew.shade_cape(RS_averaged.pressure_hPa.values * units.hPa, RS_averaged.temperature_mean.values * units.degC, parcel_prof)

# Plot a zero degree isotherm
skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

prof = cc.parcel_profile(RS_averaged.pressure_hPa.values * units.hPa, RS_averaged.temperature_mean.values[0] * units.degC, RS_averaged.temperature_d_mean.values[0] * units.degC).to('degC')
lcl_p, lcl_t = cc.lcl(RS_averaged.pressure_hPa.values[0] * units.hPa, RS_averaged.temperature_mean.values[0] * units.degC, RS_averaged.temperature_d_mean.values[0] * units.degC)
parcel_t_start = prof.T[0]
parcel_p_start = RS_averaged.pressure_hPa.values[0] * units.hPa

lfc_p, lfc_t = cc.lfc(RS_averaged.pressure_hPa.values * units.hPa,RS_averaged.temperature_mean.values * units.degC,RS_averaged.temperature_d_mean.values * units.degC,prof)
el_p,el_t = cc.el(RS_averaged.pressure_hPa.values * units.hPa,RS_averaged.temperature_mean.values * units.degC,RS_averaged.temperature_d_mean.values * units.degC)
cape,cin = cc.cape_cin(RS_averaged.pressure_hPa.values * units.hPa,RS_averaged.temperature_mean.values * units.degC,RS_averaged.temperature_d_mean.values * units.degC, parcel_prof)
sbcape,sbcin = cc.surface_based_cape_cin(RS_averaged.pressure_hPa.values * units.hPa,RS_averaged.temperature_mean.values * units.degC,RS_averaged.temperature_d_mean.values * units.degC)
mlcape,mlcin = cc.mixed_layer_cape_cin(RS_averaged.pressure_hPa.values * units.hPa,RS_averaged.temperature_mean.values * units.degC,RS_averaged.temperature_d_mean.values * units.degC)
mucape,mucin = cc.most_unstable_cape_cin(RS_averaged.pressure_hPa.values * units.hPa,RS_averaged.temperature_mean.values * units.degC,RS_averaged.temperature_d_mean.values * units.degC)

skew.ax.set_xlabel('Temperature [°C]', size = 16)
skew.ax.set_ylabel('Pressure [hPa]', size = 16)
skew.ax.tick_params(labelsize=16)
skew.ax.set_title('Skew-T-log-P Diagram: '+str(firstobj.day)+' '+str(firstobj.strftime("%B"))+' '+str(firstobj.year) , size = 20, weight = 'bold')

#skew.ax.set_ylim(1000,200)
skew.ax.set_xlim(-30,40)
ax1.axis('off')
# Zustand
ax1.text(-0.5,2.2,'Parcel P Start:',size=14, weight = 'bold')
ax1.text(-0.5,2.1,'Parcel T Start:',size=14, weight = 'bold')
ax1.text(-0.5,2,'LCL P:',size=14, weight = 'bold')
ax1.text(-0.5,1.9,'LCL T:',size=14, weight = 'bold')
ax1.text(-0.5,1.8,'LFC P:',size=14, weight = 'bold')
ax1.text(-0.5,1.7,'LFC T:',size=14, weight = 'bold')
ax1.text(-0.5,1.6,'EL P:',size=14, weight = 'bold')
ax1.text(-0.5,1.5,'EL T:',size=14, weight = 'bold')

# CIN, CAPE
ax1.text(-0.5,1.3,'CAPE:',size=14, weight = 'bold')
ax1.text(-0.5,1.2,'CIN:',size=14, weight = 'bold')
ax1.text(-0.5,1.1,'SBCAPE:',size=14, weight = 'bold')
ax1.text(-0.5,1,'SBCIN:',size=14, weight = 'bold')
ax1.text(-0.5,0.9,'MLCAPE:',size=14, weight = 'bold')
ax1.text(-0.5,0.8,'MLCIN:',size=14, weight = 'bold')
ax1.text(-0.5,0.7,'MUCAPE:',size=14, weight = 'bold')
ax1.text(-0.5,0.6,'MUCIN:',size=14, weight = 'bold')

# instability indices
#ax1.text(-0.5,0.7,'Total precipitable water (TWP):',size=8)
#ax1.text(-0.5,0.6,'Mean relative humidity in the lowest 0-3km:',size=8)
#ax1.text(-0.5,0.5,'K-index:',size=8)
#ax1.text(-0.5,0.4,'LI:',size=8)
#ax1.text(-0.5,0.3,'BLI:',size=8)
#ax1.text(-0.5,0.2,'Maximum buoyancy:',size=8)
#ax1.text(-0.5,0.1,'400/700 hPa lapse rate:',size=8)
#ax1.text(-0.5,-0.02,'600/925 hPa lapse rate:',size=8)
#ax1.text(-0.5,-0.08,'BRN',size=8)
#ax1.text(-0.5,-0.14,'BI',size=8)
#ax1.text(-0.5,-0.14,'SI',size=8)
#ax1.text(-0.5,-0.14,'SWEAT',size=8)
#ax1.text(-0.5,-0.14,'SWISS',size=8)

# Zustand
ax2.text(0.01,2.2,'{0} hPa'.format(np.round(parcel_p_start.magnitude,0)),size=14)
ax2.text(0.01,2.1,'{0} degC'.format(np.round(parcel_t_start.magnitude,0)),size=14)
ax2.text(0.01,2,'{0} hPa'.format(np.round(lcl_p.magnitude,0)), size=14)
ax2.text(0.01,1.9,'{0} degC'.format(np.round(lcl_t.magnitude,0)),size=14)
ax2.text(0.01,1.8,'{0} hPa'.format(np.round(lfc_p.magnitude,0)),size=14)
ax2.text(0.01,1.7,'{0} degC'.format(np.round(lfc_t.magnitude,0)),size=14)
ax2.text(0.01,1.6,'{0} degC'.format(np.round(el_p.magnitude,0)),size=14)
ax2.text(0.01,1.5,'{0} degC'.format(np.round(el_t.magnitude,0)),size=14)

ax2.text(0.01,1.3,'{0} J/kg'.format(np.round(cape.magnitude,0)),size=14)
ax2.text(0.01,1.2,'{0} J/kg'.format(np.round(cin.magnitude,0)),size=14)
ax2.text(0.01,1.1,'{0} J/kg'.format(np.round(sbcape.magnitude,0)),size=14)
ax2.text(0.01,1,'{0} J/kg'.format(np.round(sbcin.magnitude,0)),size=14)
ax2.text(0.01,0.9,'{0} J/kg'.format(np.round(mlcape.magnitude,0)),size=14)
ax2.text(0.01,0.8,'{0} J/kg'.format(np.round(mlcin.magnitude,0)),size=14)
ax2.text(0.01,0.7,'{0} J/kg'.format(np.round(mucape.magnitude,0)),size=14)
ax2.text(0.01,0.6,'{0} J/kg'.format(np.round(mucin.magnitude,0)),size=14)

# CIN, CAPE

# Show the plot
plt.show()

