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

def concat_combined(station_nr, firstobj, lastobj, step, leadtime):
    """To concat cosmo files over a certain time period into one xarray.

    Parameters
    ----------
    station_nr : string
        station number of which to validate
    firstobj : datetime
        starting time
    lastobj : datetime
        end time
    step : integer
        step between times
    indexes : array
        x and y index of validation point
    leadtime : integer
        leadtime to validate

    Returns
    -------
    pd_concat_cosmo : xarray
        xarray of all temperature and humidity profiles to validate with radiosonde

    """
    pd_concat_cosmo = []
    while firstobj != lastobj: # loop over days
        firstobj_cosmo = firstobj - dt.timedelta(hours=leadtime)
        try: 
            print(firstobj_cosmo.strftime('%Y%m%d%H'))
            path_cosmo_txt = f'{CFG.dir_combi}/cosmo-1e_inca_{firstobj_cosmo.strftime("%Y%m%d%H")}_{leadtime:0>2d}_00.nc'
            #path_cosmo_txt = f'{CFG.dir_COSMO_past}/{firstobj_cosmo.strftime("%Y")}/{firstobj_cosmo.strftime("%m")}/{firstobj_cosmo.strftime("%d")}/cosmo-1e_inca_{firstobj_cosmo.strftime("%Y%m%d%H")}_{leadtime:0>2d}_00.nc'

            data_cosmo = xr.open_dataset(path_cosmo_txt)
            data_cosmo = data_cosmo[['t_inca', 'qv_inca']]
            if data_cosmo.t_inca.ndim > 2: 
                data_cosmo = data_cosmo.isel(x_1 = int(indexes[1][0]), y_1 = int(indexes[0][0]))
            data_cosmo = data_cosmo.reindex(z_1=data_cosmo.z_1[::-1])
            data_cosmo["z_1"] = np.arange(0, CFG.INCA_dimension["z"])
            pd_concat_cosmo.append(data_cosmo)
        except FileNotFoundError:
            print('does not exist') 
        firstobj = firstobj + dt.timedelta(hours=step)
    pd_concat_cosmo = xr.concat(pd_concat_cosmo, dim = 'time')
    return pd_concat_cosmo

def plot_std(std,count, std_1, count_1, altitude, x_range_min_temp, x_range_max_temp, x_range_min_hum, x_range_max_hum, meas_device, firstobj):
    """To plot std

    Parameters
    ----------
    bias: xarray
        bias of temperature and specific humidity
    count: xarray
        number of measurements
    altitude : dataframe
        altitude levels
    x_range_min_temp : integer
        minimum value of temperature level 
    x_range_max_temp : integer
        maximum value of temperature level 
    x_range_min_hum : integer
        minimum value of humidity level 
    x_range_max_hum : integer
        maximum value of humidity level 
    
    Returns
    -------
    plot directly saved to predefined folder
    """
    plt.rcParams['axes.linewidth'] = 1
    fig = plt.figure(figsize = (15,18))
    ax1 = fig.add_axes([0.1,0.1,0.4,0.8])
    ax1.plot(std.std_t, altitude, color = 'red', linewidth = 3,  zorder = 10)
    ax2 = fig.add_axes([0.5,0.1,0.2,0.8])
    ax3 = ax1.twiny()
        
    ax1.set_ylabel('altitude [km]', fontsize = 35)
    ax1.set_xlabel('temperature [K]', fontsize = 35)
    ax1.tick_params(labelsize = 35)
    ax1.set_ylim(0, 15)
    ax1.set_xlim(x_range_min_temp, x_range_max_temp)
    ax1.set_yticks(np.arange(0,15))
    ax1.set_xticks(np.arange(x_range_min_temp, x_range_max_temp))
    ax1.axvspan(0, 1, alpha=0.5, color='dimgrey')
    ax1.axvspan(1, 2, alpha=0.5, color='grey')
    ax1.axvspan(2, 6, alpha=0.5, color='lightgrey')
    ax1.grid() 
    ax1.xaxis.label.set_color('red')
    ax1.plot(std.std_t, altitude, color = 'red', linewidth = 3,  zorder = 10)
    ax1.plot(std_1.std_t, altitude, color = 'orange', linewidth = 3,  zorder = 10)
        
    ax2.set_xlabel('absolute #', fontsize = 35)
    ax2.tick_params(labelsize = 35)
    ax2.set_yticks(np.arange(0,15, 1000))
    ax2.set_yticklabels(ax2.yaxis.get_ticklabels()[::4])
    ax2.yaxis.tick_right()
    ax2.set_ylim(0, 15)
    ax2.set_xlim(0,360)
    ax2.grid()
    ax2.plot(count.t_inca, altitude, color = 'red', linewidth = 3,  zorder = 10)
    ax2.plot(count.qv_inca, altitude, color = 'navy', linewidth = 3,  zorder = 10)
    ax2.plot(count_1.t_inca, altitude, color = 'red', linewidth = 3,  zorder = 10)
    ax2.plot(count_1.qv_inca, altitude, color = 'navy', linewidth = 3,  zorder = 10)
    
    ax3.tick_params(labelsize = 35, colors = 'navy', axis = 'x')
    ax3.set_xlabel('specific humidity [g/kg]', fontsize = 35, color = 'navy')
    ax3.set_xlim(x_range_min_hum, x_range_max_hum)
    ax3.set_xticks(np.arange(x_range_min_hum, x_range_max_hum))
    ax3.plot(std.std_qv * 1000, altitude, color = 'navy', linewidth = 3,  zorder = 10)
    ax3.plot(std_1.std_qv * 1000, altitude, color = 'steelblue', linewidth = 3,  zorder = 10)
    plt.savefig(CFG.dir_test_plot / f"std_{meas_device}_{firstobj}_2.png",dpi=300)
 
import matplotlib.pyplot as plt

firstdate = dt.datetime.strptime('202009270000', "%Y%m%d%H%M")
lastdate = dt.datetime.strptime('202012100000', "%Y%m%d%H%M")
step=24

altitude = sto.extract_INCA_grid_onepoint(CFG.dir_INCA, CFG.validation_station, CFG.validation_station.station, "val")
altitude_m = altitude[CFG.validation_station.station[0]][0]
indexes = altitude[CFG.validation_station.station[0]][1]

concat_RS = sun.concat_RS(CFG.validation_station.station_nr[0],firstdate, lastdate, 24, altitude_m)

### cosmo
concat_cosmo = sun.concat_cosmo(CFG.validation_station.station_nr[0], firstdate, lastdate, 24, indexes, 6)
bias_cosmo, std_cosmo, count_cosmo = sun.calculate_bias_std(concat_RS, concat_cosmo)

### comined
concat_combined = concat_combined(CFG.validation_station.station_nr[0], firstdate, lastdate, 24, CFG.validation_leadtime)
bias_combined, std_combined, count_combined = sun.calculate_bias_std(concat_RS, concat_combined)

concat_RM = sun.concat_RM(CFG.validation_station.station_nr[0], firstdate, lastdate, 24, altitude_m)
bias_RM, std_RM, count_RM = sun.calculate_bias_std(concat_RS, concat_RM)
   
plot_std(std_cosmo, count_cosmo, std_RM, count_RM, altitude_m/1000, 0, 3.9, 0,1, "cosmo", firstdate.strftime("%Y%m%d%H%M"))
 
sut.plot_bias(bias_cosmo, count_cosmo, altitude_m/1000, -4, 3.9, -0.7,0.7, "cosmo", firstdate.strftime("%Y%m%d%H%M"))
sut.plot_std(std_cosmo, count_cosmo, altitude_m/1000, 0, 2, 0,3, "cosmo", firstdate.strftime("%Y%m%d%H%M"))

sut.plot_bias(bias_combined, count_combined, altitude_m/1000, -4, 3.9, -0.7,0.7, "cosmo", firstdate.strftime("%Y%m%d%H%M"))
sut.plot_std(std_combined, count_combined, altitude_m/1000, 0, 2, 0,3, "cosmo", firstdate.strftime("%Y%m%d%H%M"))

plot_std(std_cosmo, count_cosmo, std_combined, count_combined, altitude_m/1000, -4, 3.9, -0.7,0.7, "cosmo", firstdate.strftime("%Y%m%d%H%M"))

