"""This module contains several functions to convert vertical levels of
measurements to INCA levels."""

import numpy as np
import pandas as pd
import sounding_config as CFG
import xarray as xr
from scipy import spatial
from scipy.interpolate import griddata

def average_to_INCA_grid(grid_indexes, input_data):
    """A function to average temperature and dew point temperature measurements
    to INCA grid. For datasets with a finer vertical resolution than INCA grid.

    Parameters
    ----------
    grid_indexes: list
        list of indexes and INCA grid levels for all measurement points
    input_data: dataframe
        data to be converted to INCA grid

    Returns
    -------
    smoothed_grid :
        temperature and dew point temperature on INCA grid levels
    """
    if type(input_data) == dict:
        smoothed_grid = {}
        for station in input_data.keys():
            grid = grid_indexes[station][CFG.index_INCA_grid]
            input_grid_smoothed_all = pd.DataFrame()
            input_data_time = input_data[station]
            input_interp = pd.DataFrame()
            for i in range(0, len(grid)):
                if i == 0:
                    window_h_max = grid.iloc[i] + (grid.iloc[i + 1] - grid.iloc[i]) / 2
                    window_h_min = grid.iloc[i] - (grid.iloc[i + 1] - grid.iloc[i]) / 2
                elif i == len(grid) - 1:
                    window_h_min = grid.iloc[i] - (grid.iloc[i] - grid.iloc[(i - 1)]) / 2
                    window_h_max = grid.iloc[i] + (grid.iloc[i] - grid.iloc[(i - 1)]) / 2
                else:
                    window_h_min = grid.iloc[i] - (grid.iloc[i] - grid.iloc[(i - 1)]) / 2
                    window_h_max = grid.iloc[i] + (grid.iloc[i + 1] - grid.iloc[i]) / 2
    
                input_data_within_bound = input_data_time[
                    (input_data_time.altitude_m <= float(window_h_max))
                    & (input_data_time.altitude_m >= float(window_h_min))
                ]
                aver_mean = pd.DataFrame(
                    {
                        "t_inca": np.mean(input_data_within_bound.temperature_K),
                        "qv_inca": np.mean(input_data_within_bound['specific_humidity_gkg-1']),
                        "altitude_m": (grid.iloc[i]),
                    },
                    index=[i],
                )
                input_interp = input_interp.append(aver_mean)
            input_grid_smoothed_all = input_grid_smoothed_all.append(input_interp)
            input_grid_smoothed_all = input_grid_smoothed_all.set_index(
                np.arange(1, 51)[::-1]
            )
            input_grid_smoothed_all = input_grid_smoothed_all.to_xarray().rename(
                {"index": "z_1"}
            )
            smoothed_grid[station] = input_grid_smoothed_all
        
    else: 
        smoothed_grid = {}
        grid = grid_indexes
        input_grid_smoothed_all = pd.DataFrame()
        input_data_time = input_data
        input_interp = pd.DataFrame()
        for i in range(0, len(grid)):
            if i == 0:
                window_h_max = grid.iloc[i] + (grid.iloc[i + 1] - grid.iloc[i]) / 2
                window_h_min = grid.iloc[i] - (grid.iloc[i + 1] - grid.iloc[i]) / 2
            elif i == len(grid) - 1:
                window_h_min = grid.iloc[i] - (grid.iloc[i] - grid.iloc[(i - 1)]) / 2
                window_h_max = grid.iloc[i] + (grid.iloc[i] - grid.iloc[(i - 1)]) / 2
            else:
                window_h_min = grid.iloc[i] - (grid.iloc[i] - grid.iloc[(i - 1)]) / 2
                window_h_max = grid.iloc[i] + (grid.iloc[i + 1] - grid.iloc[i]) / 2

            input_data_within_bound = input_data_time[
                (input_data_time.altitude_m <= float(window_h_max))
                & (input_data_time.altitude_m >= float(window_h_min))
            ]
            aver_mean = pd.DataFrame(
                {
                    "t_inca": np.mean(input_data_within_bound.t_inca),
                    "qv_inca": np.mean(input_data_within_bound.qv_inca),
                    "altitude_m": (grid.iloc[i]),
                },
                index=[i],
            )
            input_interp = input_interp.append(aver_mean)
        input_grid_smoothed_all = input_grid_smoothed_all.append(input_interp)
        input_grid_smoothed_all = input_grid_smoothed_all.set_index(
            np.arange(1, len(grid_indexes)+1)[::-1]
        )
        smoothed_grid = input_grid_smoothed_all
    return smoothed_grid

def interpolate_to_INCA_grid(grid_indexes, input_data):
    """A function to interpolate temperature and dew point temperature
    measurements to INCA grid. For datasets with a coarser vertical resolution
    than INCA grid.

    Parameters
    ----------
    grid_indexes: list
        list of indexes and INCA grid levels of all stations to be smoothed
    input_data: dataframe
        data to be converted to INCA grid

    Returns
    -------
    smoothed_grid :
        temperature and dew point temperature on INCA grid levels
    """
    smoothed_grid_all = {}
    if type(input_data) == dict: 
        for station in input_data.keys():
            grid = grid_indexes[station][CFG.index_INCA_grid]
            input_data_time = input_data[station]
            smoothed_grid = pd.DataFrame({"t_inca": griddata(input_data_time.altitude_m.values,input_data_time.temperature_K.values,grid,),"qv_inca": griddata(input_data_time.altitude_m.values,input_data_time['specific_humidity_gkg-1'].values,grid,),"altitude_m": grid,}).set_index(np.arange(1, 51)[::-1])
            smoothed_grid_all[station] = smoothed_grid.to_xarray().rename({"index": "z_1"})
    else:
        grid = grid_indexes
        input_data_time = input_data
        smoothed_grid_all = pd.DataFrame({"t_inca": griddata(input_data_time.altitude_m.values,input_data_time.t_inca.values,grid,),"qv_inca": griddata(input_data_time.altitude_m.values,input_data_time.qv_inca.values,grid,),"altitude_m": grid,}).set_index(np.arange(1, 51)[::-1])

    return smoothed_grid_all

def extract_INCA_grid_onepoint(
    INCA_archive, station_nr, available_stations, meas_device
):
    """A function to find indexes of nearest INCA grid point and extract INCA
    grid levels for a list of measurement stations.

    Parameters
    ----------
    INCA_archive : pathlib object
        path to INCA grid
    station_nr : list
        station numbers
    available_stations : list
        List of available station names

    Returns
    -------
    grid_indexes : dictionary
        lists number of stations with indexes of closest INCA grid point
        (position 0) and INCA grid levels (position 1)
    """
    grid_indexes = {}
    for station in available_stations:
        if meas_device == "NUCAPS":
            SP_lon = station_nr[station][CFG.index_NUCAPS_lon]
            SP_lat = station_nr[station][CFG.index_NUCAPS_lat]
        elif meas_device == "unc":
            SP_lon = station_nr[meas_device].Longitude.values[0]
            SP_lat = station_nr[meas_device].Latitude.values[0]
        elif meas_device == "val":
            SP_lon = CFG.validation_station.Longitude.values[0]
            SP_lat = CFG.validation_station.Latitude.values[0]
        else:
            SP_lon = CFG.station_info[meas_device][
                CFG.station_info[meas_device].station == station
            ].Longitude.values[0]
            SP_lat = CFG.station_info[meas_device][
                CFG.station_info[meas_device].station == station
            ].Latitude.values[0]
        inca_fn = INCA_archive / "inca_topo_levels_hsurf_ccs4.nc"
        grid = xr.open_dataset(inca_fn.as_posix())
        lon = grid.lon_1.values
        lat = grid.lat_1.values
        lonlat = np.dstack([lat.ravel(), lon.ravel()])[0, :, :]
        tree = spatial.KDTree(lonlat)
        coordinates = tree.query(([SP_lat, SP_lon]))
        coords_close = lonlat[coordinates[1]]
        indexes = np.array(np.where(grid.lon_1 == coords_close[1]))
        grid = pd.DataFrame(
            {"altitude_m": grid.HFL[:, indexes[0], indexes[1]][:, 0, 0].values}
        )[::-1]
        grid = grid.iloc[:, 0].reset_index(drop=True)
        grid_indexes[station] = [grid, indexes]
    return grid_indexes
