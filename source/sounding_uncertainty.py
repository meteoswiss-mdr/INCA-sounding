"""This module contains several functions to calculate uncertainty of datasets.
"""

import metpy
import numpy as np
import pandas as pd
import xarray as xr
from metpy import calc
from metpy.units import units
from scipy.interpolate import griddata
from scipy import interpolate
import datetime as dt

import sounding_config as CFG
import sounding_toINCAgrid as sto
#from decorators import try_wait
from sounding_utils import expand_in_space

def read_std_COSMO(
    variable_temp, variable_temp_d, hours_diff, INCA_grid_indexes, now_time
):
    """To read uncertainty of COSMO from a folder and expand it in space.
    The function is able to select the correct uncertainty with respect to
    season, area and lead time.

    Parameters
    ----------
    variable_temp : integer
        variable number of temperature
    variable_temp_d : integer
        variable number of dew point temperature
    hours_diff : integer
        lead time
    INCA_grid_indexes: list
        list of indexes and INCA grid levels

    Returns
    -------
    COSMO_std_all : numpy array
    """
    INCA_grid = sto.extract_INCA_grid_onepoint(CFG.dir_INCA,CFG.cosmo_unc_info, CFG.cosmo_unc_info['unc'].station, "unc")
    INCA_grid = INCA_grid[CFG.cosmo_unc_info['unc'].station.values[0]][CFG.index_INCA_grid]
    year_month = (
        CFG.season.name[CFG.season.number == now_time.month].iloc[0]
        + "_"
        + str(now_time.year - 1)
    )

    # print(cosmo_leadtime)
    fn = CFG.dir_std_COSMO / f"COSMO-1_verify_upper-air_{year_month}_alps.dat"
    COSMO_std = pd.read_csv(fn.as_posix(), ";")
    COSMO_std["altitude_m"] = (
        metpy.calc.pressure_to_height_std(COSMO_std.plevel.values / 100 * units.hPa)
        * 1000
    )
    COSMO_std["scores"][COSMO_std["varno"] ==  variable_temp_d] = metpy.calc.specific_humidity_from_dewpoint(COSMO_std["scores"][COSMO_std["varno"] ==  variable_temp_d].values * units.degC, COSMO_std["plevel"][COSMO_std["varno"] ==  variable_temp_d].values * units.Pa).magnitude * 1000

    COSMO_std = COSMO_std[["altitude_m", "varno", "scorename", "scores", "leadtime"]]
    COSMO_std = COSMO_std.set_index(["altitude_m", "varno", "scorename", "leadtime"])
    COSMO_std = COSMO_std[~COSMO_std.index.duplicated()]
    COSMO_std = COSMO_std.to_xarray()
    COSMO_std = COSMO_std.interp(altitude_m=INCA_grid.values)
    COSMO_std_sel = xr.Dataset()
    COSMO_std_sel["std_t"] = COSMO_std.scores.sel(
        scorename="SD", varno=2, leadtime=[0, 3, 6], drop=True
    )
    COSMO_std_sel["std_qv"] = COSMO_std.scores.sel(
        scorename="SD", varno=59, leadtime=[0, 3, 6], drop=True
    )

    LT0 = [COSMO_std_sel.std_t.sel(leadtime=0), COSMO_std_sel.std_qv.sel(leadtime=0)]
    LT3 = [COSMO_std_sel.std_t.sel(leadtime=3), COSMO_std_sel.std_qv.sel(leadtime=3)]

    unc_dist_temp = np.full((50, 711), 9999.99)
    unc_dist_hum = np.full((50, 711), 9999.99)
    unc_dist_temp[:, 0] = LT0[0].values
    unc_dist_hum[:, 0] = LT0[1].values
    unc_dist_temp[:, 100] = LT3[0].values
    unc_dist_hum[:, 100] = LT3[1].values
    unc_dist_temp[:, 101:-1] = np.repeat(LT3[0].values, 609).reshape(50, 609)
    unc_dist_hum[:, 101:-1] = np.repeat(LT3[1].values, 609).reshape(50, 609)
    unc_dist_temp[unc_dist_temp == 9999.99] = np.nan
    unc_dist_hum[unc_dist_hum == 9999.99] = np.nan

    indexes_temp = np.arange(unc_dist_temp.shape[1])
    nanval_temp = np.isfinite(unc_dist_temp).all(axis=(0))
    unc_dist_temp_interp = interpolate.interp1d(
        indexes_temp[nanval_temp],
        unc_dist_temp[:, nanval_temp],
        bounds_error=False,
        axis=1,
    )
    unc_dist_temp_interpolated = unc_dist_temp_interp(indexes_temp)
    del indexes_temp, nanval_temp, unc_dist_temp_interp

    indexes_hum = np.arange(unc_dist_hum.shape[1])
    nanval_hum = np.isfinite(unc_dist_hum).all(axis=(0))
    unc_dist_hum_interp = interpolate.interp1d(
        indexes_hum[nanval_hum], unc_dist_hum[:, nanval_hum], bounds_error=False, axis=1
    )
    unc_dist_hum_interpolated = unc_dist_hum_interp(indexes_hum)
    del indexes_hum, nanval_hum, unc_dist_hum_interp

    unc_dist = [unc_dist_temp_interpolated, unc_dist_hum_interpolated]
    INCA_grid = sto.extract_INCA_grid_onepoint(
        CFG.dir_INCA, CFG.station_info["RS"], list(CFG.station_info["RS"].station), "RS"
    )
    del unc_dist_temp_interpolated, unc_dist_hum_interpolated

    meas_std_distance = [
        np.full((50, 640, 710), 9999.99),
        np.full((50, 640, 710), 9999.99),
    ]
    for station in CFG.station_info["RS"].station.to_list():
        meas_std_distance_1 = calculate_uncertainty_dist(
            unc_dist, INCA_grid[station], CFG.index_temp, CFG.index_hum
        )
        meas_std_distance_1[0][np.isnan(meas_std_distance_1[0])] = np.nanmax(
            meas_std_distance_1[0]
        )
        meas_std_distance_1[1][np.isnan(meas_std_distance_1[1])] = np.nanmax(
            meas_std_distance_1[1]
        )
        meas_std_distance = [
            np.minimum(meas_std_distance[0], meas_std_distance_1[0]),
            np.minimum(meas_std_distance[1], meas_std_distance_1[1]),
        ]
    del INCA_grid, unc_dist, LT0, LT3, COSMO_std, unc_dist_hum, unc_dist_temp

    COSMO_std_sel = COSMO_std_sel.expand_dims({"y_1": 640, "x_1": 710}, axis=(1, 2))
    temp = np.array(COSMO_std_sel.std_t.values)
    temp[..., 0] = meas_std_distance[0]
    COSMO_std_sel["std_t"] = (("altitude_m", "y_1", "x_1", "leadtime"), temp)
    del temp
    hum = np.array(COSMO_std_sel.std_qv.values)
    hum[..., 0] = meas_std_distance[1]
    COSMO_std_sel["std_qv"] = (("altitude_m", "y_1", "x_1", "leadtime"), hum)
    COSMO_std_sel = COSMO_std_sel.rename({"altitude_m": "z_1"})
    COSMO_std_sel["z_1"] = np.arange(1, 51)[::-1]
    del hum
    del meas_std_distance

    return COSMO_std_sel



def calculate_STD_with_distance(points, n_x, n_y, n_z, data):
    """A function to calculate standard deviation with distance (in # grid points).
    The standard deviation is calculated by shifting the cosmo grid against
    itself and calculating the standard deviation between the original and
    the shifted grid.

    Parameters
    ----------
    points : integer
        number of points which grid should be shifted
    n_x : integer
        number of grid points in x direction
    n_y : integer
        number of grid points in y direction
    n_z : integer
        number of grid points in z direction
    data : 3d numpy array
        variable on cosmo grid (e.g. temperature or dew point temperature)

    Returns
    -------
    STD_temp_space : 2d numpy array
        standard deviation with distance

    """
    STD_temp_space = np.zeros((n_z, points + 1))
    for j in range(1, (points - 1)):
        for k in range(0, (n_z - 1)):
            if j >= n_y:
                STD_temp_space[k, j] = np.sqrt(
                    np.nanmean((data[k, :, 0 : (n_x - j)] - data[k, :, j:(n_x)]) ** 2) )
            else:
                var_x = (data[k, 0 : (n_y - j), :] - data[k, j:(n_y), :]) ** 2
                var_y = (data[k, :, 0 : (n_x - j)] - data[k, :, j:(n_x)]) ** 2
                STD_temp_space[k, j] = np.sqrt(
                    ( var_x.size * np.nanmean(var_x) + var_y.size * np.nanmean(var_y) ) / (var_x.size + var_y.size) )

    return STD_temp_space


def calculate_distance_from_onepoint(n_x, n_y, indexes):
    """Calculates distance of all INCA grid point to one point on the grid.

    Parameters
    ----------
    n_x : integer
        number of grid points in x direction
    n_y : integer
        number of grid points in y direction
    indexes : list of integers
        indexes of point in x and y direction

    Returns
    -------
    distance_array : 2d numpy array
        array with distance from point
    """
    distance_array = np.zeros((n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            distance_array[i, j] = np.sqrt(
                (i - indexes[0, 0]) ** 2 + (j - indexes[1, 0]) ** 2
            )
    return distance_array


def std_from_point(data, dist):
    """To calculate standard deviation from one point on a grid.

    Parameters
    ----------
    data : 2d numpy array
        standard deviation with distance
    dist : 2d numpy array
        distance from one point on the grid
    Returns
    -------
    distance_array : 3d numpy array
        standard deviation from one point on the grid
    """
    STD_temp_space_point = np.zeros(
        (CFG.INCA_dimension["z"], CFG.INCA_dimension["y"], CFG.INCA_dimension["x"])
    )
    for i in range(0, CFG.INCA_dimension["z"]):
        STD_temp_space_point[i, :, :] = griddata(np.arange(0, 711), data[i, :], dist)
    return STD_temp_space_point


def calculate_uncertainty_dist(
    STD_space, INCA_grid_indexes, var_add_1, var_add_2, convert_to_xarray=False
):
    """A function to calculate standard deviation from one point on a grid

    Parameters
    ----------
    STD_space : list of 3d numpy arrays
        standard deviation with distance for temperature (position 0) and
        dew point temperature (position 1)
    INCA_grid_indexes: list
        list of indexes and INCA grid levels of a list of stations

    Returns
    -------
    STD_distance : list of 3d numpy arrays
        standard deviation from a specific point for temperature (position 0)
        and dew point temperature (position 1)
    """
    dist = calculate_distance_from_onepoint(
        CFG.INCA_dimension["x"], CFG.INCA_dimension["y"], INCA_grid_indexes[1]
    )
    STD_distance = [
        std_from_point(STD_space[var_add_1], dist),
        std_from_point(STD_space[var_add_2], dist),
    ]
    if convert_to_xarray == True:
        STD_distance = xr.Dataset(
            data_vars={
                "std_t": (("z_1", "y_1", "x_1"), STD_distance[0]),
                "std_qv": (("z_1", "y_1", "x_1"), STD_distance[1]),
            }
        )
    return STD_distance


def total_uncertainty(std_distance, meas_std):
    """A function to calculate total standard deviation by summing up standard
    deviation of measurement device and due to distance from measurement location.

    Parameters
    ----------
    std_distance : 3d numpy array
        standard deviation with distance
    meas_std : 3d numpy array
        standard deviation of measurement device

    Returnsf
    -------
    std_total : 3d numpy array
        total standard deviation
    """
    std_distance_station = [(std_distance[0]), (std_distance[1])]
    std_total = [
        (std_distance_station[CFG.index_temp] + meas_std[CFG.index_temp]),
        (std_distance_station[CFG.index_hum] + meas_std[CFG.index_hum]),
    ]
    return std_total


#@try_wait(8)
def std_dist_interpolate(now_time_min, lead_time_first, lead_time_next, validtime):
    """A function to read standard deviation with distance from a local folder
    and inerpolate to last x minute.

    Parameters
    ----------
    now_time_min : datetime
        timestamps of the most recent cosmo run
    lead_time : integer
        lead time of last full hour
    lead_time_last: integer
        lead time of next full hour
    validtime: datetime object

    Returns
    -------
    STD_space : list with 3d numpy arrays
        standard deviation with distance interpolated to last x minute
    """
    tmstr = now_time_min.strftime("%Y%m%d%H")
    lead_time_first = int(lead_time_first)
    lead_time_next = int(lead_time_next)
    fns = [
        (
            CFG.dir_rt_val / f"sounding_stddist_{tmstr}_{lead_time_first:0>2d}.nc"
        ).as_posix(),
        (
            CFG.dir_rt_val / f"sounding_stddist_{tmstr}_{lead_time_next:0>2d}.nc"
        ).as_posix(),
    ]
    std_space = xr.open_mfdataset(fns, combine="by_coords", concat_dim="time").load()
    STD_space = std_space.interp(time=validtime)

    return STD_space


def calculate_sigma(meas_std, T_meas):
    """To calculate square of a 3d numpy array and to make sure that at points
    where measurements are nan, standard deviation is also nan.

    Parameters
    ----------
    RM_std : 3d numpy array
        standard deviation of measurement
    T_meas : 3d numpy array
        measurement (temperature or dew point temperature) of measurement device

    Returns
    -------
    meas_std : 3d numpy array
        squared standard deviation

    """
    meas_std = meas_std ** 2
    meas_std[np.isnan(T_meas)] = np.nan
    return meas_std

def concat_cosmo(station_nr, firstobj, lastobj, step, indexes, leadtime):
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
            path_cosmo_txt = f'{CFG.dir_COSMO_past}/{firstobj_cosmo.strftime("%Y")}/{firstobj_cosmo.strftime("%m")}/{firstobj_cosmo.strftime("%d")}/cosmo-1e_inca_{firstobj_cosmo.strftime("%Y%m%d%H")}_{leadtime:0>2d}_00.nc'
            data_cosmo = xr.open_dataset(path_cosmo_txt)
            data_cosmo = data_cosmo[['t_inca', 'qv_inca']]
            if data_cosmo.t_inca.ndim > 2: 
                data_cosmo = data_cosmo.isel(x_1 = int(indexes[1][0]), y_1 = int(indexes[0][0]))
            data_cosmo = data_cosmo.reindex(z_1=data_cosmo.z_1[::-1])
            data_cosmo["z_1"] = np.arange(0, CFG.INCA_dimension["z"])
            pd_concat_cosmo.append(data_cosmo)
        except FileNotFoundError:
            print('does not exist') 
        firstobj= firstobj + dt.timedelta(hours=step)
    pd_concat_cosmo = xr.concat(pd_concat_cosmo, dim = 'time')
    return pd_concat_cosmo

def concat_RS(station_nr, firstobj, lastobj, step, altitude):
    """To concat radiosonde files over a certain time period into one xarray.

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
    altitude : array
        vertical levels to average radiosonde to

    Returns
    -------
    pd_concat_RS : xarray
        xarray of all temperature and humidity profiles to validate with measurement

    """
    pd_concat_RS = []
    while firstobj != lastobj: # loop over days
        print(firstobj.strftime('%Y%m%d%H%M%S'))
        #path_RS_txt = f'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds={station_nr}&dataSourceId=34&verbose=position&parameterIds=744,745,746,742,747&date={firstobj.strftime("%Y%m%d%H%M%S")}&obsTypeIds=22&delimiter=comma'
        path_RS_txt = f'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds={station_nr}&verbose=position&parameterIds=744,745,746,742,747&date={firstobj.strftime("%Y%m%d%H%M%S")}&obsTypeIds=22&delimiter=comma'
        #path_RS_txt = RS_archive+firstobj.strftime("%Y")+'/'+firstobj.strftime("%m")+'/'+firstobj.strftime("%d")+'/RS_'+station_nr+'_'+firstobj.strftime('%Y%m%d%H')+'.txt'
        data_RS = pd.read_csv(path_RS_txt, skiprows=[1], sep=",")
        if len(data_RS) == 0 :
            time = firstobj
            data_RS = pd.DataFrame({'t_inca' :  np.full(len(altitude), np.nan), 'qv_inca' : np.full(len(altitude), np.nan), 'altitude_m' : altitude.values}) #grid_indexes['RA']['PAY'][CFG.index_INCA_grid
        else:
            time = firstobj
            data_RS = data_RS[['744', '745', '746', '742','747']]
            data_RS = data_RS[data_RS['745'] != 10000000.0]
            data_RS['746'] = metpy.calc.specific_humidity_from_dewpoint(data_RS['747'].values * units.degC, data_RS['744'].values * units.hPa)
            data_RS['745'] = data_RS['745'] + 273.15
            data_RS = data_RS.rename(columns={'744' : 'pressure_hPa', '745' : 't_inca', '746' : 'qv_inca', '742' : 'altitude_m'})
            #data_RS = sto.average_to_INCA_grid(altitude, data_RS).reset_index(drop=True)
            data_RS = sto.interpolate_to_INCA_grid(altitude, data_RS).reset_index(drop=True)
        data_RS = data_RS.to_xarray().expand_dims(time=[time])
        pd_concat_RS.append(data_RS)
        firstobj= firstobj + dt.timedelta(hours=step)
    pd_concat_RS = xr.concat(pd_concat_RS, dim = 'time').rename({"index": "z_1"})
    return pd_concat_RS  

def concat_RM(station_nr, firstobj, lastobj, step, altitude):
    """To concat radiometer files over a certain time period into one xarray.

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
   altitude : array
        vertical levels to average radiosonde to

    Returns
    -------
    pd_concat_cosmo : xarray
        xarray of all temperature and humidity profiles to validate with radiosonde

    """
    pd_concat_RM = [] # define empty dataframe
    while firstobj != lastobj: # loop over days
        print(firstobj.strftime('%Y%m%d%H%M%S'))
        path_RM_txt = f'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds={station_nr}&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&date={firstobj.strftime("%Y%m%d%H%M%S")}&obsTypeIds=31&delimiter=comma'
        path_RM_txt_QF = f'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/integral/wmo_ind?locationIds={station_nr}&measCatNr=1&dataSourceId=38&parameterIds=3150&date={firstobj.strftime("%Y%m%d%H%M%S")}&obsTypeIds=31&delimiter=comma'
        data_RM = pd.read_csv(path_RM_txt, skiprows=[1], sep=",")
        if len(data_RM) == 0 :
            time = firstobj
            data_RM = pd.DataFrame({'t_inca' :  np.full(len(altitude), np.nan), 'qv_inca' : np.full(len(altitude), np.nan), 'altitude_m' : altitude.values}) #grid_indexes['RA']['PAY'][CFG.index_INCA_grid]
        else: 
            max_alt = np.nanmax(data_RM['level'])
            min_alt = np.nanmin(data_RM['level'])
            time = firstobj
            pressure = metpy.calc.height_to_pressure_std(data_RM['level'].values * units.meters)  
            p_w = ((data_RM['3147'] * data_RM['3148']) / 2.16679).replace(0, np.nan)
            dewpoint = metpy.calc.dewpoint(p_w.values * units.Pa)
            specific_humidity = metpy.calc.specific_humidity_from_dewpoint(dewpoint, pressure)
            data_RM['3148'] = specific_humidity
            data_RM = data_RM[['3147', '3148', 'level']]
            data_RM_QF = pd.read_csv(path_RM_txt_QF)
            if (data_RM_QF['3150'].values == 1):
                data_RM['3147'] =  np.full([len(data_RM['3147'])], np.nan) 
                data_RM['3148']  = np.full([len(data_RM['3148'])], np.nan) 
            data_RM = data_RM.rename(columns={'3147' : 't_inca', '3148' : 'qv_inca', 'level' : 'altitude_m'})
            data_RM = sto.interpolate_to_INCA_grid(altitude, data_RM).reset_index(drop=True)
        data_RM = data_RM.to_xarray().expand_dims(time=[time])
        pd_concat_RM.append(data_RM)
        firstobj= firstobj + dt.timedelta(hours=step) 
    pd_concat_RM = xr.concat(pd_concat_RM, dim = 'time').rename({"index": "z_1"})
    pd_concat_RM = pd_concat_RM.where(pd_concat_RM.altitude_m < max_alt)
    pd_concat_RM = pd_concat_RM.where(pd_concat_RM.altitude_m > min_alt)
    return pd_concat_RM

def concat_RA(station_nr, firstobj, lastobj, step, altitude):
    """To concat raman lidar files over a certain time period into one xarray.

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
   altitude : array
        vertical levels to average radiosonde to

    Returns
    -------
    pd_concat_cosmo : xarray
        xarray of all temperature and humidity profiles to validate with radiosonde

    """
    pd_concat_RA = []
    while firstobj != lastobj: # loop over days   
        print(firstobj.strftime('%Y%m%d%H%M%S'))
        #path_RA_txt = f'/data/COALITION2/internships/nal/Internship_nal/2_Data_aquisition/2_1_download_past/Raman_lidar/Validation_2/Payerne/{firstobj.strftime("%Y")}/{firstobj.strftime("%m")}/{firstobj.strftime("%d")}/RALMO_{station_nr}_{firstobj.strftime("%Y%m%d%H%M%S")}.txt'       
        path_RA_txt = f'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds={station_nr}&measCatNr=2&dataSourceId=38&parameterIds=4919,4906,4907,3147,4908,4909,4910,4911,4912,4913,4914,4915&date={firstobj.strftime("%Y%m%d%H%M%S")}&profTypeIds=1104&obsTypeIds=30&delimiter=comma'
        data_RA = pd.read_csv(path_RA_txt) # open file
        if len(data_RA) == 0 :
            time = firstobj
            data_RA = pd.DataFrame({'t_inca' :  np.full(len(altitude), np.nan), 'qv_inca' : np.full(len(altitude), np.nan), 'altitude_m' : altitude.values}) #grid_indexes['RA']['PAY'][CFG.index_INCA_grid]
        else: 
            max_alt = np.nanmax(data_RA['level'])
            min_alt = np.nanmin(data_RA['level'])
            time = firstobj
            data_RA = data_RA[['4919', '3147', 'level']]
            data_RA.loc[data_RA["3147"] == int(10000000)]= np.nan
            data_RA.loc[data_RA["4919"] == int(10000000)] = np.nan
            data_RA['4919'] = data_RA['4919'] / 1000  
            data_RA = data_RA.rename(columns={'3147' : 't_inca', '4919' : 'qv_inca', 'level' : 'altitude_m'})
            data_RA = sto.average_to_INCA_grid(altitude, data_RA).reset_index(drop=True)
            data_RA.loc[data_RA.qv_inca == 0,'qv_inca'] = np.nan
            data_RA = data_RA.where(data_RA.altitude_m < max_alt)
            data_RA = data_RA.where(data_RA.altitude_m > min_alt)
        data_RA = data_RA.to_xarray().expand_dims(time=[time])
        pd_concat_RA.append(data_RA)
        firstobj= firstobj + dt.timedelta(hours=step)
    pd_concat_RA = xr.concat(pd_concat_RA, dim = 'time').rename({"index": "z_1"})
    return pd_concat_RA


def calculate_bias_std(concat_RS, concat_meas):
    """To calculate bias and rsme between radiosonde and measurement.

    Parameters
    ----------
    concat_RS : xarray
        xarray of all temperature and humidity profiles of radiosonde
    concat_meas : xarray
        xarray of all temperature and humidity profiles to validate with radiosonde
    
    Returns
    -------
    bias : xarray
        xarray of temperature and humidity bias between radiosonde and measurement
    std : xarray
        xarray of temperature and humidity rsme between radiosonde and measurement
    count : xarray
        xarray of number of measurements 

    """
    altitude = concat_RS.isel(time=0).altitude_m
    count = concat_meas.count('time') 
    count = count.where(count.t_inca != 0)
    bias = (concat_meas - concat_RS).mean('time')[['t_inca', 'qv_inca']].assign(altitude = altitude).rename({'t_inca' : 'bias_t', 'qv_inca' : 'bias_qv'})
       
    diff = (concat_meas - concat_RS)**2
    std = xr.ufuncs.sqrt(diff.sum(dim='time') / count)
    std = std.where(std.t_inca !=0)
    std = std.where(std.qv_inca !=0)
    std = std[['t_inca', 'qv_inca']].assign(altitude = altitude).rename({'t_inca' : 'std_t', 'qv_inca' : 'std_qv'})
    return bias, std, count
