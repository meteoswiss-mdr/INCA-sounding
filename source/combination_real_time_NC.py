### functions to get time
################################################################################################################################################
def get_closest_noon_midnight(now_time):
    """A function to find out if current time is closer to noon or midnight.

    Parameters
    ----------
    now_time = datetimere
        current date

    Returns
    -------
    DT : integer
        noon (12) or midnight (0)
    """
    if np.abs((12 - now_time.hour)) <= 6:
        DT = 12
    else:
        DT = 0
    return DT


def get_last_cosmo_date(delay):
    """A function to find date of current cosmo file. Consists of last cosmo run time (every 3 hours)
    and the lead time.

    Parameters
    ----------
    delay : integer
        time it normally takes until cosmo forecast is available in folder

    Returns
    -------
    now_time_cosmo : datetime
        time of last cosmo run
    hours_diff: integer
        lead time
    """
    now_time = dt.datetime.now().replace(microsecond=0, second=0, minute=0)
    now_time_delay = dt.datetime.now() - dt.timedelta(minutes=delay)
    last_cosmo_foerecast = int(now_time_delay.hour / 3) * 3
    hours_diff = (now_time.hour / 3 * 3) - (int(now_time_delay.hour / 3) * 3)
    now_time_cosmo = now_time - dt.timedelta(hours=hours_diff)
    return now_time_cosmo, hours_diff


def get_last_xmin(xminute):
    """Extracts current date and rounds down to last x minute.

    Parameters
    ----------
    xminute = integer
        minute to which it should be rounded down

    Returns
    -------
    rounded_minute : integer
        last rounded x minute
    """
    rounded_minute = int(int(dt.datetime.now().strftime("%M")) / xminute) * xminute
    return rounded_minute


### functions to read datasets
################################################################################################################################################
def read_RADIOMETER(firstobj, lastobj, station_nr):
    """Function to open radiometer measurement from a local folder, to convert
    absolute humidity [g/m3] to dew point temperature [K] and to take into consideration rain flag.

    Parameters
    ----------
    firstobj : datetime
        first date of which to download radiometer data
    lastobj : datetime
        last date of which to download radiometer data
    station_nr : string
        number of station from which measurement wants to be obtained

    Returns
    -------
    RM : dataframe
        radiometer measurement for predefined time interval
    """
    url = (
        c.dirs_in["dir_RM"]
        + "station_"
        + str(station_nr)
        + "/RM_"
        + str(firstobj.strftime("%Y%m%d%H%M"))
        + ".txt"
    )
    # -> to download with url request: url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds='+str(station_nr)+'&delimiter=comma&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'-'+str(dt.datetime.strftime(lastobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=31'

    RM = pd.read_csv(url, skiprows=[1], sep=",")
    RM = RM.rename(
        columns={
            "termin": "time_YMDHMS",
            "3147": "temperature_K",
            "3148": "absolute_humidity_gm3",
            "level": "altitude_m",
        }
    )
    # < temperature >
    RM["temperature_degC"] = RM.temperature_K - 273.15

    p_w = (RM.temperature_K * RM.absolute_humidity_gm3) / 2.16679
    RM["dew_point_degC"] = metpy.calc.dewpoint((p_w.values * units.Pa))

    url = (
        c.dirs_in["dir_RM"]
        + "station_"
        + str(station_nr)
        + "/RM_QF_"
        + str(firstobj.strftime("%Y%m%d%H%M"))
        + ".txt"
    )
    # -> to download with url request: url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/integral/wmo_ind?locationIds='+str(station_nr)+'&measCatNr=1&dataSourceId=38&parameterIds=3150,5560,5561&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&delimiter=comma&obsTypeIds=31'
    RM_quality_flag = pd.read_csv(url, skiprows=[1], sep=",")
    RM["quality_flag"] = pd.concat(
        [RM_quality_flag["3150"]] * len(RM), ignore_index=True
    )
    RM["radiometer_quality_flag_temperature_profile"] = pd.concat(
        [RM_quality_flag["5560"]] * len(RM), ignore_index=True
    )
    RM["radiometer_quality_flag_humidity_profile"] = pd.concat(
        [RM_quality_flag["5561"]] * len(RM), ignore_index=True
    )

    RM["time_YMDHMS"] = pd.to_datetime(RM.time_YMDHMS, format="%Y%m%d%H%M%S")
    RM.loc[RM.quality_flag == 1, "temperature_K"] = np.nan
    RM.loc[RM.quality_flag == 1, "dew_point_K"] = np.nan
    return RM


def read_LIDAR(firstobj, lastobj, station_nr):
    """A function to open raman lidar measurement from a local folder and to convert
    specific humidity [g/kg-1] to dew point temperature [K].

    firstobj : datetime
        first date of which to download radiometer data
    lastobj : datetime
        last date of which to download radiometer data
    station_nr : string
        number of station from which data wants to be downloaded

    Returns
    -------
    RA : dataframe
        raman lidar measurement for predefined time interval
    """

    url = (
        "http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds="
        + str(station_nr)
        + "&measCatNr=2&delimiter=comma&dataSourceId=38&parameterIds=4919,4906,4907,3147,4908,4909,4910,4911,4912,4913,4914,4915&date="
        + str(dt.datetime.strftime(firstobj, "%Y%m%d%H%M%S"))
        + "&profTypeIds=1104&obsTypeIds=30+"
    )
    RA = pd.read_csv(url, skiprows=[1], sep=",")
    RA = RA.rename(
        columns={
            "termin": "time_YMDHMS",
            "level": "altitude_m",
            "4919": "specific_humidity_gkg-1",
            "4906": "uncertainty_specific_humidity_gkg-1",
            "4907": "vertical_resolution_specific_humidity_m",
            "3147": "temperature_K",
            "4908": "uncertainty_temperature_K",
            "4909": "vertical_resolution_temperature",
            "4910": "normalised_backscatter",
            "4911": "uncertainty_backscatter",
            "4912": "vert_resolution_backscatter",
            "4913": "aerosol_dispersion_rate",
            "4914": "uncertainty_dispersion_rate",
            "4915": "vertical_resolution_aerosol_dispersion_rate",
        }
    )
    RA.loc[RA["temperature_K"] == int(10000000), "temperature_K"] = np.nan
    ## add dewpoint temperature
    pressure = metpy.calc.height_to_pressure_std(RA.altitude_m.values * units.meters)
    dewpoint_degC = (
        metpy.calc.dewpoint_from_specific_humidity(
            RA["specific_humidity_gkg-1"].values * units("g/kg"),
            (RA.temperature_K.values) * units.kelvin,
            pressure,
        ).magnitude
        + 273.15
    )
    RA.insert(value=dewpoint_degC, column="dew_point_K", loc=11)
    RA.loc[RA["specific_humidity_gkg-1"] == int(10000000), "dew_point_K"] = np.nan
    RA["time_YMDHMS"] = pd.to_datetime(RA.time_YMDHMS, format="%Y%m%d%H%M%S")
    return RA


def open_NUCAPS_file(NUCAPS_path_file):
    """To open NUCAPS file from a local folder.

    Parameters
    ----------
    NUCAPS_path_file : string
        path and filename of NUCAPS data file

    Returns
    -------
    NUCAPS_data : xarray
        NUCAPS data file
    """
    NUCAPS_data = xr.open_dataset(
        NUCAPS_path_file, decode_times=False
    )  # time units are non-standard, so we dont decode them here
    units, reference_date = NUCAPS_data.Time.attrs["units"].split(" since ")
    if units == "msec":
        ref_date = dt.datetime.strptime(
            reference_date, "%Y-%m-%dT%H:%M:%SZ"
        )  # usually '1970-01-01T00:00:00Z'
        NUCAPS_data["datetime"] = [
            -1 if np.isnan(t) else ref_date + timedelta(milliseconds=t)
            for t in NUCAPS_data.Time.data
        ]
    return NUCAPS_data


def extract_NUCAPS_dataframe(NUCAPS_data, NUCAPS_points):
    """Function extract relevant variables from NUCAPS xarray and returns it as a dataframe.

    Parameters
    ----------
    NUCAPS_data : dataframe
        dataframe containing temperature, dew point temperature and altitude in meter
    NUCAPS_points : list
        indices of measurement points

    Returns
    -------
    NUCAPS_data_extracted : dictionary of dataframes
        returns a dataframe for every NCUAPS measurement point with the variables pressure in hPa, temperature in K, dew point temperature in K and time
    """
    NUCAPS_data_extracted = {}
    for i in range(len(NUCAPS_points)):
        NUCAPS_data_extracted[NUCAPS_points[i]] = pd.DataFrame(
            {
                "pressure_hPa": NUCAPS_data.Pressure.values[NUCAPS_points[i], :]
                * units.hPa,
                "temperature_degC": (NUCAPS_data.Temperature.values - 273.15)[
                    NUCAPS_points[i], :
                ],
                "dew_point_degC": mpcalc.dewpoint(
                    mpcalc.vapor_pressure(
                        (NUCAPS_data.Pressure.values[NUCAPS_points[i], :] * units.hPa),
                        (NUCAPS_data["H2O_MR"].values)[NUCAPS_points[i], :]
                        * units("g/kg")
                        * 1000,
                    )
                ),
                "time_YMDHMS": np.repeat(
                    NUCAPS_data.datetime.values[NUCAPS_points[i]], 100
                ),
            }
        )
    return NUCAPS_data_extracted


def read_last_next_cosmo_file():
    """This function reads the cosmo file from the last and the next full hour and interpolates between the datasets to current 10 min time.

    Parameters
    ----------
    no parameters

    Returns
    -------
    COSMO_data_last : list
        last COSMO file, position 0 for temperature and position 1 for humidity
    COSMO_data_next : list
        next COSMO file, position 0 for temperature and position 1 for humidity
    now_time_cosmo : datetime
        last COSMO time
    hours_diff : integer
        forecast time
    COSMO_xarray : xarray
        COSMO file only t, qv and p extracted
    """
    now_time_cosmo, hours_diff = get_last_cosmo_date(70)
    try:
        COSMO_data_last = xr.open_dataset(
            c.dirs_in["dir_COSMO"]
            + "cosmo-1e_inca_"
            + now_time_cosmo.strftime("%Y%m%d%H")
            + "_0"
            + str(int(hours_diff))
            + "_00.nc"
        )
        COSMO_data_next = xr.open_dataset(
            c.dirs_in["dir_COSMO"]
            + "cosmo-1e_inca_"
            + now_time_cosmo.strftime("%Y%m%d%H")
            + "_0"
            + str(int(hours_diff + 1))
            + "_00.nc"
        )
        COSMO_xarray = COSMO_data_last[["t_inca", "qv_inca", "p_inca"]]
    except FileNotFoundError:
        now_time_cosmo = now_time_cosmo - dt.timedelta(hours=3)
        COSMO_data_last = COSMO_data = xr.open_dataset(
            c.dirs_in["dir_COSMO"]
            + "cosmo-1e_inca_"
            + now_time_cosmo.strftime("%Y%m%d%H")
            + "_0"
            + str(int(hours_diff))
            + "_00.nc"
        )
        COSMO_data_next = COSMO_data = xr.open_dataset(
            c.dirs_in["dir_COSMO"]
            + "cosmo-1e_inca_"
            + now_time_cosmo.strftime("%Y%m%d%H")
            + "_0"
            + str(int(hours_diff + 1))
            + "_00.nc"
        )
        COSMO_xarray = COSMO_data_last[["t_inca", "qv_inca", "p_inca"]]

    return COSMO_data_last, COSMO_data_next, now_time_cosmo, hours_diff, COSMO_xarray


def read_radiosonde(firstobj, lastobj):
    """A function to open radiosonde measurement from a local folder.

    firstobj : datetime
        first date of which to download radiometer data
    lastobj : datetime
        last date of which to download radiometer dat

    Returns
    -------
    RS_data : dataframe
        radiosonde measurement for predefined time interval
    """
    url = (
        "http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&dataSourceId=34&verbose=position&delimiter=comma&parameterIds=744,745,746,742,748,743,747&date="
        + str(dt.datetime.strftime(firstobj, "%Y%m%d%H%M%S"))
        + "-"
        + str(dt.datetime.strftime(lastobj, "%Y%m%d%H%M%S"))
        + "&obsTypeIds=22"
    )
    RS_data = pd.read_csv(url, skiprows=[1], sep=",")
    RS_data = RS_data.rename(
        columns={
            "termin": "time_YMDHMS",
            "744": "pressure_hPa",
            "745": "temperature_degC",
            "746": "relative_humidity_percent",
            "742": "altitude_m",
            "748": "wind_speed_ms-1",
            "743": "wind_dir_deg",
            "747": "dew_point_degC",
        }
    )
    RS_data = RS_data[RS_data["temperature_degC"] != 1e07]
    RS_data["temperature_K"] = RS_data["temperature_degC"] + 273.15
    RS_data["dew_point_K"] = RS_data["dew_point_degC"] + 273.15
    RS_data["time_YMDHMS"] = pd.to_datetime(RS_data.time_YMDHMS, format="%Y%m%d%H%M%S")
    return RS_data


def open_available_station_data(now_time, station_nr, measurement_device_function):
    """For a list of stations test if measurement is available and opens the file if it exists.

    Parameters
    ----------
    now_time : datetime
        current time
    station_nr : list
        list of all station numbers of which data wants to be downloaded
     measurement_device_function: function
         function to open measurement file

    Returns
    -------
    station_data : dictionary of dataframes
        contains data from all stations
    station_names : list
        all station names where a measurement exists
    """
    station_data = {}
    station_names = []
    for i in range(len(station_nr)):
        try:
            station_data[station_nr[i][c.index_temp]] = measurement_device_function(
                now_time, now_time, station_nr[i][c.index_hum]
            )
            if station_nr[i][c.index_temp] in c.tempro_stations:
                station_data[station_nr[i][c.index_temp]].dew_point_K = np.nan
            station_names.append(station_nr[i][c.index_station_name])
        except ValueError:
            pass
    return station_data, station_names


### functions to convert datasets to INCA grid
################################################################################################################################################
def limit_to_INCA_grid(INCA_archive, NUCAPS_xarray):
    """A function to extract NUCAPS measurement points within INCA grid domain.

    Parameters
    ----------
    INCA_archive : string
        path to INCA grid
    NUCAPS_xarray : xarray
        xarray with NUCAPS data

    Returns
    -------
    NUCAPS_points : list
        indices of measurement points
    NUCAPS_points_coordinates : list
        station numbers
    """
    INCA_grid = xr.open_dataset(INCA_archive + "inca_topo_levels_hsurf_ccs4.nc")
    lon_min = np.min(INCA_grid.lon_1.values)
    lon_max = np.max(INCA_grid.lon_1.values)
    lat_min = np.min(INCA_grid.lat_1.values)
    lat_max = np.max(INCA_grid.lat_1.values)
    NUCAPS_points = []
    NUCAPS_points_coords = {}
    for i in range(len(NUCAPS.Longitude.values)):
        if (
            (NUCAPS.Longitude.values[i] > lon_min)
            & (NUCAPS.Longitude.values[i] < lon_max)
            & (NUCAPS.Latitude.values[i] > lat_min)
            & (NUCAPS.Latitude.values[i] < lat_max)
        ):
            NUCAPS_points.append(i)
            NUCAPS_points_coords[i] = [
                i,
                NUCAPS.Longitude.values[i],
                NUCAPS.Latitude.values[i],
            ]
    return NUCAPS_points, NUCAPS_points_coords


def average_to_INCA_grid(INCA_grid_indexes, input_data, meas_data, station_names):
    """A function to average temperature and dew point temperature measurements to INCA grid. For datasets with a finer vertical resolution than INCA grid. 

    Parameters
    ----------
    INCA_grid_indexes: list
        list of indexes and INCA grid levels for all measurement points
    input_data: dataframe
        data to be converted to INCA grid
    station_names : list
        station names

    Returns
    -------
    smoothed_INCA_grid :
        temperature and dew point temperature on INCA grid levels
    """

    smoothed_INCA_grid = {}
    for j in range(len(station_names)):
        INCA_grid = INCA_grid_indexes[meas_data][station_names[j]][c.index_INCA_grid]
        input_grid_smoothed_all = pd.DataFrame()
        input_data_time = input_data[station_names[j]]
        input_interp = pd.DataFrame()
        for i in range(0, len(INCA_grid)):
            if i == 0:
                window_h_max = (
                    INCA_grid.iloc[i] + (INCA_grid.iloc[i + 1] - INCA_grid.iloc[i]) / 2
                )
                window_h_min = (
                    INCA_grid.iloc[i] - (INCA_grid.iloc[i + 1] - INCA_grid.iloc[i]) / 2
                )
            elif i == len(INCA_grid) - 1:
                window_h_min = (
                    INCA_grid.iloc[i]
                    - (INCA_grid.iloc[i] - INCA_grid.iloc[(i - 1)]) / 2
                )
                window_h_max = (
                    INCA_grid.iloc[i]
                    + (INCA_grid.iloc[i] - INCA_grid.iloc[(i - 1)]) / 2
                )
            else:
                window_h_min = (
                    INCA_grid.iloc[i]
                    - (INCA_grid.iloc[i] - INCA_grid.iloc[(i - 1)]) / 2
                )
                window_h_max = (
                    INCA_grid.iloc[i] + (INCA_grid.iloc[i + 1] - INCA_grid.iloc[i]) / 2
                )

            input_data_within_bound = input_data_time[
                (input_data_time.altitude_m <= float(window_h_max))
                & (input_data_time.altitude_m >= float(window_h_min))
            ]
            aver_mean = pd.DataFrame(
                {
                    "temperature_mean": np.mean(input_data_within_bound.temperature_K),
                    "temperature_d_mean": np.mean(input_data_within_bound.dew_point_K),
                    "altitude_m": (INCA_grid.iloc[i]),
                },
                index=[i],
            )
            input_interp = input_interp.append(aver_mean)
        input_grid_smoothed_all = input_grid_smoothed_all.append(input_interp)
        input_grid_smoothed_all = input_grid_smoothed_all.reset_index(drop=True)
        smoothed_INCA_grid[station_names[j]] = input_grid_smoothed_all
    return smoothed_INCA_grid


def interpolate_to_INCA_grid(INCA_grid_indexes, input_data, station_names):
    """A function to interpolate temperature and dew point temperature measurements to INCA grid. For datasets with a coarser vertical resolution than INCA grid.

    Parameters
    ----------
    INCA_grid_indexes: list
        list of indexes and INCA grid levels of all stations to be smoothed
    input_data: dataframe
        data to be converted to INCA grid
    station_names : list
        station names

    Returns
    -------
    smoothed_INCA_grid :
        temperature and dew point temperature on INCA grid levels
    """
    smoothed_INCA_grid = {}
    for j in range(len(station_names)):
        INCA_grid = INCA_grid_indexes[station_names[j]][c.index_INCA_grid]
        input_data_time = input_data[station_names[j]]
        smoothed_INCA_grid[station_names[j]] = pd.DataFrame(
            {
                "temperature_mean": griddata(
                    input_data_time.altitude_m.values,
                    input_data_time.temperature_K.values,
                    INCA_grid,
                ),
                "temperature_d_mean": griddata(
                    input_data_time.altitude_m.values,
                    input_data_time.dew_point_K.values,
                    INCA_grid,
                ),
                "altitude_m": INCA_grid,
            }
        ).reset_index(drop=True)
    return smoothed_INCA_grid


def add_altitude_m(NUCAPS_data, NUCAPS_points):
    """A function to convert NUCAPS measurements on pressure levels to altitude in meters. The function uses the metpy.calc.thickness_hydrostatic():
        https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.thickness_hydrostatic.html#metpy.calc.thickness_hydrostatic

    Parameters
    ----------
    NUCAPS_data : dataframe
        dataframe containing temperature, dew point temperature and altitude in meter
    NUCAPS_points : list
        indices of measurement points

    Returns
    -------
    NUCAPS_data : dictionary of dataframes
        returns a dataframe for every NCUAPS measurement point with the variables pressure in hPa, temperature in K, dew point temperature in K and time
    """
    for i in range(len(NUCAPS_points)):
        SMN_data = pd.read_csv(
            "http://wlsprod.meteoswiss.ch:9010/jretrievedwh/surface/wmo_ind?locationIds=06610&delimiter=comma&date="
            + str(
                dt.datetime.strftime(
                    NUCAPS_data[NUCAPS_points[i]]
                    .time_YMDHMS[0]
                    .replace(microsecond=0, second=0, minute=0),
                    "%Y%m%d%H%M%S",
                )
            )
            + "&parameterIds=90,91",
            skiprows=[1],
            sep=",",
        )
        data_comma_temp = (
            (
                NUCAPS_data[NUCAPS_points[i]][
                    [
                        "time_YMDHMS",
                        "pressure_hPa",
                        "dew_point_degC",
                        "temperature_degC",
                    ]
                ]
            )
            .append(
                {
                    "time_YMDHMS": str(
                        dt.datetime.strftime(
                            NUCAPS_data[NUCAPS_points[i]]
                            .time_YMDHMS[0]
                            .replace(microsecond=0, second=0, minute=0),
                            "%Y%m%d%H%M%S",
                        )
                    ),
                    "pressure_hPa": SMN_data["90"].values[0],
                    "temperature_degC": SMN_data["91"].values[0],
                    "altitude_m": 491,
                },
                ignore_index=True,
            )[::-1]
            .reset_index(drop=True)
        )
        for j in range(1, len(data_comma_temp)):
            p_profile = (
                data_comma_temp.pressure_hPa.iloc[j - 1],
                data_comma_temp.pressure_hPa.iloc[j],
            ) * units.hPa
            t_profile = (
                data_comma_temp.temperature_degC.iloc[j - 1],
                data_comma_temp.temperature_degC.iloc[j],
            ) * units.degC
            deltax = metpy.calc.thickness_hydrostatic(p_profile, t_profile)
            data_comma_temp.loc[j, "altitude_m"] = data_comma_temp.altitude_m.iloc[
                j - 1
            ] + (deltax.magnitude)
        NUCAPS_data[NUCAPS_points[i]]["altitude_m"] = data_comma_temp.altitude_m[
            1:101
        ].values
    return NUCAPS_data


def extract_INCA_grid_onepoint(INCA_archive, station_nr, station_names):
    """A function to find indexes of nearest INCA grid point and extract INCA grid levels for a list of measurement stations.

    Parameters
    ----------
    INCA_archive : string
        path to INCA grid
    station_nr : list
        station numbers
    station_names : list
        station names

    Returns
    -------
    INCA_grid_indexes : dictionary
        lists number of stations with indexes of closest INCA grid point (position 0) and INCA grid levels (position 1)
    """
    INCA_grid_indexes = {}
    for i in range(len(station_names)):
        SP_lon = c.coordinates_RM.Longitude[
            c.coordinates_RM.station == station_nr[i][c.index_station_name]
        ].iloc[0]
        SP_lat = c.coordinates_RM.Latitude[
            c.coordinates_RM.station == station_nr[i][c.index_station_name]
        ].iloc[0]
        INCA_grid = xr.open_dataset(INCA_archive + "inca_topo_levels_hsurf_ccs4.nc")
        lon = INCA_grid.lon_1.values
        lat = INCA_grid.lat_1.values
        lonlat = np.dstack([lat.ravel(), lon.ravel()])[0, :, :]
        tree = spatial.KDTree(lonlat)
        coordinates = tree.query(([SP_lat, SP_lon]))
        coords_close = lonlat[coordinates[1]]
        indexes = np.array(np.where(INCA_grid.lon_1 == coords_close[1]))
        INCA_grid = pd.DataFrame(
            {"altitude_m": INCA_grid.HFL[:, indexes[0], indexes[1]][:, 0, 0].values}
        )[::-1]
        INCA_grid = INCA_grid.iloc[:, 0].reset_index(drop=True)
        INCA_grid_indexes[station_names[i]] = [INCA_grid, indexes]
    return INCA_grid_indexes


def calculate_dt_COSMO_meas(station_names, INCA_grid_indexes, meas_device):
    """To calculate difference between COSMO and measurement at location of measurement station.
    Parameters
    ----------
    station_names : list of strings
        station names
    INCA_grid_indexes : dictionary
        lists number of stations with indexes of closest INCA grid point (position 0) and INCA grid levels (position 1)
    meas_device : string
        'RM' for radiometer, 'RA' for raman LIDAR

    Returns
    -------
    dT_COSMO : 1d numpy array
        difference between COSMO and measurement
    """
    dT_COSMO = {}
    for i in range(len(station_names)):
        index = INCA_grid_indexes[meas_device][station_names[i]][1]
        dT = pd.DataFrame(
            {
                "temperature_mean": (
                    COSMO_data[c.index_temp][:, index[0][0], index[1][0]]
                    - meas_data["RM"]["PAY"].temperature_mean
                ),
                "temperature_d_mean": (
                    COSMO_data[c.index_hum][:, index[0][0], index[1][0]]
                    - meas_data["RM"]["PAY"].temperature_d_mean
                ),
            }
        )
        dT_COSMO[station_names[i]] = dT
    return dT_COSMO


def calculate_new_meas(meas_data, meas_device, dT_COSMO, station_names):
    """To calculate new measurement dataset by taking into account topography.
    Parameters
    ----------
    meas_data : dictionary of dataframes
        measurements
    meas_device : string
        'RM' for radiometer, 'RA' for raman LIDAR
    dT_COSMO : 1d numpy array
        difference between COSMO and measurement
    station_names : list of strings
        station names

    Returns
    -------
    meas_data : dictionary of arrays
        measurements
    """
    for i in range(len(station_names)):
        meas_data[meas_device][station_names[i]] = (
            COSMO_data + dT_COSMO[meas_device][station_names[i]]
        )
    return meas_data

### functions to expand profile data to 3d
################################################################################################################################################
def expand_in_space(data, n_z, n_y, n_x, station_names):
    """To expand measurement profile at one point to the whole space for a list of stations.
    Parameters
    ----------
    meas_data : dataframe
        measurement profile
    n_z : integer
        number of grid points in z direction
    n_y : integer
        number of grid points in y direction
    n_x : integer
        number of grid points in x direction
    station_names : list of strings
        station names

    Returns
    -------
    meas_data_expanded : 3d numpy array
        array with expanded measurement profile
    """
    meas_data_expanded = {}
    for f in range(len(station_names)):
        data_station = data[station_names[f]]
        data_array_temp = np.tile(
            data_station.temperature_mean.values.reshape(50, 1), (1, 640 * 710)
        ).reshape((50, 640, 710))
        data_array_temp_d = np.tile(
            data_station.temperature_d_mean.values.reshape(50, 1), (1, 640 * 710)
        ).reshape((50, 640, 710))
        meas_data_expanded[station_names[f]] = [data_array_temp, data_array_temp_d]
    return meas_data_expanded


def expand_in_space_1(data, n_z, n_y, n_x):
    """A function to expand measurement profile at one point to the whole space.
    Parameters
    ----------
    meas_data : dataframe
        measurement profile
    n_z : integer
        number of grid points in z direction
    n_y : integer
        number of grid points in y direction
    n_x : integer
        number of grid points in x direction
    station_names : list of strings
        station names

    Returns
    -------
    meas_data_expanded : 3d numpy array
        array with expanded measurement profile
    """
    data_array = np.tile(data.reshape(50, 1), (1, 640 * 710)).reshape((50, 640, 710))
    return data_array


### functions to calculate bias
################################################################################################################################################
def calculate_bias_time(now_time_cosmo):
    """This function identifies at which time to start bias calculation.

    Parameters
    ----------
    now_time_cosmo : date of last cosmo file

    Returns
    -------
    bias_date : datetime
        find last time where bias can be calculated
    DT : integer
        noon (12) or midnight (0)
    """
    hour = (now_time_cosmo + timedelta(hours=hours_diff)).hour
    hour_round = hour - int(12)
    if hour_round <= 0:
        bias_date = (now_time_cosmo + timedelta(hours=hours_diff)) - timedelta(
            hours=hour
        )
    else:
        bias_date = (
            now_time_cosmo
            + timedelta(hours=hours_diff)
            - timedelta(hours=int(np.abs(hour_round)))
        )
    DT = bias_date.hour
    return bias_date, DT


def calc_run_bias_window(
    firstobj,
    INCA_grid_indexes,
    window_size_days,
    measurement_device_function,
    meas_device,
):
    """A function to calculate bias within a certain time window. Bias is calculated by validating measurement from measurement device with radiosonde measurements.

    Parameters
    ----------
    firstobj : datetime
        last time where bias can be calculated
    INCA_grid_indexes : list
        list of indexes and INCA grid levels of all stations to be smoothed
    window_size_days : integer
        time window to calculate bias
    measurement_device_function : string
        measurement device name

    Returns
    -------
    bias_t : dataframe
        bias of temperature within predefined time window for a vertical profile
    bias_t_d : dataframe
        bias of dew point temperature within predefined time window for a vertical profile
    """
    lastobj_bias = firstobj
    firstobj = firstobj - dt.timedelta(days=window_size_days)
    meas_data_1 = pd.DataFrame()
    RS_data_1 = pd.DataFrame()
    while firstobj != lastobj_bias:
        RS_data = {"PAY": read_radiosonde(firstobj, firstobj)}
        RS_data = average_to_INCA_grid(INCA_grid_indexes, RS_data, meas_device, ["PAY"])

        meas_data = {"PAY": measurement_device_function(firstobj, firstobj, "06610")}
        meas_data = average_to_INCA_grid(INCA_grid_indexes, meas_data, "RM", ["PAY"])

        RS_data_1 = RS_data_1.append(RS_data["PAY"])
        meas_data_1 = meas_data_1.append(meas_data["PAY"])
        firstobj = firstobj + dt.timedelta(days=1)

    bias_t = np.subtract(
        meas_data_1.temperature_mean.reset_index(drop=True),
        RS_data_1.temperature_mean.reset_index(drop=True),
    )
    bias_t = pd.DataFrame(
        {
            "diff_temp": bias_t.reset_index(drop=True),
            "altitude_m": RS_data_1.altitude_m.reset_index(drop=True),
        }
    )
    bias_t = bias_t.astype(float)
    bias_t = (
        bias_t.groupby("altitude_m")["diff_temp"]
        .mean()
        .to_frame(name="mean_all")
        .reset_index(drop=True)
    )
    bias_t[np.isnan(bias_t)] = 0

    bias_t_d = np.subtract(
        meas_data_1.temperature_d_mean.reset_index(drop=True),
        RS_data_1.temperature_d_mean.reset_index(drop=True),
    )
    bias_t_d = pd.DataFrame(
        {
            "diff_temp_d": bias_t_d.reset_index(drop=True),
            "altitude_m": RS_data_1.altitude_m.reset_index(drop=True),
        }
    )
    bias_t_d = bias_t_d.astype(float)
    bias_t_d = (
        bias_t_d.groupby("altitude_m")["diff_temp_d"]
        .mean()
        .to_frame(name="mean_all")
        .reset_index(drop=True)
    )
    bias_t_d[np.isnan(bias_t_d)] = 0

    return bias_t, bias_t_d


### functions to calculate uncertainty
################################################################################################################################################
def read_std_COSMO(
    variable_temp, variable_temp_d, hours_diff, station_nr, INCA_grid_indexes
):
    """To read uncertainty of COSMO from a folder and expand it in space. The function is able to select the correct uncertainty with respect to season, area and lead time.

    Parameters
    ----------
    variable_temp : integer
        variable number of temperature
    variable_temp_d : integer
        variable number of dew point temperature
    hours_diff : integer
        lead time
    station_nr : string
        station number
    INCA_grid_indexes: list
        list of indexes and INCA grid levels

    Returns
    -------
    COSMO_std_all : numpy array
    """
    INCA_grid = INCA_grid_indexes["RM"]["PAY"][c.index_INCA_grid]
    year_month = (
        c.season.name[c.season.number == now_time.month].iloc[0]
        + "_"
        + str(now_time.year - 1)
    )
    cosmo_leadtime = int(hours_diff / 3) * 3
    # print(cosmo_leadtime)
    COSMO_std = pd.read_csv(
        c.dirs_in["dir_std_COSMO"]
        + "/COSMO/"
        + year_month
        + "/scratch/owm/verify/upper-air/"
        + year_month
        + "/COSMO-1/output_all_alps//allscores.dat",
        ";",
    )
    COSMO_std["altitude_m"] = (
        metpy.calc.pressure_to_height_std(COSMO_std.plevel.values / 100 * units.hPa)
        * 1000
    )
    COSMO_std = COSMO_std[COSMO_std.scorename == "SD"]
    COSMO_std = COSMO_std[COSMO_std.leadtime == int(cosmo_leadtime)]

    COSMO_std_temp = griddata(
        COSMO_std[COSMO_std.varno == variable_temp][0:20].altitude_m.values,
        COSMO_std[COSMO_std.varno == variable_temp][0:20].scores.values,
        (INCA_grid),
    )
    COSMO_std_temp = expand_in_space_1(COSMO_std_temp, n_z, n_y, n_x)

    COSMO_std_temp_d = griddata(
        COSMO_std[COSMO_std.varno == variable_temp_d][0:20].altitude_m.values,
        COSMO_std[COSMO_std.varno == variable_temp_d][0:20].scores.values,
        (INCA_grid),
    )
    COSMO_std_temp_d = expand_in_space_1(COSMO_std_temp_d, n_z, n_y, n_x)
    COSMO_std_all = [COSMO_std_temp, COSMO_std_temp_d]
    return COSMO_std_all


def calculate_STD_with_distance(points, n_x, n_y, n_z, data):
    """A function to calculate standard deviation with distance (in # grid points). The standard deviation is calculated by shifting the cosmo grid against
    itself and calculating the standard deviation between the original and the shifted grid.

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
    STD_temp_space = np.zeros((n_z, (points + 1)))
    for j in range(0, (points - 1)):
        for k in range(0, (n_z - 1)):
            if j >= n_y:
                STD_temp_space[k, j] = np.nanmean(
                    np.sqrt(
                        ((data[k, :, 0 : (n_x - j)] - data[k, :, j:(n_x)]) ** 2) / 2
                    )
                )
            else:
                std_x = np.sqrt(
                    ((data[k, 0 : (n_y - j), :] - data[k, j:(n_y), :]) ** 2) / 2
                )
                num_x = std_x.shape[0] * std_x.shape[1]
                std_x = np.nanmean(std_x)
                std_y = np.sqrt(
                    ((data[k, :, 0 : (n_x - j)] - data[k, :, j:(n_x)]) ** 2) / 2
                )
                num_y = std_y.shape[0] * std_y.shape[1]
                std_y = np.nanmean(std_y)
                STD_temp_space[k, j] = (num_x / (num_x + num_y)) * std_x + (
                    num_y / (num_x + num_y)
                ) * std_y
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
    STD_temp_space_point = np.zeros((n_z, n_y, n_x))
    for i in range(0, n_z):
        STD_temp_space_point[i, :, :] = griddata(np.arange(0, 711), data[i, :], dist)
    return STD_temp_space_point


def calculate_uncertainty_dist(STD_space, INCA_grid_indexes, n_x, n_y, station_names):
    """A function to calculate standard deviation from one point on a grid

     Parameters
     ----------
     STD_space : list of 3d numpy arrays
         standard deviation with distance for temperature (position 0) and dew point temperature (position 1)
     INCA_grid_indexes: list
         list of indexes and INCA grid levels of a list of stations
     n_x : integer
         number of grid points in x direction
     n_y : integer
         number of grid points in y direction
    station_names : list of strings
        station names

     Returns
     -------
     STD_distance : list of 3d numpy arrays
         standard deviation from a specific point for temperature (position 0) and dew point temperature (position 1)
    """
    dist = calculate_distance_from_onepoint(n_x, n_y, INCA_grid_indexes[1])
    # STD_distance = [
    #    std_from_point(STD_space[0], dist),
    #    std_from_point(STD_space[1], dist),
    # ]
    STD_distance = [
        std_from_point(STD_space[0], dist),
        std_from_point(STD_space[1], dist),
    ]
    end = dt.datetime.now()
    logger.info(end - start)
    return STD_distance


def total_uncertainty(std_distance, meas_std, station_names):
    """A function to calculate total standard deviation by summing up standard deviation of measurement device and due to distance from measurement location.

    Parameters
    ----------
    std_distance : 3d numpy array
        standard deviation with distance
    meas_std : 3d numpy array
        standard deviation of measurement device
    station_names : list of strings
        station names

    Returns
    -------
    std_total : 3d numpy array
        total standard deviation
    """
    std_distance_station = [(std_distance[0]), (std_distance[1])]
    std_total = [
        (std_distance_station[c.index_temp] + meas_std[c.index_temp]),
        (std_distance_station[c.index_hum] + meas_std[c.index_hum]),
    ]
    return std_total


def calculate_sigma(meas_std, T_meas):
    """To calculate square of a 3d numpy array and to make sure that at points where measurements are nan, standard deviation is also nan.

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


################################################################################################################################################
import copy
import datetime as dt
from datetime import timedelta
import logging
import numpy as np
import pandas as pd
import time

import INCA3d_config as c
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import metpy
from metpy import calc
from metpy.units import units
import pysteps
from pysteps import nowcasts
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from scipy import spatial
from scipy.interpolate import griddata
import xarray as xr
import metpy.calc as mpcalc
import bottleneck as bn
import logging

logging.basicConfig(
    filename=c.dirs_out["dir_sigma_COSMO"] + c.logger_filename,
    level=c.logger_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

n_z = c.INCA_dimension["z"]
n_y = c.INCA_dimension["y"]
n_x = c.INCA_dimension["x"]

all_station_names = {}

################################################################################################################################################
# ----------------------------------------------------------- READ DATA -------------------------------------------------------------------
################################################################################################################################################
logger.info("Read data...")
start = dt.datetime.now()
########## COSMO ##########
(
    COSMO_data_last,
    COSMO_data_next,
    now_time_cosmo,
    hours_diff,
    COSMO_xarray,
) = read_last_next_cosmo_file()
COSMO_data_last = [
    COSMO_data_last.t_inca.values[0, :, :, :][::-1],
    metpy.calc.dewpoint_from_specific_humidity(
        COSMO_data_last.qv_inca, COSMO_data_last.t_inca, COSMO_data_last.p_inca
    )[0, :, :, :][::-1].magnitude,
]
COSMO_data_next = [
    COSMO_data_next.t_inca.values[0, :, :, :][::-1],
    metpy.calc.dewpoint_from_specific_humidity(
        COSMO_data_next.qv_inca, COSMO_data_next.t_inca, COSMO_data_next.p_inca
    )[0, :, :, :][::-1].magnitude,
]
logger.debug("COSMO time:%", now_time_cosmo)
logger.debug("COSMO time:%", hours_diff)
a_COSMO_last = int(dt.datetime.now().strftime("%M")) / 60
a_COSMO_next = 1 - a_COSMO_last
COSMO_data = [
    (
        (a_COSMO_last * COSMO_data_last[c.index_temp])
        + (a_COSMO_next * COSMO_data_next[c.index_temp])
    ),
    (
        (a_COSMO_last * COSMO_data_last[c.index_hum])
        + (a_COSMO_next * COSMO_data_next[c.index_hum])
    ),
]

# plt.contourf(COSMO_data[c.index_temp][20,:,:], cmap = cm.Spectral_r) # index_hum
# plt.contourf(COSMO_data_last[c.index_temp][20,:,:]-COSMO_data_next[c.index_temp][20,:,:], cmap = cm.Spectral_r) # index_hum
# plt.colorbar()

meas_data = {}
INCA_grid_indexes = {}
dT_COSMO = {}
########## RADIOMETER ##########
if c.RM_data == True:
    now_time = now_time_cosmo + timedelta(hours=hours_diff)
    DT = get_closest_noon_midnight(now_time)
    RM_data, RM_station_names = open_available_station_data(
        now_time + dt.timedelta(minutes=get_last_xmin(10)),
        c.RM_station_nr,
        read_RADIOMETER,
    )
    INCA_grid_indexes["RM"] = extract_INCA_grid_onepoint(
        c.dirs_in["dir_INCA"], c.RM_station_nr, RM_station_names
    )
    # plt.plot(RM_data['PAY'].temperature_K, RM_data['PAY'].altitude_m, color = 'red', zorder = 2)
    # plt.scatter(RM_data['PAY'].temperature_K, RM_data['PAY'].altitude_m, s = 10, color = 'red')
    meas_data["RM"] = interpolate_to_INCA_grid(
        INCA_grid_indexes["RM"], RM_data, RM_station_names
    )
    # plt.plot(meas_data['RM']['PAY'].temperature_mean, meas_data['RM']['PAY'].altitude_m, zorder = 1)
    # plt.scatter(meas_data['RM']['PAY'].temperature_mean, meas_data['RM']['PAY'].altitude_m, s = 10)
    all_station_names["RM"] = RM_station_names

    dT_COSMO["RM"] = calculate_dt_COSMO_meas(RM_station_names, INCA_grid_indexes, "RM")
########## LIDAR ##########
if c.RA_data == True:
    start = dt.datetime.now()
    RA_data, RA_station_names = open_available_station_data(
        now_time + dt.timedelta(minutes=get_last_xmin(30)) - dt.timedelta(days=300),
        c.RA_station_nr,
        read_LIDAR,
    )
    INCA_grid_indexes["RA"] = extract_INCA_grid_onepoint(
        c.dirs_in["dir_INCA"], c.RA_station_nr, c.lidar_station
    )
    # plt.plot(RA_data['PAY'].temperature_K, RA_data['PAY'].altitude_m, color = 'blue')
    # plt.scatter(RA_data['PAY'].temperature_K, RA_data['PAY'].altitude_m)
    meas_data["RA"] = interpolate_to_INCA_grid(
        INCA_grid_indexes["RA"], RA_data, RA_station_names
    )
    # plt.plot(meas_data['RA']['PAY'].temperature_mean, meas_data['RA']['PAY'].altitude_m, color = 'orange')
    # plt.scatter(meas_data['RA']['PAY'].temperature_mean, meas_data['RA']['PAY'].altitude_m, color= 'orange')
    all_station_names["RA"] = RA_station_names
    del RA_data
    del RA_station_names
    end = dt.datetime.now()
    logger.info(end - start)
########## NUCAPS ##########
if c.NUCAPS_data == True:
    NUCAPS = open_NUCAPS_file(
        c.dirs_in["dir_NUCAPS"]
        + "/2019/05/01/NUCAPS-EDR_v2r0_j01_s201905011251590_e201905011252290_c201905011325090.nc"
    )
    NUCAPS_points, NUCAPS_points_coords = limit_to_INCA_grid(
        c.dirs_in["dir_INCA"], NUCAPS
    )
    INCA_grid_indexes["NUCAPS"] = extract_INCA_grid_onepoint_NUCAPS(
        c.dirs_in["dir_INCA"], NUCAPS_points_coords, NUCAPS_points
    )

    NUCAPS_data = extract_NUCAPS_dataframe(NUCAPS, NUCAPS_points)
    # plt.plot(NUCAPS_data[118].temperature_degC, NUCAPS_data[118].pressure_hPa)
    NUCAPS_data = add_altitude_m(NUCAPS_data, NUCAPS_points)
    # plt.plot(NUCAPS_data[118].temperature_degC, NUCAPS_data[118].altitude_m)
    meas_data["NUCAPS"] = average_to_INCA_grid(
        INCA_grid_indexes["NUCAPS"], NUCAPS_data, NUCAPS_points
    )

    all_station_names["NUCAPS"] = NUCAPS_points
    del NUCAPS_data
    del NUCAPS_points

end = dt.datetime.now()
logger.info(end - start)

################################################################################################################################################
# ----------------------------------------------------------- BIAS CORRECTION -------------------------------------------------------------------
################################################################################################################################################
########## RADIOMETER ##########
logger.info("Bias correction...")
start = dt.datetime.now()
if c.bias_correction == True & c.RM_data == True:
    bias_date, DT = calculate_bias_time(now_time_cosmo)
    bias_temp, bias_temp_d = calc_run_bias_window(
        bias_date, INCA_grid_indexes, c.bias_window_size, read_RADIOMETER, "RM"
    )
    # plt.plot(bias_temp, INCA_grid_indexes['RM']['PAY'][c.index_temp]) # index_hum

    meas_data["RM"]["PAY"].temperature_mean = np.subtract(
        meas_data["RM"]["PAY"].temperature_mean.values, bias_temp.iloc[:, 0].values
    )
    meas_data["RM"]["PAY"].temperature_d_mean = np.subtract(
        meas_data["RM"]["PAY"].temperature_d_mean, bias_temp_d.iloc[:, 0].values
    )
    # plt.plot(meas_data['RM']['PAY'].temperature_mean, meas_data['RM']['PAY'].altitude_m)

########## LIDAR ##########
if c.bias_correction == True & c.RA_data == True:
    bias_date, DT = calculate_bias_time(now_time_cosmo)
    bias_temp, bias_temp_d = calc_run_bias_window(
        bias_date, INCA_grid_indexes, c.bias_window_size, read_LIDAR, "RA"
    )
    # plt.plot(bias_temp, INCA_grid_indexes['RA']['PAY'][c.index_temp]) # index_hum

    meas_data["RA"]["PAY"].temperature_mean = np.subtract(
        meas_data["RA"]["PAY"].temperature_mean.values, bias_temp.iloc[:, 0].values
    )
    meas_data["RA"]["PAY"].temperature_d_mean = np.subtract(
        meas_data["RA"]["PAY"].temperature_d_mean, bias_temp_d.iloc[:, 0].values
    )
    # plt.plot(meas_data['RA']['PAY'].temperature_mean, meas_data['RA']['PAY'].altitude_m)

end = dt.datetime.now()
logger.info(end - start)

################################################################################################################################################
# ----------------------------------------------------------- EXPAND IN SPACE -------------------------------------------------------------------
################################################################################################################################################
########## RADIOMETER ##########
start = dt.datetime.now()
logger.info("Expand in space...")
if c.RM_data == True:
    dT_COSMO["RM"] = expand_in_space(
        dT_COSMO["RM"], n_z, n_y, n_x, all_station_names["RM"]
    )
    # plt.contourf(dT_COSMO["RM"]['PAY'][c.index_temp][20,:,:], cmap = cm.Spectral_r) # index_hum
    # plt.colorbar()
    meas_data = calculate_new_meas(meas_data, "RM", dT_COSMO, all_station_names["RM"])

########## LIDAR ##########
start = dt.datetime.now()
logger.info("Expand in space...")
if c.RA_data == True:
    dT_COSMO["RA"] = expand_in_space(
        dT_COSMO["RA"], n_z, n_y, n_x, all_station_names["RA"]
    )
    # plt.contourf(dT_COSMO["RA"]['PAY'][c.index_temp][20,:,:], cmap = cm.Spectral_r) # index_hum
    # plt.colorbar()
    meas_data = calculate_new_meas(meas_data, "RA", dT_COSMO, all_station_names["RA"])
    
########## NUCAPS ##########e
if c.NUCAPS_data == True:
    meas_data["NUCAPS"] = expand_in_space(
        meas_data["NUCAPS"], n_z, n_y, n_x, all_station_names["NUCAPS"]
    )
    # plt.contourf(NUCAPS_data[118][c.index_temp][:,340,:], levels = np.arange(-70,20,1)) # index_hum
    # plt.colorbar()
end = dt.datetime.now()
logger.info(end - start)

################################################################################################################################################
# ----------------------------------------------------------- UNCERTAINTY ESTIMATION ------------------------------------------------------------
################################################################################################################################################
start = dt.datetime.now()
logger.info("Calculate uncertainty...")
########## COSMO ##############
## uncertainty measurement device
COSMO_total_std = read_std_COSMO(
    c.variables_std_COSMO.number[c.variables_std_COSMO.name == "temperature"].iloc[0],
    c.variables_std_COSMO.number[c.variables_std_COSMO.name == "humidity"].iloc[0],
    hours_diff,
    c.RM_station_nr,
    INCA_grid_indexes,
)
# plt.plot(COSMO_total_std[c.index_temp][:,341,:], INCA_grid_indexes['RM']['PAY'][c.index_INCA_grid],  label = 'combined', linewidth = 1, zorder = 0) # index_hum
# plt.contourf(COSMO_total_std[0][:,341,:], cmap = cm.Spectral_r, levels = np.arange(0,2,0.1))
# plt.colorbar()

########## SPACE ##############
## uncertainty with distance
STD_space = [
    calculate_STD_with_distance(n_x, n_x, n_y, n_z, COSMO_data[c.index_temp]),
    calculate_STD_with_distance(n_x, n_x, n_y, n_z, COSMO_data[c.index_hum]),
]
plt.contourf(STD_space[c.index_temp])  # c.index_hum
plt.colorbar()

end = dt.datetime.now()
logger.info(end - start)

#### loop trough stations
station_names = list(set(all_station_names))
for i in range(len(all_station_names)):
    meas_device = station_names[i]
    logger.info("Measurement device:" + str(meas_device))
    for j in range(len(all_station_names[meas_device])):
        start = dt.datetime.now()
        station = all_station_names[meas_device][j]
        logger.info("Station:" + str(station))

        ## uncertainty with distance from measurement station
        meas_std_distance = calculate_uncertainty_dist(
            STD_space, INCA_grid_indexes[meas_device][station], n_x, n_y, station
        )

        ## uncertainty of measurement device
        meas_std = c.factor * pd.read_csv(
            c.dirs_in["dir_" + str(DT) + "_std_" + str(meas_device)], delimiter=";"
        )
        meas_std = [
            expand_in_space_1(meas_std.std_temp.values, n_z, n_y, n_x),
            expand_in_space_1(meas_std.std_temp_d.values, n_z, n_y, n_x),
        ]
        ## total uncertainty
        meas_total_std = total_uncertainty(meas_std_distance, meas_std, station)

        # plt.contourf(meas_total_std[0][20,:,:], levels = np.arange(0,5,0.1), cmap = cm.Spectral_r)
        # plt.colorbar(label = 'std [K]')
        # plt.xlabel('lon [# grid points]')
        # plt.ylabel('lat  [# grid points]')

        del meas_std_distance
        del meas_std
        ################################################################################################################################################
        # ----------------------------------------------------------- DEFINITION OF WEIGHTS & COMBINATION -------------------------------------------------------------
        ################################################################################################################################################
        ## calculation of squared standard deviation
        ########## COSMO ##########
        COSMO_sigma = [
            calculate_sigma(COSMO_total_std[c.index_temp], COSMO_data[c.index_temp]),
            calculate_sigma(COSMO_total_std[c.index_hum], COSMO_data[c.index_hum]),
        ]
        ########## measurement device ##########
        meas_sigma = [
            calculate_sigma(
                meas_total_std[c.index_temp],
                meas_data[meas_device][station][c.index_temp],
            ),
            calculate_sigma(
                meas_total_std[c.index_hum],
                meas_data[meas_device][station][c.index_hum],
            ),
        ]
        ########## total ##########
        STD_total = [
            bn.nansum(
                np.stack((COSMO_sigma[c.index_temp], meas_sigma[c.index_temp])), axis=0
            ),
            bn.nansum(
                np.stack((COSMO_sigma[c.index_hum], meas_sigma[c.index_hum])), axis=0
            ),
        ]

        ## calculation of weights
        ########## COSMO ##########
        # a_COSMO = [
        #    (meas_sigma[c.index_temp] / STD_total[c.index_temp]),
        #    (meas_sigma[c.index_hum] / STD_total[c.index_hum]),
        # ]
        # a_COSMO[c.index_temp][np.isnan(meas_sigma[c.index_temp])] = 1
        # a_COSMO[c.index_hum][np.isnan(meas_sigma[c.index_hum])] = 1

        ########## measurement device ##########
        a_meas = [
            (COSMO_sigma[c.index_temp] / STD_total[c.index_temp]),
            (COSMO_sigma[c.index_hum] / STD_total[c.index_hum]),
        ]
        a_meas[c.index_temp][a_meas[c.index_temp] == 1] = 0
        a_meas[c.index_temp][np.isnan(a_meas[c.index_temp])] = 0
        a_meas[c.index_hum][np.isnan(meas_sigma[c.index_hum])] = 0

        ## calculation of new standard deviation COSMO (combination of std COSMO and meas device)
        COSMO_total_std = [
            np.sqrt(
                (COSMO_total_std[c.index_temp] ** 2 * meas_total_std[c.index_temp] ** 2)
                / (
                    COSMO_total_std[c.index_temp] ** 2
                    + meas_total_std[c.index_temp] ** 2
                )
            ),
            np.sqrt(
                (COSMO_total_std[c.index_hum] ** 2 * meas_total_std[c.index_hum] ** 2)
                / (COSMO_total_std[c.index_hum] ** 2 + meas_total_std[c.index_hum] ** 2)
            ),
        ]

        #################################################################################################################################################
        # ----------------------------------------------------------- COMBINATION DATASETS -------------------------------------------------------------
        ################################################################################################################################################
        COMBINED = [
            (
                bn.nansum(
                    np.stack(
                        (
                            COSMO_data[c.index_temp],
                            (
                                a_meas[c.index_temp]
                                * dT_COSMO[meas_device][station][c.index_temp]
                            ),
                        )
                    ),
                    axis=0,
                )
            ),
            bn.nansum(
                np.stack(
                    (
                        COSMO_data[c.index_hum],
                        (
                            a_meas[c.index_hum]
                            * dT_COSMO[meas_device][station][c.index_hum]
                        ),
                    )
                ),
                axis=0,
            ),
        ]

        del a_meas
        end = dt.datetime.now()
        logger.info("loop run time :" + str(end - start))
        
################################################################################################################################################
# ----------------------------------------------------------- SAVE NEW DATASET ------------------------------------------------------------
################################################################################################################################################
COSMO_xarray["t_inca"] = (["z", "y", "x"], COMBINED[c.index_temp])
COSMO_xarray["td_inca"] = (["z", "y", "x"], COMBINED[c.index_hum])
COSMO_xarray["qv_inca"] = (
    ["z", "y", "x"],
    metpy.calc.specific_humidity_from_dewpoint(
        COMBINED[1] * units.degC, COSMO_xarray.p_inca.values * units.hPa
    )[::-1][0, :, :, :].magnitude,
)
COSMO_xarray.to_netcdf(
    path=c.dirs_out["dir_combi"]
    + "cosmo_1e_inca_"
    + str(now_time_cosmo.strftime("%Y%m%d%H"))
    + "_0"
    + str(int(hours_diff))
    + "_00.nc"
)