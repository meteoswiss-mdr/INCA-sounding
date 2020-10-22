#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#def loop_over_all_stations(function_name, station_names):
#    output_array = {}
#    for i in range(len(station_names)):
#        output_array[station_names[i]] = function_name
#    return output_array

#def calculate_STD_with_distance(points, n_x, n_y, n_z, data):
#    STD_temp_space=np.zeros((n_z,(points+1)))
#    for j in range(0, (points-1)):
#        for k in range(0, (n_z-1)):
#            std_x = np.sqrt(((data[k,0:(n_y-j),:] - data[k,j:(n_y),:])**2)/2)
#            std_y = np.sqrt(((data[k,:,0:(n_x-j)] - data[k,:,j:(n_x)])**2)/2)
#            STD_temp_space[k,j] = np.mean(0.5 * (std_x[:, j:(n_x)] + std_y[j:(n_y),:]))
#    return STD_temp_space

#def calculate_distance_from_onepoint(n_x, n_y, indexes):
#    distance_array = np.zeros((n_y,n_x))
#    for i in range(n_y):
#        for j in range(n_x):
#            distance_array[i,j] = np.sqrt((i-indexes[0,0])**2 + (j-indexes[1,0])**2)
#    return distance_array

#def calculate_uncertainty_dist(COSMO_data, INCA_grid_indexes, n_x, n_y, station_names):
#    T_COSMO = COSMO_data[0]
#    T_d_COSMO = COSMO_data[1]
#    STD_temp_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_COSMO)
#    STD_temp_d_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_d_COSMO)
#    STD_distance = {}
#    for i in range(len(station_names)):
#        indexes = INCA_grid_indexes[station_names[i]][1]
#        dist = calculate_distance_from_onepoint(n_x, n_y, indexes)
#        STD_distance[station_names[i]] = [std_from_point(STD_temp_space, dist), std_from_point(STD_temp_d_space, dist)]  
#    return STD_distance 

def get_last_cosmo_date(delay):
    now_time = dt.datetime.now().replace(microsecond=0, second=0, minute=0)
    now_time_delay = dt.datetime.now() - dt.timedelta(minutes=delay)
    last_cosmo_forecast = int(now_time_delay.hour/3)*3
    hours_diff = (now_time.hour/3 * 3) - (int(now_time_delay.hour/3) * 3)
    now_time_cosmo = now_time - dt.timedelta(hours=hours_diff)
    return now_time_cosmo, hours_diff

def get_last_10min():
    rounded_minute = int(int(dt.datetime.now().strftime('%M')) / 10) * 10
    return rounded_minute

def find_closest_noon_midnight(now_time):
    if np.abs((12 - now_time.hour)) <= 6:
        DT = 12
    else:
        DT = 0
    return DT

################################################################################################################################################
def read_last_next_cosmo_file():
    now_time_cosmo, hours_diff = get_last_cosmo_date(70)
    try: 
        COSMO_data_last = xr.open_dataset(c.dirs_in['dir_COSMO']+'cosmo-1e_inca_'+now_time_cosmo.strftime('%Y%m%d%H')+'_0'+str(int(hours_diff))+'_00.nc')
        COSMO_data_next = xr.open_dataset(c.dirs_in['dir_COSMO']+'cosmo-1e_inca_'+now_time_cosmo.strftime('%Y%m%d%H')+'_0'+str(int(hours_diff+1))+'_00.nc')
        COSMO_xarray = COSMO_data_last[['t_inca', 'qv_inca', 'p_inca']]
    except FileNotFoundError: 
        now_time_cosmo = now_time_cosmo - dt.timedelta(hours=3)
        COSMO_data_last = COSMO_data = xr.open_dataset(c.dirs_in['dir_COSMO']+'cosmo-1e_inca_'+now_time_cosmo.strftime('%Y%m%d%H')+'_0'+str(int(hours_diff))+'_00.nc')
        COSMO_data_next = COSMO_data = xr.open_dataset(c.dirs_in['dir_COSMO']+'cosmo-1e_inca_'+now_time_cosmo.strftime('%Y%m%d%H')+'_0'+str(int(hours_diff+1))+'_00.nc')
        COSMO_xarray = COSMO_data_last[['t_inca', 'qv_inca', 'p_inca']]
        
    return COSMO_data_last, COSMO_data_next, now_time_cosmo, hours_diff, COSMO_xarray

def read_cosmo_file_future(now_time_cosmo, hours_diff, delta_t):
    hours_diff = hours_diff + 1
    COSMO_data = xr.open_dataset(c.dirs_in['dir_COSMO']+'cosmo-1e_inca_'+now_time_cosmo.strftime('%Y%m%d%H')+'_0'+str(int(hours_diff))+'_00.nc')
    return COSMO_data, now_time_cosmo

def open_NUCAPS_file(NUCAPS_file):       
    ds = xr.open_dataset(NUCAPS_file, decode_times=False)  # time units are non-standard, so we dont decode them here 
    units, reference_date = ds.Time.attrs['units'].split(' since ')
    if units=='msec':
        ref_date = dt.datetime.strptime(reference_date,"%Y-%m-%dT%H:%M:%SZ") # usually '1970-01-01T00:00:00Z'
        ds['datetime'] = [ -1 if np.isnan(t) else ref_date + timedelta(milliseconds=t) for t in ds.Time.data]
    return ds

def read_radiosonde(firstobj, lastobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&dataSourceId=34&verbose=position&delimiter=comma&parameterIds=744,745,746,742,748,743,747&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'-'+str(dt.datetime.strftime(lastobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=22'
    RS_data = pd.read_csv(url, skiprows = [1], sep=',')
    RS_data = RS_data.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent','742':'altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })
    RS_data = RS_data[RS_data['temperature_degC'] != 1e+07]
    RS_data['time_YMDHMS'] = pd.to_datetime(RS_data.time_YMDHMS, format = '%Y%m%d%H%M%S')
    return RS_data

def read_HATPRO(firstobj, lastobj, station_nr):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds='+str(station_nr)+'&delimiter=comma&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'-'+str(dt.datetime.strftime(lastobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=31'

    RM = pd.read_csv(url, skiprows = [1], sep=',') 
    RM = RM.rename(columns = {'termin' : 'time_YMDHMS' , '3147' : 'temperature_K', '3148' : 'absolute_humidity_gm3', 'level' : 'altitude_m'})
    # < temperature >
    RM['temperature_degC'] = RM.temperature_K - 273.15
        
    p_w = ((RM.temperature_K * RM.absolute_humidity_gm3) / 2.16679)
    RM['dew_point_degC'] = metpy.calc.dewpoint((p_w.values * units.Pa))
        
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/integral/wmo_ind?locationIds='+str(station_nr)+'&measCatNr=1&dataSourceId=38&parameterIds=3150,5560,5561&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&delimiter=comma&obsTypeIds=31'
    RM_quality_flag = pd.read_csv(url, skiprows = [1], sep=',')
    RM['quality_flag'] = pd.concat([RM_quality_flag['3150']] * len(RM), ignore_index=True)
    RM['radiometer_quality_flag_temperature_profile'] = pd.concat([RM_quality_flag['5560']] * len(RM), ignore_index=True)
    RM['radiometer_quality_flag_humidity_profile'] = pd.concat([RM_quality_flag['5561']] * len(RM), ignore_index=True)
           
    RM['time_YMDHMS'] = pd.to_datetime(RM.time_YMDHMS, format = '%Y%m%d%H%M%S') 
    RM.loc[RM.quality_flag == 1, 'temperature_degC'] = np.nan
    RM.loc[RM.quality_flag == 1, 'dew_point_degC'] = np.nan                                        
    return RM

def open_available_RM_data(station_nr, now_time):
    station_data = {}
    station_names = []
    for i in range(len(station_nr)):
        try:
            station_data[station_nr[i][c.index_temp]] = read_HATPRO(now_time, now_time, station_nr[i][c.index_hum])
            station_names.append(station_nr[i][c.index_station_name])
        except ValueError:
            pass
    return station_data, station_names
################################################################################################################################################
def extract_INCA_grid_onepoint(INCA_archive, lon_SP, lat_SP):
    INCA_grid = xr.open_dataset(INCA_archive+'inca_topo_levels_hsurf_ccs4.nc')
    lon = INCA_grid.lon_1.values
    lat = INCA_grid.lat_1.values
    lonlat = np.dstack([lat.ravel(), lon.ravel()])[0,:,:]
    tree = spatial.KDTree(lonlat)
    coordinates = tree.query(([lat_SP, lon_SP]))
    coords_close = lonlat[coordinates[1]]
    indexes = np.array(np.where(INCA_grid.lon_1 == coords_close[1]))
    INCA_grid_PAY = pd.DataFrame({'altitude_m' : INCA_grid.HFL[:, indexes[0], indexes[1]][:,0,0].values})[::-1]
    INCA_grid_PAY = INCA_grid_PAY.iloc[:,0].reset_index(drop=True)
    return INCA_grid_PAY, indexes

def extract_INCA_grid_onepoint(INCA_archive, station_nr, station_names):
    INCA_grid_indexes = {}
    for i in range(len(station_names)):
        SP_lon = c.coordinates_RM.Longitude[c.coordinates_RM.station == station_nr[i][c.index_station_name]].iloc[0]
        SP_lat = c.coordinates_RM.Latitude[c.coordinates_RM.station == station_nr[i][c.index_station_name]].iloc[0]
        INCA_grid = xr.open_dataset(INCA_archive+'inca_topo_levels_hsurf_ccs4.nc')
        lon = INCA_grid.lon_1.values
        lat = INCA_grid.lat_1.values
        lonlat = np.dstack([lat.ravel(), lon.ravel()])[0,:,:]
        tree = spatial.KDTree(lonlat)
        coordinates = tree.query(([SP_lat, SP_lon]))
        coords_close = lonlat[coordinates[1]]
        indexes = np.array(np.where(INCA_grid.lon_1 == coords_close[1]))
        INCA_grid = pd.DataFrame({'altitude_m' : INCA_grid.HFL[:, indexes[0], indexes[1]][:,0,0].values})[::-1]
        INCA_grid = INCA_grid.iloc[:,0].reset_index(drop=True)
        INCA_grid_indexes[station_names[i]] = [INCA_grid, indexes]
    return INCA_grid_indexes

def average_to_INCA_grid(INCA_grid_indexes, input_data, station_names):
    """ A simple code to average vertical profil to INCA_grid
    
    Parameters:
        INCA_grid: 1D vertical INCA coordinates
        input_data_time: dataset to be averaged to the INCA grid as a dataframe
        
    Returns: e
        input_grid_smoothed_acalculatell: input_data_time smoothed to INCA grid
        
    """
    smoothed_INCA_grid = {}
    for j in range(len(station_names)):
        INCA_grid = INCA_grid_indexes[station_names[j]][c.index_INCA_grid]
        #INCA_grid = INCA_grid[::-1].reset_index(drop=True)
        input_grid_smoothed_all = pd.DataFrame()  
        input_data_time = input_data[station_names[j]]
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
            if window_h_max < np.min(input_data_time.altitude_m):
                aver_mean = pd.DataFrame({'temperature_mean' : np.nan, 'temperature_d_mean' : np.nan, 'altitude_m' : INCA_grid.loc[i]}, index = [i])
            elif input_data_within_bound.altitude_m.count() == 0:
                aver_mean = pd.DataFrame({'temperature_mean' : griddata(input_data_time.altitude_m.values, input_data_time.temperature_degC.values, INCA_grid.loc[i]), 'temperature_d_mean' : griddata(input_data_time.altitude_m.values, input_data_time.dew_point_degC.values, INCA_grid.loc[i]),'altitude_m' : INCA_grid.loc[i]}, index = [i]).reset_index(drop=True) 
            else: 
                aver_mean = pd.DataFrame({'temperature_mean': np.mean(input_data_within_bound.temperature_degC), 'temperature_d_mean' : np.mean(input_data_within_bound.dew_point_degC), 'altitude_m' : (INCA_grid.iloc[i])}, index = [i])
            input_interp = input_interp.append(aver_mean)
        input_grid_smoothed_all = input_grid_smoothed_all.append(input_interp)
        input_grid_smoothed_all = input_grid_smoothed_all.reset_index(drop=True)
        smoothed_INCA_grid[station_names[j]] = input_grid_smoothed_all
    return smoothed_INCA_grid

def interpolate_to_INCA_grid(INCA_grid_indexes, input_data, station_names):
    smoothed_INCA_grid = {}
    for j in range(len(station_names)):
        INCA_grid = INCA_grid_indexes[station_names[j]][c.index_INCA_grid]
        input_data_time = input_data[station_names[j]]
        smoothed_INCA_grid[station_names[j]] = pd.DataFrame({'temperature_mean' : griddata(input_data_time.altitude_m.values, input_data_time.temperature_degC.values, INCA_grid), 'temperature_d_mean' : griddata(input_data_time.altitude_m.values, input_data_time.dew_point_degC.values, INCA_grid),'altitude_m' : INCA_grid}).reset_index(drop=True)
    return smoothed_INCA_grid
################################################################################################################################################
def expand_in_space(data, n_z, n_y, n_x, station_names):
    RM_data_expanded = {}
    for f in range(len(station_names)):
        data_station = data[station_names[f]]
        data_array_temp = np.zeros((n_z, n_y, n_x))
        data_array_temp_d = np.zeros((n_z, n_y, n_x))
        for i in range(n_y):
            for j in range(n_x):
                data_array_temp[:, i,j] = data_station.temperature_mean.values
                data_array_temp_d[:, i,j] = data_station.temperature_d_mean.values
        RM_data_expanded[station_names[f]] = [data_array_temp, data_array_temp_d]
    return RM_data_expanded

def expand_in_space_1(data, n_z, n_y, n_x):
    data_array = np.zeros((n_z, n_y, n_x))
    for i in range(n_y):
        for j in range(n_x):
            data_array[:, i,j] = data
    return data_array
################################################################################################################################################
def calculate_bias_time(now_time_cosmo):
    hour = (now_time_cosmo + timedelta(hours = hours_diff)).hour
    hour_round = hour - int(12)
    if hour_round <= 0:
        bias_date = (now_time_cosmo + timedelta(hours= hours_diff))- timedelta(hours=hour)
    else: 
        bias_date = (now_time_cosmo + timedelta(hours= hours_diff) - timedelta(hours=int(np.abs(hour_round))))
    DT = bias_date.hour
    return bias_date, DT

def calc_run_bias_window(firstobj, INCA_grid_indexes, window_size_days):
    lastobj_bias = firstobj
    firstobj = firstobj - dt.timedelta(days=window_size_days)
    RM_data_1 = pd.DataFrame()
    RS_data_1 = pd.DataFrame()
    while firstobj != lastobj_bias:
        RS_data = {'PAY' : read_radiosonde(firstobj, firstobj)}
        RS_data = average_to_INCA_grid(INCA_grid_indexes, RS_data, ['PAY'])
            
        RM_data = {'PAY' : read_HATPRO(firstobj, firstobj, '06610')}
        RM_data = average_to_INCA_grid(INCA_grid_indexes, RM_data, ['PAY'])
        
        RS_data_1 = RS_data_1.append(RS_data['PAY'])
        RM_data_1 = RM_data_1.append(RM_data['PAY'])
        firstobj = firstobj + dt.timedelta(days=1)
              
    bias_t = np.subtract(RM_data_1.temperature_mean.reset_index(drop=True), RS_data_1.temperature_mean.reset_index(drop=True))
    bias_t = pd.DataFrame({'diff_temp':bias_t.reset_index(drop=True), 'altitude_m': RS_data_1.altitude_m.reset_index(drop=True)})  
    bias_t = bias_t.astype(float)
    bias_t = bias_t.groupby('altitude_m')['diff_temp'].mean().to_frame(name='mean_all').reset_index(drop=True)  
    bias_t[np.isnan(bias_t)] = 0
    
    bias_t_d = np.subtract(RM_data_1.temperature_d_mean.reset_index(drop=True), RS_data_1.temperature_d_mean.reset_index(drop=True))
    bias_t_d = pd.DataFrame({'diff_temp_d':bias_t_d.reset_index(drop=True), 'altitude_m': RS_data_1.altitude_m.reset_index(drop=True)})  
    bias_t_d = bias_t_d.astype(float)
    bias_t_d = bias_t_d.groupby('altitude_m')['diff_temp_d'].mean().to_frame(name='mean_all').reset_index(drop=True)
    bias_t_d[np.isnan(bias_t_d)] = 0

    return bias_t, bias_t_d

################################################################################################################################################
def attribute_grid_points_to_closest_measurement_point(INCA_archive, data, station_names):
    INCA_grid = xr.open_dataset(INCA_archive+'inca_topo_levels_hsurf_ccs4.nc')
    lon = np.array(data.Longitude.values)
    lat = np.array(data.Latitude.values)
    lonlat = np.dstack([lat.ravel(), lon.ravel()])[0,:,:]
    tree = spatial.KDTree(lonlat)    
    lonlat_grid = np.dstack([INCA_grid.lat_1.values, INCA_grid.lon_1.values])
    indices  = tree.query(lonlat_grid)
    dist_array = lonlat[indices[1]][:,:,0]
    for j in range(len(lat)):
        dist_array[dist_array == lat[j]] = j+1
    dist_array_binary = {}
    for i in range(len(station_names)): 
        dist_array_1 = copy.deepcopy(dist_array)
        dist_array_1[dist_array_1 != i+1] = 0
        dist_array_1[dist_array_1 == i+1] = 1
        dist_array_binary[station_names[i]] = dist_array_1
    return dist_array_binary

def limit_area_to_station(RM_data, dist_array_RM, station_names):
    all_RM_temp_sum = np.zeros(shape=(50,640,710))
    all_RM_temp_d_sum = np.zeros(shape=(50,640,710))
    RM_data_all = {}
    for i in range(len(station_names)):
        all_RM_temp_sum = copy.deepcopy(np.nansum(np.stack((all_RM_temp_sum, (RM_data[station_names[i]][c.index_temp] * dist_array_RM[station_names[i]]))), axis = 0))
        all_RM_temp_d_sum  = copy.deepcopy(np.nansum(np.stack((all_RM_temp_d_sum, (RM_data[station_names[i]][c.index_hum] * dist_array_RM[station_names[i]]))), axis = 0) ) 
    RM_data_all['ALL'] = [all_RM_temp_sum, all_RM_temp_d_sum]
    return RM_data_all

################################################################################################################################################
def read_std_COSMO(variable_temp, variable_temp_d, hours_diff, station_nr, INCA_grid_indexes):
    INCA_grid = INCA_grid_indexes['PAY'][c.index_INCA_grid]
    year_month = c.season.name[c.season.number == now_time.month].iloc[0]+'_'+str(now_time.year-1)
    cosmo_leadtime = int(hours_diff/3) * 3
    COSMO_std = pd.read_csv(c.dirs_in['dir_std_COSMO']+'/COSMO/'+year_month+'/scratch/owm/verify/upper-air/'+year_month+'/COSMO-1/output_all_alps//allscores.dat', ';')  
    COSMO_std['altitude_m'] = metpy.calc.pressure_to_height_std(COSMO_std.plevel.values/100 * units.hPa) * 1000
    COSMO_std = COSMO_std[COSMO_std.scorename == 'SD']
    COSMO_std = COSMO_std[COSMO_std.leadtime == int(cosmo_leadtime)]
    
    COSMO_std_temp = copy.deepcopy(COSMO_std[COSMO_std.varno == variable_temp][0:20])
    COSMO_std_temp = copy.deepcopy(griddata(COSMO_std_temp.altitude_m.values, COSMO_std_temp.scores.values, (INCA_grid)) )   
    COSMO_std_temp = copy.deepcopy(expand_in_space_1(COSMO_std_temp, n_z, n_y, n_x))
    
    COSMO_std_temp_d = copy.deepcopy(COSMO_std[COSMO_std.varno == variable_temp_d][0:20])
    COSMO_std_temp_d = copy.deepcopy(griddata(COSMO_std_temp_d.altitude_m.values, COSMO_std_temp_d.scores.values, (INCA_grid)))    
    COSMO_std_temp_d = copy.deepcopy(expand_in_space_1(COSMO_std_temp_d, n_z, n_y, n_x))
    COSMO_std_all = [COSMO_std_temp, COSMO_std_temp_d]
    return COSMO_std_all

def calculate_sigma(RM_std, T_RM):
   RM_std = copy.deepcopy(RM_std**2)
   RM_std[np.isnan(T_RM)] = np.nan
   return RM_std   

def total_uncertainty(RM_std_distance, RM_std_temp_absolute, RM_std_temp_d_absolute, station_names):
    RM_std_total = {}
    for i in range(len(station_names)):
        RM_std_distance_station = copy.deepcopy(RM_std_distance[station_names[i]])
        RM_std_temp_absolute_station = copy.deepcopy(RM_std_temp_absolute)
        RM_std_total[station_names[i]] = [RM_std_distance_station + RM_std_temp_absolute, RM_std_distance_station + RM_std_temp_d_absolute]
    return RM_std_total

##############################################################################################################
def calculate_STD_with_distance(points, n_x, n_y, n_z, data):
    STD_temp_space=np.zeros((n_z, (points+1)))
    for j in range(0,(points-1)):
        for k in range(0,(n_z-1)):
            if j >= n_y:
                STD_temp_space[k,j] = np.mean(np.sqrt(((data[k,:,0:(n_x-j)] - data[k,:,j:(n_x)])**2)/2))
            else: 
                std_x = np.sqrt(((data[k,0:(n_y-j),:] - data[k,j:(n_y),:])**2)/2)
                num_x = std_x.shape[0] * std_x.shape[1]
                std_x = np.mean(std_x)
                std_y = np.sqrt(((data[k,:,0:(n_x-j)] - data[k,:,j:(n_x)])**2)/2)
                num_y = std_y.shape[0] * std_y.shape[1]
                std_y = np.mean(std_y)
                STD_temp_space[k,j] = (num_x / (num_x + num_y)) * std_x + (num_y / (num_x + num_y)) * std_y
    return STD_temp_space

def calculate_distance_from_onepoint(n_x, n_y, indexes):
    distance_array = np.zeros((n_y,n_x))
    for i in range(n_y):
        for j in range(n_x):
            distance_array[i,j] = np.sqrt((i-indexes[0,0])**2 + (j-indexes[1,0])**2)
    return distance_array

def std_from_point(data, dist):
    STD_temp_space_point = np.zeros((n_z, n_y,n_x))
    #STD_temp_space = data[::-1]
    for i in range(0, n_y):
        for j in range(0, n_x):
            distance = copy.deepcopy(dist[i,j])
            dist_max = copy.deepcopy(np.ceil(distance))
            dist_min = copy.deepcopy(np.floor(distance))
            diff_max = copy.deepcopy(dist_max - distance)
            diff_min = copy.deepcopy(1 - diff_max)
            STD_temp_space_point[:, i, j]  = copy.deepcopy((diff_min / (diff_min + diff_max)  * data[:, int(dist_max)]) + (diff_max / (diff_min + diff_max) * data[:, int(dist_min)]) )
    return STD_temp_space_point

def calculate_uncertainty_dist(COSMO_data, INCA_grid_indexes, n_x, n_y, station_names):
    T_COSMO = COSMO_data[c.index_temp]
    T_d_COSMO = COSMO_data[c.index_hum]
    STD_temp_space = calculate_STD_with_distance(n_x, n_x, n_y, n_z, T_COSMO)
    STD_temp_d_space = calculate_STD_with_distance(n_x, n_x, n_y, n_z, T_d_COSMO)
    STD_distance = {}
    for i in range(len(station_names)):
        indexes = copy.deepcopy(INCA_grid_indexes[station_names[i]][1])
        dist = copy.deepcopy(calculate_distance_from_onepoint(n_x, n_y, indexes))
        STD_distance[station_names[i]] = [std_from_point(STD_temp_space, dist), std_from_point(STD_temp_d_space, dist)]  
    return STD_distance 

def total_uncertainty(RM_std_distance, RM_std, station_names):
    RM_std_total = {}
    for i in range(len(station_names)):
        RM_std_distance_station = copy.deepcopy(RM_std_distance[station_names[i]])
        RM_std_total[station_names[i]] = [RM_std_distance_station[c.index_temp] + RM_std[c.index_temp], RM_std_distance_station[c.index_hum] + RM_std[c.index_hum]]
    return RM_std_total

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
from scipy import spatial
from scipy.interpolate import griddata 
import xarray as xr

logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(levelname)s %(name)s %(message)s', )
logger = logging.getLogger(__name__)

n_z = c.INCA_dimension['z']
n_y = c.INCA_dimension['y']
n_x = c.INCA_dimension['x']

################################################################################################################################################
#----------------------------------------------------------- READ DATA -------------------------------------------------------------------
################################################################################################################################################
logger.info('Read data...')
start = dt.datetime.now()
########## COSMO ##########
COSMO_data_last, COSMO_data_next, now_time_cosmo, hours_diff, COSMO_xarray = read_last_next_cosmo_file()  
COSMO_data_last = [COSMO_data_last.t_inca.values[0,:,:,:][::-1] - 273.15, metpy.calc.dewpoint_from_specific_humidity(COSMO_data_last.qv_inca, COSMO_data_last.t_inca, COSMO_data_last.p_inca)[0,:,:,:][::-1].magnitude]
COSMO_data_next = [COSMO_data_next.t_inca.values[0,:,:,:][::-1] - 273.15, metpy.calc.dewpoint_from_specific_humidity(COSMO_data_next.qv_inca, COSMO_data_next.t_inca, COSMO_data_next.p_inca)[0,:,:,:][::-1].magnitude]

a_COSMO_last = (int(dt.datetime.now().strftime('%M'))/60)
a_COSMO_next = 1 - a_COSMO_last
COSMO_data = [((a_COSMO_last * COSMO_data_last[c.index_temp]) + (a_COSMO_next * COSMO_data_next[c.index_temp])),((a_COSMO_last * COSMO_data_last[c.index_hum]) + (a_COSMO_next * COSMO_data_next[c.index_hum]))]

#plt.contourf(COSMO_data[index_temp][:,341,:], levels = np.arange(-60, 20), cmap = cm.Spectral_r) # index_hum
#plt.contourf(COSMO_data[index_temp][20,:,:], levels = np.arange(0, 20, 0.5), cmap = cm.Spectral_r) # index_hum
#plt.colorbar()

########## RADIOMETER ##########
now_time = now_time_cosmo + timedelta(hours=hours_diff)
DT = find_closest_noon_midnight(now_time)
rounded_minutes = get_last_10min()
RM_data, station_names = open_available_RM_data(c.station_nr, now_time + dt.timedelta(minutes=rounded_minutes))
INCA_grid_indexes = extract_INCA_grid_onepoint(c.dirs_in['dir_INCA'],  c.station_nr, station_names)  
#plt.plot(RM_data['PAY'].temperature_degC, RM_data['PAY'].altitude_m)   
#plt.scatter(RM_data['PAY'].temperature_degC, RM_data['PAY'].altitude_m)
RM_data = interpolate_to_INCA_grid(INCA_grid_indexes,  RM_data, station_names)
#plt.plot(RM_data['PAY'].temperature_mean, RM_data['PAY'].altitude_m) 
#plt.scatter(RM_data['PAY'].temperature_mean, RM_data['PAY'].altitude_m)
end = dt.datetime.now()
logger.info(end-start)
################################################################################################################################################
#----------------------------------------------------------- BIAS CORRECTION -------------------------------------------------------------------
################################################################################################################################################
if c.bias_correction == True: 
    start = dt.datetime.now()
    logger.info('Subtract bias...')
    bias_date, DT = calculate_bias_time(now_time_cosmo) 
    bias_temp, bias_temp_d = calc_run_bias_window(bias_date, INCA_grid_indexes, c.bias_window_size)
    #plt.plot(bias_temp, INCA_grid_indexes['PAY'][c.index_temp]) # index_hum
    
    RM_data['PAY'].temperature_mean = np.subtract(RM_data['PAY'].temperature_mean.values,bias_temp.iloc[:,0].values)
    RM_data['PAY'].temperature_d_mean = np.subtract(RM_data['PAY'].temperature_d_mean,-bias_temp_d.iloc[:,0].values)
    #plt.plot(RM_data['PAY'].temperature_mean, RM_data['PAY'].altitude_m)
    end = dt.datetime.now()
    logger.info(end-start)
################################################################################################################################################
#----------------------------------------------------------- EXPAND IN SPACE -------------------------------------------------------------------
################################################################################################################################################
start = dt.datetime.now()
logger.info('Expand in space...')
RM_data = expand_in_space(RM_data, n_z, n_y, n_x, station_names)
#plt.contourf(RM_data['PAY'][index_temp][:,340,:], levels = np.arange(-60,20,1)) # index_hum
#plt.plot(RM_data['PAY'][index_temp][:,341,:], INCA_grid_indexes['PAY'][c.index_INCA], color = 'black') # index_hum
#plt.colorbar()
end = dt.datetime.now()
logger.info(end-start)
################################################################################################################################################
#----------------------------------------------------------- UNCERTAINTY ESTIMATION ------------------------------------------------------------
################################################################################################################################################
start = dt.datetime.now()
logger.info('Calculate uncertainty...')
########## COSMO ############## 
## uncertainty measurement device   
#COSMO_total_std = read_std_COSMO(c.variables_std_COSMO.number[c.variables_std_COSMO.name == 'temperature'].iloc[0], c.variables_std_COSMO.number[c.variables_std_COSMO.name == 'humidity'].iloc[0], hours_diff,  c.station_nr,  INCA_grid_indexes)   

#fig, ax = plt.subplots(figsize = (6, 12))
#ax.plot(COSMO_total_std[index_temp][:,341,:], INCA_grid_indexes['PAY'][c.index_INCA_point], color = 'red', label = 'combined', linewidth = 5, zorder = 0) # index_hum
#ax.set_xlim(0,4)

########## RADIOMETER ##########
## uncertainty with distance
RM_std_distance = calculate_uncertainty_dist(COSMO_data, INCA_grid_indexes, n_x, n_y, station_names)
plt.contourf(RM_std_distance['GRE'][1][20, :, :], cmap = cm.Spectral_r, levels = np.arange(0,5,0.1))
plt.contourf(RM_std_distance['GRE'][1][:, 433, :], cmap = cm.Spectral_r, levels = np.arange(0,6,0.1))
plt.colorbar()
## uncertainty measurement device
RM_std= c.factor * pd.read_csv(c.dirs_in['dir_'+str(DT)+'_std_RM'], delimiter=';')
RM_std = [expand_in_space_1(RM_std.std_temp.values, n_z, n_y, n_x), expand_in_space_1(RM_std.std_temp_d.values, n_z, n_y, n_x)]  
#plt.contourf(RM_std[c.index_temp][:, 341,:], cmap = cm.Spectral_r, levels = np.arange(0,18))    # c.index_hum
#plt.colorbar()         
## total uncertainty
RM_total_std = total_uncertainty(RM_std_distance, RM_std, station_names)
#plt.contourf(RM_total_std['PAY'][c.index_temp][:,340,:], levels = np.arange(0,6, 0.2), cmap = cm.Spectral_r) # c.index_hum
#plt.colorbar()
end = dt.datetime.now()
logger.info(end-start)
################################################################################################################################################
#----------------------------------------- CONCAT MEASUREMENTS OF DIFFERENT STATIONS TO ONE LAYER ----------------------------------------------
################################################################################################################################################
start = dt.datetime.now()
logger.info('Concat to one layer...')
########## RADIOMETER ##########
RM_dist_array_binary = attribute_grid_points_to_closest_measurement_point(c.dirs_in['dir_INCA'], c.coordinates_RM[c.coordinates_RM.station.isin(station_names)], station_names)
plt.contourf(RM_dist_array_binary['PAY'])
plt.colorbar()
RM_data = limit_area_to_station(RM_data, RM_dist_array_binary, station_names)
#plt.contourf(RM_data['ALL'][c.index_temp][:,341,:], levels = np.arange(-60,20)) # c.index_hum
#plt.colorbar()
#plt.contourf(RM_data['ALL'][c.index_temp][25,:,:]) # c.index_hum
#plt.colorbar()
RM_total_std = limit_area_to_station(RM_total_std, RM_dist_array_binary, station_names)

fig, ax = plt.subplots(figsize = (12, 12))
ax.contourf(RM_total_std['ALL'][0][20,:,:])

#plt.contourf(RM_total_std['ALL'][c.index_temp][:,341,:], levels = np.arange(0,10)) # c.index_hum
#plt.contourf(RM_total_std['ALL'][c.index_temp][10,:,:], levels = np.arange(0,10)) # c.index_hum
#plt.colorbar()
end = dt.datetime.now()
logger.info(end-start)


fig, ax = plt.subplots(figsize = (12, 12))
#ax.contourf(x[:,306,:])
#ax.contourf(y[:,306,:])
ax.contourf(x[:,306,:] + y[:,306,:])
ax.scatter(x = INCA_grid_indexes['PAY'][1][1], y = 3, color = 'red', s = 100)
ax.scatter(x = INCA_grid_indexes['GRE'][1][1], y = 3, color = 'red', s = 100)

fig, ax = plt.subplots(figsize = (12, 12))
ax.contourf(RM_total_std['ALL'][0][20,:,:])

fig, ax = plt.subplots(figsize = (12, 12))
ax.contourf(RM_dist_array_binary['GRE'])
#ax.contourf(RM_data_all['ALL'][0][20,:,:])
ax.scatter(x = INCA_grid_indexes['PAY'][1][1], y = INCA_grid_indexes['PAY'][1][0], color = 'red', s = 100)
ax.scatter(x = INCA_grid_indexes['GRE'][1][1], y = INCA_grid_indexes['GRE'][1][0], color = 'red', s = 100)




################################################################################################################################################
def attribute_grid_points_to_closest_measurement_point(INCA_archive, data, station_names):
    INCA_grid = xr.open_dataset(INCA_archive+'inca_topo_levels_hsurf_ccs4.nc')
    lon = np.array(data.Longitude.values)
    lat = np.array(data.Latitude.values)
    lonlat = np.dstack([lat.ravel(), lon.ravel()])[0,:,:]
    tree = spatial.KDTree(lonlat)    
    lonlat_grid = np.dstack([INCA_grid.lat_1.values, INCA_grid.lon_1.values])
    indices  = tree.query(lonlat_grid)
    dist_array = lonlat[indices[1]][:,:,0]
    for j in range(len(lat)):
        dist_array[dist_array == lat[j]] = j+1
    dist_array_binary = {}
    for i in range(len(station_names)): 
        dist_array_1 = copy.deepcopy(dist_array)
        dist_array_1[dist_array_1 != i+1] = 0
        dist_array_1[dist_array_1 == i+1] = 1
        dist_array_binary[station_names[i]] = dist_array_1
    return dist_array_binary

def limit_area_to_station(RM_data, dist_array_RM, station_names):
    all_RM_temp_sum = np.zeros(shape=(50,640,710))
    all_RM_temp_d_sum = np.zeros(shape=(50,640,710))
    RM_data_all = {}
    for i in range(len(station_names)):
        all_RM_temp_sum = copy.deepcopy(np.nansum(np.stack((all_RM_temp_sum, (RM_data[station_names[i]][c.index_temp] * dist_array_RM[station_names[i]]))), axis = 0))
        all_RM_temp_d_sum  = copy.deepcopy(np.nansum(np.stack((all_RM_temp_d_sum, (RM_data[station_names[i]][c.index_hum] * dist_array_RM[station_names[i]]))), axis = 0) ) 
    RM_data_all['ALL'] = [all_RM_temp_sum, all_RM_temp_d_sum]
    return RM_data_all



################################################################################################################################################
#----------------------------------------------------------- DEFINITION OF WEIGHTS -------------------------------------------------------------
################################################################################################################################################
start = dt.datetime.now()
logger.info('Calculation of weights...')
RM_sigma = [calculate_sigma(RM_total_std['ALL'][c.index_temp], RM_data['ALL'][c.index_temp]), calculate_sigma(RM_total_std['ALL'][c.index_hum], RM_data['ALL'][c.index_hum])]
COSMO_sigma = [calculate_sigma(COSMO_total_std[c.index_temp], COSMO_data[c.index_temp]), calculate_sigma(COSMO_total_std[c.index_hum], COSMO_data[c.index_hum])]
STD_total = [np.nansum(np.stack((COSMO_sigma[c.index_temp], RM_sigma[c.index_temp])), axis = 0), np.nansum(np.stack((COSMO_sigma[c.index_hum], RM_sigma[c.index_hum])), axis = 0)]  
########## COSMO ############## 
a_COSMO = [(RM_sigma[c.index_temp] / STD_total[c.index_temp]), (RM_sigma[c.index_hum] / STD_total[c.index_hum])]
a_COSMO[c.index_temp][np.isnan(RM_sigma[c.index_temp])] = 1 
a_COSMO[c.index_hum][np.isnan(RM_sigma[c.index_hum])] = 1
#plt.contourf(a_COSMO[c.index_temp][:,341,:], levels = np.arange(0,1.1,0.1),cmap = cm.Spectral_r) # c.index_hum
#plt.contourf(a_COSMO[c.index_temp][20,:,:], levels = np.arange(0,1.1,0.1),cmap = cm.Spectral_r) # c.index_hum
#plt.colorbar()
########## DATASET ##########
a_RM = [(COSMO_sigma[c.index_temp] / STD_total[c.index_temp]), (COSMO_sigma[c.index_hum] / STD_total[c.index_hum])]
a_RM[c.index_temp][np.isnan(RM_sigma[c.index_temp])] = 0
a_RM[c.index_hum][np.isnan(RM_sigma[c.index_hum])] = 0
a_RM[c.index_temp][a_RM[c.index_temp]==1] =0
#plt.contourf(a_RM[c.index_temp][:,341,:], levels = np.arange(0,1.1,0.1),cmap = cm.Spectral_r) # c.index_hum
#plt.contourf(a_RM[c.index_temp][20,:,:], levels = np.arange(0,1.1,0.1),cmap = cm.Spectral_r) 
#plt.colorbar()
end = dt.datetime.now()
logger.info(end-start)

#plt.contourf(a_RM[c.index_temp][:,341,:] + a_COSMO[0][:,341,:], levels = np.arange(0,1.1,0.1),cmap = cm.Spectral_r) # c.index_hum
#plt.colorbar()

################################################################################################################################################
#----------------------------------------------------------- COMBINATION DATASETS -------------------------------------------------------------
################################################################################################################################################
start = dt.datetime.now()
logger.info('Combination of datasets...')
COMBINED = [(np.nansum(np.stack(((a_COSMO[c.index_temp] * COSMO_data[c.index_temp]) ,  (a_RM[c.index_temp] * RM_data['ALL'][c.index_temp]))), axis = 0)), (np.nansum(np.stack(((a_COSMO[c.index_hum] * COSMO_data[c.index_hum]) ,  (a_RM[c.index_hum] * RM_data['ALL'][c.index_hum]))), axis = 0))]
plt.contourf(COMBINED[c.index_hum][20,:,:], levels = np.arange(-10, 20, 0.5), cmap = cm.Spectral_r)
plt.colorbar()
COSMO_xarray['t_inca'] = (['z', 'y', 'x' ], COMBINED[c.index_temp])
COSMO_xarray['td_inca'] =  (['z', 'y', 'x' ], COMBINED[1])
COSMO_xarray['qv_inca'] =  (['z', 'y', 'x' ], metpy.calc.specific_humidity_from_dewpoint(COMBINED[1] * units.degC, COSMO_xarray.p_inca.values * units.hPa)[::-1][0,:,:,:].magnitude)
COSMO_xarray.to_netcdf(path = c.dirs_out['dir_combi'] + 'cosmo_1e_inca_'+str(now_time_cosmo.strftime('%Y%m%d%H'))+'_0'+str(int(hours_diff))+'_00.nc')
end = dt.datetime.now()
logger.info(end-start)

#plt.contourf(np.nansum((COMBINED[c.index_temp], -COSMO_data[c.index_temp]), axis = 0)[20,:,:], cmap = cm.Spectral_r) # c.index_hum
#plt.colorbar()

#plt.contourf(a_COSMO[c.index_temp][20,:,:], cmap = cm.Spectral_r) # c.index_hum
#plt.colorbar()

#plt.contourf(a_RM[c.index_temp][20,:,:], cmap = cm.Spectral_r) # c.index_hum
#plt.colorbar()

################################################################################################################################################
#----------------------------------------------------------- NOWCASTING -------------------------------------------------------------
################################################################################################################################################
logger.info('Nowcast...')
start = datetime.now()
TAU = 6 * 60
FITC = 0
step=np.arange(0,360,10)

# calculate weights 
gew = np.zeros(len(step))
for i in range(len(step)):
    gew[i] = np.exp(-(step[i] - FITC) / TAU)
plt.plot(step, gew)

# calculate increment
increment = [(COSMO_data[c.index_temp] - COMBINED[c.index_temp]) , (COSMO_data[c.index_hum] - COSMO_data[c.index_hum])]
plt.contourf(increment[0][20,:,:])
plt.colorbar()

# calculate COSMO
a_step_next = np.zeros(len(step))
for i in range(len(a_step_next)):    
    a_step_next[i] = ((step[i] - (int(step[i] / 60) * 60))/60)
a_step_last = 1- a_step_next
plt.plot(step, a_step_last)
plt.plot(step, a_step_next)

COSMO_data_future = {}
for i in range(0,7):
    COSMO_data =  xr.open_dataset(c.dirs_in['dir_COSMO']+'cosmo-1e_inca_'+now_time_cosmo.strftime('%Y%m%d%H')+'_0'+str(int(hours_diff+i))+'_00.nc')[['t_inca', 'qv_inca', 'p_inca']]
    COSMO_data_future[i] = [COSMO_data.t_inca.values[0,:,:,:][::-1] - 273.15, metpy.calc.dewpoint_from_specific_humidity(COSMO_data.qv_inca, COSMO_data.t_inca, COSMO_data.p_inca)[0,:,:,:][::-1].magnitude]
    #plt.contourf(COSMO_data_future[i][0][20,:,:], cmap = cm.Spectral_r, levels = np.arange(-5,25))
    #plt.colorbar()
    #plt.show()
 
COSMO_data_future_all = {}
f = 0
for i in range(0,6):
    for j in range(0,6):
        COSMO_data_future_all[f] = [COSMO_data_future[i+1][0] * a_step_next[f] + COSMO_data_future[i][0] * a_step_last[f] , COSMO_data_future[i+1][1] * a_step_next[f] + COSMO_data_future[i][1] * a_step_last[f]]
        #fig, ax = plt.subplots(figsize = (15,12))
        #im = ax.contourf(COSMO_data_future_all[f][0][20,:,:], cmap = cm.Spectral_r, levels = np.arange(-5,25))
        #fig.colorbar(im)
        #fig.savefig('/data/COALITION2/internships/nal/videos/fig_'+str(f)+'.png', bbox_inches='tight')
        f = f+1
  
# calculate Nowcast
COMBINED = {}
for i in range(len(step)):
    COMBINED[i] = [gew[i]*(COSMO_data_future_all[i][c.index_temp] + increment[0])+(1-gew[i])*COSMO_data_future_all[i][c.index_temp] , gew[i]*(COSMO_data_future_all[i][c.index_hum] + increment[c.index_hum])+(1-gew[i])*COSMO_data_future_all[i][c.index_hum]]
    #fig, ax = plt.subplots(figsize = (15,12))
    #im = ax.contourf(COSMO_data_future_all[i][0][20,:,:], cmap = cm.Spectral_r, levels = np.arange(-5,25))
    #fig.colorbar(im)
    #fig.savefig('/data/COALITION2/internships/nal/videos/fig_combined_'+str(i)+'.png', bbox_inches='tight')

end = datetime.now()
logger.info(end-start)
