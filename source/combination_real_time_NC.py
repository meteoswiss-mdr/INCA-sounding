#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 08:52:50 2020
r
@author: nal
"""
def average_to_INCA_grid(INCA_grid_indexes, input_data, station_names):
    """ A simple code to average vertical profil to INCA_grid
    
    Parameters:
        INCA_grid: 1D vertical INCA coordinates
        input_data_time: dataset to be averaged to the INCA grid as a dataframe
        
    Returns: 
        input_grid_smoothed_acalculatell: input_data_time smoothed to INCA grid
        
    """
    smoothed_INCA_grid = {}
    for j in range(len(station_names)):
        INCA_grid = INCA_grid_indexes[station_names[j]][0]
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
            if window_h_min < np.min(input_data_time.altitude_m):
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

    
def loop_over_all_stations(function_name, station_names):
    output_array = {}
    for i in range(len(station_names)):
        output_array[station_names[i]] = function_name
    return output_array

def read_radiosonde(firstobj, lastobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&dataSourceId=34&verbose=position&delimiter=comma&parameterIds=744,745,746,742,748,743,747&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'-'+str(dt.datetime.strftime(lastobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=22'
    RS_data = pd.read_csv(url, skiprows = [1], sep=',')
    RS_data = RS_data.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent', '742':'altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })
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
    RM['temperature_degC'][RM.quality_flag == 1] = np.nan
    RM['dew_point_degC'][RM.quality_flag == 1] = np.nan                                        
    return RM



def calculate_STD_with_distance(points, n_x, n_y, n_z, data):
    STD_temp_space=np.zeros((n_z,(points+1)))
    for j in range(0, (points-1)):
        for k in range(0, (n_z-1)):
            std_x = np.sqrt(((data[k,0:(n_y-j),:] - data[k,j:(n_y),:])**2)/2)
            std_y = np.sqrt(((data[k,:,0:(n_x-j)] - data[k,:,j:(n_x)])**2)/2)
            STD_temp_space[k,j] = np.mean(0.5 * (std_x[:, j:(n_x)] + std_y[j:(n_y),:]))
    return STD_temp_space

def calculate_distance_from_onepoint(n_x, n_y, indexes):
    distance_array = np.zeros((n_y,n_x))
    for i in range(n_y):
        for j in range(n_x):
            distance_array[i,j] = np.sqrt((i-indexes[0,0])**2 + (j-indexes[1,0])**2)
    return distance_array

def std_from_point(data, distance_array):
    STD_temp_space_point = np.zeros((n_z, n_y,n_x))
    STD_temp_space = data[::-1]
    for i in range(0, n_y):
        for j in range(0, n_x):
            dist = distance_array[i,j]
            dist_max = np.ceil(dist)
            dist_min = np.floor(dist)
            diff_max = dist_max - dist
            diff_min = 1 - diff_max
            if (dist_max >= 345) or (dist_min >= 345):
                STD_temp_space_point[:, i, j] = np.full((50,), np.nan)
            else: 
                data_1 = (diff_min / (diff_min + diff_max)  * data[:, int(dist_max)]) + (diff_max / (diff_min + diff_max) * data[:, int(dist_min)]) 
                STD_temp_space_point[:, i, j] = data_1
    return STD_temp_space_point

def plot_profile(T_COMBINED, T_COSMO, T_RM, indexes, INCA_grid): 
    fig, ax = plt.subplots(figsize = (5, 12))
    ax.plot(T_COMBINED[:,indexes[0,0],indexes[1,0]], INCA_grid, color = 'red', label = 'combined', linewidth = 5, zorder = 0)
    ax.plot(T_COSMO[:,indexes[0,0],indexes[1,0]], INCA_grid, color = 'green', label = 'COSMO', linewidth = 3)
    ax.plot(T_RM[:, indexes[0,0], indexes[1,0]], INCA_grid, color = 'navy', label = 'HATPRO', linewidth = 3)
    #ax.plot(T_NUCAPS[:, indexes[0,0], indexes[1,0]], INCA_grid, color = 'purple', label = 'NUCPAS', linewidth = 3)
    ax.legend(fontsize = 20)
    plt.xticks(np.arange(-50, 30, 20), fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.set_xlabel('Temperature [K]', fontsize = 20)
    ax.set_ylabel('Altitude [m]', fontsize = 20)

    ax.legend(fontsize = 20)
    
def calc_run_bias_window(firstobj, INCA_grid_indexes, window_size_days):
    lastobj_bias = firstobj
    firstobj = firstobj - dt.timedelta(days=window_size_days)
    RS_data = {'PAY' : read_radiosonde(firstobj, lastobj_bias)}
    RS_data = average_to_INCA_grid(INCA_grid_indexes, RS_data, ['PAY'])
        
    RM_data = {'PAY' : read_HATPRO(firstobj, lastobj_bias, '06610')}
    RM_data = average_to_INCA_grid(INCA_grid_indexes, RM_data, ['PAY'])
          
    bias_t = np.subtract(RM_data['PAY'].temperature_mean.reset_index(drop=True), RS_data['PAY'].temperature_mean.reset_index(drop=True))
    bias_t = pd.DataFrame({'diff_temp':bias_t.reset_index(drop=True), 'altitude_m':RS_data['PAY'].altitude_m.reset_index(drop=True)})  
    bias_t = bias_t.astype(float)
    bias_t = bias_t.groupby('altitude_m')['diff_temp'].mean().to_frame(name='mean_all').reset_index(drop=True)
            
    bias_t_d = np.subtract(RM_data['PAY'].temperature_mean.reset_index(drop=True), RS_data['PAY'].temperature_mean.reset_index(drop=True))
    bias_t_d = pd.DataFrame({'diff_temp_d':bias_t_d.reset_index(drop=True), 'altitude_m': RS_data['PAY'].altitude_m.reset_index(drop=True)})  
    bias_t_d = bias_t_d.astype(float)
    bias_t_d = bias_t_d.groupby('altitude_m')['diff_temp_d'].mean().to_frame(name='mean_all').reset_index(drop=True)
            
    return bias_t, bias_t_d

def get_last_cosmo_date(delay):
    now_time = dt.datetime.now().replace(microsecond=0, second=0, minute=0)
    now_time_delay = dt.datetime.now() - dt.timedelta(minutes=delay)
    last_cosmo_forecast = int(now_time_delay.hour/3)*3
    hours_diff = (now_time.hour/3 * 3) - (int(now_time_delay.hour/3) * 3)
    now_time_cosmo = now_time - dt.timedelta(hours=hours_diff)
    return now_time_cosmo, hours_diff

def get_last_10min():
    rounded_minute = int(int(datetime.now().strftime('%M')) / 10) * 10
    return rounded_minute

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

def calculate_uncertainty_dist(COSMO_data, INCA_grid_indexes, n_x, n_y, station_names):
    T_COSMO = COSMO_data[0]
    T_d_COSMO = COSMO_data[1]
    STD_temp_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_COSMO)
    STD_temp_d_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_d_COSMO)
    for i in range(len(station_names)):
        indexes = INCA_grid_idnexes[station_names[i]][1]
        dist = calculate_distance_from_onepoint(n_x, n_y, indexes)
        STD_distance = [std_from_point(STD_temp_space, dist), std_from_point(STD_temp_space, dist)]  
    return STD_distance 

def open_NUCAPS_file(NUCAPS_file):       
    ds = xr.open_dataset(NUCAPS_file, decode_times=False)  # time units are non-standard, so we dont decode them here 
    units, reference_date = ds.Time.attrs['units'].split(' since ')
    if units=='msec':
        ref_date = dt.datetime.strptime(reference_date,"%Y-%m-%dT%H:%M:%SZ") # usually '1970-01-01T00:00:00Z'
        ds['datetime'] = [ -1 if np.isnan(t) else ref_date + timedelta(milliseconds=t) for t in ds.Time.data]
    return ds

def calculate_bias_time(now_time_cosmo):
    hour = (now_time_cosmo + timedelta(hours = hours_diff)).hour
    hour_round = hour - int(12)
    if hour_round <= 0:
        bias_date = (now_time_cosmo + timedelta(hours= hours_diff))- timedelta(hours=hour)
    else: 
        bias_date = (now_time_cosmo + timedelta(hours= hours_diff) - timedelta(hours=int(np.abs(hour_round))))
    DT = bias_date.hour
    return bias_date, DT

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
    



def read_std_COSMO(variable_temp, variable_temp_d, hours_diff, station_nr, INCA_grid_indexes):
    INCA_grid = INCA_grid_indexes['PAY'][0]
    year_month = c.season.name[c.season.number == now_time.month].iloc[0]+'_'+str(now_time.year-1)
    cosmo_leadtime = int(hours_diff/3) * 3
    COSMO_std = pd.read_csv(c.dirs_in['dir_std']+'/COSMO/'+year_month+'/scratch/owm/verify/upper-air/'+year_month+'/COSMO-1/output_all_all//allscores.dat', ';')  
    COSMO_std['altitude_m'] = metpy.calc.pressure_to_height_std(COSMO_std.plevel.values/100 * units.hPa) * 1000
    COSMO_std = COSMO_std[COSMO_std.scorename == 'SD']
    COSMO_std = COSMO_std[COSMO_std.leadtime == int(cosmo_leadtime)]
    
    COSMO_std_temp = COSMO_std[COSMO_std.varno == variable_temp][0:20]
    COSMO_std_temp = griddata(COSMO_std_temp.altitude_m.values, COSMO_std_temp.scores.values, (INCA_grid))    
    COSMO_std_temp = expand_in_space_1(COSMO_std_temp, n_z, n_y, n_x)
    
    COSMO_std_temp_d = COSMO_std[COSMO_std.varno == variable_temp_d][0:20]
    COSMO_std_temp_d = griddata(COSMO_std_temp_d.altitude_m.values, COSMO_std_temp_d.scores.values, (INCA_grid))    
    COSMO_std_temp_d = expand_in_space_1(COSMO_std_temp_d, n_z, n_y, n_x)
    COSMO_std = [COSMO_std_temp, COSMO_std_temp_d]
    return COSMO_std

def open_available_RM_data(station_nr, now_time):
    station_data = {}
    station_names = []
    for i in range(len(station_nr)):
        try:
            station_data[station_nr[i][0]] = read_HATPRO(now_time, now_time, station_nr[i][1])
            station_names.append(station_nr[i][0])
        except ValueError:
            pass
    return station_data, station_names

def calculate_uncertainty_dist(T_COSMO, T_d_COSMO, indexes, n_x, n_y, station_names):
    uncertainty_dist = {}
    STD_temp_space = calilculate_STD_with_distance(345, n_x, n_y, n_z, T_COSMO)
    end = datetime.now()
    STD_temp_d_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_d_COSMO)
    for i in range(len(station_names)):
        dist = calculate_distance_from_onepoint(n_x, n_y, indexes[station_names[i]][1]) 
        STD_temp_space_point= std_from_point(STD_temp_space, dist)
        STD_temp_d_space_point = std_from_point(STD_temp_d_space, dist)  
        uncertainty_dist[station_names[i]] = [STD_temp_space_point, STD_temp_d_space_point]
    return uncertainty_dist

def total_uncertainty(RM_std_distance, RM_std_temp_absolute, RM_std_temp_d_absolute, station_names):
    RM_std_total = {}
    for i in range(len(station_names)):
        RM_std_distance_station = RM_std_distance[station_names[i]]
        RM_std_temp_absolute_station = RM_std_temp_absolute
        RM_std_total[station_names[i]] = [RM_std_distance + RM_std_temp_absolute, RM_std_distance + RM_std_temp_d_absolute]
    return RM_std_total

def find_closest_noon_midnight(now_time):
    if np.abs((12 - now_time.hour)) <= 6:
        DT = 12
    else:
        DT = 0
    return DT

def calculate_sigma(RM_std, T_RM):
   RM_std[np.isnan(T_RM)] = np.nan
   RM_std = RM_std**2
   return RM_std            

def extract_INCA_grid_onepoint(INCA_archive, station_nr, station_names):
    INCA_grid_indexes = {}
    for i in range(len(station_names)):
        SP_lon = c.coordinates_RM.Longitude[c.coordinates_RM.station == station_nr[i][0]].iloc[0]
        SP_lat = c.coordinates_RM.Latitude[c.coordinates_RM.station == station_nr[i][0]].iloc[0]
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

def limit_area_to_station(RM_data, dist_array_RM, station_names):
    all_RM_temp_sum = np.zeros(shape=(50,640,710))
    all_RM_temp_d_sum = np.zeros(shape=(50,640,710))
    for i in range(len(station_names)):
        all_RM_temp_sum  = all_RM_temp_sum + (RM_data[station_names[i]][0] * dist_array_RM[station_names[0]])
        all_RM_temp_d_sum = all_RM_temp_d_sum + (RM_data[station_names[i]][1] * dist_array_RM[station_names[0]])    
    RM_data_all['ALL'] = [all_RM_temp_sum, all_RM_temp_d_sum]
    return RM_data_all

def total_uncertainty(RM_std_distance, RM_std, station_names):
    RM_std_total = {}
    for i in range(len(station_names)):
        RM_std_distance_station = copy.deepcopy(RM_std_distance[station_names[i]])
        RM_std_total[station_names[i]] = [RM_std_distance_station[0] + RM_std[0], RM_std_distance_station[1] + RM_std[1]]
    return RM_std_total

def calculate_sigma(RM_data, RM_std):
   RM_std[np.isnan(RM_data)] = np.nan
   RM_std = RM_std**2
   return RM_std

def plot_in_latlon_dir(data, levels, xlabel, ylabel, cbarlabel, cmap):
    fig, ax = plt.subplots(figsize = (12, 12))
    im = ax.contourf(np.arange(0,710),np.arange(0,640), data, cmap =  cmap, levels = levels) # in y direction
    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
    cbar.set_label(label=cbarlabel, size = 20)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_xticklabels([np.arange(0,4.1,0.2)])
    ax.scatter(indexes[1,0], indexes[0,0] , color = 'black')
    
def plot_in_lat_dir(data, levels, xlabel, ylabel, cbarlabel, cmap, points):
    fig, ax = plt.subplots(figsize = (12, 12))
    im = ax.contourf(np.arange(0,points), INCA_grid_indexes[station_names[0]][0], data, cmap =  cmap, levels = levels) # in y direction
    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im,  cax = cax, orientation= 'vertical')
    cbar.set_label(label=cbarlabel, size = 20)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_xticklabels([np.arange(0,4.1,0.2)])
    ax.axvline(indexes[0,0], color = 'black')

def calculate_uncertainty_dist(COSMO_data, INCA_grid_indexes, n_x, n_y, station_names):
    T_COSMO = COSMO_data[0]
    T_d_COSMO = COSMO_data[1]
    STD_temp_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_COSMO)
    STD_temp_d_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_d_COSMO)
    for i in range(len(station_names)):
        indexes = INCA_grid_indexes[station_names[i]][1]
        dist = calculate_distance_from_onepoint(n_x, n_y, indexes)
        STD_distance = [std_from_point(STD_temp_space, dist), std_from_point(STD_temp_space, dist)]  
    return STD_distance 

def calculate_distance_from_onepoint(n_x, n_y, indexes):
    distance_array = np.zeros((n_y,n_x))
    for i in range(n_y):
        for j in range(n_x):
            distance_array[i,j] = np.sqrt((i-indexes[0,0])**2 + (j-indexes[1,0])**2)
    return distance_array

def calculate_uncertainty_dist(COSMO_data, INCA_grid_indexes, n_x, n_y, station_names):
    T_COSMO = COSMO_data[0]
    T_d_COSMO = COSMO_data[1]
    STD_temp_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_COSMO)
    STD_temp_d_space = calculate_STD_with_distance(345, n_x, n_y, n_z, T_d_COSMO)
    STD_distance = {}
    for i in range(len(station_names)):
        indexes = INCA_grid_indexes[station_names[i]][1]
        dist = calculate_distance_from_onepoint(n_x, n_y, indexes)
        STD_distance[station_names[i]] = [std_from_point(STD_temp_space, dist), std_from_point(STD_temp_d_space, dist)]  
    return STD_distance 

def total_uncertainty(RM_std_distance, RM_std, station_names):
    RM_std_total = {}
    for i in range(len(station_names)):
        RM_std_distance_station = copy.deepcopy(RM_std_distance[station_names[i]])
        RM_std_total[station_names[i]] = [RM_std_distance_station[0] + RM_std[0], RM_std_distance_station[1] + RM_std[1]]
    return RM_std_total

def get_last_10min():
    rounded_minute = int(int(datetime.now().strftime('%M')) / 10) * 10
    return rounded_minute

from datetime import datetime, timedelta
import logging
import time
import datetime as dt
import matplotlib.cm as cm
import copy

import xarray as xr
import pandas as pd
import metpy
from metpy import calc
from metpy.units import units
import numpy as np
from scipy import spatial
from scipy.interpolate import griddata 
import matplotlib.pyplot as plt

import pysteps
from pysteps import nowcasts
import INCA3d_config as c 
from scipy.spatial import KDTree

logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(levelname)s %(name)s %(message)s', )
logger = logging.getLogger(__name__)

n_z = c.INCA_dimension['z']
n_y = c.INCA_dimension['y']
n_x = c.INCA_dimension['x']

############################################### READ DATA #################################################################################
logger.info('Read data...')
start = datetime.now()
########## COSMO ##########
COSMO_data_last, COSMO_data_next, now_time_cosmo, hours_diff, COSMO_xarray = read_last_next_cosmo_file()  
COSMO_data_last = [COSMO_data_last.t_inca.values[0,:,:,:][::-1] - 273.15, metpy.calc.dewpoint_from_specific_humidity(COSMO_data_last.qv_inca, COSMO_data_last.t_inca, COSMO_data_last.p_inca)[0,:,:,:][::-1].magnitude]
COSMO_data_next = [COSMO_data_next.t_inca.values[0,:,:,:][::-1] - 273.15, metpy.calc.dewpoint_from_specific_humidity(COSMO_data_next.qv_inca, COSMO_data_next.t_inca, COSMO_data_next.p_inca)[0,:,:,:][::-1].magnitude]

a_COSMO_last = (int(datetime.now().strftime('%M'))/60)
a_COSMO_next = 1 - a_COSMO_last
COSMO_data = [((a_COSMO_last * COSMO_data_last[0]) + (a_COSMO_next * COSMO_data_next[0])),((a_COSMO_last * COSMO_data_last[1]) + (a_COSMO_next * COSMO_data_next[1]))]

#plt.contourf(COSMO_data[0][20,:,:], levels = np.arange(-13,10), cmap = cm.coolwarm)
#plt.colorbar()
now_time = now_time_cosmo + timedelta(hours=hours_diff)
DT = find_closest_noon_midnight(now_time)
#plt.plot(COSMO_data[1][:,345,350], INCA_grid_indexes['PAY'][0])
########## RADIOMETER ##########
rounded_minutes = get_last_10min()
RM_data, station_names = open_available_RM_data(c.station_nr, now_time + dt.timedelta(minutes=rounded_minutes))
#plt.plot(RM_data['PAY'].temperature_degC, RM_data['PAY'].altitude_m)
INCA_grid_indexes = extract_INCA_grid_onepoint(c.dirs_in['dir_INCA'],  c.station_nr, station_names)     
RM_data = average_to_INCA_grid(INCA_grid_indexes,  RM_data, station_names)
#plt.plot(RM_data['PAY'].temperature_mean, INCA_grid_indexes['PAY'][0])
end = datetime.now()
print((end-start))
############################################# BIAS CORRECTION ##############################################################################
start = datetime.now()
logger.info('Subtract bias...')
bias_date, DT = calculate_bias_time(now_time_cosmo) 
bias_temp, bias_temp_d = calc_run_bias_window(bias_date, INCA_grid_indexes, c.bias_window_size)
RM_data['PAY'].temperature_mean = np.subtract(RM_data['PAY'].temperature_mean,bias_temp.iloc[:,0])
RM_data['PAY'].temperature_d_mean = np.subtract(RM_data['PAY'].temperature_d_mean,bias_temp_d.iloc[:,0])
plt.plot(RM_data['PAY'].temperature_mean, INCA_grid_indexes['PAY'][0])
end = datetime.now()
print((end-start))
############################################# EXPAND IN SPACE ##############################################################################
start = datetime.now()
logger.info('Expand in space...')
RM_data = expand_in_space(RM_data, n_z, n_y, n_x, station_names)
plt.contourf(RM_data['PAY'][0][:,340,:])
plt.colorbar()
end = datetime.now()
print((end-start))
############################################# UNCERTAINTY ESTIMATION ########################################################################
start = datetime.now()
logger.info('Calculate uncertainty...')
########## COSMO ############## 
## uncertainty measurement device   
COSMO_total_std = read_std_COSMO(c.variables_std_COSMO.number[c.variables_std_COSMO.name == 'temperature'].iloc[0], c.variables_std_COSMO.number[c.variables_std_COSMO.name == 'humidity'].iloc[0], 6,  c.station_nr,  INCA_grid_indexes)   
plt.plot(COSMO_total_std[0][:, 341,345], INCA_grid_indexes['PAY'][0])

########## RADIOMETER ##########
## uncertainty with distance
RM_std_distance = calculate_uncertainty_dist(COSMO_data, INCA_grid_indexes, n_x, n_y, station_names)

## uncertainty measurement device
RM_std_temp = c.factor * pd.read_csv(c.dirs_in['dir_std']+'/std_RM_temp_'+str(DT)+'_new.csv')
RM_std_temp_d = c.factor * pd.read_csv(c.dirs_in['dir_std']+'std_RM_temp_d_'+str(DT)+'_new.csv')
RM_std = [expand_in_space_1(RM_std_temp.std_temp.values, n_z, n_y, n_x), expand_in_space_1(RM_std_temp_d.std_temp_d.values, n_z, n_y, n_x)]              
## total uncertainty
RM_total_std = total_uncertainty(RM_std_distance, RM_std, station_names)
plt.contourf(RM_total_std['PAY'][0][:,340,:], levels = np.arange(0,6, 0.2), cmap = cm.Spectral_r)
plt.colorbar()
plt.contourf(RM_total_std['PAY'][0][20,:,:], levels = np.arange(0,6, 0.2), cmap = cm.Spectral_r)
plt.colorbar()
end = datetime.now()
print((end-start))
############################################# CONCAT MEASUREMENTS OF DIFFERENT STATIONS TO ONE LAYER ########################################
start = datetime.now()
logger.info('Concat to one layer...')
########## RADIOMETER ##########
RM_dist_array_binary = attribute_grid_points_to_closest_measurement_point(c.dirs_in['dir_INCA'], c.coordinates_RM[c.coordinates_RM.station.isin(station_names)], station_names)
plt.contourf(RM_dist_array_binary['PAY'])
plt.colorbar()
RM_data_1 = limit_area_to_station(RM_data, RM_dist_array_binary, station_names)
plt.contourf(RM_data['ALL'][1][20,:,:])
RM_total_std = limit_area_to_station(RM_total_std, RM_dist_array_binary, station_names)
end = datetime.now()
print((end-start))
############################################################ DEFINITION OF WEIGHTS ###########################################################
start = datetime.now()
logger.info('Calculation of weights...')
RM_sigma = [calculate_sigma(RM_total_std['ALL'][0], RM_data['ALL'][0]), calculate_sigma(RM_total_std['ALL'][1], RM_data['ALL'][1])]
COSMO_sigma = [calculate_sigma(COSMO_total_std[0], COSMO_data[0]), calculate_sigma(COSMO_total_std[1], COSMO_data[1])]
STD_total = [np.nansum(np.stack((COSMO_sigma[0], RM_sigma[0])), axis = 0), np.nansum(np.stack((COSMO_sigma[1], RM_sigma[1])), axis = 0)]
    
a_COSMO = [(RM_sigma[0] / STD_total[0]), (RM_sigma[1] / STD_total[1])]
a_COSMO[0][np.isnan(RM_sigma[0])] = 1 
a_COSMO[1][np.isnan(RM_sigma[1])] = 1 

a_RM = [(COSMO_sigma[0] / STD_total[0]), (COSMO_sigma[1] / STD_total[1])]
a_RM[0][np.isnan(RM_sigma[0])] = 0
a_RM[1][np.isnan(RM_sigma[1])] = 0
end = datetime.now()
print((end-start))
############################################################ COMBINATION DATASETS ###########A################################################
start = datetime.now()
logger.info('Combination of datasets...')
COMBINED = [np.nansum(np.stack(((a_COSMO[0] * COSMO_data[0]) ,  (a_RM[0] * RM_data['ALL'][0]))), axis = 0), np.nansum(np.stack(((a_COSMO[1] * COSMO_data[1]) ,  (a_RM[1] * RM_data['ALL'][1]))), axis = 0)]
COSMO_xarray['t_inca'] = (['z', 'y', 'x' ], COMBINED[0])
COSMO_xarray['td_inca'] =  (['z', 'y', 'x' ], COMBINED[1])
COSMO_xarray['qv_inca'] =  (['z', 'y', 'x' ], metpy.calc.specific_humidity_from_dewpoint(COMBINED[1] * units.degC, COSMO_xarray.p_inca.values * units.hPa)[::-1][0,:,:,:].magnitude)
COSMO_xarray.to_netcdf(path = c.dirs_out['dir_combi'] + 'cosmo_1e_inca_'+str(now_time_cosmo.strftime('%Y%m%d%H'))+'_0'+str(int(hours_diff))+'_00.nc')
end = datetime.now()
print((end-start))

############################################################ NOWCASTING ###########A##########################################################
TAU = 6
FITC = 0
#step=?
#INCA_analyse = INCA ANALYSIS+DT
#gew = np.exp(-(step - FITC) / TAU)
#T=gew*(INCA_analyse)+(1-gew)*COSMO_data[0]
