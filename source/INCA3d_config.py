# -*- coding: utf-8 -*-
#### paths and file names 
import pandas as pd
import logging
basedir_in = '/data/mch/INCA/' 
basedir_out = '/data/COALITION2/internships/nal/combined_profiles_output/'

dirs_in = dict(
dir_COSMO = '/data/mch/INCA/cosmo1/',
dir_INCA = '/data/COALITION2/internships/nal/data/COSMO/',
dir_RM = '/data/COALITION2/internships/nal/download_realtime/',
dir_0_std_RM = '/data/COALITION2/internships/nal/std_files/std_RM_temp_temp_d_0.csv',
dir_12_std_RM = '/data/COALITION2/internships/nal/std_files/std_RM_temp_temp_d_12.csv',
dir_0_std_RA = '/data/COALITION2/internships/nal/std_files/std_RA_temp_temp_d_0.csv',
dir_12_std_RA = '/data/COALITION2/internships/nal/std_files/std_RA_temp_temp_d_12.csv',
dir_0_std_NUCAPS = '/data/COALITION2/internships/nal/std_files/std_NUCAPS_temp_temp_d_0.csv',
dir_12_std_NUCAPS = '/data/COALITION2/internships/nal/std_files/std_NUCAPS_temp_temp_d_12.csv',
dir_std_COSMO = '/data/COALITION2/internships/nal/std_files/',
dir_NUCAPS = '/data/COALITION2/database/NUCAPS/'
)

dirs_out = dict(
dir_combi = '/data/COALITION2/internships/nal/INCA3d/combi_output/',
dir_sigma_COSMO= '/data/COALITION2/internships/nal/INCA3d/sigma_cosmo/'
)

#### variables
factor = 1
index_temp = 0
index_hum = 1
index_INCA_grid = 0
index_INCA_point = 1
index_station_name = 0
index_station_nr = 1
bias_window_size = 3
bias_stations = ['PAY']
RM_station_nr = [['PAY','06610'],['SHA' , '06620'], ['GRE' , '06632'], ['GVE' , '06700'], ['KLO' , '06670']]
RA_station_nr = [['PAY','06610']]
tempro_stations =  ['SHA']
lidar_station = ['PAY']
INCA_dimension = {'z' : 50, 'y' : 640, 'x' : 710}
season = pd.DataFrame({'name' : ['DJF', 'DJF', 'DJF', 'MAM', 'MAM', 'MAM', 'JJA', 'JJA', 'JJA', 'SON', 'SON', 'SON'], 'number' : [12,1,2,3,4,5,6,7,8,9,10,11]})
coordinates_RM = pd.DataFrame({'station' : ['PAY', 'SHA', 'GRE'], 'Longitude' : [6.93601, 8.620142, 7.415144], 'Latitude' : [46.8220, 47.689842, 47.179097], 'indexes_y' : [345,442,385], 'indexes_x' : [306,433,343]})
variables_std_COSMO = pd.DataFrame({'name' : ['temperature', 'humidity'], 'number' : [2,59]})

#### module parts
bias_correction = False
RM_data = True
RA_data = False # currently not available
NUCAPS_data = False # currently not available
#save_log_figures = True
logger_level = logging.INFO
logger_filename = 'logfile.log'
