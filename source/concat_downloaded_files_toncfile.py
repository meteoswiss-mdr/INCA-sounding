# A script to concat measurement files over a certain period (e.g. 1-yr) to one file and to save it as a nc-fil
import pandas as pd
import numpy as np
import datetime as dt
import netCDF4 as nc4
import xarray as xr
import metpy.calc
from metpy.units import units
   
######################################## define paths ############################################ 
RS_archive   = '/data/COALITION2/internships/nal/Internship_nal/2_Data_aquisition/2_1_download_past/Radiosonde/Validation_2/Payerne/' # path of radiosonde data
RA_archive = '/data/COALITION2/internships/nal/Internship_nal/2_Data_aquisition/2_1_download_past/Raman_lidar/Validation_2/Payerne/' # path of raman lidar data
RM_archive = '/data/COALITION2/internships/nal/Internship_nal/2_Data_aquisition/2_1_download_past/Radiometer/Validation_2/Payerne/' # path of SwissMetNet data

######################################## define variables ######################################## 
# radiosonde 
## Payerne -> 06610, Munich -> 10868, Milano -> 16080, Stuttgart -> 10739
location_name_RS = 'Payerne' # location name 
location_id_RS = '06610' # location ID for 
firstdate_RS = '2019050100' # define time span
lastdate_RS = '2020050100'
step_RS = 12 # define time step in hours
nc_name_RS = 'RS_concat.nc'

# raman lidar
location_name_RA = 'Payerne' # location name
location_id_RA = '06610' # location ID 
firstdate_RA = '2019050100' # define time span
lastdate_RA = '2020050100'
step_RA = 12 # define time step in minutes
nc_name_RA = 'RA_concat.nc'

# Radiometer
location_name_RM = 'Payerne' # location name
location_id_RM = '06610' # location ID
firstdate_RM = '2019050100' # define time span
lastdate_RM = '2020050100'
step_RM = 12 # define time step in minutes
nc_name_RM = 'RM_concat.nc'
nc_name_RM_quality_flag= 'RM_concat_quality_flag.nc'
######################################## read txt files and concat to one dataframe ######################################## 
##### Radiosondes #####  
firstobj=dt.datetime.strptime(firstdate_RS,'%Y%m%d%H')
lastobj=dt.datetime.strptime(lastdate_RS,'%Y%m%d%H')

pd_concat_RS = pd.DataFrame() # define empty dataframe
while firstobj != lastobj: # loop over days
    print(firstobj.strftime('%Y%m%d%H%M%S'))
    path_RS_txt = RS_archive+firstobj.strftime("%Y")+'/'+firstobj.strftime("%m")+'/'+firstobj.strftime("%d")+'/RS_'+location_id_RS+'_'+firstobj.strftime('%Y%m%d%H')+'.txt'
    data_RS = pd.read_csv(path_RS_txt)
    pd_concat = pd_concat_RS.append(data_RS)
    
    firstobj= firstobj + dt.timedelta(hours=step_RS)
 
# name variables
pd_concat_RS = pd_concat.rename(columns = {'termin':'time_YMDHMS', '744': 'pressure_hPa', '745':'temperature_degC', '746':'relative_humidity_percent', '742':'geopotential_altitude_m', '748':'wind_speed_ms-1', '743': 'wind_dir_deg', '747':'dew_point_degC' })
ds = xr.Dataset(pd_concat_RS)
ds.to_netcdf(RS_archive+'/'+nc_name_RS) # save dataframe to nc file 

##### Raman lidar #####
firstobj=dt.datetime.strptime(firstdate_RA,'%Y%m%d%H')
lastobj=dt.datetime.strptime(lastdate_RA,'%Y%m%d%H')
pd_concat_RA = pd.DataFrame()
while firstobj != lastobj: # loop over days   
    print(firstobj.strftime('%Y%m%d%H%M%S'))
    path_RA_txt = RALMO_archive+firstobj.strftime("%Y")+'/'+firstobj.strftime("%m")+'/'+firstobj.strftime("%d")+'/'+'RALMO_'+location_id_RA+'_'+firstobj.strftime('%Y%m%d%H%M%S')+'.txt'
    data_RA = pd.read_csv(path_RA_txt) # open file
    pd_concat_RA = pd_concat_RA.append(data_RA) # append file to a dataframe

    firstobj= firstobj + dt.timedelta(hours=step_RA)

# name variables
pd_concat_RA = pd_concat.rename(columns = {'termin':'time_YMDHMS', 'level':'altitude_m', '4919': 'specific_humidity_gkg-1', '4906':'uncertainty_specific_humidity_gkg-1', '4907':'vertical_resolution_specific_humidity_m', '3147':'temperature_K', '4908':'uncertainty_temperature_K', '4909': 'vertical_resolution_temperature', '4910':'normalised_backscatter', '4911':'uncertainty_backscatter', '4912': 'vert_resolution_backscatter', '4913': 'aerosol_dispersion_rate', '4914': 'uncertainty_dispersion_rate', '4915' : 'vertical_resolution_aerosol_dispersion_rate'})
ds = xr.Dataset(pd_concat_RA)
ds.to_netcdf(RA_archive+'/'+nc_name_RA) # save dataframe to nc file

##### Radiometer #####  
firstobj=dt.datetime.strptime(firstdate_RM,'%Y%m%d%H')
lastobj=dt.datetime.strptime(lastdate_RM,'%Y%m%d%H')

pd_concat_RM = pd.DataFrame() # define empty dataframe
while firstobj != lastobj: # loop over days
    print(firstobj.strftime('%Y%m%d%H%M%S'))
    path_RM_txt = RM_archive+firstobj.strftime("%Y")+'/'+firstobj.strftime("%m")+'/'+firstobj.strftime("%d")+'/Radiometer_'+location_id_RM+'_'+firstobj.strftime('%Y%m%d%H%M%S')+'.txt'
    data_RM = pd.read_csv(path_RM_txt)
    pd_concat_RM = pd_concat_RM.append(data_RM)
    
    firstobj= firstobj + dt.timedelta(hours=step_RM)
 
# name variables
pd_concat_RM = pd_concat.rename(columns = {'termin' : 'time_YMDHMS', '3147':'temperature_K', 'level':'absolute_humidity_gm3'})
ds = xr.Dataset(pd_concat_RM)
ds.to_netcdf(RA_archive+'/'+nc_name_RM) # save dataframe to nc file

# quality flag
firstobj=dt.datetime.strptime(firstdate_RM,'%Y%m%d%H')
lastobj=dt.datetime.strptime(lastdate_RM,'%Y%m%d%H')

pd_concat_RM = pd.DataFrame() # define empty dataframe
while firstobj != lastobj: # loop over days
    print(firstobj.strftime('%Y%m%d%H%M%S'))
    path_RM_txt = RM_archive+firstobj.strftime("%Y")+'/'+firstobj.strftime("%m")+'/'+firstobj.strftime("%d")+'/Radiometer_quality_flag_'+location_id_RM+'_'+firstobj.strftime('%Y%m%d%H%M%S')+'.txt'
    data_RM = pd.read_csv(path_RM_txt)
    pd_concat_RM = pd_concat_RM.append(data_RM)
    
    firstobj= firstobj + dt.timedelta(minutes=step_RM)
 
# name variables
pd_concat_RM = pd_concat.rename(columns = {'3150' : 'rainflag', '4902' : 'cloudbase_height','5552' : 'environ_temp','5553' : 'environ_pres','5554' : 'environ_rel_humidity','5555' : 'humidity','5556' : 'receiver_2','5557':'fstatus_boundary_layer_mode','5558':'quality_flag_LWP','5559':'quality_flag_IWV','5560':'quality_flag_temp','5561':'quality_flag_hum','5562':'quality_flag_boundary_layer_temp','5563':'radiometer_alarm','5547':'liquid_water_path'})
ds = xr.Dataset(pd_concat_RM)
ds.to_netcdf(RA_archive+'/'+nc_name_RM_quality_flag) # save dataframe to nc file