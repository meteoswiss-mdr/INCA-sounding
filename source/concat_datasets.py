import pandas as pd
import numpy as np
import datetime as dt
import netCDF4 as nc4
import xarray as xr
import metpy.calc
from metpy.units import units

def read_HATPRO(firstobj):
    url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&delimiter=comma&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&obsTypeIds=31'
    RM = pd.read_csv(url, skiprows = [1], sep=',')
    if RM.empty:
        firstobj= firstobj + dt.timedelta(minutes=step_RM)
    else: 
        RM = RM.rename(columns = {'termin' : 'time_YMDHMS' , '3147' : 'temperature_K', '3148' : 'absolute_humidity_gm3', 'level' : 'altitude_m'})
        # < temperature >
        RM['temperature_degC'] = RM.temperature_K - 273.15
        
        
        p_w = ((RM.temperature_K * RM.absolute_humidity_gm3) / 2.16679)
        RM['dew_point_degC'] = metpy.calc.dewpoint((p_w.values * units.Pa))
        
        url = 'http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/integral/wmo_ind?locationIds=06610&measCatNr=1&dataSourceId=38&parameterIds=3150,5560,5561&date='+str(dt.datetime.strftime(firstobj, '%Y%m%d%H%M%S'))+'&delimiter=comma&obsTypeIds=31'
        RM_quality_flag = pd.read_csv(url, skiprows = [1], sep=',')
        if RM_quality_flag.empty:
            RM['quality_flag'] = np.nan * len(RM)
            RM['radiometer_quality_flag_temperature_profile'] = np.nan * len(RM)
            RM['radiometer_quality_flag_humidity_profile'] = np.nan * len(RM)
        else: 
            RM_quality_flag = RM_quality_flag.rename(columns = {'3150' : 'radiometer_rainflag', '5560': 'radiometer_quality_flag_temperature_profile', '5561' : 'radiometer_quality_flag_humidity_profile'})
            RM['quality_flag'] = pd.concat([RM_quality_flag] * len(RM)).radiometer_rainflag.values
            RM['radiometer_quality_flag_temperature_profile'] = pd.concat([RM_quality_flag] * len(RM)).radiometer_quality_flag_temperature_profile.values
            RM['radiometer_quality_flag_humidity_profile'] = pd.concat([RM_quality_flag] * len(RM)).radiometer_quality_flag_humidity_profile.values
            
        RM['time_YMDHMS'] = pd.to_datetime(RM.time_YMDHMS, format = '%Y%m%d%H%M%S')                                         
        return RM
    
######################################## define paths ############################################ 

RS_archive   = '/data/COALITION2/internships/nal/data/Radiosondes/Payerne/' # path of radiosonde data
RALMO_archive = '/data/COALITION2/internships/nal/data/RALMO/Payerne/' # path of raman lidar data
SMN_archive = '/data/COALITION2/internships/nal/data/SwissMetNet/Payerne' # path of SwissMetNet data
RM_archive = '/data/COALITION2/internships/nal/data/Radiometer/Payerne/' # path of SwissMetNet data

######################################## define variables ######################################## 
# radiosonde 
## Payerne -> 06610, Munich -> 10868, Milano -> 16080, Stuttgart -> 10739
location_name_RS = 'Payerne' # location name 
location_id_RS = '06610' # location ID for 
firstdate_RS = '2019050100' # define time span
lastdate_RS = '2020050100'
step_RS = 12 # define time step in hours
nc_name_RS = 'test_RS_concat.nc'

# raman lidar
location_name_RA = 'Payerne' # location name
location_id_RA = '06610' # location ID 
firstdate_RA = '2019050100' # define time span
lastdate_RA = '2020050100'
step_RA = 30 # define time step in minutes
nc_name_RA = 'test_RA_concat.nc'

# SwissMetNet
location_name_SMN = 'Payerne' # location name
location_id_SMN = '06610' # location ID
firstdate_SMN = '2019050100' # define time span
lastdate_SMN = '2020050100'
step_SMN = 30 # define time step in minutes
nc_name_SMN = 'SMN_concat1.nc'

# Radiometer
location_name_RM = 'Payerne' # location name
location_id_RM = '06610' # location ID
firstdate_RM = '2019050100' # define time span
lastdate_RM = '2020050100'
step_RM = 30 # define time step in minutes
nc_name_RM = 'RM_concat_new1.nc'

######################################## read txt files and concat to one dataframe ######################################## 
##### Radiometer #####  
firstobj=dt.datetime.strptime(firstdate_RM,'%Y%m%d%H')
lastobj=dt.datetime.strptime(lastdate_RM,'%Y%m%d%H')

pd_concat_RM = pd.DataFrame() # define empty dataframe
while firstobj != lastobj: # loop over days
    print(firstobj.strftime('%Y%m%d%H%M%S'))
    RM_data = read_HATPRO(firstobj)
    pd_concat_RM = pd_concat_RM.append(RM_data)
    firstobj= firstobj + dt.timedelta(minutes=step_RM)
 
# name variables
pd_concat_RM = xr.Dataset(pd_concat_RM)
pd_concat_RM.to_netcdf(RM_archive+'/'+nc_name_RM) # save dataframe to nc file 

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

    firstobj= firstobj + dt.timedelta(minutes=step_RA)

# name variables
pd_concat_RA = pd_concat.rename(columns = {'termin':'time_YMDHMS', 'level':'altitude_m', '4919': 'specific_humidity_gkg-1', '4906':'uncertainty_specific_humidity_gkg-1', '4907':'vertical_resolution_specific_humidity_m', '3147':'temperature_K', '4908':'uncertainty_temperature_K', '4909': 'vertical_resolution_temperature', '4910':'normalised_backscatter', '4911':'uncertainty_backscatter', '4912': 'vert_resolution_backscatter', '4913': 'aerosol_dispersion_rate', '4914': 'uncertainty_dispersion_rate', '4915' : 'vertical_resolution_aerosol_dispersion_rate'})
ds = xr.Dataset(pd_concat_RA)
ds.to_netcdf(archive_RA+'/'+nc_name_RA) # save dataframe to nc file

##### SMN #####
firstobj=dt.datetime.strptime(firstdate_SMN,'%Y%m%d%H')
lastobj=dt.datetime.strptime(lastdate_SMN,'%Y%m%d%H')
pd_concat_SMN = pd.DataFrame()
while firstobj != lastobj: # loop over days 
    print(firstobj.strftime('%Y%m%d%H%M%S'))  
    path_SMN_txt = SMN_archive+'/'+firstobj.strftime("%Y")+'/'+firstobj.strftime("%m")+'/'+firstobj.strftime("%d")+'/'+'SMN_'+location_id_SMN+'_'+firstobj.strftime('%Y%m%d%H%M%S')+'.txt'
    data_SMN = pd.read_csv(path_SMN_txt) # open file
    pd_concat_SMN = pd_concat_SMN.append(data_SMN) # append file to a dataframe

    firstobj= firstobj + dt.timedelta(minutes=step_SMN)

# name variables
pd_concat_SMN = pd_concat_SMN.rename(columns = {'termin':'time_YMDHMS', '90': 'pressure_hPa', '91': 'temperature_degC', '98': 'relative_humidity_percent', '196':'wind_speed_mean10min_ms-1', '197':'wind_dir_deg', 
                                            '2761': 'mean_x wind', '2762':'mean y wind', '2763':'mean z wind', '2764': 'mean_temperature_degC', '2765': 'std_x', '2766': 'std_y','2767': 'STd_z', '2768':'std_T','2775':'std_wind_parallel_to_mean',
                                            '2776': 'std_wind_horizontally_perpendicular','2777': 'std_wind_vertically_perpendicular','2778':'longitudinal_turbulence_intensity','2779':'transversal_turbulence_intensity','2780':'vertical_turbulence_velocity','2781':'friction_velocity','2784':'Monin_Obukhov_stability_parameter','2785':'vertical_momentum_flux',
                                            '2786':'vertical_heat_flux','2790':'mean_horizontal_velocity','2792': 'diffusion_class','93': 'precipitation_mm','94': 'sunshine_duration_mean10min'})
ds = xr.Dataset(pd_concat_SMN)
ds.to_netcdf(SMN_archive+'/'+nc_name_SMN) # save dataframe to nc file