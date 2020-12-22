import os
import datetime as dt

# calculate uncertainty for months in the past
firstdate='202009271200'
lastdate='202012171200'
step=24

firstobj = dt.datetime.strptime(firstdate, "%Y%m%d%H%M")
lastobj = dt.datetime.strptime(lastdate, "%Y%m%d%H%M")

firstobj_cosmo = firstobj - dt.timedelta(hours=3)
lastobj_cosmo = lastobj - dt.timedelta(hours=3)
while firstobj_cosmo != lastobj_cosmo: # loop over days
    print(firstobj)
    os.system(f'python /home/zue/users/nal/inca_validation/SCR/sounding_stddist.py {firstobj_cosmo.strftime("%Y%m%d%H")}')
    firstobj_cosmo = firstobj_cosmo + dt.timedelta(hours=step)


while firstobj != lastobj: # loop over days
    print(firstobj)
    os.system(f'python /home/zue/users/nal/inca_validation/SCR/sounding_validation.py {firstobj.strftime("%Y%m%d%H%M")}')
    firstobj = firstobj + dt.timedelta(hours=step)
    
    # change filename with date
    # save to path: path='/data/COALITION2/internships/nal/validation/uncertainty/'
    

# calculate combination 
while firstobj != lastobj: # loop over days
    print(firstobj)
    os.system(f'python /home/zue/users/nal/inca_validation/SCR/sounding_main.py {firstobj.strftime("%Y%m%d%H%M")}')
    firstobj= firstobj + dt.timedelta(hours=step)


#x = xr.open_dataset('/data/COALITION2/internships/nal/validation/combi_retrieval/cosmo-1e_inca_2020092621_03_00.nc')
#f = xr.open_dataset('/data/COALITION2/database/cosmo/T-TD_3D/2020/09/26/cosmo-1e_inca_2020092621_03_00.nc')


