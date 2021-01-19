#!/bin/bash

currentdate=$1
loopenddate=$(/bin/date --date "$2 1 day" +%Y%m%d)

until [ "$currentdate" == "$loopenddate" ]
do
  for HH in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 
  do
  for MMMM in 0000 1000 2000 3000 4000 5000  
  do
    echo $currentdate$HH$MMMM
    curl "http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=06610&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&delimiter=comma&obsTypeIds=31&date=$currentdate$HH$MMMM" > /data/COALITION2/internships/nal/Internship_nal/2_Data_aquisition/2_1_download_past/Radiometer/Validation_2/Payerne/Radiometer_06610_$currentdate$HH$MMMM.txt
    curl "http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/integral/wmo_ind?locationIds=06620&delimiter=comma&measCatNr=1&dataSourceId=38&parameterIds=3150,4902,5552,5553,5554,5555,5556,5557,5558,5559,5560,5561,5562,5563,2537,5547&obsTypeIds=31&date=$currentdate$HH$MMMM" > //data/COALITION2/internships/nal/Internship_nal/2_Data_aquisition/2_1_download_past/Radiometer/Validation_2/Payerne/Radiometer_quality_flag_06610_$currentdate$HH$MMMM.txt    
    done
  done
  currentdate=$(/bin/date --date "$currentdate 1 day" +%Y%m%d)
done
