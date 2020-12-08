#!/bin/bash

dir1=/data/COALITION2/internships/nal/anaconda3/bin/
dir2=/data/COALITION2/internships/nal/download_realtime/

cd ${dir1}
source activate INCA3d
echo "*** Activate environment" 
cd ${dir2}
echo "*** Start to download observation data (loop until all data is there)"

currentdate=$(echo "$(date "+%Y%m%d%H%M") - ($(date +%M)%10)" | bc)
echo $currentdate
end_time=$(date -d "$date +180 seconds" "+%Y%m%d%H%M")

################################## define veriables #########################################
station_list='06610 06632 06620 06670 06700' # define stations
waiting_time=10 # define waiting time between loops 
MS=00 # define seconds
############################################################################################

### download measurements of all stations ###
for station in $station_list
do
echo $station
curl "http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=$station&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&delimiter=comma&obsTypeIds=31&date=$currentdate$MS" > /data/COALITION2/internships/nal/download_realtime/station_$station/RM_$station_$currentdate.txt
curl "http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/integral/wmo_ind?locationIds=$station&delimiter=comma&measCatNr=1&dataSourceId=38&parameterIds=3150,4902,5552,5553,5554,5555,5556,5557,5558,5559,5560,5561,5562,5563,2537,5547&obsTypeIds=31&date=$currentdate$MS" > /data/COALITION2/internships/nal/download_realtime/station_$station/RM_QF_$station_$currentdate.txt
done

### check which files are empty and downlaod them again ###
while (($(date "+%Y%m%d%H%M") != $end_time))
	do 
	for station in $station_list
	do
	echo $station
	cd /data/COALITION2/internships/nal/download_realtime/station_$station
	size=$(wc -l RM_$station_$currentdate.txt | awk '{print $1}')
	echo "$size"
	
	if (($size < 3))
		then 
		echo $station
		curl "http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/data/wmo_ind?locationIds=$station&measCatNr=1&dataSourceId=38&parameterIds=3147,3148&delimiter=comma&obsTypeIds=31&date=$currentdate$MS" > /data/COALITION2/internships/nal/download_realtime/station_$station/RM_$station_$currentdate.txt
		curl "http://wlsprod.meteoswiss.ch:9010/jretrievedwh/profile/integral/wmo_ind?locationIds=$station&delimiter=comma&measCatNr=1&dataSourceId=38&parameterIds=3150,4902,5552,5553,5554,5555,5556,5557,5558,5559,5560,5561,5562,5563,2537,5547&obsTypeIds=31&date=$currentdate$MS" > /data/COALITION2/internships/nal/download_realtime/station_$station/RM_QF_$station_$currentdate.txt
		fi

	sleep $waiting_time
done
done



