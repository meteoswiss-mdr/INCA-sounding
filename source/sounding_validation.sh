#!/bin/bash

######################## define variables and paths##########################################
dir_RS_in=/data/mch/INCA/OBS_PROFILES/Payerne/
dir_script=/data/mch/INCA/SCR

waiting_time=20 # define waiting time in seconds between loops 

######################## activate environment ###############################################
#source /data/COALITION2/internships/nal/anaconda3/bin/activate INCA3d
export PKG_CONFIG_PATH=/data/mch/INCA/python/3.6.9/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/data/mch/INCA/python/3.6.9/lib:$LD_LIBRARY_PATH
# activate python environnment
source /data/mch/INCA/python36/bin/activate

######################## define variable and end time#########################################
current_hour=$(date "+%H")
twelve=12
hours_diff="$(($twelve-$current_hour))"

if (("$hours_diff" > "6")); then
	hours_diff="$(($hours_diff-$twelve))"
	fi

RS_time=$(date -d "$date +${hours_diff} hours" "+%Y%m%d%H")
end_time=$(date -d "$date +2 hours" "+%Y%m%d%H%M")

######################## check if file is available and start calculation ####################
while (($(date "+%Y%m%d%H%M") < $end_time))
	do
	if [[ -f ${dir_RS_in}/"RS_06610_${RS_time}00.txt" ]]; then
		echo "RS file of ${RS_time} is available now"
		python ${dir_script}/sounding_validation.py "${RS_time}00"
		break
	fi
	sleep $waiting_time

done