dir="/data/COALITION2/internships/nal/Internship_nal/2_Data_aquisition/2_1_download_past/Radiometer/Validation_2/Payerne/"

myarchive="/data/COALITION2/internships/nal/Internship_nal/2_Data_aquisition/2_1_download_past/Radiometer/Validation_2/Payerne/"
cd $dir

# sort files into folder
RM_files=`ls Radiometer_06610_2019*.txt`

for RM_file in $RM_files
do
    year=`echo "$RM_file" | awk '{print substr($0,18,4)}'`
    echo $year
    month=`echo "$RM_file" | awk '{print substr($0,20,2)}'`
    echo $month
    day=`echo "$RM_file" | awk '{print substr($0,22,2)}'`
    echo $day
    if [ ! -d $myarchive/$year/$month/$day ]; then
       echo mkdir -p $myarchive/$year/$month/$day
       mkdir -p $myarchive/$year/$month/$day
    fi

    echo move  $RM_file to $myarchive/$year/$month/$day/
    mv -f $RM_file $myarchive/$year/$month/$day/

  done

cd -

