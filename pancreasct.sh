#! /bin/bash

set -euxo pipefail

root_path="$HOME/Downloads/Pancreas-CT"
data_path="$root_path"/manifest-1599750808610/Pancreas-CT

mkdir "$root_path"/raw

for folder in "$data_path"/PANCREAS_*
do
    echo "${folder##*/}"
    mkdir "$root_path"/raw/"${folder##*/}"
    for dated_folder in "$folder"/*
    do  
        for named_folder in "$dated_folder"/Pancreas-*
        do
            # echo "$path"/"$folder"/"$dated_folder"/"$named_folder"/*.dcm
            for file in "$named_folder"/*.dcm
            do  
                cp -v "$file" "$root_path"/raw/"${folder##*/}"/"${file##*/}"
            done
        done
    done 
done
