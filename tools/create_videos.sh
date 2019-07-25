#!/bin/bash

# dependency: ffmpeg (https://ffmpeg.org/)

# Use this script to create video files from the images included in the ucsd anomaly dataset

for set in Train Test; do
    input_folders=`find ./data -type d -name "${set}0*"`

    for input_folder in $input_folders; do
        output_folder=ucsd_ad_vid/`echo ${input_folder} | cut -f 3- -d "/"`
        echo $output_folder
        mkdir -p $output_folder
        ffmpeg -framerate 12 -i ${input_folder}/%03d.tif ${output_folder}/video.mp4
    done
done
