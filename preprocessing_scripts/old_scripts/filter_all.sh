#! /bin/bash

parent='/home/gurveer/Desktop/cst-pointcloud-annotation_web-2023/data/'

for folder in $parent/[05]*; do
    if [ -d "$folder" ]; then
        python3 filter_by_scan_angle_rank.py --folder "$folder"/las_files
    fi
done