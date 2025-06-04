import numpy as np
import laspy
import math
import os
import argparse
from tkinter import Tk, filedialog
import pandas as pd
import json
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import pairwise_distances_argmin_min
import open3d as o3d

DESKTOP = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

def clean_label(name):
    return '-'.join([s.lower() for s in name.split()])

def obtain_data_path():
    """
    Lets the user choose a file through the UI
    """
    # Manually obtain file via UI
    Tk().withdraw()
    las_filename = filedialog.askdirectory(
        initialdir=DESKTOP, title="Please select the data folder"
    )

    print(f"You have chosen to open the point cloud:\n{las_filename}")

    return os.path.abspath(las_filename)

def main():
    # Sample point cloud data (replace this with your actual point cloud data)
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="The full path to the file that is to be split into sections.")

    args = parser.parse_args()
    data = args.folder if args.folder is not None else obtain_data_path()
    parsed = os.path.join(os.getcwd(), "parsed")
    las_check = os.path.join(os.getcwd(), "las_check")
    class_names = set()

    if not os.path.exists(parsed):
        os.makedirs(parsed)
    for scene in os.listdir(data):
        scene = os.path.join(data, scene)
        las_files = os.path.join(scene, "las_files")
        if not os.path.exists(las_files):
            print(f"No files foudn for {scene}")
            continue
        for filename in os.listdir(las_files):
            filename_cut, _ = os.path.splitext(filename)
            las_file = os.path.join(las_files, filename)
            labels_file = os.path.join(scene, "label", f"{filename_cut}_converted.json")
            centre_file = os.path.join(scene, "centres", f"{filename_cut}_converted_centre.json")
            lane_marking_las = os.path.join(
                scene, "las_files", f"{filename_cut}_lane_marking.las")
            pavement_las = os.path.join(
                scene, "las_files", f"{filename_cut}_pavement.las")
            lane_marking_labels = os.path.join(
                scene, "label", f"{filename_cut}_converted_lane_marking.json")
            pavement_labels = os.path.join(
                scene, "label", f"{filename_cut}_pavement_converted.json")
            if not os.path.exists(labels_file):
                print(f"No labels found for {filename}, moving on...")
                continue
            if not os.path.exists(centre_file):
                print(f"No centre found for {filename}, moving on...")
                continue
                
            if os.path.exists(lane_marking_las):
                with open(lane_marking_labels, "r") as f:
                    box_list = json.load(f)

                for box in box_list:
                    label_name = clean_label(box["obj_type"])    
                    class_names.add(label_name)
            
            if os.path.exists(pavement_las):
                with open(pavement_labels, "r") as f:
                    box_list = json.load(f)

                for box in box_list:
                    label_name = clean_label(box["obj_type"])    
                    class_names.add(label_name)
            with open(labels_file, "r") as f:
                box_list = json.load(f)

            for box in box_list:
                label_name = clean_label(box["obj_type"])    
                class_names.add(label_name)
    
    print("The following classes were found in all files")
    print(list(class_names))
    return

if __name__ == "__main__":
    main()