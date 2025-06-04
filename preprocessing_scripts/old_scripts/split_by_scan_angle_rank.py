import laspy
import numpy as np
import math
import os
import argparse
from tkinter import Tk, filedialog
import pandas as pd
from copy import deepcopy

DESKTOP = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

# Open our .las point cloud into memory
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="The full path to the folder that is to be split into sections.")

    args = parser.parse_args()
    datapath = args.folder if args.folder is not None else obtain_data_path()

    for file in os.listdir(datapath):
        filename = os.path.join(datapath, file)
        las = laspy.read(filename)

        if "scan_angle_rank" not in list(las.point_format.dimension_names) or np.all(las.scan_angle_rank == 0):
            print("No scan angle rank found in", filename)
            continue
        
        mask_negative = las.scan_angle_rank > 0
        mask = np.logical_not(mask_negative)
        negative_points = las.points[mask_negative]
        positive_points = las.points[mask]


        print(f"Filtering {filename}...")


        filename_cut, _ = os.path.splitext(os.path.basename(filename))

        folder = os.path.join(os.getcwd(), "filtered")
        if not os.path.exists(folder):
            os.makedirs(folder)

        outpath = os.path.join(folder, f"{filename_cut}_sar_points_negative.las")
        print(f"Writing to {outpath}...")
        new_las = laspy.LasData(las.header)
        new_las.points = negative_points
        new_las.write(outpath)

        outpath = os.path.join(folder, f"{filename_cut}_sar_points_positive.las")
        print(f"Writing to {outpath}...")
        new_las = laspy.LasData(las.header)
        new_las.points = positive_points
        new_las.write(outpath)

    print(f"Done filetering...")

if __name__ == "__main__":
    main()