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
    parser.add_argument("--folder", type=str, default=None, help="The full path to the file that is to be split into sections.")
    parser.add_argument("--p", type=int, default=40, help="Distance from the centre line beyond which all points will be removed")

    args = parser.parse_args()

    p = args.p
    p = 0 if p < 0 else 100 if p > 100 else p
    p /= 100
    args.p = p

    datapath = args.folder if args.folder is not None else obtain_data_path()

    for file in os.listdir(datapath):
        filename = os.path.join(datapath, file)
        las = laspy.read(filename)

        filename_cut, _ = os.path.splitext(os.path.basename(filename))

        data = np.vstack((las.x, las.y, las.z)).T

        average_x = np.mean(data[:, 0])
        average_y = np.mean(data[:, 1])

        range_x = np.max(data[:, 0]) - np.min(data[:, 0])
        range_y = np.max(data[:, 1]) - np.min(data[:, 1])

        print(range_x, range_y)
        # Filter points based on the range
        mask_x = (data[:, 0] >= average_x - (range_x/2)*args.p) & (data[:, 0] <= average_x + (range_x/2)*args.p)
        new_x = data[mask_x]
        mask_y = (data[:, 1] >= average_y - (range_x/2)*args.p) & (data[:, 1] <= average_y + (range_y/2)*args.p)
        new_y = data[mask_y]
        print(len(las.points))
        print(len(new_x), len(new_y))

        folder = os.path.join(os.getcwd(), "lanes")
        if not os.path.exists(folder):
            os.makedirs(folder)
        outpath = os.path.join(folder, f"{filename_cut}_distance_{(range_x/2)*args.p}_x.las")
        print(f"{len(new_x)} points kept in the x direction")
        new_las = laspy.LasData(las.header)
        new_las.points = las.points[mask_x]
        new_las.write(outpath)

        outpath = os.path.join(folder, f"{filename_cut}_distance_{(range_y/2)*args.p}_y.las")
        print(f"{len(new_y)} points kept in the y direction")
        new_las = laspy.LasData(las.header)
        new_las.points = las.points[mask_y]
        new_las.write(outpath)

if __name__ == "__main__":
    main()