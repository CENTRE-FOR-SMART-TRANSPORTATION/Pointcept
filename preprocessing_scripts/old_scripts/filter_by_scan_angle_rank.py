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
    parser.add_argument("--percentage", type=int, default=50, help="The length of the smaller section")
    parser.add_argument("--d", type=int, default=10, help="The distance from the centre line beyond which all points are discarded")
    args = parser.parse_args()
    p = args.percentage
    p = 0 if p < 0 else 100 if p > 100 else p
    p /= 100
    p = 1 - p

    datapath = args.folder if args.folder is not None else obtain_data_path()

    for file in os.listdir(datapath):
        filename = os.path.join(datapath, file)
        las = laspy.read(filename)

        if "scan_angle_rank" not in list(las.point_format.dimension_names) or np.all(las.scan_angle_rank == 0):
            print("No scan angle rank found in", filename)
            continue
        
        centre_line_mask = las.scan_angle_rank == 0
        points = np.vstack((las.x, las.y, las.z)).T
        centre_line_points = points[centre_line_mask]
        centre_line_y_points = centre_line_points[:, 1]
        min_point = centre_line_points[0]
        max_point = centre_line_points[-1]
        direction_vector = max_point - min_point

        # v1 = points - min_point
        # v2 = np.cross(direction_vector, v1)
        # perpendicular_distances = np.linalg.norm(v2, axis=1) / np.linalg.norm(direction_vector)

        v1_xy = points[:, :2] - min_point[:2]
        direction_vector_xy = max_point[:2] - min_point[:2]

        # Pad vectors to 3D for cross product
        v1_3d = np.column_stack([v1_xy, np.zeros_like(v1_xy[:, 0])])
        direction_vector_3d = np.array([direction_vector_xy[0], direction_vector_xy[1], 0])
        
        # Cross product in 3D
        v2_3d = np.cross(direction_vector_3d, v1_3d)
        print(direction_vector_3d.shape, v2_3d.shape, v1_3d.shape)
        perpendicular_distances = np.linalg.norm(v2_3d, axis=1) / np.linalg.norm(direction_vector_3d)

        # Step 3: Create a new array with points having perpendicular distance less than 10
        filtered_points = las.points[perpendicular_distances < args.d]

        print(f"Filtering {filename}...")


        filename_cut, _ = os.path.splitext(os.path.basename(filename))

        folder = os.path.join(os.getcwd(), "filtered")
        if not os.path.exists(folder):
            os.makedirs(folder)
        outpath = os.path.join(folder, f"{filename_cut}_sar__d_points_{args.d}.las")
        print(f"Writing to {outpath}...")

        new_las = laspy.LasData(las.header)
        new_las.points = filtered_points
        new_las.write(outpath)

    print(f"Done filetering...")

if __name__ == "__main__":
    main()