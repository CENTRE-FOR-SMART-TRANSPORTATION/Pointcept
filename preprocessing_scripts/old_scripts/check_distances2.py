import laspy
import numpy as np
import math
import os
import argparse
from tkinter import Tk, filedialog
import pandas as pd

DESKTOP = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 
SECTION_LENGTH = 150 # metres

def sanity_check(num_sections, folder, filename_cut):
    '''
    To check if every section has the same information as the original file
    '''
    for i in range(num_sections):
        filename = os.path.join(folder, f"{filename_cut}_section_{i}.las")
        las = laspy.read(filename)

        print(las.header)
        print(las.points)
        print(list(las.point_format.dimension_names))

# Open our .las point cloud into memory
def obtain_las_path():
    """
    Lets the user choose a file through the UI
    """
    # Manually obtain file via UI
    Tk().withdraw()
    las_filename = filedialog.askopenfilename(
        filetypes=[(".las files", "*.las"), ("All files", "*")],
        initialdir=DESKTOP,
        title="Please select the main point cloud",
    )

    print(f"You have chosen to open the point cloud:\n{las_filename}")

    return os.path.abspath(las_filename)
def calc_dist(las):
    x, y = las.xyz[:, 0], las.xyz[:, 1]
    assert(len(x) == len(y))

    first_x, first_y = x[0], y[0]
    # Create a DataFrame from points


    # Calculate distances using vectorized operations
    # along the x-axis, assuming the road is longer along the x axis
    distances = ((y - first_y)**2)**0.5
    sorted_idx = np.argsort(distances)
    distances = distances[sorted_idx]
    return distances.tolist()

def main():
    sections = os.path.join(os.getcwd(), "sections")
    for l in os.listdir(sections):
        las_file = os.path.join(sections, l)
        print(f"Checking for file {l}...\n")
        for d in os.listdir(las_file):
            print(f"\tChecking for sections of length {d}")
            dist = int(d)
            err = dist//10
            folder = os.path.join(las_file, d)
            for filename in os.listdir(folder):
                temp = filename # for printing if required
                filename = os.path.join(folder, filename)
                las = laspy.read(filename)
                dista = calc_dist(las)
                print(f"\t\t{temp} is {max(dista)} metres instead of {dist} metres")
            print()
        print()
    print("Done...")  

if __name__ == "__main__":
    main()