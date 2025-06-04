import laspy
import numpy as np
import math
import os
import argparse
from tkinter import Tk, filedialog
import pandas as pd
from copy import deepcopy

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
    distances = ((x - first_x)**2)**0.5
    sorted_idx = np.argsort(distances)
    distances = distances[sorted_idx]
    return distances.tolist(), sorted_idx

def binary_search(dist, d):
    low, high = 0, len(dist)
    while low < high:
        mid = (low + high)//2
        if dist[mid] > d:
            high = mid - 1
        elif dist[mid] < d:
            low = mid + 1
        else:
            print(mid, dist[mid])
            return mid
    print(mid, dist[mid])
    return mid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="The full path to the file that is to be split into sections.")
    parser.add_argument("--length", type=int, default=150, help="The length of the smaller section")

    args = parser.parse_args()

    section_length = args.length
    filename = args.file
    if filename is None:
        filename = obtain_las_path()

    print(f"Cutting {filename} into sections of {section_length} metres...")

    las = laspy.read(filename)

    x, y = las.xyz[:, 0], las.xyz[:, 1]
    arr = None
    if abs(max(x) - min(x)) < abs(max(y)-min(y)):
        arr = deepcopy(x)
    else:
        arr = deepcopy(y)
    sorted_indices = np.argsort(arr)
    points = las.points[sorted_indices]

    idxs = [0]

    arr = arr[sorted_indices]
    d = section_length
    cur = arr[0]
    val = max(arr)
    while cur < val:
        cur += section_length
        idxs.append(binary_search(arr, cur))

    print(idxs)
    print(max(arr), min(arr))

    filename_cut, _ = os.path.splitext(os.path.basename(filename))

    folder = os.path.join(os.getcwd(), "sections", filename_cut, str(section_length))
    if not os.path.exists(folder):
        os.makedirs(folder)


    # discard the last section
    for i in range(len(idxs)-1):
        start = idxs[i]
        end = idxs[i+1]

        section_points = points[start:end]

        outpath = os.path.join(folder, f"{filename_cut}_section_{i}.las")

        new_las = laspy.LasData(las.header)
        new_las.points = section_points
        new_las.write(outpath)

        print(f"Section {i} written to {outpath}...")

if __name__ == "__main__":
    main()