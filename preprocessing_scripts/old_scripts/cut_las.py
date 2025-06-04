import laspy
import numpy as np
import math
import os
import argparse
from tkinter import Tk, filedialog

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
    gps_time = las.gps_time
    sorted_indices = np.argsort(gps_time)
    points = las.points[sorted_indices]
    total_points = len(points)

    # assuming the road is wider alogn x
    x_range = las.header.x_max - las.header.x_min

    points_per_metre = total_points / (x_range)

    points_per_section = math.ceil(points_per_metre * section_length)

    num_sections = math.ceil(total_points/points_per_section)


    filename_cut, _ = os.path.splitext(filename.split("/")[-1])

    folder = os.path.join(os.getcwd(), "sections", filename_cut, str(section_length))
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename_cut, _ = os.path.splitext(os.path.basename(filename))

    for i in range(num_sections):
        start = i*points_per_section
        end = min((i+1)*points_per_section, total_points)

        section_points = points[start:end]

        outpath = os.path.join(folder, f"{filename_cut}_section_{i}.las")

        new_las = laspy.LasData(las.header)
        new_las.points = section_points
        new_las.write(outpath)

        print(f"Section {i} written to {outpath}...")

if __name__ == "__main__":
    main()