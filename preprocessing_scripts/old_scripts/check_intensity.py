import laspy
import numpy as np
import math
import os
import argparse
from tkinter import Tk, filedialog
import pandas as pd
from copy import deepcopy

# Open our .las point cloud into memory
def obtain_las_path():
    """
    Lets the user choose a file through the UI
    """
    # Manually obtain file via UI
    Tk().withdraw()
    las_filename = filedialog.askopenfilename(
        filetypes=[(".las files", "*.las"), ("All files", "*")],
        initialdir=os.getcwd(),
        title="Please select the main point cloud",
    )

    print(f"You have chosen to open the point cloud:\n{las_filename}")

    return os.path.abspath(las_filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="The full path to the file that is to be split into sections.")
    args = parser.parse_args()

    filename = args.file
    if filename is None:
        filename = obtain_las_path()

    las = laspy.read(filename)

    x, y, z = las.xyz[:, 0], las.xyz[:, 1], las.xyz[:, 2]
    intensity = las.intensity
    with open(f"{os.path.splitext(os.path.basename(filename))[0]}.txt", "w") as f:
        for a, b, c, i in zip(x, y, z, intensity):
            f.write(f"{a},{b},{c},{i}\n")

if __name__ == "__main__":
    main()