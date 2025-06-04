import laspy
import numpy as np
import os

file = os.path.join(os.path.expanduser("~"), "Desktop", "las_files", "03702E_C1R1_R1R1_18000_20000_section_12_sar_d_points_35_pavement.las")
las = laspy.read(file)

points = np.vstack((las.x, las.y, las.z)).T
with open("points.txt", "w") as f:
    for x, y, z in points:
        f.write(f"{x},{y},{z}\n")