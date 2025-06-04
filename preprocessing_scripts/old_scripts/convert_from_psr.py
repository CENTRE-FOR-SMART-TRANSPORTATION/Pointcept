import numpy as np
import laspy
import math
import os
import argparse
from tkinter import Tk, filedialog
import pandas as pd
from copy import deepcopy

DESKTOP = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

def rotate_matrix(theta_x, theta_y, theta_z):
    """
    Create a 3D rotation matrix based on rotation angles around x, y, and z axes.
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def compute_bounding_box_corners(position, rotation, scale):
    # Define local coordinates of the bounding box corners
    local_corners = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ])

    # Apply scale
    scaled_corners = local_corners * scale

    # Create rotation matrix
    rotation_matrix = rotate_matrix(*rotation)

    # Apply rotation
    rotated_corners = np.dot(rotation_matrix, scaled_corners.T).T

    # Apply translation
    final_corners = rotated_corners + position

    return final_corners

def points_in_box(corners, points):
    """
    Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
    p1 = np.array(corners[0])
    p_x = np.array(corners[1])
    p_y = np.array(corners[2])
    p_z = np.array(corners[4])

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape(-1,1)

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask

def main():
    # Sample point cloud data (replace this with your actual point cloud data)
    las = laspy.read("/Users/Gurveer/Desktop/CST/repos/cst-pointcloud-annotation_web-2023/data/comparison/las_files/00248N_C1R1_38000_40000_section_3.las")
    points = np.vstack((las.x, las.y, las.z))

    psr = box_list[0]["psr"]
    position = np.array([psr["position"]["x"], psr["position"]["y"], psr["position"]["z"]])
    rotation = np.array([psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"]])
    scale = np.array([psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"]])

    corners = compute_bounding_box_corners(position, rotation, scale)
    for c in corners:
      c[0] += centre[0]
      c[1] += centre[1]
      c[2] += centre[2]

    mask = points_in_box(corners, points)

    selected_points = las.points[mask]
    new_las = laspy.LasData(las.header)
    new_las.points = selected_points
    new_las.write("points.las")
    info = np.vstack((new_las.x, new_las.y, new_las.z, new_las.intensity))
    with open("points.txt", "w") as f:
      for x,y,z,i in info.T:
        f.write(f"{x},{y},{z},{i}\n")

if __name__ == "__main__":
    main()