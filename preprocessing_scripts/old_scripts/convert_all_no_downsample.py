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

global_count = dict()
global_point_count = dict()

def intensity_normalize(intensity_arr):
    min = np.min(intensity_arr)
    intensity_arr = intensity_arr - min
    m = np.max(np.abs(intensity_arr))
    intensity_arr = intensity_arr / m
    return intensity_arr

def pc_normalize(pc):
    pc = pc.T
    min = np.min(pc, axis=0)
    for p in pc:
        if p[0] < min[0]:
            print("min x")
    pc = pc - min
    for p in pc:
        if p[0] <   0:
            print("really x")
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

   
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

def write_points_to_txt(outfile, points):
    with open(outfile, "w") as f:
        for x,y,z,i in points:
            f.write(f"{round(x,6)} {round(y,6)} {round(z,6)} {(round(i,6))}\n")
def main():
    # Sample point cloud data (replace this with your actual point cloud data)
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="The full path to the file that is to be split into sections.")

    args = parser.parse_args()
    data = args.folder if args.folder is not None else obtain_data_path()
    parsed = os.path.join(os.getcwd(), "parsed")
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
            if not os.path.exists(labels_file):
                print(f"No labels found for {filename}, moving on...")
                continue
            if not os.path.exists(centre_file):
                print(f"No centre found for {filename}, moving on...")
                continue
            with open(labels_file, "r") as f:
                box_list = json.load(f)
            with open(centre_file, "r") as f:
                centre = json.load(f)

            outfolder = os.path.join(parsed, filename_cut)
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            annotations = os.path.join(outfolder, "Annotations")
            if not os.path.exists(annotations):
                os.makedirs(annotations)
            
            # write the full file to .txt
            las = laspy.read(las_file)
            points = np.vstack((las.x, las.y, las.z, las.intensity)).T
            print(f"Total number of points in file", len(points))

            label_count = defaultdict(int)
            points_with_intensity = points
            points = np.vstack((points[:, 0], points[:, 1], points[:, 2]))

            cumulative_mask = np.zeros_like(points[0,:], dtype=bool)
            total_labelled_points = 0
            for box in box_list:
                psr = box["psr"]
                position = np.array([psr["position"]["x"], psr["position"]["y"], psr["position"]["z"]])
                rotation = np.array([psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"]])
                scale = np.array([psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"]])
                corners = compute_bounding_box_corners(position, rotation, scale)
                for c in corners:
                    c[0] += centre[0]
                    c[1] += centre[1]
                    c[2] += centre[2]
                mask = points_in_box(corners, points)
                selected_points = points_with_intensity[mask]

                label_name = clean_label(box["obj_type"])   
                if len(selected_points) == 0:
                    print(f"0 points found for {label_name}") 
                    continue

                label_count[label_name] += 1
                outfile = os.path.join(annotations, f"{label_name}_{label_count[label_name]}.txt")
                points_arr = np.vstack((selected_points[:, 0], selected_points[:, 1], selected_points[:, 2]))
                intensity_arr = selected_points[:, 3]
                points_arr = pc_normalize(points_arr)
                intensity_arr = intensity_normalize(intensity_arr)

                info = np.vstack((points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], intensity_arr))
                print(f"Writing points for {label_name} to {outfile}...")
                write_points_to_txt(outfile, info.T)

                total_labelled_points += info.T.shape[0]
                
                if label_name not in global_count:
                    global_count[label_name] = 0
                global_count[label_name] += 1

                if label_name not in global_point_count:
                    global_point_count[label_name] = 0
                global_point_count[label_name] += info.T.shape[0]

                cumulative_mask |= mask

            unlabeled_points = points_with_intensity[~cumulative_mask]
            print("Labelled", total_labelled_points, "Unlabelled/Clutter", len(unlabeled_points))
            outfile = os.path.join(annotations, "clutter_0.txt")
    
            points_arr = np.vstack((unlabeled_points[:, 0], unlabeled_points[:, 1], unlabeled_points[:, 2]))
            intensity_arr = unlabeled_points[:, 3]
            points_arr = pc_normalize(points_arr)
            intensity_arr = intensity_normalize(intensity_arr)
            info = np.vstack((points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], intensity_arr))
            print(f"Writing unlabelled points to {outfile}...")
           # write_points_to_txt(outfile, info.T)

            label_name = "clutter"
            if label_name not in global_count:
                global_count[label_name] = 0
            global_count[label_name] += 1

            if label_name not in global_point_count:
                global_point_count[label_name] = 0
            global_point_count[label_name] += info.T.shape[0]

            for label in label_count:
                class_names.add(label)
    
    print("The following classes were found in all files")
    print(list(class_names))
    print("Total number of each label")
    for k in global_count:
        print(k, global_count[k])
    print("Total points for each label")
    for k in global_point_count:
        print(k, global_point_count[k])

    # print nicely
    print(f"""
    Stop sign (Octagon)                     {global_point_count["stop-sign"]}
    Regulatory sign (Vertical Rectangle)    {global_point_count["regulatory-sign"]}
    Guide sign (Horizontal Rectangle)       {global_point_count["guide-sign"]}
    Warning sign (Diamond)      {global_point_count["warning-sign"]}
    Crossbuck (X shape)         {global_point_count["crossbuck"]}
    Highway Guardrails          {global_point_count["highway-guardrails"]}
    Transmission tower          {global_point_count["transmission-tower"]}
    Delineator post             {global_point_count["delineator-post"]}
    Wooden utility pole         {global_point_count["wooden-utility-pole"]}
    Wires                       {global_point_count["wires"]}
    Highway fence               {global_point_count["highway-fence"]}
    Fence                       {global_point_count["fence"]}
    Street lights               {global_point_count["street-lights"]}
    Vegetation                  {global_point_count["vegetation"]}
    Clutter                     {global_point_count["clutter"]}
    """)
    return

if __name__ == "__main__":
    main()