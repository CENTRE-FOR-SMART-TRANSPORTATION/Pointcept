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
WRITE_LAS = True
MAX_CLUTTER_POINTS = 50000
VOXEL_SIZE = 0.4
DOWNSAMPLE_LABELS = ["vegetation"]
MAX_LABEL_SIZE = 100000

def voxel_downsample(points, voxel_size):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    intensity = points[:, 3]
    new_pointcloud, original_indices, _ = (o3d.geometry.PointCloud.voxel_down_sample_and_trace(point_cloud, voxel_size, point_cloud.get_min_bound(), point_cloud.get_max_bound(), False))

    new_intensity = []

    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        avg = np.mean(intensity[idx])
        new_intensity.append(avg)
    # print(len(new_pointcloud.points))
    # print(len(original_indices))
    # print(len(new_intensity))
    # print(np.asarray(new_pointcloud.points).shape)
    # print(np.array(new_intensity).shape)
    new_points = np.hstack((new_pointcloud.points, np.array(new_intensity).reshape(-1, 1)))
    # print(new_points.shape)
    # print(new_points)
    return new_points


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

def main():
    # Sample point cloud data (replace this with your actual point cloud data)
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="The full path to the file that is to be split into sections.")

    args = parser.parse_args()
    data = args.folder if args.folder is not None else obtain_data_path()
    parsed = os.path.join(os.getcwd(), "parsed")
    las_check = os.path.join(os.getcwd(), "las_check")
    class_names = set()

    if WRITE_LAS and not os.path.exists(las_check):
        os.makedirs(las_check)
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
            # print(box_list)
            # print(centre)

            outfolder = os.path.join(parsed, filename_cut)
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            annotations = os.path.join(outfolder, "Annotations")
            if not os.path.exists(annotations):
                os.makedirs(annotations)
            
            # write the full file to .txt
            las = laspy.read(las_file)
            max_i = np.max(las.intensity)
            max_x = np.max(las.x)
            max_y = np.max(las.y)
            max_z = np.max(las.z)
            points = np.vstack((las.x, las.y, las.z, las.intensity)).T
            print("Number of points before voxel downsample", len(points))
            # points = voxel_downsample(points, VOXEL_SIZE)
            print("Number of points after voxel downsample", len(points))
            outfile = os.path.join(outfolder, f"{filename_cut}.txt")

            # print(f"Writing full file to {outfile}...")
            # with open(outfile, "w") as f:
            #     for x,y,z,i in points:
            #         f.write(f"{x-centre[0]} {y-centre[1]} {z-centre[2]} {(i)}\n")

            label_count = defaultdict(int)
            points_with_intensity = points
            points = np.vstack((points[:, 0], points[:, 1], points[:, 2]))

            cumulative_mask = np.zeros_like(points[0,:], dtype=bool)
            total_labelled_points = 0
            for box in box_list:
                if clean_label(box["obj_type"]) == "vegetation":
                    continue
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
                label_count[label_name] += 1
                outfile = os.path.join(annotations, f"{label_name}_{label_count[label_name]}.txt")
                info = np.vstack((selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], selected_points[:, 3]))

                print(f"Writing points for {label_name} to {outfile}...")
                with open(outfile, "w") as f:
                    for x,y,z,i in info.T:
                        f.write(f"{round((x)/max_x, 8)} {round((y)/max_y, 6)} {round((z)/max_z, 6)} {round((i)/max_i, 6)}\n")
                        total_labelled_points += 1

                cumulative_mask |= mask

            unlabeled_points = points_with_intensity[~cumulative_mask]
            # Randomly select up to 50,000 points if there are more than 50,000 points
            print("Labelled", total_labelled_points, "Unlabelled/Clutter", len(unlabeled_points))
            outfile = os.path.join(annotations, "clutter_0.txt")
            info = np.vstack((unlabeled_points[:, 0], unlabeled_points[:, 1], unlabeled_points[:, 2], unlabeled_points[:, 3]))
            # print(f"Writing unlabelled points to {outfile}...")
            # with open(outfile, "w") as f:
            #     for x,y,z,i in info.T:
                        # f.write(f"{(x)/max_x} {(y)/max_y} {(z)/max_z} {(i)/max_i}\n")

            for label in label_count:
                class_names.add(label)
    
    print("The following classes were found in all files")
    print(list(class_names))
    return

if __name__ == "__main__":
    main()