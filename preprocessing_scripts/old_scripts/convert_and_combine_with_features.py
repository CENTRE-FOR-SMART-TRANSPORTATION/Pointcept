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
import sys

from gendata import getSampleCloud2, getSamplePoly2, dataDir, isCoordEqual, createSymbolicLinks
import cloudComPy as cc

DESKTOP = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

ROUGHNESS_RADIUS = 1
DENSITY_RADIUS = 0.06

COMBINE_LABELS = ["lane_marking", "objects_vegetation",
                  "objects", "vegetation", "pavement"]

global_count = dict()
global_point_count = dict()

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

    return Rx @ Ry @ Rz


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

    v = points - p1.reshape(-1, 1)

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask


def clean_label(name):
    label_name = '-'.join([s.lower() for s in name.split()])
    return label_name


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
        for x, y, z, i, r, d, g in points:
            f.write(f"{round(x,15)} {round(y,15)} {round(z,15)} {(round(i,15))} {(round(r,15))} {(round(d,15))} {(round(g,15))}\n")


def get_label_points(name, scene, filename_cut, orig_points, cumulative_mask, 
annotations, label_count, class_names):
    total_labelled_points = 0
    las_file = os.path.join(
        scene, "las_files", f"{filename_cut}_{name}.las")
    labels = os.path.join(
        scene, "label", f"{filename_cut}_converted_{name}.json")
    centre_file = os.path.join(
        scene, "centres", f"{filename_cut}_converted_{name}_centre.json")
    with open(labels, "r") as f:
        box_list = json.load(f)
    las = laspy.read(las_file)

    cloud = cc.loadPointCloud(las_file)

    ret = cc.computeRoughness(ROUGHNESS_RADIUS, [cloud])

    if not ret:
        raise RuntimeError
    nsf = cloud.getNumberOfScalarFields()
    roughness = cloud.getScalarField(nsf-1)
    roughness = roughness.toNpArrayCopy()

    if np.isnan(roughness).any():
        roughness = np.nan_to_num(roughness)
        # assert False, "nan value in roughness"

    ret = cc.computeLocalDensity(
        cc.Density.DENSITY_KNN, DENSITY_RADIUS, [cloud])
    if not ret:
        raise RuntimeError

    nsf = cloud.getNumberOfScalarFields()
    density = cloud.getScalarField(nsf-1)
    density = density.toNpArrayCopy()

    # Z coordinate as a scalar Field
    ok = cloud.exportCoordToSF(False, False, True)
    nsf = cloud.getNumberOfScalarFields()
    cloud.computeScalarFieldGradient(nsf-1, 0.06, True)

    nsf = cloud.getNumberOfScalarFields()

    z_gradient = cloud.getScalarField(nsf-1)
    z_gradient = z_gradient.toNpArrayCopy()

    points_with_intensity = np.vstack(
        (las.x, las.y, las.z, las.intensity, roughness, density, z_gradient)).T
    points = np.vstack((las.x, las.y, las.z))
    seen = set()
    with open(centre_file, "r") as f:
        global_centre = json.load(f)

    outer_seen = set()
    for x, y, z, i, r, d, g in np.round(orig_points[cumulative_mask], 6):
        outer_seen.add((x, y, z, i, r, d, g))

    label_mask = np.zeros_like(points[0, :], dtype=bool)

    for idx, t in enumerate(np.round(points_with_intensity, 6)):
        x, y, z, i, r, d, g = t
        if (x, y, z, i, r, d, g) in outer_seen:
            label_mask[idx] = True

    for x, y, z, i, r, d, g in np.round(points_with_intensity, 6):
        seen.add((x, y, z, i, r, d, g))

    # so that we don't select the same points later from the full file
    for idx, t in enumerate(np.round(orig_points, 6)):
        x, y, z, i, r, d, g = t
        if (x, y, z, i, r, d, g) in seen:
            cumulative_mask[idx] = True

    for box in box_list:
        psr = box["psr"]
        position = np.array(
            [psr["position"]["x"], psr["position"]["y"], psr["position"]["z"]])
        rotation = np.array(
            [psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"]])
        scale = np.array(
            [psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"]])
        corners = compute_bounding_box_corners(
            position, rotation, scale)
        for c in corners:
            c[0] += global_centre[0]
            c[1] += global_centre[1]
            c[2] += global_centre[2]
        mask = points_in_box(corners, points)
        mask[label_mask] &= False  # xor
        selected_points = points_with_intensity[mask]

        label_name = clean_label(box["obj_type"])
        if len(selected_points) == 0:
            print(f"0 points found for {label_name}")
            continue

        label_count[label_name] += 1
        outfile = os.path.join(
            annotations, f"{label_name}_{label_count[label_name]}.txt")

        points_arr = np.vstack(
            (selected_points[:, 0], selected_points[:, 1], selected_points[:, 2])).T
        print(selected_points.shape)
        intensity_arr = selected_points[:, 3]
        roughness_arr = selected_points[:, 4]
        density_arr = selected_points[:, 5]
        z_gradient_arr = selected_points[:, 6]

        info = np.vstack(
            (points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], intensity_arr, roughness_arr, density_arr, z_gradient_arr))
        print(f"Writing points for {label_name} to {outfile}...")
        write_points_to_txt(outfile, info.T)
        if label_name != "clutter":
            total_labelled_points += info.T.shape[0]

        if label_name not in global_count:
            global_count[label_name] = 0
        global_count[label_name] += 1

        if label_name not in global_point_count:
            global_point_count[label_name] = 0
        global_point_count[label_name] += info.T.shape[0]

        label_mask |= mask

    unlabeled_points = points_with_intensity[~label_mask]
    if len(unlabeled_points) > 0:
        print("Labelled", total_labelled_points,
              "Unlabelled/Clutter", len(unlabeled_points))
        label_count["clutter"] += 1
        outfile = os.path.join(
            annotations, f'{"clutter"}_{label_count["clutter"]}.txt')

        points_arr = np.vstack(
            (selected_points[:, 0], selected_points[:, 1], selected_points[:, 2])).T
        intensity_arr = selected_points[:, 3]
        roughness_arr = selected_points[:, 4]
        density_arr = selected_points[:, 5]
        z_gradient_arr = selected_points[:, 6]

        info = np.vstack(
            (points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], intensity_arr, roughness_arr, density_arr, z_gradient_arr))
        print(f"Writing unlabelled points to {outfile}...")
        write_points_to_txt(outfile, info.T)

        label_name = "clutter"
        if label_name not in global_count:
            global_count[label_name] = 0
        global_count[label_name] += 1

        if label_name not in global_point_count:
            global_point_count[label_name] = 0
        global_point_count[label_name] += info.T.shape[0]

    for label in label_count:
        class_names.add(label)

    return total_labelled_points


def main():
    # Sample point cloud data (replace this with your actual point cloud data)
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None,
                        help="The full path to the file that is to be split into sections.")

    args = parser.parse_args()
    data = args.folder if args.folder is not None else obtain_data_path()
    parsed = os.path.join(os.getcwd(), "parsed")
    las_check = os.path.join(os.getcwd(), "las_check")
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
            labels_file = os.path.join(
                scene, "label", f"{filename_cut}_converted.json")
            centre_file = os.path.join(
                scene, "centres", f"{filename_cut}_converted_centre.json")

            next = False
            for label in COMBINE_LABELS:
                if label in filename:
                    next = True
            if next:
                print(f"{filename} is not the original file, moving on...")
                continue

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

            cloud = cc.loadPointCloud(las_file)

            ret = cc.computeRoughness(ROUGHNESS_RADIUS, [cloud])
            if not ret:
                raise RuntimeError
            nsf = cloud.getNumberOfScalarFields()
            roughness = cloud.getScalarField(nsf-1)
            roughness = roughness.toNpArrayCopy()

            if np.isnan(roughness).any():
                roughness = np.nan_to_num(roughness)

            ret = cc.computeLocalDensity(
                cc.Density.DENSITY_KNN, DENSITY_RADIUS, [cloud])
            if not ret:
                raise RuntimeError

            nsf = cloud.getNumberOfScalarFields()
            density = cloud.getScalarField(nsf-1)
            density = density.toNpArrayCopy()

            # Z coordinate as a scalar Field
            ok = cloud.exportCoordToSF(False, False, True)
            nsf = cloud.getNumberOfScalarFields()
            cloud.computeScalarFieldGradient(nsf-1, 0.06, True)

            nsf = cloud.getNumberOfScalarFields()

            z_gradient = cloud.getScalarField(nsf-1)
            z_gradient = z_gradient.toNpArrayCopy()

            points = np.vstack(
                (las.x, las.y, las.z, las.intensity, roughness, density, z_gradient)).T

            print(
                f"Total number of points", len(points))
            # points = downsample(points)
            outfile = os.path.join(outfolder, f"{filename_cut}.txt")
            # write_points_to_txt(outfile, points)

            label_count = defaultdict(int)
            label_count["clutter"] += 1
            points_with_intensity = points
            points = np.vstack((points[:, 0], points[:, 1], points[:, 2]))

            cumulative_mask = np.zeros_like(points[0, :], dtype=bool)
            total_labelled_points = 0

            for label in COMBINE_LABELS:
                if os.path.exists(os.path.join(las_files, f"{filename_cut}_{label}.las")):
                    t = get_label_points(
                        label, scene, filename_cut, points_with_intensity, cumulative_mask, annotations, 
                        label_count, class_names)
                    total_labelled_points += t
            for box in box_list:
                psr = box["psr"]
                position = np.array(
                    [psr["position"]["x"], psr["position"]["y"], psr["position"]["z"]])
                rotation = np.array(
                    [psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"]])
                scale = np.array(
                    [psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"]])
                corners = compute_bounding_box_corners(
                    position, rotation, scale)
                for c in corners:
                    c[0] += centre[0]
                    c[1] += centre[1]
                    c[2] += centre[2]
                mask = points_in_box(corners, points)
                mask[cumulative_mask] &= False  # xor
                selected_points = points_with_intensity[mask]

                label_name = clean_label(box["obj_type"])
                if len(selected_points) == 0:
                    print(f"0 points found for {label_name}")
                    continue

                label_count[label_name] += 1
                outfile = os.path.join(
                    annotations, f"{label_name}_{label_count[label_name]}.txt")

                points_arr = np.vstack(
                    (selected_points[:, 0], selected_points[:, 1], selected_points[:, 2])).T
                intensity_arr = selected_points[:, 3]
                roughness_arr = selected_points[:, 4]
                density_arr = selected_points[:, 5]
                z_gradient_arr = selected_points[:, 6]

                info = np.vstack(
                    (points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], intensity_arr, roughness_arr, density_arr, z_gradient_arr))
                print(f"Writing points for {label_name} to {outfile}...")
                write_points_to_txt(outfile, info.T)
                if label_name in FIXED_LABEL:
                    total_labelled_points += info.T.shape[0]

                if label_name not in global_count:
                    global_count[label_name] = 0
                global_count[label_name] += 1

                if label_name not in global_point_count:
                    global_point_count[label_name] = 0
                global_point_count[label_name] += info.T.shape[0]

                cumulative_mask |= mask

            unlabeled_points = points_with_intensity[~cumulative_mask]
            print("Labelled", total_labelled_points,
                  "Unlabelled/Clutter", len(unlabeled_points))
            label_count["clutter"] += 1
            outfile = os.path.join(
                annotations, f'{"clutter"}_{label_count["clutter"]}.txt')

            points_arr = np.vstack(
                (unlabeled_points[:, 0], unlabeled_points[:, 1], unlabeled_points[:, 2])).T
            intensity_arr = unlabeled_points[:, 3]
            roughness_arr = unlabeled_points[:, 4]
            density_arr = unlabeled_points[:, 5]
            z_gradient_arr = unlabeled_points[:, 6]

            info = np.vstack(
                (points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], intensity_arr, roughness_arr, density_arr, z_gradient_arr))
            print(f"Writing unlabelled points to {outfile}...")
            write_points_to_txt(outfile, info.T)

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
    return


if __name__ == "__main__":
    main()
