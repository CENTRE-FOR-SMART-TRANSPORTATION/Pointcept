import laspy
import numpy as np
import os
import argparse
from tkinter import Tk, filedialog
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import matplotlib.pyplot as plt
from collections import defaultdict
import random

DESKTOP = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

import numpy as np
from sklearn.neighbors import NearestNeighbors
    
def obtain_las_path():
    """
    Lets the user choose a file through the UI
    """
    Tk().withdraw()
    las_filename = filedialog.askopenfilename(
        filetypes=[(".las files", "*.las"), ("All files", "*")],
        initialdir=DESKTOP,
        title="Please select the main point cloud",
    )

    print(f"You have chosen to open the point cloud:\n{las_filename}")

    return os.path.abspath(las_filename)

def cluster_points(points, epsilon=1.0, min_pts=50):
    # Standardize the features (important for DBSCAN)
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
    labels = dbscan.fit_predict(points_scaled)

    return labels

def filter_points(points, p_height=0.9, p_intensity=0.1):
    # Sort points by height (z-axis)
    sorted_indices_height = points[:, 2].argsort()
    sorted_points_height = points[sorted_indices_height]

    print("Number of points before height filtering...", len(sorted_points_height))
    # Keep the lowest p_height% based on height
    num_to_keep = int((p_height) * len(sorted_points_height))
    height_filtered_points = sorted_points_height[num_to_keep:]
    print("Number of points after height filtering...", len(height_filtered_points), num_to_keep, p_height * len(sorted_points_height))
    # Sort points by intensity
    sorted_indices_intensity = (-height_filtered_points[:, 3]).argsort()
    sorted_points_intensity = height_filtered_points[sorted_indices_intensity]

    # Keep the highest p_intensity% based on intensity
    num_to_keep = int(len(height_filtered_points))
    filtered_points = sorted_points_intensity[:num_to_keep]
    print("Number of points after intensity filtering...", len(filtered_points))

    return filtered_points


def generate_label_colors(labels):
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(labels)))
    label_colors = {label: tuple([random.randint(0,255), random.randint(0,255), random.randint(0,255)]) for label in labels}
    return label_colors

def save_to_las(points, cluster_labels, filename="output.txt"):
    labels = set()
    count = defaultdict(int)

    for i in cluster_labels:
        count[i] += 1
    
    for label in count:
        if count[label] < 150:
            labels.add(label)
        else:
            labels.add(label)
            print("removed", label)

    colors = generate_label_colors(labels)

    print(colors)

    with open(filename, "w") as f:
        for idx, (x, y, z, i) in enumerate(points):
            # Assign a unique color to each cluster
            if cluster_labels[idx] in labels:
                cluster_color = colors[label]
                f.write(f"{x},{y},{z},{cluster_color[0]},{cluster_color[1]},{cluster_color[2]}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="The full path to the file that is to be split into sections.")
    parser.add_argument("--ph", type=int, default=90, help="The bottom percent of height to keep")
    parser.add_argument("--pi", type=int, default=1, help="The top percent of intensity to keep")
    parser.add_argument("--epsilon", type=float, default=1.0, help="DBSCAN epsilon parameter")
    parser.add_argument("--min_pts", type=int, default=5, help="DBSCAN minPts parameter")

    args = parser.parse_args()
    p = args.ph
    p = 0 if p < 0 else 100 if p > 100 else p
    p /= 100
    args.ph = p

    p = args.pi
    p = 0 if p < 0 else 100 if p > 100 else p
    p /= 100
    args.pi = p

    filename = args.file
    if filename is None:
        filename = obtain_las_path()

    las = laspy.read(filename)
    points = np.vstack((las.x, las.y, las.z, las.intensity)).T

    # filter by height and intensity
    filtered_points = filter_points(points, args.ph, args.pi)

    print(len(filtered_points))
    # cluster the points
    # cluster_labels = cluster_points(filtered_points, epsilon=args.epsilon, min_pts=args.min_pts)
    # print(set(cluster_labels))


    save_to_las(filtered_points, [0 for i in filtered_points], f"output_e_{args.epsilon}_m_{args.min_pts}.txt")

    # draw the bounding boxes

    # save them all in a file

if __name__ == "__main__":
    main()
