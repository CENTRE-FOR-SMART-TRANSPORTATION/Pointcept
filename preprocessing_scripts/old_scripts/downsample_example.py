import numpy as np
import laspy
import os
from tkinter import Tk, filedialog
import open3d as o3d

DESKTOP = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
VOXEL_SIZE = 0.5

def obtain_data_path():
    """
    Lets the user choose a file through the UI
    """
    # Manually obtain file via UI
    Tk().withdraw()
    las_filename = filedialog.askopenfilename(
        filetypes=[(".las files", "*.las")],
        initialdir=DESKTOP, title="Please select the data file",
    )

    print(f"You have chosen to open the point cloud:\n{las_filename}")

    return os.path.abspath(las_filename)


def voxel_downsample(points, voxel_size):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    intensity = points[:, 3]
    new_pointcloud, original_indices, _ = (o3d.geometry.PointCloud.voxel_down_sample_and_trace(
        point_cloud, voxel_size, point_cloud.get_min_bound(), point_cloud.get_max_bound(), False))

    new_intensity = []

    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        avg = np.mean(intensity[idx])
        new_intensity.append(avg)

    new_points = np.hstack(
        (new_pointcloud.points, np.array(new_intensity).reshape(-1, 1)))
    return new_points


def main():
    # Sample point cloud data (replace this with your actual point cloud data)
    las = laspy.read(obtain_data_path())
    points = np.vstack((las.x, las.y, las.z, las.intensity)).T

    selected_points = voxel_downsample(points, VOXEL_SIZE)

    with open("downsample.txt", "w") as f:
        for x, y, z, i in selected_points:
            f.write(f"{x},{y},{z},{i}\n")


if __name__ == "__main__":
    main()
