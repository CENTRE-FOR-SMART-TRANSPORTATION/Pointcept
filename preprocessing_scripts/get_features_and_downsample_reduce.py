import os
import numpy as np
import open3d as o3d
import argparse
from statistics import mode

from gendata import getSampleCloud2, getSamplePoly2, dataDir, isCoordEqual, createSymbolicLinks
import cloudComPy as cc

ROUGHNESS_RADIUS = 1
DENSITY_RADIUS = 1
Z_GRADIENT_RADIUS = 0.2
INTENSITY_GRADIENT_RADIUS = 0.875

VOXEL_SIZE = 0.1

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default=None, required=True,
                        help="The full path to the file that is to be split into sections.")
parser.add_argument("--outdir", type=str, default=None, required=True, help="The outfolder for the preprocessed files")

args = parser.parse_args()
folder = args.folder
outfolder = args.outdir

if not os.path.exists(outfolder):
    os.makedirs(outfolder)


classes = ['concrete-barriers', 'wires', 'traffic-sign', 'clutter', 'lane', 'shoulder', 'light-pole', 'vegetation', 'broken-line', 'solid-line', 'traffic-cones','highway-guardrails', 'chevrons', 'arrows', 'delineator-post']
# classes = ['concrete-barriers', 'traffic-sign', 'clutter', 'pavement', 'light-pole', 'vegetation', 'broken-line', 'solid-line','highway-guardrails', 'chevrons', 'arrows']
class2label = {cls: i for i, cls in enumerate(classes)}

def normalise(points):
    const = 100000
    pc = np.vstack((points[:,0], points[:,1], points[:,2])).T
    pc_min = np.min(pc, axis=0)
    pc_m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc - pc_min
    pc = pc / pc_m
    pc *= const

    intensity = points[:,3]
    intensity_min = np.min(intensity)
    intensity_m = np.max(np.abs(intensity))
    intensity = intensity - intensity_min
    intensity = intensity / intensity_m

    roughness = points[:,4]
    roughness_min = np.nanmin(roughness)
    roughness_m = np.nanmax(np.abs(roughness))
    roughness = roughness - roughness_min
    roughness = roughness / roughness_m

    if np.isnan(roughness).any():
        roughness = np.nan_to_num(roughness, nan=0, posinf=1, neginf=0)
    density = points[:,5]
    density_min = np.nanmin(density)
    density_m = np.nanmax(np.abs(density))
    density = density - density_min
    density = density / density_m

    if np.isnan(density).any():
        density = np.nan_to_num(density, nan=0, posinf=1, neginf=0)

    z_gradient = points[:,6]
    # z_gradient_min = np.min(z_gradient)
    # z_gradient_m = np.max(np.abs(z_gradient))
    # z_gradient = z_gradient - z_gradient_min
    # z_gradient = z_gradient / z_gradient_m

    if np.isnan(z_gradient).any():
        z_gradient = np.nan_to_num(z_gradient, nan=0, posinf=1, neginf=0)

    intensity_gradient = points[:, 7]
    intensity_gradient_min = np.nanmin(intensity_gradient)
    intensity_gradient_m = np.nanmax(np.abs(intensity_gradient))
    print(intensity_gradient, intensity_gradient_min, intensity_gradient_m)
    intensity_gradient = intensity_gradient - intensity_gradient_min
    intensity_gradient = intensity_gradient / intensity_gradient_m
    print(intensity_gradient)
    if np.isnan(intensity_gradient).any():
        intensity_gradient = np.nan_to_num(intensity_gradient, nan=0, posinf=1, neginf=0)

    print(pc.shape, intensity.shape)

    return np.vstack((pc[:, 0], pc[:, 1], pc[:, 2], intensity, roughness, density, z_gradient, intensity_gradient, points[:, 8])).T

def voxel_downsample(points, voxel_size):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    min_bound = np.min(points[:, :3], axis=0)
    max_bound = np.max(points[:, :3], axis=0)
    intensity = points[:, 3]
    roughness = points[:, 4] 
    density = points[:, 5]
    z_gradient = points[:, 6] 
    intensity_gradient = points[:, 7]
    labels = points[:, 8]
    new_pointcloud, original_indices, _ = (o3d.geometry.PointCloud.voxel_down_sample_and_trace(
        point_cloud, voxel_size, min_bound, max_bound, False))
    new_intensity = []
    new_roughness = []
    new_density = []
    new_z_gradient = []
    new_intensity_gradient = []
    new_label = []

    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        avg = np.mean(intensity[idx])
        new_intensity.append(avg)

    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        avg = np.mean(roughness[idx])
        new_roughness.append(avg)

    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        avg = np.mean(density[idx])
        new_density.append(avg)

    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        avg = np.mean(z_gradient[idx])
        new_z_gradient.append(avg)

    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        avg = np.mean(intensity_gradient[idx])
        new_intensity_gradient.append(avg)
    
    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        point_labels = labels[idx]
        avg = mode(point_labels)
        new_label.append(avg)

    new_points = np.hstack(
        (new_pointcloud.points, 
        np.array(new_intensity).reshape(-1, 1),
        np.array(new_roughness).reshape(-1, 1),
        np.array(new_density).reshape(-1, 1),
        np.array(new_z_gradient).reshape(-1, 1),
        np.array(new_intensity_gradient).reshape(-1, 1),
        np.array(new_label).reshape(-1, 1)))

    print(new_points.shape)
    return new_points

for section in os.listdir(folder):
    print(f"Working on {section}...")
    annotations = os.path.join(folder, section, "Annotations")

    all_points = []
    total = 0
    for file in os.listdir(annotations):
        label_name, _ = file.split("_")
        points = np.loadtxt(os.path.join(annotations, file), dtype=float).reshape([-1,4])
        label = np.repeat(class2label[label_name], points.shape[0]).reshape([-1,1])
        # print(points.shape, label_name)
        points = np.hstack((points, label))
        all_points.append(points)
        total += points.shape[0]
    all_points = np.vstack(all_points)

    with open("temp.xyz", "w") as f:
        for p in all_points:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")

    cloud = cc.loadPointCloud("temp.xyz")

    nsf = cloud.getNumberOfScalarFields()
    cloud.computeScalarFieldGradient(nsf-1, INTENSITY_GRADIENT_RADIUS, False)

    nsf = cloud.getNumberOfScalarFields()

    intensity_gradient = cloud.getScalarField(nsf-1)
    intensity_gradient = intensity_gradient.toNpArrayCopy()

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
    cloud.computeScalarFieldGradient(nsf-1, Z_GRADIENT_RADIUS, True)

    nsf = cloud.getNumberOfScalarFields()

    z_gradient = cloud.getScalarField(nsf-1)
    z_gradient = z_gradient.toNpArrayCopy()

    all_points = np.vstack((all_points[:, 0], all_points[:, 1], all_points[:, 2], all_points[:, 3], roughness, density, z_gradient, intensity_gradient, all_points[:, 4])).T

    all_points = voxel_downsample(all_points, VOXEL_SIZE)
    print(total, all_points.shape)
    print(section)
    all_points = normalise(all_points)
    annotations = os.path.join(outfolder, section, "Annotations")
    if not os.path.exists(annotations):
        os.makedirs(annotations)

    # with open(os.path.join(annotations, f"hello.txt"), "w") as f:
    #     for x,y,z,i,r,d,g in points:
    #         f.write(f"{x} {y} {z} {i} {r} {d} {g}\n")
    count = 2
    marking_count = 1
    pavement_count = 1
    clutter_labels = ["wires", "delineator-post", "traffic-cones"]
    marking_labels =  [ 'broken-line', 'solid-line', 'chevrons', 'arrows']
    pavement_labels = ['lane', 'shoulder']
    for name in class2label:
        num = class2label[name]
        points = all_points[np.where(all_points[:, 8] == num)]
        if len(points) > 0:
            if name in clutter_labels:
                print(f"{name} converted to clutter")
                with open(os.path.join(annotations, f"clutter_{count}.txt"), "w") as f:
                    for x,y,z,i,r,d,g,ig,l in points:
                        f.write(f"{x} {y} {z} {i} {r} {d} {g} {ig}\n") 
                count += 1 
            elif name in marking_labels:
                print(f"{name} converted to marking")
                with open(os.path.join(annotations, f"marking_{marking_count}.txt"), "w") as f:
                    for x,y,z,i,r,d,g,ig,l in points:
                        f.write(f"{x} {y} {z} {i} {r} {d} {g} {ig}\n") 
                marking_count += 1 
            elif name in pavement_labels:
                print(f"{name} converted to pavement")
                with open(os.path.join(annotations, f"pavement_{pavement_count}.txt"), "w") as f:
                    for x,y,z,i,r,d,g,ig,l in points:
                        f.write(f"{x} {y} {z} {i} {r} {d} {g} {ig}\n") 
                pavement_count += 1 
            else:       
                with open(os.path.join(annotations, f"{name}_1.txt"), "w") as f:
                    for x,y,z,i,r,d,g,ig,l in points:
                        f.write(f"{x} {y} {z} {i} {r} {d} {g} {ig}\n")
