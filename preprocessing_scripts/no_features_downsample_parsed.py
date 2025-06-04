import os
import numpy as np
import open3d as o3d
import argparse
from statistics import mode

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


classes = ['concrete-barriers', 'wires', 'traffic-sign', 'clutter', 'shoulder', 'lane', 'light-pole', 'vegetation', 'broken-line', 'solid-line', 'traffic-cones','highway-guardrails', 'chevrons', 'arrows', 'delineator-post']
# classes = ['solid-line', 'traffic-sign', 'wooden-utility-pole', 'clutter', 'road', 'wires', 'delineator-post', 'broken-line', 'vegetation']
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

    print(pc.shape, intensity.shape)
    return np.vstack((pc[:, 0], pc[:, 1], pc[:, 2], intensity, points[:, 4])).T

def voxel_downsample(points, voxel_size):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    min_bound = np.min(points[:, :3], axis=0)
    max_bound = np.max(points[:, :3], axis=0)
    intensity = points[:, 3]
    labels = points[:, 4]
    new_pointcloud, original_indices, _ = (o3d.geometry.PointCloud.voxel_down_sample_and_trace(
        point_cloud, voxel_size, min_bound, max_bound, False))
    new_intensity = []
    new_label = []

    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        avg = np.mean(intensity[idx])
        new_intensity.append(avg)
    
    for vec in original_indices:
        idx = [x for x in vec if x != -1]
        point_labels = labels[idx]
        avg = mode(point_labels)
        new_label.append(avg)

    new_points = np.hstack(
        (new_pointcloud.points, np.array(new_intensity).reshape(-1, 1), np.array(new_label).reshape(-1, 1)))
    print(new_points.shape)
    return new_points

for section in os.listdir(folder):
    print(f"Working on {section}...")
    annotations = os.path.join(folder, section, "Annotations")

    all_points = []
    total = 0
    for file in os.listdir(annotations):
        label_name, _ = file.split("_")
        if label_name == "highway-guardrail":
        	label_name = "highway-guardrails"
        if label_name == 'arrow':
        	label_name = "arrows"
       	if label_name == 'chevron':
       		label_name = 'chevrons'
        points = np.loadtxt(os.path.join(annotations, file), dtype=float).reshape([-1,4])
        label = np.repeat(class2label[label_name], points.shape[0]).reshape([-1,1])
        # print(points.shape, label_name)
        points = np.hstack((points, label))
        all_points.append(points)
        total += points.shape[0]
    all_points = np.vstack(all_points)
    all_points = voxel_downsample(all_points, VOXEL_SIZE)
    print(total, all_points.shape)
    print(section)
    all_points = normalise(all_points)
    annotations = os.path.join(outfolder, section, "Annotations")
    if not os.path.exists(annotations):
        os.makedirs(annotations)
        
    count = 2
    #marking_count = 1
    #pavement_count = 1
    clutter_labels = ["wires", "delineator-post", "traffic-cones"]
    #marking_labels =  [ 'broken-line', 'solid-line', 'chevrons', 'arrows']
    #pavement_labels = ['lane', 'shoulder']
    for name in class2label:
        num = class2label[name]
        points = all_points[np.where(all_points[:, 4] == num)]
        if len(points) > 0:
            if name in clutter_labels:
                print(f"{name} converted to clutter")
                with open(os.path.join(annotations, f"clutter_{count}.txt"), "w") as f:
                    for x,y,z,i,l in points:
                        f.write(f"{x} {y} {z} {i}\n") 
                count += 1 
    #        elif name in marking_labels:
    #            print(f"{name} converted to marking")
    #            with open(os.path.join(annotations, f"marking_{marking_count}.txt"), "w") as f:
    #                for x,y,z,i,l in points:
    #                    f.write(f"{x} {y} {z} {i}\n") 
    #            marking_count += 1 
    #        elif name in pavement_labels:
    #            print(f"{name} converted to pavement")
    #           with open(os.path.join(annotations, f"pavement_{pavement_count}.txt"), "w") as f:
    #                for x,y,z,i,l in points:
    #                    f.write(f"{x} {y} {z} {i}\n") 
    #            pavement_count += 1 
            else:       
                with open(os.path.join(annotations, f"{name}_1.txt"), "w") as f:
                    for x,y,z,i,l in points:
                        f.write(f"{x} {y} {z} {i}\n")
    
