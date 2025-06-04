import os
import numpy as np
import open3d as o3d
import argparse
from statistics import mode

colors = {
    0: [255, 255, 255],  # white, lane
    1: [0, 0, 255],      # blue, shoulder
    2: [128, 128, 0],    # olive, chevrons
    3: [255, 0, 255],    # purple, broken-line
    4: [0, 255, 255],    # cyan, solid-line
    5: [255, 165, 0],    # orange, arrows
    6: [0, 128, 0],      # green, vegetation
    7: [255, 0, 0],      # red, traffic-sign
    8: [128, 0, 128],    # magenta, highway-guardrails
    9: [255, 255, 0],    # yellow, concrete-barriers
    10: [0, 0, 0],       # black, light-pole
    11: [192, 192, 192]  # silver, clutter
}

VOXEL_SIZE = 0.10

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default=None, required=True,
                        help="The full path to the file that is to be split into sections.")
parser.add_argument("--outdir", type=str, default=None, required=True, help="The outfolder for the preprocessed files")

args = parser.parse_args()
folder = args.folder
outfolder = args.outdir

if not os.path.exists(outfolder):
    os.makedirs(outfolder)


class_names = ['lane', 'shoulder', 'chevrons', 'broken-line', 'solid-line', 'arrows', 'vegetation', 'traffic-sign', 'highway-guardrails', 'concrete-barriers', 'light-pole', 'clutter']
# classes = ['solid-line', 'traffic-sign', 'wooden-utility-pole', 'clutter', 'road', 'wires', 'delineator-post', 'broken-line', 'vegetation']
class2label = {cls: i for i, cls in enumerate(class_names)}

for section in os.listdir(folder):
    print(f"Working on {section}...")
    annotations = os.path.join(folder, section, "Annotations")

    all_points = []
    total = 0
    for file in os.listdir(annotations):
        label_name, _ = file.split("_")
        points = np.loadtxt(os.path.join(annotations, file), dtype=float).reshape([-1,8])
        label = np.repeat(class2label[label_name], points.shape[0]).reshape([-1,1])
        # print(points.shape, label_name)
        points = np.hstack((points, label))
        all_points.append(points)
        total += points.shape[0]

    all_points = np.vstack(all_points)
    print(total, all_points.shape)
    annotations = os.path.join(outfolder, section, "Annotations")
    if not os.path.exists(annotations):
        os.makedirs(annotations)


    with open(os.path.join(annotations, f"{section}.txt"), "w") as f:
        for t in all_points:
            x, y, z, l = t[0], t[1], t[2], t[-1]
            f.write(f"{x} {y} {z} {colors[l][0]} {colors[l][1]} {colors[l][2]}\n")

        #    "{x} {y} {z} {i} {r} {d} {g} {ig}\n")

