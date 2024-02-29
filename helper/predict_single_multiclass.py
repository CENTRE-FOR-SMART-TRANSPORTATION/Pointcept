import torch
from pointcept.models import build_model
from collections import OrderedDict
from torch.nn.functional import softmax
import os
import laspy
import numpy as np
import open3d as o3d
from collections import defaultdict
from random import randint
from tabulate import tabulate
import pandas as pd

print(torch.backends.cudnn.version())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.is_available())


torch.cuda.empty_cache()
model_saved = torch.load(
    '/home/gurveer/Desktop/Pointcept/exp/cstdataset/semseg-pt-v2m2-0-base-multiclass/model/model_best.pth')
single_sample = torch.load(
    '/home/gurveer/Desktop/datasets/preprocessed_new_norm_v2/train/03702E_C1R1_R1R1_18000_20000_section_24_sar_d_points_35.pth')
original = torch.load(
    '/home/gurveer/Desktop/datasets/preprocessed_50m_nonunified_orig/train/03702E_C1R1_R1R1_18000_20000_section_24_sar_d_points_35.pth')
folder = "/home/gurveer/Desktop/model_data/data/some_labels/las_files"

colors = dict()
colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255),
          3: (255, 255, 0), 4: (255, 0, 255), 5: (255, 255, 255), 6: (255, 255, 255)}
num_classes = 7
class_names = ['traffic-sign', 'delineator-post', 'wires', 'wooden-utility-pole', 'road', 'vegetation', 'clutter']

def print_matrix(matrix, filename):
    headers = ["", *class_names]
    data = [[class_names[i], *matrix[i]] for i in range(len(matrix))]
    table = tabulate(data, headers, tablefmt="grid")
    print(table)
    df = pd.DataFrame(data, columns=headers)
    df.to_excel(filename, index=False)
    
print(colors)


def intensity_normalize(intensity_arr):
    min = np.min(intensity_arr)
    intensity_arr = intensity_arr - min
    m = np.max(np.abs(intensity_arr))
    intensity_arr = intensity_arr / m
    return np.round(intensity_arr, 6)


def pc_normalize(pc):
    pc = pc.T
    min = np.min(pc, axis=0)
    for p in pc:
        if p[0] < min[0]:
            print("min x")
    pc = pc - min
    for p in pc:
        if p[0] < 0:
            print("really x")
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return np.round(pc, 6)


def farthest_point_downsample(points, numpoints):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.hstack((points[:, 3:], points[:, 3:], points[:, 3:])))
    new_pointcloud = point_cloud.farthest_point_down_sample(numpoints)
    new_points = np.hstack((np.asarray(new_pointcloud.points), np.asarray(
        new_pointcloud.colors)[:, 0].reshape(-1, 1)))
    return new_points


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


predictions_folder = os.path.join(
    os.path.expanduser('~'), 'Desktop', 'predictions_single')
if not os.path.exists(predictions_folder):
    os.makedirs(predictions_folder)


state_dict = model_saved["state_dict"]
model = build_model(dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m2",
        in_channels=4,
        num_classes=7,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.15, 0.375, 0.9375, 2.34375),  # x3, x2.5, x2.5, x2.5
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="map",  # map / interp
    ),
    criteria=[dict(type="FocalLoss", gamma=2.0, alpha=0.5,
                   loss_weight=1.0, ignore_index=-1)],
))


# for some reason this broke when it was working before
# new_state_dict = OrderedDict()
# for k in state_dict:
#     # module.a.b = a.b
#     new_k = k[7:]
#     new_state_dict[new_k] = state_dict[k]


model.load_state_dict(state_dict)

model.eval()

dev = 'cuda'
model = model.to(dev)
data_dict = OrderedDict()
data_dict["coord"] = torch.from_numpy(single_sample["coord"]).clone().to(
    torch.float).contiguous().detach().to(dev)
data_dict["feat"] = torch.from_numpy(np.vstack((single_sample["coord"][:, 0], single_sample["coord"][:, 1], single_sample["coord"]
                                     [:, 2], single_sample["intensity"][:, 0])).T).clone().to(torch.float).contiguous().detach().to(dev)
data_dict["offset"] = torch.tensor(
    [single_sample["coord"].shape[0]],  device=dev)
print('running')
with torch.no_grad():
    seg_logits = model(data_dict)['seg_logits']
    sm = softmax(seg_logits, -1)

labels = sm.max(1)[1].data.cpu().numpy()
ground_truth = single_sample["semantic_gt"]

matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
totals = [0 for _ in range(num_classes)]
for i in ground_truth:
    totals[i[0]] += 1
for i in range(len(totals)):
    if totals[i] == 0:
        totals[i] = -1

for gt, l in zip(ground_truth, labels):
    matrix[gt[0]][l] += 1

print_matrix(matrix, "multiclass.xlsx")
for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        matrix[i][j] /= totals[i]
        matrix[i][j] = round(matrix[i][j], 3)


print(matrix)
print(totals)
print_matrix(matrix, "multiclass.xlsx")
outfile = os.path.join(
    predictions_folder, f"section10_non_unified_lovasz_v2.txt")
with open(outfile, "w") as f:
    for c, l in zip(original["coord"], labels):
        f.write(
            f"{c[0]},{c[1]},{c[2]},{colors[l][0]},{colors[l][1]},{colors[l][2]}\n")
count = defaultdict(int)

for l in labels:
    count[l] += 1

print(count)
