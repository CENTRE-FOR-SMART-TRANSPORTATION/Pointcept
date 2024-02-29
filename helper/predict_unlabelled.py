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

print(torch.backends.cudnn.version())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.is_available())


torch.cuda.empty_cache()
model_saved = torch.load(
    'exp/cstdataset/semseg-pt-v2m2-0-base-multiclass/model/model_best.pth')

folder = "/home/gurveer/Desktop/model_data/50m/segments/03702E_C1R1_R1R1_18000_20000/las_files/"

colors = dict()
colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255),
          3: (255, 255, 0), 4: (255, 0, 255), 5: (255, 255, 255), 6: (100, 100, 100)}
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
    os.path.expanduser('~'), 'Desktop', 'predictions_newnorm')
if not os.path.exists(predictions_folder):
    os.makedirs(predictions_folder)

for file in os.listdir(folder):
    filename = os.path.join(folder, file)
    las = laspy.read(filename)
    unlabelled_data = np.vstack((las.x, las.y, las.z, las.intensity)).T
    unlabelled_data = voxel_downsample(unlabelled_data, 0.4)
    points_arr = np.vstack(
        (unlabelled_data[:, 0], unlabelled_data[:, 1], unlabelled_data[:, 2]))
    points = points_arr.copy()
    intensity_arr = unlabelled_data[:, 3]
    points_arr = pc_normalize(points_arr)
    intensity_arr = intensity_normalize(intensity_arr)
    unlabelled_data = np.vstack(
        (points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], intensity_arr)).T
    unlabelled_data = torch.from_numpy(
        unlabelled_data).to(torch.float).contiguous()
    # unlabelled_data = torch.randn((1000, 4), dtype=torch.float)
    print(unlabelled_data.shape)
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
    data_dict["coord"] = unlabelled_data[:, :3].clone().detach().to(dev)
    data_dict["feat"] = unlabelled_data.clone().detach().to(dev)
    data_dict["offset"] = torch.tensor([unlabelled_data.shape[0]],  device=dev)
    with torch.no_grad():
        seg_logits = model(data_dict)['seg_logits']
        sm = softmax(seg_logits, -1)
    labels = sm.max(1)[1].data.cpu().numpy()
    outfile = os.path.join(
        predictions_folder, f"{os.path.splitext(file)[0]}.txt")
    with open(outfile, "w") as f:
        for c, l in zip(points.T, labels):
            f.write(
                f"{c[0]},{c[1]},{c[2]},{colors[l][0]},{colors[l][1]},{colors[l][2]}\n")
    count = defaultdict(int)
    for l in labels:
        count[l] += 1
    print(filename)
    print(count)
