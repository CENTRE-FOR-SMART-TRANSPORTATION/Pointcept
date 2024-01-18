import torch
from pointcept.models import build_model
from collections import OrderedDict
from torch.nn.functional import softmax
import os
import laspy
import numpy as np
import open3d as o3d
from collections import defaultdict

print(torch.backends.cudnn.version())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.is_available())


torch.cuda.empty_cache()
model_saved = torch.load('/home/gurveer/Desktop/saved/model_best.pth')

folder = "/home/gurveer/Desktop/model_data/data/some_labels/las_files"

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

def farthest_point_downsample(points, numpoints):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(np.hstack((points[:, 3:],points[:, 3:],points[:, 3:])))
    new_pointcloud = point_cloud.farthest_point_down_sample(numpoints)
    new_points = np.hstack((np.asarray(new_pointcloud.points), np.asarray(new_pointcloud.colors)[:, 0].reshape(-1, 1)))
    return new_points

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

    new_points = np.hstack((new_pointcloud.points, np.array(new_intensity).reshape(-1, 1)))
    return new_points

for file in os.listdir(folder):
    filename = os.path.join(folder, file)
    las = laspy.read(filename)
    unlabelled_data = np.vstack((las.x, las.y, las.z, las.intensity)).T
    # unlabelled_data = voxel_downsample(unlabelled_data, 0.1)
    points_arr = np.vstack((unlabelled_data[:, 0], unlabelled_data[:, 1], unlabelled_data[:, 2]))
    intensity_arr = unlabelled_data[:, 3]
    points_arr = pc_normalize(points_arr)
    intensity_arr = intensity_normalize(intensity_arr)
    unlabelled_data = np.vstack((points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], intensity_arr)).T
    unlabelled_data = torch.from_numpy(unlabelled_data).to(torch.float).contiguous()
    # unlabelled_data = torch.randn((1000, 4), dtype=torch.float)
    print(unlabelled_data.shape)
    state_dict = model_saved["state_dict"]
    model = build_model(dict(
        type="DefaultSegmentor",
        backbone=dict(
            type="PointTransformer-Seg50",
            in_channels=4,
            num_classes=12,
        ),
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
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
    count = defaultdict(int)
    for l in labels:
        count[l] += 1
    print(count)
