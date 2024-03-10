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
colors = dict()
# colors = {
#     0: [255,255,0], # yellow, solid-line
#     1: [0,255,0], # green, traffic-sign
#     2: [0,0,255], # blue, wooden-utility-pole
#     3: [255,0,0], # red, clutter
#     4: [255,255,255], # white, road
#     5: [0,0,0],     # black, wires
#     6: [0,0,128],   # light blue, delineator post
#     7: [255,0,255], # purple, broken-line
#     8: [0,255,255]  # cyan, vegetatoin
# }
colors = {
    0: [255,255,0], # yellow, solid-edge-line
    1: [0,255,0], # green, dashed-lane-line
    2: [0,0,255], # blue, gore-area
    3: [255,0,0], # red, vegetation
    4: [255,255,255], # white, shoulder
    5: [0,0,0],     # black, clutter
    6: [0,0,128],   # light blue, traffic-sign
    7: [255,0,255], # purple, light-pole
    8: [0,255,255],  # cyan, concrete-barriers
    9: [128,0,128],  # dark purple, lane
}

num_classes = 10
# class_names = ['traffic-sign', 'delineator-post', 'wires', 'wooden-utility-pole', 'road', 'vegetation', 'clutter']
# class_names = ['solid-line', 'traffic-sign', 'wooden-utility-pole', 'clutter', 'road', 'wires', 'delineator-post', 'broken-line', 'vegetation']
class_names = ['solid-edge-line', 'dashed-lane-line', 'gore-area', 'vegetation', 'shoulder', 'clutter', 'traffic-sign', 'light-pole', 'concrete-barriers', 'lane']
def print_matrix(matrix, filename):
    headers = ["", *class_names]
    data = [[class_names[i], *matrix[i]] for i in range(len(matrix))]
    table = tabulate(data, headers, tablefmt="grid")
    print(table)
    df = pd.DataFrame(data, columns=headers)
    df.to_excel(filename, index=False)
    
print(colors)

predictions_folder = os.path.join(
    os.path.expanduser('~'), 'Desktop', 'predictions_single')
if not os.path.exists(predictions_folder):
    os.makedirs(predictions_folder)

model_saved = torch.load(
    '/home/gurveer/Desktop/saved/exp_different_selective/cstdataset/combined_config_different/model/model_best.pth')
folder = "/home/gurveer/Desktop/datasets/preprocessed_different_selective/test/"

state_dict = model_saved["state_dict"]
model = build_model(dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m2",
        in_channels=4,
        num_classes=10,
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


predictions_folder = os.path.join(
    os.path.expanduser('~'), 'Desktop', 'predictions_test')
if not os.path.exists(predictions_folder):
    os.makedirs(predictions_folder)

for file in os.listdir(folder):
    single_sample = torch.load(os.path.join(folder, file))
    original = torch.load(os.path.join(folder, file))
    filename_cut, ext = os.path.splitext(file)
    filename = os.path.join(folder, file)
    data_dict = OrderedDict()
    data_dict["coord"] = torch.from_numpy(single_sample["coord"]).clone().to(
        torch.float).contiguous().detach().to(dev)
    data_dict["feat"] = torch.from_numpy(np.vstack((single_sample["coord"][:, 0], single_sample["coord"][:, 1], single_sample["coord"]
                                        [:, 2], single_sample["intensity"][:, 0])).T).clone().to(torch.float).contiguous().detach().to(dev)
    data_dict["offset"] = torch.tensor(
        [single_sample["coord"].shape[0]],  device=dev)

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

    print_matrix(matrix, os.path.join(
        predictions_folder, f"{filename_cut}_num.xlsx"))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] /= totals[i]
            matrix[i][j] = round(matrix[i][j], 3)


    print(matrix)
    print(totals)
    print_matrix(matrix, os.path.join(
        predictions_folder, f"{filename_cut}_perc.xlsx"))
    outfile = os.path.join(
        predictions_folder, f"{filename_cut}_prediction.txt")
    with open(outfile, "w") as f:
        for c, l in zip(original["coord"], labels):
            f.write(
                f"{c[0]},{c[1]},{c[2]},{colors[l][0]},{colors[l][1]},{colors[l][2]}\n")
    count = defaultdict(int)

    for l in labels:
        count[l] += 1

    print(count)
