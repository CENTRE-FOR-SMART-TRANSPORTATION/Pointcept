"""
Preprocessing Script for S3DIS
Parsing normal vectors has a large consumption of memory. Please reduce max_workers if memory is limited.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import glob
import torch
import numpy as np
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

area_mesh_dict = {}


def parse_room(
    room, dataset_root, output_root
):
    print("Parsing: {}".format(room))
    classes = ['solid-edge-lines', 'dash-solid-center-lines', 'lane', 'dashed-center-line', 'shoulder', 'vegetation', 'clutter']

    class2label = {cls: i for i, cls in enumerate(classes)}
    # class2label['clutter'] = -1
    source_dir = os.path.join(dataset_root, room)
    save_path = os.path.join(output_root, room) + ".pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    object_path_list = sorted(glob.glob(os.path.join(source_dir, "Annotations/*.txt")))

    print(class2label, source_dir ,save_path)

    room_coords = []
    room_intensity = []
    room_semantic_gt = []
    room_instance_gt = []

    for object_id, object_path in enumerate(object_path_list):
        object_name = os.path.basename(object_path).split("_")[0]
        obj = np.loadtxt(object_path)
        try:
            coords = obj[:, :3]
            intensity = obj[:, 3]
            intensity = intensity.reshape([-1, 1])
        except IndexError:
            print("#################### error", object_path)
            continue
        class_name = object_name if object_name in classes else "clutter"
        semantic_gt = np.repeat(class2label[class_name], coords.shape[0])
        semantic_gt = semantic_gt.reshape([-1, 1])
        instance_gt = np.repeat(object_id, coords.shape[0])
        instance_gt = instance_gt.reshape([-1, 1])

        print(f"{coords.shape} points for {class_name}/{object_name}")
        room_coords.append(coords)
        room_intensity.append(intensity)
        room_semantic_gt.append(semantic_gt)
        room_instance_gt.append(instance_gt)


    room_coords = np.ascontiguousarray(np.vstack(room_coords))
    room_intensity = np.ascontiguousarray(np.vstack(room_intensity))
    room_semantic_gt = np.ascontiguousarray(np.vstack(room_semantic_gt))
    room_instance_gt = np.ascontiguousarray(np.vstack(room_instance_gt))

    save_dict = dict(
        coord=room_coords,
        intensity=room_intensity,
        semantic_gt=room_semantic_gt,
        instance_gt=room_instance_gt,
    )

    torch.save(save_dict, save_path)


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Path to CST dataset"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where area folders will be located",
    )
    parser.add_argument(
        "--raw_root",
        default=None,
        help="Path to Stanford2d3dDataset_noXYZ dataset (optional)",
    )
    parser.add_argument(
        "--align_angle", action="store_true", help="Whether align room angles"
    )
    parser.add_argument(
        "--parse_normal", action="store_true", help="Whether process normal"
    )
    args = parser.parse_args()

    if args.parse_normal:
        assert args.raw_root is not None

    room_list = []

    # Load room information
    print("Loading room information ...")
    for d in os.listdir(args.dataset_root):
        room_list.append(d)


    # Preprocess data.
    print("Processing scenes...")
    # pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    pool = ProcessPoolExecutor(max_workers=8)  # peak 110G memory when parsing normal.
    _ = list(
        pool.map(
            parse_room,
            room_list,
            repeat(args.dataset_root),
            repeat(args.output_root),
        )
    )


if __name__ == "__main__":
    main_process()
