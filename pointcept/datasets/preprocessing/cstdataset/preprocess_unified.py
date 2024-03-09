

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
    args = parser.parse_args()

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
    for file in os.listdir(args.dataset_root):
        print(f"Processing {file}...")
        points = np.loadtxt(os.path.join(args.dataset_root, file))

        room_coords = np.ascontiguousarray(points[:, :3])
        room_intensity = np.ascontiguousarray(points[:, 3])
        room_semantic_gt = np.ascontiguousarray(points[:, 4])
        room_instance_gt = np.ascontiguousarray(np.ones_like(points[:, 4])) # each object has only one instance

        save_dict = dict(
            coord=room_coords,
            intensity=room_intensity,
            semantic_gt=room_semantic_gt,
            instance_gt=room_instance_gt,
        )

        filename_cut = os.path.splitext(file)[0]
        save_path = os.path.join(args.output_root, f"{filename_cut}.pth")
        torch.save(save_dict, save_path)

if __name__ == "__main__":
    main_process()
