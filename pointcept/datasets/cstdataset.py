"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
import traceback
import sys
import argparse

@DATASETS.register_module()
class CSTDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/cstdataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        # print("init function for CSTDataset called...")
        #traceback.print_stack(file=sys.stdout)
        super(CSTDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        # print("Printing attributes...")
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        # print(self.data_list)
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            data = torch.load(data_path)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)
        # name = (
        #     os.path.basename(self.data_list[idx % len(self.data_list)]).split('.')[0]
        # )
        coord = np.asarray(data["coord"])
        intensity = np.asarray(data["intensity"])
        try:
            roughness = np.asarray(data["roughness"])
            density = np.asarray(data["density"])  
            z_gradient = np.asarray(data["z_gradient"]) 
            intensity_gradient = np.asarray(data["intensity_gradient"])   
        except KeyError:
            print("No extra features found")   

        scene_id = data_path
        if "semantic_gt" in data.keys():
            segment = np.asarray(data["semantic_gt"].reshape([-1]))
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            # name=name,
            coord=coord,
            intensity=intensity,
            roughness=roughness,
            density=density,
            z_gradient=z_gradient,
            intensity_gradient=intensity_gradient,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if "normal" in data.keys():
            data_dict["normal"] = data["normal"]
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict["index"] = np.arange(len(data_dict["coord"]))
        # removing transform
        # data_dict = self.transform(data_dict)
        data_dict_list = [data_dict]
        # for aug in self.aug_transform:
        #     data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = [data_dict]
        # for data in data_dict_list:
        #     data_part_list = self.test_voxelize(data)
        #     for data_part in data_part_list:
        #         if self.test_crop:
        #             data_part = self.test_crop(data_part)
        #         else:
        #             data_part = [data_part]
        #         input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Path to Stanford3dDataset_v1.2 dataset"
    )
    args = parser.parse_args()
    data = CSTDataset("train", args.dataset_root)
    print(data.get_data_list())
    print(data.get_data_name(0))
    print(data.prepare_train_data(0))