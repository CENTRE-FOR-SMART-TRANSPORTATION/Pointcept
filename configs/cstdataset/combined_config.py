_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 1  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True


# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointTransformer-Seg50",
        in_channels=4,
        num_classes=13,
    ),
    criteria=[
        # dict(type="FocalLoss", gamma=2.0, alpha=0.5,
        #      loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),],
)


# scheduler settings
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.007, weight_decay=0.05)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)

# dataset settings
dataset_type = "CSTDataset"
data_root = "data/cstdataset"

##########
'''
change the number of classes, names as required
make sure the transformations are right
change the features and keys in the collect transformation
'''
##########
data = dict(
    num_classes=13,
    ignore_index=-1,
    names=['concrete-barriers', 'wires', 'traffic-sign', 'clutter', 'pavement', 'light_pole', 'vegetation', 'broken-line', 'solid-line', 
           'traffic-cones', 'gore_area', 'highway-guardrails', 'delineator-post'],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            # dict(
            #     type="GridSample",
            #     grid_size=0.04,
            #     hash_type="fnv",
            #     mode="train",
            #     keys=("coord", "intensity", "segment"),
            #     return_discrete_coord=True,
            # ),
            # dict(type="SphereCrop", point_max=80000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=["coord", "intensity"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord",
                           "segment": "origin_segment"},
            ),
            # dict(
            #     type="GridSample",
            #     grid_size=1.5,
            #     hash_type="fnv",
            #     mode="train",
            #     keys=("coord", "intensity", "segment"),
            #     return_discrete_coord=True,
            # ),
            # dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                offset_keys_dict=dict(offset="coord"),
                feat_keys=["coord", "intensity"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[dict(type="CenterShift", apply_z=True)],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="test",
                keys=("coord", "intensity"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "intensity"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]),
                 dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)
