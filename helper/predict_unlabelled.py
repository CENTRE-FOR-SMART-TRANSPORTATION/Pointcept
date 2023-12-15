import torch
from pointcept.models import build_model
from collections import OrderedDict

model = torch.load('/home/gurveer/Desktop/models/model_best.pth')

unlabelled_data = torch.randn((10000, 4))

state_dict = model["state_dict"]

model = build_model(dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointTransformer-Seg50",
        in_channels=4,
        num_classes=8,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
))

new_state_dict = OrderedDict()
for k in state_dict:
    # module.a.b = a.b
    new_k = k[7:]
    new_state_dict[new_k] = state_dict[k]   


model.load_state_dict(new_state_dict)

model.eval()

data_dict = OrderedDict()
data_dict["coord"] = unlabelled_data[:, 3]
data_dict["feat"] = unlabelled_data
data_dict["offset"] = 1
print(model(data_dict))