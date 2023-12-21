import torch
from pointcept.models import build_model
from collections import OrderedDict

print(torch.backends.cudnn.version())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.is_available())


torch.cuda.empty_cache()
model = torch.load('/home/gurveer/Desktop/models/model_best.pth')

unlabelled_data = torch.randn((1000, 4))

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

dev = 'cuda'
model = model.to(dev)
data_dict = OrderedDict()
data_dict["coord"] = torch.tensor(unlabelled_data[:, :3], device=dev)
data_dict["feat"] = torch.tensor(unlabelled_data,  device=dev)
data_dict["offset"] = torch.tensor([1],  device=dev)
model(data_dict)
