import torch
from pointcept.models import build_model
from collections import OrderedDict
from torch.nn.functional import softmax
import os
import laspy
import numpy as np

print(torch.backends.cudnn.version())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.is_available())


torch.cuda.empty_cache()
model_saved = torch.load('/home/gurveer/Desktop/saved/model_best.pth')

folder = "/home/gurveer/Desktop/model_data/data/some_labels/las_files"

for file in os.listdir(folder):
    filename = os.path.join(folder, file)
    las = laspy.read(filename)
    unlabelled_data = np.vstack((las.x, las.y, las.z, las.intensity)).T
    unlabelled_data = torch.from_numpy(unlabelled_data[:1000, :]).to(torch.float)
    # unlabelled_data = torch.randn((1000, 4), dtype=torch.float)
    print(unlabelled_data.shape)
    print(unlabelled_data.is_contiguous())
    print(unlabelled_data)
    exit()
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
    print(sm.max(1)[1].data.cpu().numpy())
    break