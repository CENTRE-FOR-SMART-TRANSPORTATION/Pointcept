import os
import torch
folder = os.path.join(os.getcwd(), "data", "cstdataset")
for f in os.listdir(folder):
    for file in os.listdir(os.path.join(folder, f)):
        data = torch.load(os.path.join(folder, f, file))
        print(file, len(data["coord"]))