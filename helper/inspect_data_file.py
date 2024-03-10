import torch
import os
# Load the .pth file
file_path = os.path.join(os.path.expanduser('~'), "Desktop", "datasets", "preprocessed_features", "train", "00215S_L1L1_24000_22000_section_21_sar__d_points_40_sar__d_points_35.pth")
data = torch.load(file_path)

# Print the keys in the loaded data
print("Keys in the loaded data:", data.keys())

# Access and print specific information
print("Coordinates shape:", data['coord'].shape)
# print("Color shape:", data['intensity'].shape)

# print("Printing first 10 coordinates,their color values and their labels...")
# i = 0
# for c in zip(data['coord'], data['intensity'], data['semantic_gt'], data['instance_gt']):
#     if c[2][0] < 4:
#         print(c)
#         i += 1
# print(i)
# If 'normal' is present
if 'normal' in data:
    print("Normals shape:", data['normal'].shape)
