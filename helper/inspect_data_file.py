import torch

# Load the .pth file
file_path = '/home/gurveer/Desktop/datasets/preprocessed/Area_1/conferenceRoom_1.pth'
data = torch.load(file_path)

# Print the keys in the loaded data
print("Keys in the loaded data:", data.keys())

# Access and print specific information
print("Coordinates shape:", data['coord'].shape)
print("Color shape:", data['color'].shape)
print("Semantic labels shape:", data['semantic_gt'].shape)
print("Instance labels shape:", data['instance_gt'].shape)

print("Printing first 10 coordinates,their color values and their labels...")
i = 0
for c in zip(data['coord'], data['color'], data['semantic_gt'], data['instance_gt']):
    i += 1
    if i == 10:
        break
    print(c)

# If 'normal' is present
if 'normal' in data:
    print("Normals shape:", data['normal'].shape)
