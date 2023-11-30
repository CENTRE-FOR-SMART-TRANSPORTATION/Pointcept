import torch

# Load the .pth file
file_path = 'path/to/your/file.pth'
data = torch.load(file_path)

# Print the keys in the loaded data
print("Keys in the loaded data:", data.keys())

# Access and print specific information
print("Coordinates shape:", data['coord'].shape)
print("Color shape:", data['color'].shape)
print("Semantic labels shape:", data['semantic_gt'].shape)
print("Instance labels shape:", data['instance_gt'].shape)

# If 'normal' is present
if 'normal' in data:
    print("Normals shape:", data['normal'].shape)
