import struct

def read_binary_label_file(file_path):
    labels = []

    with open(file_path, 'rb') as file:
        # Read binary data
        data = file.read()

    # Assuming each label entry has a fixed size of 15 floats
    entry_size = 15
    num_entries = len(data) // (entry_size * 4)  # Each float is 4 bytes

    for i in range(num_entries):
        start = i * entry_size * 4
        end = start + entry_size * 4

        # Unpack binary data into a list of floats
        values = struct.unpack('f' * entry_size, data[start:end])

        # Extract relevant information
        obj_type = values[0]
        truncated = values[1]
        occluded = values[2]
        alpha = values[3]
        bbox = values[4:8]
        dimensions = values[8:11]
        location = values[11:14]
        rotation_y = values[14]

        # Store the information in a dictionary or data structure of your choice
        label_info = {
            'type': obj_type,
            'truncated': truncated,
            'occluded': occluded,
            'alpha': alpha,
            'bbox': bbox,
            'dimensions': dimensions,
            'location': location,
            'rotation_y': rotation_y
        }

        labels.append(label_info)

    return labels

filename = os.path.join(os.path.expanduser('~'), 'Desktop/fresh/datasets/kitti/dataset/sequences/00/labels/000001.label')
binary_labels_data = read_binary_label_file(filename)

# Print the extracted information
for label_info in binary_labels_data:
    print(label_info)
