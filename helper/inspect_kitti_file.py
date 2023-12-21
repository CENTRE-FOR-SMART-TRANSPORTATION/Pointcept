import os

filname = os.path.join(os.path.expanduser('~'), 'Desktop/fresh/datasets/kitti/dataset/sequences/00/labels/000001.label')

with open(filname, 'r') as f:
    print()
    print(f.readline()) 