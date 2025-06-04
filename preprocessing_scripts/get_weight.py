from argparse import ArgumentParser
import os
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument("--folder", type=str, default=None, required=True,
                        help="The full path to the file that is to be split into sections.")

args = parser.parse_args()
folder = os.path.abspath(args.folder)

if not os.path.exists(folder):
    print(f"{folder} does not exist.")
    exit()

count = defaultdict(int)

for subfolder in os.listdir(folder):
    annotations = os.path.join(folder, subfolder, "Annotations")

    for file in os.listdir(annotations):
        fname, _ = os.path.splitext(file)
        label_name, _ = fname.split("_")
        points = 0
        with open(os.path.join(annotations, file), "r") as f:
            for i in f:
                points += 1
            count[label_name] += points

total = 0
for k, v in count.items():
    print(k, v)
    total += v

labels = ['concrete-barriers', 'traffic-sign', 'clutter', 'pavement', 'light-pole', 'vegetation', 'broken-line', 'solid-line','highway-guardrails', 'chevrons', 'arrows']

weights = []

for l in labels:
    if (count[l] == 0):
        print(f"No points found for {l}")
        continue
    weights.append((total/count[l]))

print(weights)
