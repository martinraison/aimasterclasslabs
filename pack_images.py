from __future__ import print_function
import argparse
import numpy as np
import os
from PIL import Image
import re
import torch
import numpy as np

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--images-path', type=str, required=True,
                    help='folder containing preprocessed png images')
parser.add_argument('--labels-path', type=str, required=True,
                    help='folder containing image labels')
parser.add_argument('--output-path', type=str, required=True,
                    help='output file path')
parser.add_argument('--val-size', type=int, default=100)
args = parser.parse_args()

images = []
labels = []
for f in os.listdir(args.images_path):
    if not f.endswith('.png'):
        continue
    m = re.match('^image-(\d+).png$', f)
    if not m:
        print('no label for file {}, skipping'.format(f))

    with open(os.path.join(args.labels_path, 'label-{}.txt'.format(m.group(1)))) as label_f:
        label = label_f.read()
        labels.append(ord(label) - 96)  # 'a' is 97 but label 0 is not used in EMNIST
    image = Image.open(os.path.join(args.images_path, f))
    image = torch.from_numpy(np.array(image)).t().unsqueeze(0)
    images.append(image)
images = torch.cat(images)
labels = torch.LongTensor(labels)

n = args.val_size
torch.save([images[:n], labels[:n]], args.output_path.replace(".pt", "-val.pt"))
torch.save([images[n:], labels[n:]], args.output_path.replace(".pt", "-test.pt"))
