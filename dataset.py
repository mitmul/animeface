# Copyright (c) 2017 Shunta Saito

import glob
import os

import matplotlib  # isort:skip
matplotlib.use('Agg')  # isort:skip

import numpy as np
from PIL import Image

from chainer import dataset
from chainer import datasets


class AnimeFaceDataset(dataset.DatasetMixin):
    IMG_DIR = 'animeface-character-dataset/thumb'
    IMG_SIZE = (160, 160)
    MEAN_FILE = 'image_mean.npy'

    def __init__(self):

        # Enumerate all directories under "thumb" dir
        img_dirs = [d for d in glob.glob('{}/*'.format(self.IMG_DIR))
                    if os.path.isdir(d)]

        # Prepare an empty list for storing image file paths
        self.img_fns = []

        for dname in img_dirs:

            # If "ignore" file exists, the directory is empty or all files
            # under the directory is marked as "difficult". So let's skip them
            if len(glob.glob('{}/ignore'.format(dname))):
                continue

            # Add image file paths to the list
            self.img_fns += glob.glob('{}/*.png'.format(dname))

        # This is a unique list for all directory names under "thumb"
        self.cls_labels = list(set(os.path.dirname(fn)
                                   for fn in self.img_fns))

        # The mean vector of Illustartion2Vec dataset
        self.mean = np.load(self.MEAN_FILE)
        self.mean = self.mean.mean(axis=(1, 2))

    def __len__(self):
        return len(self.img_fns)

    def get_example(self, i):

        # Load the image and resize it using PIL
        img = Image.open(self.img_fns[i])
        img = img.resize(self.IMG_SIZE, Image.BICUBIC)
        img = np.asarray(img, dtype=np.float)

        # The PNG image could be 4-channel, so crop the first 3 channels
        if img.shape[0] > 3:
            img = img[:, :, :3]

        # Convert R-G-B order image to B-G-R order image
        img = img.transpose(2, 0, 1)[::-1, ...]

        # Subtract mean from the image per each channel
        img -= self.mean[:, None, None]

        # Convert the image type to float32
        img = img.astype(np.float32)

        # The same dir name should indicates the same label ID
        dname = os.path.dirname(self.img_fns[i])
        label = np.asarray(self.cls_labels.index(dname), dtype=np.int32)

        return img, label
