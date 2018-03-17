import os.path
import random
import torch
from PIL import Image
import numpy as np
from glob import glob
import scipy.io

class Dataset():
    def __init__(self):
        self.opts = []
        self.num_imgs = 0
        self.img_list = []

    def initialize(self, opts):
        self.opts = opts
        self.num_imgs = 0
        self.db_name = opts.db_name
        self.edgeBoxResol = opts.edge_box_resol
        self.output_size = opts.output_size

        if self.db_name == 'celebA':
            # Load image list
            img_path = os.path.join(opts.dataset_root, opts.db_name, "*.jpg")
            img_list = glob(img_path)
            self.img_list = np.array(sorted(img_list))
            self.num_imgs = len(self.img_list)

            # Load part BBoxes
            bbs = scipy.io.loadmat("celebA_allbbs.mat")['allbbs']
            changeRatio = float(self.output_size) / 128.0
            bbs = np.floor(bbs * changeRatio).astype(np.int)

            self.bbs = bbs

        elif self.db_name == 'compcar_128' or self.db_name == 'compcar_256' or self.db_name == 'compcar_256_bilinear':

            img_path = os.path.join(opts.dataset_root, opts.db_name, "*.jpg")
            img_list = glob(img_path)
            self.img_list = np.array(sorted(img_list))
            self.num_imgs = len(self.img_list)

            # Load part BBoxes
            bbs = scipy.io.loadmat("compcar_allbbs.mat")['allbbs']
            changeRatio = opts.output_size
            bbs = np.floor(bbs * changeRatio).astype(np.int)

            self.bbs = bbs

        else:
            print ('Not ready for this dataset ...')

    def __getitem__(self, index):
        img_path = self.img_list[index]
        bbs = self.bbs[index]
        return img_path, bbs

    def __len__(self):
        # return len(self.paths)
        return self.num_imgs

    def name(self):
        return 'Dataset'






