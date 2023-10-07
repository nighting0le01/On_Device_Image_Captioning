import numpy as np
import random
import copy
import torch
from time import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.vizwiz_dataset import VizWizDataset
from utils import language_utils
from data.transparent_data_loader import TransparentDataLoader

from PIL import Image as PIL_Image
import torchvision

import functools
print = functools.partial(print, flush=True)


class VizWizDataLoader(TransparentDataLoader):
    NOT_DEFINED = -1

    def __init__(self, vizwiz_dataset,
                       array_of_init_seeds,
                       batch_size, rank=0, num_procs=1,
                       dataloader_mode='caption_wise',
                       resize_image_size=None,
                       verbose=False):
        super(TransparentDataLoader, self).__init__()
        assert (dataloader_mode == 'caption_wise' or dataloader_mode == 'image_wise'), \
            "dataloader_mode must be either caption_wise or image_wise"

        self.dataset = vizwiz_dataset
        self.dataloader_mode = dataloader_mode

        self.num_procs = num_procs
        self.rank = rank

        self.epoch_it = 0
        self.array_of_init_seeds = array_of_init_seeds * 10
        self.max_num_epoch = len(array_of_init_seeds)

        self.max_num_regions = None

        self.batch_size = batch_size

        self.num_procs = num_procs
        self.num_batches = VizWizDataLoader.NOT_DEFINED
        self.batch_it = []
        self.image_idx_x = []
        self.caption_y = []
        for idx_proc in range(num_procs):
            self.batch_it.append(0)
            self.image_idx_x.append([])
            self.caption_y.append([])

    
        preprocess_layers_1 = [torchvision.transforms.Resize((resize_image_size, resize_image_size))]
        preprocess_layers_2 = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        self.image_preprocess_1 = torchvision.transforms.Compose(preprocess_layers_1)
        self.image_preprocess_2 = torchvision.transforms.Compose(preprocess_layers_2)
        self.debug_counter = 0
        self.set_epoch_it(epoch=0, verbose=verbose)
    def init_epoch(self, epoch_it, verbose=False):
        init_timer_start = time()

        batch_size = self.batch_size
        random.seed(self.array_of_init_seeds[epoch_it])