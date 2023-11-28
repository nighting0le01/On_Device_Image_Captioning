import numpy as np
import random
import copy
import json
import os
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

    def __init__(
        self,
        vizwiz_dataset: VizWizDataset,
        array_of_init_seeds,
        batch_size,
        rank=0,
        num_procs=1,
        dataloader_mode="caption_wise",
        resize_image_size=None,
        image_folder="/home/arpitsah/Desktop/Fall-2023/odml/vizWiz/images",
        verbose=False,
    ):
        super(TransparentDataLoader, self).__init__()
        assert (
            dataloader_mode == "caption_wise" or dataloader_mode == "image_wise"
        ), "dataloader_mode must be either caption_wise or image_wise"

        self.dataset = vizwiz_dataset
        self.dataloader_mode = dataloader_mode
        self.image_folder = image_folder

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

        preprocess_layers_1 = [
            torchvision.transforms.Resize((resize_image_size, resize_image_size))
        ]
        preprocess_layers_2 = [
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]

        self.image_preprocess_1 = torchvision.transforms.Compose(preprocess_layers_1)
        self.image_preprocess_2 = torchvision.transforms.Compose(preprocess_layers_2)
        self.debug_counter = 0
        self.set_epoch_it(epoch=0, verbose=verbose)

    def init_epoch(self, epoch_it, verbose=False):
        init_timer_start = time()
        batch_size = self.batch_size
        random.seed(self.array_of_init_seeds[epoch_it])

        if self.dataset.split_name() == "train":
            random.shuffle(self.dataset.train_list)
        elif self.dataset.split_name() == "validation":
            random.shuffle(self.dataset.val_list)
        else:
            random.shuffle(self.dataset.test_list)

        self.batch_it = []
        self.image_idx_x = []
        self.caption_y = []
        for idx_proc in range(self.num_procs):
            self.batch_it.append(0)
            self.image_idx_x.append([])
            self.caption_y.append([])

        tailing_elements = len(self.dataset) % (batch_size * self.num_procs)
        if tailing_elements != 0:
            if self.dataset.split_name() == "train":
                self.dataset.train_list = self.dataset.train_list[:-tailing_elements]
                self.dataset.train_num_images = len(self.dataset.train_list)
            elif self.dataset.split_name() == "validation":
                self.dataset.val_list = self.dataset.val_list[:-tailing_elements]
                self.dataset.val_num_images = len(self.dataset.val_list)
            else:
                self.dataset.test_list = self.dataset.test_list[:-tailing_elements]
                self.dataset.test_num_images = len(self.dataset.test_list)

        image_idx_batch = []
        caption_y_batch = []
        for idx_proc in range(self.num_procs):
            image_idx_batch.append([])
            caption_y_batch.append([])
        i = 0

        while i < len(self.dataset):
            for idx_proc in range(self.num_procs):
                img_idx = i
                if self.dataloader_mode == "caption_wise":
                    caption = self.dataset[i]["tokenized_caption"]
                    caption = self.preprocess(caption)
                else:
                    caption = self.dataset[i]["all_captions"]
                image_idx_batch[idx_proc].append(img_idx)
                caption_y_batch[idx_proc].append(caption)
                i += 1
            if i % batch_size == 0:
                for idx_proc in range(self.num_procs):
                    self.image_idx_x[idx_proc].append(image_idx_batch[idx_proc])
                    self.caption_y[idx_proc].append(caption_y_batch[idx_proc])
                    image_idx_batch[idx_proc] = []
                    caption_y_batch[idx_proc] = []
        self.num_batches = len(self.image_idx_x[0])

    def get_next_batch(self, verbose=False, get_also_image_idxes=False):
        if self.batch_it[self.rank] >= self.num_batches:
            if verbose:
                print("Proc: " + str(self.rank) + " re-initialization")
            self.epoch_it += 1
            if self.epoch_it >= len(self.array_of_init_seeds):
                raise Exception(
                    "Please increase number of random seed in the array of initialization seed."
                )

            self.init_epoch(epoch_it=self.epoch_it, verbose=verbose)
        image_idx_batch = self.image_idx_x[self.rank][self.batch_it[self.rank]]
        batch_x, batch_x_num_pads = self.get_padded_img_batch(image_idx_batch)

        caption_str_batch = copy.copy(
            self.caption_y[self.rank][self.batch_it[self.rank]]
        )
        if self.dataloader_mode == "caption_wise":
            caption_encoded_batch = language_utils.convert_allsentences_word2idx(
                caption_str_batch, self.dataset.caption_word2idx_dict
            )
            batch_y, batch_y_num_pads = self.add_pad_according_to_batch(
                caption_encoded_batch, self.dataset.caption_word2idx_dict["PAD"]
            )
            batch_y = torch.tensor(batch_y)
        else:
            batch_y = caption_str_batch

        if verbose:
            mean_src_len = "Constant"
            if self.dataloader_mode == "caption_wise":
                mean_trg_len = int(
                    sum(
                        [
                            (len(batch_y[i]) - batch_y_num_pads[i])
                            for i in range(len(batch_y))
                        ]
                    )
                    / len(batch_y)
                )
            else:
                mean_trg_len = "variable"
            print(
                str(self.rank)
                + "] "
                + __name__
                + ") batch "
                + str(self.batch_it[self.rank])
                + " / "
                + str(self.num_batches)
                + " batch_size: "
                + str(len(batch_x))
                + " epoch: "
                + str(self.epoch_it)
                + " avg_src_seq_len: "
                + str(mean_src_len)
                + " avg_trg_seq_len: "
                + str(mean_trg_len)
            )

        self.batch_it[self.rank] += 1

        if self.dataloader_mode == "caption_wise":
            if get_also_image_idxes:
                return (
                    batch_x,
                    batch_y,
                    batch_x_num_pads,
                    batch_y_num_pads,
                    image_idx_batch,
                )
            else:
                return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads
        else:
            if get_also_image_idxes:
                return batch_x, batch_y, batch_x_num_pads, image_idx_batch
            else:
                return batch_x, batch_y, batch_x_num_pads

    def get_batch_samples(self, dataset_split, img_idx_batch_list):
        batch_captions_y = []
        idx_list = img_idx_batch_list
        for i in range(len(idx_list)):
            idx = idx_list[i]
            if dataset_split == VizWizDataset.ValidationSet_ID:
                caption = self.dataset.val_list[idx]["tokenized_caption"]
            if dataset_split == VizWizDataset.TrainSet_ID:
                caption = self.dataset.train_list[idx]["tokenized_caption"]

            preprocessed_caption = self.preprocess(caption)
            if dataset_split != VizWizDataset.TestSet_ID:
                batch_captions_y.append(preprocessed_caption)

        self.dataset.current_split = dataset_split
        batch_x, batch_x_num_pads = self.get_padded_img_batch(
            img_idxes=img_idx_batch_list
        )

        if dataset_split != VizWizDataset.TestSet_ID:
            batch_caption_y_encoded = language_utils.convert_allsentences_word2idx(
                batch_captions_y, self.dataset.caption_word2idx_dict
            )
            batch_y, batch_y_num_pads = self.add_pad_according_to_batch(
                batch_caption_y_encoded, self.dataset.get_pad_token_idx()
            )
        batch_y = torch.tensor(batch_y)
        if dataset_split == VizWizDataset.TestSet_ID:
            return batch_x, batch_x_num_pads
        else:
            return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads

    def get_padded_img_batch(self, img_idxes):
        image_tensors = []
        if self.dataset.current_split == VizWizDataset.ValidationSet_ID:
            subfolder = "val"
        else:
            subfolder = "train"

        for image_idx in img_idxes:
            image_file = self.dataset[image_idx]["image_path"]
            full_image_path = os.path.join(self.image_folder, subfolder, image_file)
            pil_image = PIL_Image.open(full_image_path)
            if pil_image.mode != "RGB":
                pil_image = PIL_Image.new("RGB", pil_image.size)
            preprocess_pil_image = self.image_preprocess_1(pil_image)
            tens_image_1 = torchvision.transforms.ToTensor()(preprocess_pil_image)
            tens_image_2 = self.image_preprocess_2(tens_image_1)
            image_tensors.append(tens_image_2)

        self.debug_counter += 1
        return torch.stack(image_tensors, dim=0), None

    def get_captions_by_idx(self, idx, dataset_split):
        if dataset_split == VizWizDataset.TestSet_ID:
            raise ValueError("No captions exist for the VizWiz test set")
        elif dataset_split == VizWizDataset.ValidationSet_ID:
            caption = self.dataset.val_list[idx]["all_captions"]
        else:
            caption = self.dataset.train_list[idx]["all_captions"]
        return caption

    def get_images_by_idx(self, idx, dataset_split, is_tensor=True):
        if dataset_split == VizWizDataset.TestSet_ID:
            image_file = self.dataset.test_list[idx]["image_path"]
            full_image_path = os.path.join(self.image_folder, "test", image_file)
        elif dataset_split == VizWizDataset.ValidationSet_ID:
            image_file = self.dataset.val_list[idx]["image_path"]
            full_image_path = os.path.join(self.image_folder, "val", image_file)
        else:
            image_file = self.dataset.train_list[idx]["image_path"]
            full_image_path = os.path.join(self.image_folder, "train", image_file)
        # full_image_path  = os.path.join(self.image_folder, image_file)
        pil_image = PIL_Image.open(full_image_path)
        if pil_image.mode != "RGB":
            pil_image = PIL_Image.new("RGB", pil_image.size)
        if not is_tensor:
            return pil_image
        else:
            preprocess_pil_image = self.image_preprocess_1(pil_image)
            tens_image_1 = torchvision.transforms.ToTensor()(preprocess_pil_image)
            tens_image_2 = self.image_preprocess_2(tens_image_1)
            return tens_image_2

    def preprocess(self, caption):
        if isinstance(caption, str):
            caption = language_utils.lowercase_and_clean_trailing_spaces([caption])
            caption = language_utils.add_space_between_non_alphanumeric_symbols(caption)
            caption = language_utils.remove_punctuations(caption)
            caption = (
                [self.dataset.get_sos_token_str()]
                + language_utils.tokenize(caption)[0]
                + [self.dataset.get_eos_token_str()]
            )
        preprocessed_tokenized_caption = []
        for word in caption:
            if word not in self.dataset.caption_word2idx_dict.keys():
                preprocessed_tokenized_caption.append(self.dataset.get_unk_token_str())
            else:
                preprocessed_tokenized_caption.append(word)
        return preprocessed_tokenized_caption

    def preprocess_list(self, caption_list):
        for i in range(len(caption_list)):
            caption_list[i] = self.preprocess(caption_list[i])
        return caption_list


if __name__ == "__main__":
    with open("vocab/coco_vocab_idx_dict.json", "r") as vocab_json:
        coco_vocab_idx_dict = json.load(vocab_json)

    dataset_w_coco_vocab = VizWizDataset(
        2, train=False, coco_vocab_dict=coco_vocab_idx_dict
    )
    val_dataloader_rank_0 = VizWizDataLoader(
        dataset_w_coco_vocab,
        array_of_init_seeds=[42],
        batch_size=8,
        resize_image_size=384,
        num_procs=2,
        rank=0,
    )
    print(val_dataloader_rank_0.get_batch_it())
    image_tensor, text_tensor, _, _ = val_dataloader_rank_0.get_next_batch(verbose=True)
    print(image_tensor.shape, text_tensor.shape)
    print(val_dataloader_rank_0.get_batch_it())

    val_dataloader_rank_1 = VizWizDataLoader(
        dataset_w_coco_vocab,
        array_of_init_seeds=[42],
        batch_size=8,
        resize_image_size=384,
        num_procs=2,
        rank=1,
    )

    print(val_dataloader_rank_1.get_batch_it())
    image_tensor, text_tensor, _, _ = val_dataloader_rank_1.get_next_batch(verbose=True)
    print(image_tensor.shape, text_tensor.shape)
    print(val_dataloader_rank_1.get_batch_it())

    print(val_dataloader_rank_0.get_batch_it())
    image_tensor, text_tensor, _, _ = val_dataloader_rank_0.get_next_batch(verbose=True)
    print(image_tensor.shape, text_tensor.shape)
    print(val_dataloader_rank_0.get_batch_it())

    print(val_dataloader_rank_1.get_batch_it())
    image_tensor, text_tensor, _, _ = val_dataloader_rank_1.get_next_batch(verbose=True)
    print(image_tensor.shape, text_tensor.shape)
    print(val_dataloader_rank_1.get_batch_it())
