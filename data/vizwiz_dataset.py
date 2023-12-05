import json
from time import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import language_utils
from torch.utils.data import Dataset
from data.coco_dataset import CocoDatasetKarpathy
import os

import functools

print = functools.partial(print, flush=True)


class VizWizDataset(Dataset):
    TrainSet_ID = 1
    ValidationSet_ID = 2
    TestSet_ID = 3

    def __init__(
        self,
        current_split: int,
        vizwiz_annotations_dir: str = "/home/arpitsah/Desktop/Fall-2023/odml/vizWiz/annotations",
        annotations_filtered: bool = False,
        train: bool = True,
        val: bool = True,
        test: bool = False,
        verbose: bool = False,
        dict_min_occurrences=5,
        coco_vocab_dict: dict = None,
        max_seq_length=54,
    ):
        super(Dataset, self).__init__()
        if not train and not val:
            raise ValueError("Need at least train or val to be true")
        if not annotations_filtered:
            if train:
                train_path = os.path.join(vizwiz_annotations_dir, "train.json")
                save_path = os.path.join(vizwiz_annotations_dir, "processed_train.json")
                self.process_annotations(train_path, save_path)
            if val:
                val_path = os.path.join(vizwiz_annotations_dir, "val.json")
                save_path = os.path.join(vizwiz_annotations_dir, "processed_val.json")
                self.process_annotations(val_path, save_path)
            if test:
                test_path = os.path.join(vizwiz_annotations_dir, "test.json")
                save_path = os.path.join(vizwiz_annotations_dir, "processed_test.json")
                self.process_annotations(test_path, save_path, is_test=True)

        if train:
            train_load_path = os.path.join(
                vizwiz_annotations_dir, "processed_train.json"
            )
            with open(train_load_path) as f:
                self.train_dict = json.load(f)
        if val:
            val_load_path = os.path.join(vizwiz_annotations_dir, "processed_val.json")
            with open(val_load_path) as f:
                self.val_dict = json.load(f)
        if test:
            test_load_path = os.path.join(vizwiz_annotations_dir, "processed_test.json")
            with open(test_load_path) as f:
                self.test_dict = json.load(f)

        self.current_split = current_split

        self.train_list = []
        self.val_list = []
        self.test_list = []
        self.max_seq_len = max_seq_length
        tokenized_captions_list = []
        if train:
            for key in self.train_dict.keys():
                tokenized_captions_list.append(
                    self.train_dict[key]["tokenized_caption"]
                )
                if len(self.train_dict[key]["tokenized_caption"]) > self.max_seq_len:
                    self.train_dict[key]["tokenized_caption"] = self.train_dict[key][
                        "tokenized_caption"
                    ][: self.max_seq_len - 1] + ["EOS"]
                self.train_list.append(self.train_dict[key])
        if val:
            for key in self.val_dict.keys():
                if not train:
                    tokenized_captions_list.append(
                        self.val_dict[key]["tokenized_caption"]
                    )
                if len(self.val_dict[key]["tokenized_caption"]) > self.max_seq_len:
                    self.val_dict[key]["tokenized_caption"] = self.val_dict[key][
                        "tokenized_caption"
                    ][: self.max_seq_len - 1] + ["EOS"]
                self.val_list.append(self.val_dict[key])
        if test:
            for key in self.test_dict.keys():
                if len(self.test_dict[key]["tokenized_caption"]) > self.max_seq_len:
                    self.test_dict[key]["tokenized_caption"] = self.test_dict[key][
                        "tokenized_caption"
                    ][: self.max_seq_len - 1] + ["EOS"]
                self.test_list.append(self.test_dict[key])

        self.train_num_images = len(self.train_list)
        self.val_num_images = len(self.val_list)
        self.test_num_images = len(self.test_list)
        counter_dict = dict()
        for i in range(len(tokenized_captions_list)):
            for word in tokenized_captions_list[i]:
                if word not in counter_dict:
                    counter_dict[word] = 1
                else:
                    counter_dict[word] += 1

        less_than_min_occurrences_set = set()
        for k, v in counter_dict.items():
            if v < dict_min_occurrences:
                less_than_min_occurrences_set.add(k)
        if verbose:
            print(
                "tot tokens "
                + str(len(counter_dict))
                + " less than "
                + str(dict_min_occurrences)
                + ": "
                + str(len(less_than_min_occurrences_set))
                + " remaining: "
                + str(len(counter_dict) - len(less_than_min_occurrences_set))
            )

        self.num_caption_vocab = 4

        discovered_words = ["PAD", "SOS", "EOS", "UNK"]
        for i in range(len(tokenized_captions_list)):
            caption = tokenized_captions_list[i]

            for word in caption:
                if (word not in discovered_words) and (
                    not word in less_than_min_occurrences_set
                ):
                    discovered_words.append(word)
                    self.num_caption_vocab += 1

        discovered_words.sort()
        self.caption_word2idx_dict = dict()
        self.caption_idx2word_list = []
        if coco_vocab_dict is None:
            for i in range(len(discovered_words)):
                self.caption_word2idx_dict[discovered_words[i]] = i
                self.caption_idx2word_list.append(discovered_words[i])
        else:
            self.caption_word2idx_dict = coco_vocab_dict
            idx2word_dict = {v: k for k, v in coco_vocab_dict.items()}
            self.caption_idx2word_list = [""] * len(idx2word_dict)
            for idx in idx2word_dict.keys():
                self.caption_idx2word_list[idx] = idx2word_dict[idx]
        if verbose:
            print("There are " + str(self.num_caption_vocab) + " vocabs in dict")

    def split_name(self):
        if self.current_split == VizWizDataset.TrainSet_ID:
            return "train"
        elif self.current_split == VizWizDataset.ValidationSet_ID:
            return "validation"
        else:
            return "test"

    def set_split(self, split):
        if split == "train":
            self.current_split = VizWizDataset.TrainSet_ID
        elif split == "validation":
            self.current_split = VizWizDataset.ValidationSet_ID
        else:
            self.current_split = VizWizDataset.TestSet_ID

    def __len__(self):
        if self.current_split == VizWizDataset.TrainSet_ID:
            return len(self.train_list)
        elif self.current_split == VizWizDataset.ValidationSet_ID:
            return len(self.val_list)
        else:
            return len(self.test_list)

    def __getitem__(self, idx):
        if self.current_split == VizWizDataset.TrainSet_ID:
            return self.train_list[idx]
        elif self.current_split == VizWizDataset.ValidationSet_ID:
            return self.val_list[idx]
        else:
            return self.test_list[idx]

    def process_annotations(self, load_path, save_file, is_test: bool = False):
        if not os.path.isfile(load_path):
            raise FileNotFoundError("There is no such VizWiz Annotations Json")
        with open(load_path) as f:
            annotation_json = json.load(f)
        full_dict = self.consolidated_dict(annotation_json, is_test)
        filtered_dict = self.filter_annotations(full_dict, is_test)
        processed_dict = {}
        for image_id in filtered_dict.keys():
            if image_id not in processed_dict:
                processed_dict[image_id] = {}
            raw_caption = filtered_dict[image_id]["caption"]
            if raw_caption is None:
                del processed_dict[image_id]
                continue
            processed_dict[image_id]["image_path"] = filtered_dict[image_id][
                "image_path"
            ]
            processed_dict[image_id]["raw_caption"] = raw_caption
            processed_dict[image_id]["tokenized_caption"] = self.tokenize_caption(
                raw_caption
            )
            processed_dict[image_id]["all_captions"] = filtered_dict[image_id][
                "all_captions"
            ]
        with open(save_file, "w") as fp:
            json.dump(processed_dict, fp)

    @staticmethod
    def filter_annotations(unfiltered_dict: dict, is_test: bool, strict_filt=True):
        filtered_dict = {}
        for image_id, annotation_dict in unfiltered_dict.items():
            image_path = annotation_dict["image_path"]
            if image_id not in filtered_dict:
                filtered_dict[image_id] = {}
            filtered_dict[image_id]["image_path"] = image_path
            chosen_caption = None
            filtered_dict[image_id]["caption"] = chosen_caption

            if is_test:
                filtered_dict[image_id]["caption"] = chosen_caption
                continue
            all_captions = annotation_dict["annotations"]
            filtered_captions = []
            if strict_filt:
                if (
                    "Quality issues are too severe to recognize visual content."
                    in all_captions
                ):
                    continue
            for caption in all_captions:
                if (
                    caption
                    != "Quality issues are too severe to recognize visual content."
                ):
                    chosen_caption = caption
                    filtered_captions.append(caption)

            filtered_dict[image_id]["caption"] = chosen_caption
            filtered_dict[image_id]["all_captions"] = filtered_captions
        return filtered_dict

    @staticmethod
    def consolidated_dict(annotation_json, is_test):
        annotation_dict = {}
        image_list = annotation_json["images"]
        for image in image_list:
            annotation_dict[image["id"]] = {"image_path": image["file_name"]}
        if is_test:
            return annotation_dict
        annotation_list = annotation_json["annotations"]
        for annotation in annotation_list:
            image_id = annotation["image_id"]
            caption = annotation["caption"]
            if "annotations" not in annotation_dict[image_id]:
                annotation_dict[image_id]["annotations"] = [caption]
            else:
                annotation_dict[image_id]["annotations"].append(caption)
        return annotation_dict

    @staticmethod
    def tokenize_caption(caption: str):
        tmp = language_utils.lowercase_and_clean_trailing_spaces([caption])
        tmp = language_utils.add_space_between_non_alphanumeric_symbols(tmp)
        tmp = language_utils.remove_punctuations(tmp)
        tokenized_caption = ["SOS"] + language_utils.tokenize(tmp)[0] + ["EOS"]
        return tokenized_caption

    def get_all_images_captions(self, dataset_split):
        all_image_references = []

        if dataset_split == VizWizDataset.TestSet_ID:
            dataset = self.test_list
        elif dataset_split == VizWizDataset.ValidationSet_ID:
            dataset = self.val_list
        else:
            dataset = self.train_list

        for img_idx in range(len(dataset)):
            all_image_references.append(dataset[img_idx]["all_captions"])
        return all_image_references

    def get_eos_token_idx(self):
        return self.caption_word2idx_dict["EOS"]

    def get_sos_token_idx(self):
        return self.caption_word2idx_dict["SOS"]

    def get_pad_token_idx(self):
        return self.caption_word2idx_dict["PAD"]

    def get_unk_token_idx(self):
        return self.caption_word2idx_dict["UNK"]

    def get_eos_token_str(self):
        return "EOS"

    def get_sos_token_str(self):
        return "SOS"

    def get_pad_token_str(self):
        return "PAD"

    def get_unk_token_str(self):
        return "UNK"


if __name__ == "__main__":
    dataset = VizWizDataset(2, train=False)
    print(len(dataset))
    print(dataset[7])
    print(dataset.caption_word2idx_dict)
    print(dataset.caption_idx2word_list)
    print(len(dataset.caption_word2idx_dict))
    print(len(dataset.caption_idx2word_list))
    """
    coco_dataset = CocoDatasetKarpathy(
        images_path=None,
        train2014_bboxes_path = None, 
        val2014_bboxes_path=None, 
        precalc_features_hdf5_filepath=None, 
        coco_annotations_path="/usr0/home/nvaikunt/On_Device_Image_Captioning/coco_annotations/dataset_coco.json"
    )

    coco_vocab_idx_dict = coco_dataset.caption_word2idx_dict
    with open("vocab/coco_vocab_idx_dict.json", "w") as vocab_json: 
        json.dump(coco_vocab_idx_dict, vocab_json)
    """
    with open("/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/vocab/coco_vocab_idx_dict.json", "r") as vocab_json:
        coco_vocab_idx_dict = json.load(vocab_json)

    dataset_w_new_vocab = VizWizDataset(
        2, train=False, coco_vocab_dict=coco_vocab_idx_dict
    )
    print(dataset_w_new_vocab.caption_word2idx_dict)
    print(dataset_w_new_vocab.caption_idx2word_list)
    print(len(dataset_w_new_vocab.caption_word2idx_dict))
    print(len(dataset_w_new_vocab.caption_idx2word_list))
