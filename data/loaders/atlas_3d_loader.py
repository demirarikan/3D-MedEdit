from core.DataLoader import DefaultDataset, DefaultDataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import os
import logging
import torchvision.transforms as transforms
from transforms.preprocessing import (
    AddChannelIfNeeded3D,
    AtlasAssertChannelFirst,
    ReadImage,
    To01,
    Pad3d,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dl_utils.config_utils import import_module
from dl_utils.data_utils import get_data_from_csv

class Atlas3DDataset():
    def __init__(
        self,
        data_dir,
        file_type="",
        label_dir=None,
        mask_dir=None,
        target_size=(64, 64),
        test=False,
    ):
        self.label_dir = label_dir
        self.target_size = target_size
        if data_dir[0].endswith(".csv"):
            self.files = get_data_from_csv(data_dir)
        else:
            self.files = glob.glob(os.path.join(data_dir, "*" + file_type)) 

        self.nr_items = len(self.files)
        logging.info(
            "Atlas3DLoader::init(): Loading {} files from: {}".format(
                self.nr_items, data_dir
            )
        )
        self.im_t = (
            self.get_image_transform_test() if test else self.get_image_transform()
        )
        if label_dir is not None:
            if data_dir[0].endswith(".csv"):
                self.label_files = get_data_from_csv(label_dir)
            else:
                self.label_files = glob.glob(os.path.join(data_dir, "*" + file_type)) 
            self.seg_t = (
                self.get_label_transform_test() if test else self.get_label_transform()
            )

    def get_image_transform(self):
        default_t = transforms.Compose(
            [
                ReadImage(),
                To01(), 
                AddChannelIfNeeded3D(),
                AtlasAssertChannelFirst(),
                transforms.Resize(self.target_size),
                Pad3d((0,0,0,0,3,3))
            ]
        )
        return default_t

    def get_image_transform_test(self):
        default_t = transforms.Compose(
            [
                ReadImage(),
                To01(),
                AddChannelIfNeeded3D(),
                AtlasAssertChannelFirst(),
                transforms.Resize(self.target_size),
                Pad3d((0,0,0,0,3,3))
            ]
        )
        return default_t

    def get_label_transform(self):
        default_t = transforms.Compose(
            [
                ReadImage(),
                To01(),
                AddChannelIfNeeded3D(),
                AtlasAssertChannelFirst(),
                transforms.Resize(self.target_size),
            ]
        )
        return default_t

    def get_label_transform_test(self):
        default_t = transforms.Compose(
            [
                ReadImage(),
                To01(),
                AddChannelIfNeeded3D(),
                AtlasAssertChannelFirst(),
                transforms.Resize(self.target_size),
            ]
        )
        return default_t

    def get_label(self, idx):
        if self.label_dir is not None:
            return self.seg_t(self.label_files[idx])
        else:
            return 0

    def __getitem__(self, idx):
        return self.im_t(self.files[idx]), self.get_label(idx)

    def __len__(self):
        return self.nr_items


class Atlas3DLoader(DefaultDataLoader):
    def __init__(self, args):
        akeys = args.keys()
        dataset_module = args["dataset_module"] if "dataset_module" in akeys else None
        self.data_dir = args["data_dir"] if "data_dir" in akeys else None
        self.file_type = args["file_type"] if "file_type" in akeys else ""

        self.target_size = args["target_size"] if "target_size" in akeys else (64, 64)
        self.batch_size = args["batch_size"] if "batch_size" in akeys else 8
        self.num_workers = args["num_workers"] if "num_workers" in akeys else 2
        self.shuffle = args["shuffle"] if "shuffle" in akeys else True
        self.drop_last = args["drop_last"] if "drop_last" in akeys else False

        assert (
            type(self.data_dir) is dict
        ), "DefaultDataset::init():  data_dir variable should be a dictionary"
        if dataset_module is not None:
            assert (
                "module_name" in dataset_module.keys()
                and "class_name" in dataset_module.keys()
            ), "DefaultDataset::init(): Please use the keywords [module_name|class_name] in the dataset_module dictionary"
            self.ds_module = import_module(
                dataset_module["module_name"], dataset_module["class_name"]
            )
            print(dataset_module["class_name"])
        else:
            self.ds_module = import_module("core.DataLoader", "DefaultDataset")
            print("Default DATASET!")

    def train_dataloader(self):
        return DataLoader(
            self.ds_module(
                data_dir=self.data_dir["train"],
                file_type=self.file_type,
                label_dir=None,
                target_size=self.target_size,
                test=False,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_module(
                data_dir=self.data_dir["val"],
                file_type=self.file_type,
                label_dir=None,
                target_size=self.target_size,
                test=False,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.ds_module(
                data_dir=self.data_dir["test"],
                file_type=self.file_type,
                label_dir=None,
                target_size=self.target_size,
                test=False,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

if __name__ == "__main__":
    from torch.nn import functional as F
    dataset_path = "/vol/ciamspace/datasets/atlas/processed/lin/"

    test_vol = "patient_1_sub-r001s001_T1_lin.nii.gz"

    file_type = ".nii.gz"

    target_size = (128, 128)

    ds = Atlas3DDataset(dataset_path, file_type, target_size=target_size)
    print(ds[0][0].shape)
    out = F.pad(ds[0][0], (0, 0, 0, 0, 1,1), "constant", 0)
    print(out.shape)

    # print(len(ds))
    # print(ds[0][0].shape)

    
    
    
