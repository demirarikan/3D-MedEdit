import os
import glob
import random 

"""
    This script is used to split the files in the dataset into training, validation and test sets.
    It is assumed that all the data is in the same folder with the same file extension.
    The result will be a csv file for each specified set.
"""

upenn = "/vol/ciamspace/datasets/upenn-gbm/processed/registered_to_mni152"
atlas_skull_stripped = "/vol/ciamspace/datasets/atlas/processed/skull_stripped"
brats_train = "/vol/ciamspace/datasets/brats/brats_20/processed/registered_mni152_1mm/train_data"
brats_val = "/vol/ciamspace/datasets/brats/brats_20/processed/registered_mni152_1mm/val_data"
ixi_skull_stipped = "/vol/ciamspace/datasets/ixi/processed/skull_stripped"

DATASET_DIR = os.path.abspath(atlas_skull_stripped)
OUTPUT_DIR = os.path.abspath(".") # Output directory for the csv files
DATASET_NAME = "atlas_skull_stripped"
EXTENSION = ".nii.gz" # File extension of the dataset
TRAIN_RATIO = 0.8 # Ratio of the training set
VAL_RATIO = 0.1 # Ratio of the validation set
TEST_RATIO = 0.1 # Ratio of the test set
SEED = 42 # Seed for the random number generator

def split_data(dataset_dir, dataset_name, output_dir, extension, train_ratio, val_ratio, test_ratio, seed):
    random.seed(seed)
    files = glob.glob(
        fr'{DATASET_DIR}/*T1_lin{EXTENSION}'
        # os.path.join(dataset_dir, "*" + extension)
        )
    files = [file for file in files if "_mask" not in file]
    random.shuffle(files)
    total_files = len(files)
    train_files = files[:int(total_files * train_ratio)]
    val_files = files[int(total_files * train_ratio):int(total_files * (train_ratio + val_ratio))]
    test_files = files[int(total_files * (train_ratio + val_ratio)):]
    with open(os.path.join(output_dir, f"{dataset_name}_train.csv"), "w") as f:
        f.write(",\n".join(train_files))

    with open(os.path.join(output_dir, f"{dataset_name}_val.csv"), "w") as f:
        f.write(",\n".join(val_files))

    with open(os.path.join(output_dir, f"{dataset_name}_test.csv"), "w") as f:
        f.write(",\n".join(test_files))

split_data(DATASET_DIR, DATASET_NAME, OUTPUT_DIR, EXTENSION, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)