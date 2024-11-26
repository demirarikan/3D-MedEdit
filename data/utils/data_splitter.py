import os
import glob
import random 

"""
    This script is used to split the files in the dataset into training, validation and test sets.
    It is assumed that all the data is in the same folder with the same file extension.
    The result will be a csv file for each specified set.
"""

DATASET_DIR = os.path.abspath("/vol/ciamspace/datasets/atlas/processed/lin")
OUTPUT_DIR = os.path.abspath(".") # Output directory for the csv files
DATASET_NAME = "atlas_3d"
EXTENSION = ".nii.gz" # File extension of the dataset
TRAIN_RATIO = 0.8 # Ratio of the training set
VAL_RATIO = 0.1 # Ratio of the validation set
TEST_RATIO = 0.1 # Ratio of the test set
SEED = 42 # Seed for the random number generator

def split_data(dataset_dir, dataset_name, output_dir, extension, train_ratio, val_ratio, test_ratio, seed):
    random.seed(seed)
    files = glob.glob(os.path.join(dataset_dir, "*" + extension))
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