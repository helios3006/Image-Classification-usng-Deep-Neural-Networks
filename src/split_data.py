import os
import random
import shutil
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

def make_dirs():
    for split in ["train", "val", "test"]:
        for cls in ["cats", "dogs"]:
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def split_files(file_list):
    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]
    return train_files, val_files, test_files

def copy_files(files, split, cls_name):
    for f in files:
        dest = OUT_DIR / split / cls_name / f.name
        shutil.copy(f, dest)

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR}")

    make_dirs()

    cat_files = list(RAW_DIR.glob("cat.*.jpg"))
    dog_files = list(RAW_DIR.glob("dog.*.jpg"))

    if len(cat_files) == 0 or len(dog_files) == 0:
        raise ValueError("No cat/dog images found. Check filenames like cat.0.jpg and dog.0.jpg")

    cat_train, cat_val, cat_test = split_files(cat_files)
    dog_train, dog_val, dog_test = split_files(dog_files)

    copy_files(cat_train, "train", "cats")
    copy_files(cat_val, "val", "cats")
    copy_files(cat_test, "test", "cats")

    copy_files(dog_train, "train", "dogs")
    copy_files(dog_val, "val", "dogs")
    copy_files(dog_test, "test", "dogs")

    print("Done splitting dataset!")
    print("Train cats:", len(cat_train), "Train dogs:", len(dog_train))
    print("Val cats:", len(cat_val), "Val dogs:", len(dog_val))
    print("Test cats:", len(cat_test), "Test dogs:", len(dog_test))

if __name__ == "__main__":
    main()
