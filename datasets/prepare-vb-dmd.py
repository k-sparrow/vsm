import shutil
import os
from pathlib import Path
from torch.utils.data import random_split
from tqdm import tqdm


VB_DMD_TRAIN_SET_ROOT = Path("vb-dmd/train/")
VB_DMD_TRAIN_NOISY_ROOT = Path(VB_DMD_TRAIN_SET_ROOT) / "noisy"
VB_DMD_TRAIN_CLEAN_ROOT = Path(VB_DMD_TRAIN_SET_ROOT) / "clean"
VB_DMD_VALID_NOISY_ROOT = Path(str(VB_DMD_TRAIN_NOISY_ROOT).replace("train", "valid"))
VB_DMD_VALID_CLEAN_ROOT = Path(str(VB_DMD_TRAIN_CLEAN_ROOT).replace("train", "valid"))

print(VB_DMD_TRAIN_SET_ROOT,
      VB_DMD_TRAIN_NOISY_ROOT,
      VB_DMD_TRAIN_CLEAN_ROOT,
      VB_DMD_VALID_NOISY_ROOT,
      VB_DMD_VALID_CLEAN_ROOT)

assert VB_DMD_TRAIN_SET_ROOT.exists()
assert VB_DMD_TRAIN_NOISY_ROOT.exists()
assert VB_DMD_TRAIN_CLEAN_ROOT.exists()
assert VB_DMD_VALID_NOISY_ROOT.exists()
assert VB_DMD_VALID_CLEAN_ROOT.exists()

# get all training data
train_clean_full = list(VB_DMD_TRAIN_CLEAN_ROOT.glob("*.wav"))
train_noisy_full = list(VB_DMD_TRAIN_NOISY_ROOT.glob("*.wav"))

# split train into train-valid sets
train_clean, valid_clean = random_split(train_clean_full, [0.8, 0.2])

for valid_clean_file in tqdm(valid_clean):
    valid_noisy_file = Path(str(valid_clean_file).replace("clean", "noisy"))
    assert valid_noisy_file.exists()
    assert valid_noisy_file != valid_clean_file
    assert valid_noisy_file in train_noisy_full

    shutil.copy2(valid_clean_file, VB_DMD_VALID_CLEAN_ROOT)
    valid_clean_file.unlink()
    shutil.copy2(valid_noisy_file, VB_DMD_VALID_NOISY_ROOT)
    valid_noisy_file.unlink()


