import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# PATHS
# =========================
lr_dir = "data/train/LR"
hr_dir = "data/train/HR"

out_lr_dir = "Dataset_aug/train/LR"
out_hr_dir = "Dataset_aug/train/HR"

os.makedirs(out_lr_dir, exist_ok=True)
os.makedirs(out_hr_dir, exist_ok=True)

# =========================
# AUGMENTATIONS
# =========================
def augment_pair(lr, hr):
    augmented = []

    # original
    augmented.append(("orig", lr, hr))

    # horizontal flip
    augmented.append(("hflip", cv2.flip(lr, 1), cv2.flip(hr, 1)))

    # vertical flip
    augmented.append(("vflip", cv2.flip(lr, 0), cv2.flip(hr, 0)))

    # rotate 90
    augmented.append(("rot90", np.rot90(lr), np.rot90(hr)))

    # rotate 180
    augmented.append(("rot180", np.rot90(lr, 2), np.rot90(hr, 2)))

    # rotate 270
    augmented.append(("rot270", np.rot90(lr, 3), np.rot90(hr, 3)))

    # flip + rotate (optional but useful)
    augmented.append(("hflip_rot90", np.rot90(cv2.flip(lr,1)), np.rot90(cv2.flip(hr,1))))

    return augmented

# =========================
# PROCESS
# =========================
lr_files = sorted(os.listdir(lr_dir))

for file in tqdm(lr_files):
    lr_path = os.path.join(lr_dir, file)
    hr_path = os.path.join(hr_dir, file)

    if not os.path.exists(hr_path):
        print(f"Skipping {file} (no HR match)")
        continue

    lr_img = cv2.imread(lr_path)
    hr_img = cv2.imread(hr_path)

    if lr_img is None or hr_img is None:
        print(f"Error reading {file}")
        continue

    # sanity check (IMPORTANT)
    if hr_img.shape[0] != lr_img.shape[0] * 4 or hr_img.shape[1] != lr_img.shape[1] * 4:
        print(f"Size mismatch in {file}")
        continue

    augmented = augment_pair(lr_img, hr_img)

    base_name = os.path.splitext(file)[0]

    for aug_name, lr_aug, hr_aug in augmented:
        out_name = f"{base_name}_{aug_name}.png"

        cv2.imwrite(os.path.join(out_lr_dir, out_name), lr_aug)
        cv2.imwrite(os.path.join(out_hr_dir, out_name), hr_aug)