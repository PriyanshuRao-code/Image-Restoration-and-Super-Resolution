import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def add_gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    noisy = img.copy()
    h, w, c = img.shape

    num_salt = int(salt_prob * h * w)
    num_pepper = int(pepper_prob * h * w)

    # Salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

    # Pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1], :] = 0

    return noisy

def add_poisson_noise(img):
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img * vals) / float(vals)
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    gaussian_dir = os.path.join(output_dir, "gaussian")
    sp_dir = os.path.join(output_dir, "salt_pepper")
    poisson_dir = os.path.join(output_dir, "poisson")

    os.makedirs(gaussian_dir, exist_ok=True)
    os.makedirs(sp_dir, exist_ok=True)
    os.makedirs(poisson_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        g = add_gaussian_noise(img)
        sp = add_salt_pepper_noise(img)
        p = add_poisson_noise(img)

        cv2.imwrite(os.path.join(gaussian_dir, img_name), g)
        cv2.imwrite(os.path.join(sp_dir, img_name), sp)
        cv2.imwrite(os.path.join(poisson_dir, img_name), p)


if __name__ == "__main__":
    input_dir = "data/valid/LR"
    output_dir = "noisy_images"

    process_folder(input_dir, output_dir)