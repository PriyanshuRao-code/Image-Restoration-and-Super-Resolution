import torch
import os
import csv
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import cv2

from models.team07_DVMSR import DVMSR
import utils.utils_image as util


def load_model(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DVMSR().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    return model, device


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def match_size(img, ref):
    if img.shape != ref.shape:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
    return img


def run_inference_and_metrics(lr_dir, hr_dir, checkpoint, csv_path):
    model, device = load_model(checkpoint)

    transform = transforms.ToTensor()

    psnr_list = []
    ssim_list = []

    rows = []

    image_names = sorted(os.listdir(lr_dir))

    for name in tqdm(image_names):
        lr_path = os.path.join(lr_dir, name)
        hr_path = os.path.join(hr_dir, name)

        if not os.path.exists(hr_path):
            continue

        # Load images
        lr_img = load_image(lr_path)
        hr_img = load_image(hr_path)

        lr_tensor = transform(lr_img).unsqueeze(0).to(device)
        hr_tensor = transform(hr_img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # Convert to numpy
        sr_img = util.tensor2uint(sr_tensor.squeeze(0).cpu(), data_range=1)
        hr_img = util.tensor2uint(hr_tensor.squeeze(0).cpu(), data_range=1)

        # Match size if needed
        sr_img = match_size(sr_img, hr_img)

        # Metrics
        psnr_val = psnr(hr_img, sr_img, data_range=255)
        ssim_val = ssim(hr_img, sr_img, channel_axis=2, data_range=255)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        rows.append([name, psnr_val, ssim_val])

    # Averages
    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "psnr", "ssim"])

        for row in rows:
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["AVERAGE", avg_psnr, avg_ssim])

    print("\n📊 FINAL RESULTS:")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    lr_dir = "data/valid/LR"   # change this
    hr_dir = "data/valid/HR"   # change this
    checkpoint = "checkpoints/best_model.pth"

    csv_path = "inference_metrics.csv"

    run_inference_and_metrics(lr_dir, hr_dir, checkpoint, csv_path)