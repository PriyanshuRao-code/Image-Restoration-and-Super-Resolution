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


def match_size(img, ref):
    if img.shape != ref.shape:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
    return img


def compare_models(lr_dir, hr_dir, ckpt1, ckpt2, csv_path=None):
    model1, device = load_model(ckpt1)
    model2, _ = load_model(ckpt2)

    transform = transforms.ToTensor()

    psnr1_list, ssim1_list = [], []
    psnr2_list, ssim2_list = [], []

    rows = []

    image_names = sorted(os.listdir(lr_dir))

    for name in tqdm(image_names):
        lr_path = os.path.join(lr_dir, name)
        hr_path = os.path.join(hr_dir, name)

        if not os.path.exists(hr_path):
            continue

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_tensor = transform(lr_img).unsqueeze(0).to(device)
        hr_tensor = transform(hr_img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr1 = model1(lr_tensor)
            sr2 = model2(lr_tensor)

        sr1_img = util.tensor2uint(sr1.squeeze(0).cpu(), data_range=1)
        sr2_img = util.tensor2uint(sr2.squeeze(0).cpu(), data_range=1)
        hr_img = util.tensor2uint(hr_tensor.squeeze(0).cpu(), data_range=1)

        # Ensure same size
        sr1_img = match_size(sr1_img, hr_img)
        sr2_img = match_size(sr2_img, hr_img)

        # Metrics
        psnr1 = psnr(hr_img, sr1_img, data_range=255)
        ssim1 = ssim(hr_img, sr1_img, channel_axis=2, data_range=255)

        psnr2 = psnr(hr_img, sr2_img, data_range=255)
        ssim2 = ssim(hr_img, sr2_img, channel_axis=2, data_range=255)

        psnr1_list.append(psnr1)
        ssim1_list.append(ssim1)

        psnr2_list.append(psnr2)
        ssim2_list.append(ssim2)

        rows.append([name, psnr1, ssim1, psnr2, ssim2])

    # Averages
    avg_psnr1 = np.mean(psnr1_list)
    avg_ssim1 = np.mean(ssim1_list)

    avg_psnr2 = np.mean(psnr2_list)
    avg_ssim2 = np.mean(ssim2_list)

    print("\n📊 FINAL COMPARISON:")
    print("Model 1:")
    print(f"  PSNR: {avg_psnr1:.4f}")
    print(f"  SSIM: {avg_ssim1:.4f}")

    print("\nModel 2:")
    print(f"  PSNR: {avg_psnr2:.4f}")
    print(f"  SSIM: {avg_ssim2:.4f}")

    print("\nΔ Improvement (Model2 - Model1):")
    print(f"  PSNR Gain: {avg_psnr2 - avg_psnr1:.4f}")
    print(f"  SSIM Gain: {avg_ssim2 - avg_ssim1:.4f}")

    # Optional CSV
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "psnr_model1", "ssim_model1", "psnr_model2", "ssim_model2"])
            writer.writerows(rows)

            writer.writerow([])
            writer.writerow(["AVERAGE",
                             avg_psnr1, avg_ssim1,
                             avg_psnr2, avg_ssim2])


if __name__ == "__main__":
    lr_dir = "data/valid/LR"
    hr_dir = "data/valid/LR"

    ckpt1 = "model_zoo/team07_DVMSR.pth"      # pretrained
    ckpt2 = "checkpoints/best_model.pth"      # fine-tuned

    csv_path = "model_comparison.csv"

    compare_models(lr_dir, hr_dir, ckpt1, ckpt2, csv_path)