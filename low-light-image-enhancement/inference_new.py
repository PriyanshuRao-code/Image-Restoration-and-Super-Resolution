import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
import cv2

from models.team07_DVMSR import DVMSR
import utils.utils_image as util
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ================= MODULES =================
import torch.nn as nn

class IlluminationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        illum = self.net(x)
        return x * illum


class TextureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.net(x)


class FullModel(nn.Module):
    def __init__(self, dvmsr):
        super().__init__()
        self.illum = IlluminationModule()
        self.dvmsr = dvmsr
        self.texture = TextureBlock()

    def forward(self, x):
        x = self.illum(x)
        x = self.dvmsr(x)
        x = self.texture(x)
        return x


# ================= LOAD MODEL =================
def load_model(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dvmsr = DVMSR().to(device)
    model = FullModel(dvmsr).to(device)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    return model, device


# ================= UTILS =================
def match_size(img, ref):
    if img.shape != ref.shape:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
    return img


# ================= INFERENCE =================
def run_inference(lr_dir, hr_dir, checkpoint, output_dir, csv_path):
    model, device = load_model(checkpoint)

    transform = transforms.ToTensor()

    # Create folders
    lr_out = os.path.join(output_dir, "LR")
    sr_out = os.path.join(output_dir, "SR")
    hr_out = os.path.join(output_dir, "HR")

    os.makedirs(lr_out, exist_ok=True)
    os.makedirs(sr_out, exist_ok=True)
    os.makedirs(hr_out, exist_ok=True)

    psnr_list = []
    ssim_list = []

    rows = []

    for name in tqdm(sorted(os.listdir(lr_dir))):
        lr_path = os.path.join(lr_dir, name)
        hr_path = os.path.join(hr_dir, name)

        if not os.path.exists(hr_path):
            continue

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_tensor = transform(lr_img).unsqueeze(0).to(device)
        hr_tensor = transform(hr_img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # Convert
        sr_img = util.tensor2uint(sr_tensor.squeeze(0).cpu(), data_range=1)
        hr_img = util.tensor2uint(hr_tensor.squeeze(0).cpu(), data_range=1)
        lr_img = util.tensor2uint(lr_tensor.squeeze(0).cpu(), data_range=1)

        # Match size
        sr_img = match_size(sr_img, hr_img)

        # Metrics
        psnr_val = psnr(hr_img, sr_img, data_range=255)
        ssim_val = ssim(hr_img, sr_img, channel_axis=2, data_range=255)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        rows.append([name, psnr_val, ssim_val])

        # Save images
        Image.fromarray(lr_img).save(os.path.join(lr_out, name))
        Image.fromarray(sr_img).save(os.path.join(sr_out, name))
        Image.fromarray(hr_img).save(os.path.join(hr_out, name))

    # Averages
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "psnr", "ssim"])

        for row in rows:
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["AVERAGE", avg_psnr, avg_ssim])

    print("\n📊 FINAL RESULTS:")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")


# ================= MAIN =================
if __name__ == "__main__":
    lr_dir = "data/valid/LR"
    hr_dir = "data/valid/HR"

    checkpoint = "checkpoints_new/latest_model.pth"

    output_dir = "results"
    csv_path = "results/metrics.csv"

    run_inference(lr_dir, hr_dir, checkpoint, output_dir, csv_path)