import os
import argparse
import json
import csv
import math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm

from models.wrappers import get_model


# ================= DATASET =================
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=32):  # CHANGE
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.patch_size = patch_size  # CHANGE

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        lr = np.array(lr)
        hr = np.array(hr)

        h, w, _ = lr.shape
        ps = self.patch_size

        # ===== RANDOM CROP (FIX SIZE ISSUE) =====
        # CHANGE
        if h >= ps and w >= ps:
            top = np.random.randint(0, h - ps + 1)
            left = np.random.randint(0, w - ps + 1)

            lr = lr[top:top+ps, left:left+ps]
            hr = hr[top*4:(top+ps)*4, left*4:(left+ps)*4]

        # ===== TO TENSOR =====
        # CHANGE
        lr = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        hr = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0

        return lr, hr


# ================= GAMMA =================
def gamma_correction(x):
    return x ** (1 / 2.2)


# ================= PERCEPTUAL LOSS =================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights="IMAGENET1K_V1").features[:35].eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, sr, hr):
        sr = self.normalize(sr)
        hr = self.normalize(hr)
        return nn.functional.l1_loss(self.vgg(sr), self.vgg(hr))


# ================= METRICS =================
def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / torch.sqrt(mse).item())


def calculate_ssim(sr, hr):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = sr.mean()
    mu_y = hr.mean()
    sigma_x = ((sr - mu_x) ** 2).mean()
    sigma_y = ((hr - mu_y) ** 2).mean()
    sigma_xy = ((sr - mu_x) * (hr - mu_y)).mean()

    return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))


# ================= TRAIN =================
def train_single(args, model_type):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = f"experiments/{model_type}"
    ckpt_dir = f"{base}/Checkpoint/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    model = get_model(model_type).to(device)

    model.dvmsr.load_state_dict(
        torch.load("model_zoo/team07_DVMSR.pth", map_location=device)
    )

    # ===== FREEZE =====
    for p in model.dvmsr.parameters():
        p.requires_grad = False
    for p in model.dvmsr.layers[-1].parameters():
        p.requires_grad = True
    for p in model.dvmsr.upsample.parameters():
        p.requires_grad = True

    # ===== DATA LOADERS =====
    # CHANGE
    train_loader = DataLoader(
        SRDataset(f"{args.train_dir}/LR", f"{args.train_dir}/HR", patch_size=32),
        batch_size=args.batch_size,
        shuffle=True
    )

    # CHANGE
    val_loader = DataLoader(
        SRDataset(f"{args.val_dir}/LR", f"{args.val_dir}/HR", patch_size=32),
        batch_size=1,
        shuffle=False
    )

    criterion = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    log_path = f"{base}/Checkpoint/log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_psnr", "train_ssim", "val_psnr", "val_ssim"])

    best_loss = float("inf")
    best_psnr = -float("inf")
    best_ssim = -float("inf")

    for epoch in range(args.epochs):

        model.train()
        train_loss = train_psnr = train_ssim = 0
        train_samples = 0

        for lr, hr in tqdm(train_loader):

            lr, hr = lr.to(device), hr.to(device)

            if model_type == "gamma":
                lr = gamma_correction(lr)

            sr = model(lr)

            if model_type in ["perceptual", "full"]:
                loss = criterion(sr, hr) + 0.01 * perceptual_loss(sr, hr)
            else:
                loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = lr.size(0)
            train_loss += loss.item() * bs

            for i in range(bs):
                train_psnr += calculate_psnr(sr[i], hr[i])
                train_ssim += calculate_ssim(sr[i], hr[i]).item()

            train_samples += bs

        train_loss /= train_samples
        train_psnr /= train_samples
        train_ssim /= train_samples

        # ===== VALIDATION =====
        model.eval()
        val_loss = val_psnr = val_ssim = 0
        val_samples = 0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)

                if model_type == "gamma":
                    lr = gamma_correction(lr)

                sr = model(lr)

                if model_type in ["perceptual", "full"]:
                    loss = criterion(sr, hr) + 0.01 * perceptual_loss(sr, hr)
                else:
                    loss = criterion(sr, hr)

                val_loss += loss.item()
                val_psnr += calculate_psnr(sr[0], hr[0])
                val_ssim += calculate_ssim(sr[0], hr[0]).item()
                val_samples += 1

        val_loss /= val_samples
        val_psnr /= val_samples
        val_ssim /= val_samples

        print(f"{model_type} Epoch {epoch+1} | PSNR: {val_psnr:.2f}")

        # ===== SAVE =====
        torch.save(model.state_dict(), f"{ckpt_dir}/initial_format_epoch_{epoch+1}.pth")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, f"{ckpt_dir}/current.pth")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{ckpt_dir}/best_loss.pth")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), f"{ckpt_dir}/best_psnr.pth")

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), f"{ckpt_dir}/best_ssim.pth")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss, train_psnr, train_ssim, val_psnr, val_ssim])

    with open(f"{base}/Checkpoint/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"✅ Finished training {model_type}")


# ================= MAIN =================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="Dataset_aug/train")
    parser.add_argument("--val_dir", type=str, default="Dataset_aug/valid")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=["baseline", "gamma", "illum", "perceptual", "full", "all"]
    )

    args = parser.parse_args()

    MODEL_LIST = ["baseline", "gamma", "illum", "perceptual", "full"]

    if args.model_type == "all":
        for m in MODEL_LIST:
            print(f"\n🚀 Training {m}")
            train_single(args, m)
    else:
        train_single(args, args.model_type)


if __name__ == "__main__":
    main()