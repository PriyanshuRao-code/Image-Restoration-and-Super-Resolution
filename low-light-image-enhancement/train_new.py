import os
import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

from models.team07_DVMSR import DVMSR


# ================= DATASET =================
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))

        assert len(self.lr_images) == len(self.hr_images), "Dataset size mismatch"

        for l, h in zip(self.lr_images, self.hr_images):
            assert l == h, f"Mismatch: {l} vs {h}"

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = Image.open(os.path.join(self.lr_dir, self.lr_images[idx])).convert("RGB")
        hr = Image.open(os.path.join(self.hr_dir, self.hr_images[idx])).convert("RGB")

        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)

        return lr, hr


# ================= ILLUMINATION MODULE =================
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


# ================= FULL MODEL =================
class FullModel(nn.Module):
    def __init__(self, dvmsr):
        super().__init__()
        self.illum = IlluminationModule()
        self.dvmsr = dvmsr

    def forward(self, x):
        x = self.illum(x)
        x = self.dvmsr(x)
        return x


# ================= METRICS =================
def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / torch.sqrt(mse).item())


from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim(sr, hr):
    sr = sr.detach().cpu().numpy().transpose(1,2,0)
    hr = hr.detach().cpu().numpy().transpose(1,2,0)
    return ssim(hr, sr, channel_axis=2, data_range=1)


# ================= PERCEPTUAL LOSS =================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights="IMAGENET1K_V1").features[:35].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, sr, hr):
        return nn.functional.l1_loss(self.vgg(sr), self.vgg(hr))


# ================= TRAIN =================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained DVMSR
    dvmsr = DVMSR().to(device)
    dvmsr.load_state_dict(torch.load("model_zoo/team07_DVMSR.pth", map_location=device))

    model = FullModel(dvmsr).to(device)

    # Freeze DVMSR except last layers
    for p in model.dvmsr.parameters():
        p.requires_grad = False

    for p in model.dvmsr.layers[-1].parameters():
        p.requires_grad = True

    for p in model.dvmsr.conv_after_body.parameters():
        p.requires_grad = True

    for p in model.dvmsr.upsample.parameters():
        p.requires_grad = True

    # Dataset
    transform = transforms.ToTensor()

    train_loader = DataLoader(
        SRDataset("data/train/LR", "data/train/HR", transform),
        batch_size=8, shuffle=True
    )

    val_loader = DataLoader(
        SRDataset("data/valid/LR", "data/valid/HR", transform),
        batch_size=1, shuffle=False
    )

    # Loss
    perceptual_loss = PerceptualLoss().to(device)

    def total_loss(sr, hr):
        l1 = nn.functional.l1_loss(sr, hr)
        ssim_loss = 1 - calculate_ssim(sr[0], hr[0])  # batch=1 safe usage
        perc = perceptual_loss(sr, hr)

        return l1 + 0.05 * ssim_loss + 0.005 * perc

    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5
    )

    # CSV Logging
    os.makedirs("checkpoints_new", exist_ok=True)
    csv_path = "checkpoints_new/training_log.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "psnr", "ssim"])

    epochs = 100
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        count = 0

        for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)
            loss = total_loss(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += 1

        avg_train = train_loss / count

        # Validation
        model.eval()
        val_loss = 0
        psnr_total = 0
        ssim_total = 0
        count = 0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)

                sr = model(lr)
                loss = total_loss(sr, hr)

                val_loss += loss.item()
                psnr_total += calculate_psnr(sr, hr)
                ssim_total += calculate_ssim(sr[0], hr[0])

                count += 1

        avg_val = val_loss / count
        avg_psnr = psnr_total / count
        avg_ssim = ssim_total / count

        print(f"\nEpoch {epoch+1}: Train={avg_train:.4f} Val={avg_val:.4f} PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f}")

        # Save CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train, avg_val, avg_psnr, avg_ssim])

        # Save best model
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "checkpoints_new/best_model.pth")
            print("✅ Saved best model")


if __name__ == "__main__":
    train()