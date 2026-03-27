import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.team07_DVMSR import DVMSR
from torchsummary import summary

import json   # CHANGE
import csv    # CHANGE
import math   # CHANGE


# ================= DATASET =================
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
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


# ================= METRICS =================
def calculate_psnr(sr, hr):  # CHANGE
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / torch.sqrt(mse).item())


def calculate_ssim(sr, hr):  # CHANGE
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = sr.mean()
    mu_y = hr.mean()

    sigma_x = ((sr - mu_x) ** 2).mean()
    sigma_y = ((hr - mu_y) ** 2).mean()
    sigma_xy = ((sr - mu_x) * (hr - mu_y)).mean()

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return ssim.item()


# ================= TRAIN =================
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs,
          start_epoch=0, best_loss=float('inf'), best_epoch=0,
          args=None, log_path=None):  # CHANGE

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(start_epoch, epochs):

        # -------- TRAIN --------
        model.train()
        total_train_loss = 0   # CHANGE
        total_train_samples = 0  # CHANGE

        for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = lr.size(0)  # CHANGE
            total_train_loss += loss.item() * batch_size  # CHANGE
            total_train_samples += batch_size  # CHANGE

        avg_train_loss = total_train_loss / total_train_samples  # CHANGE

        # -------- VALIDATION --------
        model.eval()
        total_val_loss = 0  # CHANGE
        total_val_samples = 0  # CHANGE
        psnr_total = 0
        ssim_total = 0
        total_images = 0  # CHANGE

        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)

                loss = criterion(sr, hr)

                batch_size = lr.size(0)
                total_val_loss += loss.item() * batch_size  # CHANGE
                total_val_samples += batch_size  # CHANGE

                # -------- PER-IMAGE METRICS --------  # CHANGE
                for i in range(batch_size):
                    psnr_total += calculate_psnr(sr[i], hr[i])
                    ssim_total += calculate_ssim(sr[i], hr[i])

                total_images += batch_size  # CHANGE

        avg_val_loss = total_val_loss / total_val_samples  # CHANGE
        avg_psnr = psnr_total / total_images  # CHANGE
        avg_ssim = ssim_total / total_images  # CHANGE

        print(f"Epoch [{epoch+1}/{epochs}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

        # -------- LOGGING --------
        if log_path:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_psnr, avg_ssim])

        # -------- CURRENT CHECKPOINT --------
        if args and args.checkpoint_type == "full":
            current_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'args': vars(args)
            }
            torch.save(current_checkpoint, "checkpoints/current_checkpoint.pth")

        # -------- BEST CHECKPOINT --------
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch + 1

            torch.save(model.state_dict(), "checkpoints/best_model.pth")

            if args and args.checkpoint_type == "full":
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'psnr': avg_psnr,
                    'ssim': avg_ssim,
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'args': vars(args)
                }
                torch.save(best_checkpoint, "checkpoints/best_checkpoint.pth")

            print(f"✅ Saved BEST model at epoch {best_epoch}")

    print("Training complete.")


# ================= MAIN =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/valid")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--show_summary", type=int, default=1)
    parser.add_argument("--checkpoint_type", type=str, default="full",
                        choices=["simple", "full"])  # CHANGE

    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    # -------- CONFIG --------
    config_path = "checkpoints/config.json"  # CHANGE
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)

    # -------- LOG FILE --------
    log_path = "checkpoints/logs.csv"  # CHANGE
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "psnr", "ssim"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DVMSR().to(device)

    pretrained_path = "model_zoo/team07_DVMSR.pth"
    checkpoint_full = "checkpoints/current_checkpoint.pth"

    start_epoch = 0
    best_loss = float('inf')
    best_epoch = 0

    if args.resume and os.path.isfile(checkpoint_full):
        checkpoint = torch.load(checkpoint_full, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']

        print(f"⏯ Resumed from epoch {start_epoch}")
    else:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("Loaded pretrained weights")

    transform = transforms.Compose([transforms.ToTensor()])

    train_loader = DataLoader(
        SRDataset(os.path.join(args.train_dir, "LR"),
                  os.path.join(args.train_dir, "HR"), transform),
        batch_size=args.batch_size, shuffle=True
    )

    val_loader = DataLoader(
        SRDataset(os.path.join(args.val_dir, "LR"),
                  os.path.join(args.val_dir, "HR"), transform),
        batch_size=1, shuffle=False
    )

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume and os.path.isfile(checkpoint_full):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args.show_summary:
        summary(model, input_size=(3, 64, 64))

    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        args.epochs,
        start_epoch=start_epoch,
        best_loss=best_loss,
        best_epoch=best_epoch,
        args=args,
        log_path=log_path
    )