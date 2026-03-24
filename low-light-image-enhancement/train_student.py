import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.team07_DVMSR import DVMSR


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


def train(student, teacher, train_loader, val_loader, criterion_rec, criterion_kd, optimizer, device, epochs):
    best_loss = float('inf')
    os.makedirs("checkpoints_distilled", exist_ok=True)

    λ_rec = 1.0  # Reconstruction loss weight
    λ_kd = 0.5   # Distillation loss weight

    for epoch in range(epochs):
        student.train()
        train_loss = 0

        for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            lr, hr = lr.to(device), hr.to(device)

            with torch.no_grad():
                sr_teacher = teacher(lr)

            sr_student = student(lr)

            loss_rec = criterion_rec(sr_student, hr)
            loss_kd = criterion_kd(sr_student, sr_teacher)

            loss = λ_rec * loss_rec + λ_kd * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | KD Train Loss: {avg_train_loss:.4f}")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(student.state_dict(), "checkpoints_distilled/student_best.pth")
            print("Saved best student model")

    print("\n Knowledge Distillation Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/valid")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Using device:", device)

    teacher = DVMSR().to(device)
    teacher.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Teacher Loaded (Frozen)")

    student = DVMSR(depths=[2,2]).to(device)
    print("Student Initialized (depths=[2,2])")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = SRDataset(os.path.join(args.train_dir, "LR"), os.path.join(args.train_dir, "HR"), transform)
    val_dataset   = SRDataset(os.path.join(args.val_dir, "LR"),   os.path.join(args.val_dir, "HR"), transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)


    criterion_rec = nn.L1Loss()
    criterion_kd  = nn.L1Loss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)


    train(student, teacher, train_loader, val_loader, criterion_rec, criterion_kd, optimizer, device, args.epochs)
