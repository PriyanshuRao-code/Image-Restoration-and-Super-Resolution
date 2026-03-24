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
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)

        return lr, hr


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                val_loss += criterion(sr, hr).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Saved best model")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/train", help="Path to training folder")
    parser.add_argument("--val_dir", type=str, default="data/valid", help="Path to validation folder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_layers", type=int, default=10, help="Number of layers to unfreeze (0 = freeze all)")
    parser.add_argument("--resume", type=int, default=1, help="0 = use pretrained, 1 = resume from checkpoint")
    parser.add_argument("--show_summary", type=int, default=1, help="Show model summary (1 = yes, 0 = no)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Using device: {device}")

    model = DVMSR().to(device)


    pretrained_path = "model_zoo/team07_DVMSR.pth"
    checkpoint_path = "checkpoints/best_model.pth"
    if args.resume and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"⏯ Resumed training from checkpoint: {checkpoint_path}")
    else:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Loaded pretrained weights from: {pretrained_path}")


    all_params = list(model.parameters())
    if args.unfreeze_layers == 0:
        for p in all_params:
            p.requires_grad = False
    elif args.unfreeze_layers >= len(all_params):
        for p in all_params:
            p.requires_grad = True
    else:
        for p in all_params[:-args.unfreeze_layers]:
            p.requires_grad = False
        for p in all_params[-args.unfreeze_layers:]:
            p.requires_grad = True

    frozen = sum(not p.requires_grad for p in all_params)
    trainable = sum(p.requires_grad for p in all_params)
    print(f"Frozen layers: {frozen} | Trainable layers: {trainable}")


    if args.show_summary:
        summary(model, input_size=(3, 64, 64))


    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = SRDataset(os.path.join(args.train_dir, "LR"), os.path.join(args.train_dir, "HR"), transform)
    val_dataset = SRDataset(os.path.join(args.val_dir, "LR"), os.path.join(args.val_dir, "HR"), transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    criterion = nn.L1Loss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)
