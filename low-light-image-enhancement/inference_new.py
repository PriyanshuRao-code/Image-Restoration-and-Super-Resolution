import os
import torch
import numpy as np
import csv
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2

from models.wrappers import get_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ================= GAMMA =================
def gamma_correction(x):
    return torch.clamp(x, 0, 1) ** (1/2.2)


# ================= LOAD =================
def load_model(model_type, ckpt_path, device):

    model = get_model(model_type).to(device)

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


# ================= MATCH SIZE =================
def match_size(img, ref):
    if img.shape != ref.shape:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
    return img


# ================= MAIN =================
def evaluate(models_list, lr_dir, hr_dir, output_root):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    for m in models_list:

        name = m["name"]
        model_type = m["type"]
        ckpt = m["ckpt"]
        use_gamma = m.get("gamma", False)

        print(f"\n🔷 Evaluating {name}")

        model = load_model(model_type, ckpt, device)

        base = f"{output_root}/{name}"
        os.makedirs(base, exist_ok=True)

        lr_out = f"{base}/LR"
        hr_out = f"{base}/HR"
        sr_out = f"{base}/SR"

        os.makedirs(lr_out, exist_ok=True)
        os.makedirs(hr_out, exist_ok=True)
        os.makedirs(sr_out, exist_ok=True)

        rows = []
        psnr_list = []
        ssim_list = []

        for img_name in tqdm(sorted(os.listdir(lr_dir))):

            lr = Image.open(f"{lr_dir}/{img_name}").convert("RGB")
            hr = Image.open(f"{hr_dir}/{img_name}").convert("RGB")

            lr_tensor = transform(lr).unsqueeze(0).to(device)
            hr_tensor = transform(hr).unsqueeze(0).to(device)

            if use_gamma:
                lr_tensor = gamma_correction(lr_tensor)

            with torch.no_grad():
                sr_tensor = model(lr_tensor)

            sr = sr_tensor.squeeze().cpu().numpy().transpose(1,2,0)
            hr_np = hr_tensor.squeeze().cpu().numpy().transpose(1,2,0)
            lr_np = lr_tensor.squeeze().cpu().numpy().transpose(1,2,0)

            sr = (sr*255).clip(0,255).astype(np.uint8)
            hr_np = (hr_np*255).clip(0,255).astype(np.uint8)
            lr_np = (lr_np*255).clip(0,255).astype(np.uint8)

            sr = match_size(sr, hr_np)

            p = psnr(hr_np, sr, data_range=255)
            s = ssim(hr_np, sr, channel_axis=2, data_range=255)

            psnr_list.append(p)
            ssim_list.append(s)

            rows.append([img_name, p, s])

            Image.fromarray(lr_np).save(f"{lr_out}/{img_name}")
            Image.fromarray(sr).save(f"{sr_out}/{img_name}")
            Image.fromarray(hr_np).save(f"{hr_out}/{img_name}")

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        with open(f"{base}/metrics.csv","w") as f:
            writer = csv.writer(f)
            writer.writerow(["image","psnr","ssim"])
            writer.writerows(rows)
            writer.writerow([])
            writer.writerow(["AVERAGE", avg_psnr, avg_ssim])

        print(f"✅ {name} → PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.3f}")