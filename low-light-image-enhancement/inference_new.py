import os
import torch
import numpy as np
import csv
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.wrappers import get_model


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= GAMMA =================
def gamma_correction(x):
    return torch.clamp(x, 0, 1) ** (1 / 2.2)


# ================= METRICS =================
def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


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


# ================= LOAD MODEL =================
def load_model(model_type, ckpt_path):
    model = get_model(model_type).to(device)

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


# ================= FLOPs =================
def compute_flops(model, input_tensor):
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops, params
    except ImportError:
        print("⚠️ Install thop: pip install thop")
        return None, None


# ================= SIZE MATCH =================
def match_size(sr, hr):
    sr_h, sr_w = sr.shape[-2:]
    hr_h, hr_w = hr.shape[-2:]

    h = min(sr_h, hr_h)
    w = min(sr_w, hr_w)

    return sr[:, :h, :w], hr[:, :h, :w]


# ================= MAIN =================
def evaluate(models_list, lr_dir, hr_dir, output_root):

    transform = transforms.ToTensor()

    # ===== GLOBAL LR/HR OUTPUT (ONLY ONCE) =====
    lr_out_global = os.path.join(output_root, "LR")
    hr_out_global = os.path.join(output_root, "HR")

    os.makedirs(lr_out_global, exist_ok=True)
    os.makedirs(hr_out_global, exist_ok=True)

    saved_once = set()  # track saved images

    for m in models_list:

        name = m["name"]
        model_type = m["type"]
        ckpt = m["ckpt"]

        print(f"\n🔷 Evaluating {name}")

        model = load_model(model_type, ckpt)

        # ===== FLOPs (ONCE) =====
        dummy = torch.randn(1, 3, 64, 64).to(device)
        flops, params = compute_flops(model, dummy)

        if flops:
            print(f"⚙️ FLOPs: {flops/1e9:.3f} GFLOPs | Params: {params/1e6:.3f}M")

        # ===== OUTPUT =====
        base = os.path.join(output_root, name)
        sr_out = os.path.join(base, "SR")

        os.makedirs(sr_out, exist_ok=True)

        rows = []
        psnr_list = []
        ssim_list = []

        for img_name in tqdm(sorted(os.listdir(lr_dir))):

            lr = Image.open(os.path.join(lr_dir, img_name)).convert("RGB")
            hr = Image.open(os.path.join(hr_dir, img_name)).convert("RGB")

            lr_tensor = transform(lr).unsqueeze(0).to(device)
            hr_tensor = transform(hr).unsqueeze(0).to(device)

            # ===== GAMMA =====
            if model_type=="gamma" or model_type=="gamma_perceptual":
                lr_tensor = gamma_correction(lr_tensor)

            # ===== INFERENCE =====
            with torch.no_grad():
                sr_tensor = model(lr_tensor)

            sr_tensor = torch.clamp(sr_tensor, 0, 1)

            sr_tensor, hr_tensor = match_size(sr_tensor, hr_tensor)

            # ===== METRICS =====
            p = calculate_psnr(sr_tensor[0], hr_tensor[0]).item()
            s = calculate_ssim(sr_tensor[0], hr_tensor[0]).item()

            psnr_list.append(p)
            ssim_list.append(s)

            rows.append([img_name, p, s])

            # ===== SAVE =====
            lr_np = (lr_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            hr_np = (hr_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            sr_np = (sr_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Save LR/HR only once
            if img_name not in saved_once:
                Image.fromarray(lr_np).save(os.path.join(lr_out_global, img_name))
                Image.fromarray(hr_np).save(os.path.join(hr_out_global, img_name))
                saved_once.add(img_name)

            # Always save SR (per model)
            Image.fromarray(sr_np).save(os.path.join(sr_out, img_name))

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        # ===== CSV =====
        with open(os.path.join(base, "metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "psnr", "ssim"])
            writer.writerows(rows)
            writer.writerow([])
            writer.writerow(["AVERAGE", avg_psnr, avg_ssim])

        print(f"✅ {name} → PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}")


# ================= RUN =================
if __name__ == "__main__":

    metric ="ssim"
    models_list = [
        {"name": "baseline", "type": "baseline", "ckpt": f"experiments/baseline/Checkpoint/checkpoints/best_{metric}.pth"},
        {"name": "gamma", "type": "gamma", "ckpt": f"experiments/gamma/Checkpoint/checkpoints/best_{metric}.pth"},
        {"name": "perceptual", "type": "perceptual", "ckpt": f"experiments/perceptual/Checkpoint/checkpoints/best_{metric}.pth"},
        {"name": "full", "type": "full", "ckpt": f"experiments/full/Checkpoint/checkpoints/best_{metric}.pth"},
        {"name": "illum", "type": "illum", "ckpt": f"experiments/illum/Checkpoint/checkpoints/best_{metric}.pth"},
        {"name": "gamma_perceptual", "type": "gamma_perceptual", "ckpt": f"experiments/gamma_perceptual/Checkpoint/checkpoints/best_{metric}.pth"},
    ]

    evaluate(
        models_list,
        lr_dir="Dataset_aug/valid/LR",
        hr_dir="Dataset_aug/valid/HR",
        output_root=f"inference_models/eval/{metric}"
    )