import torch
import os
import csv
from PIL import Image
from torchvision import transforms
from models.team07_DVMSR import DVMSR
import utils.utils_image as util
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def load_model(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DVMSR().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    return model, device


def process_folder(model, device, noisy_dir, hr_dir, noise_type, csv_writer, output_root):
    transform = transforms.ToTensor()

    # 🔥 Folder structure
    base_dir = os.path.join(output_root, noise_type)
    lr_dir = os.path.join(base_dir, "LR")
    sr_dir = os.path.join(base_dir, "SR")
    hr_out_dir = os.path.join(base_dir, "HR")

    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(hr_out_dir, exist_ok=True)

    # 🔥 For averages
    psnr_list = []
    ssim_list = []

    for img_name in tqdm(os.listdir(noisy_dir)):
        noisy_path = os.path.join(noisy_dir, img_name)
        hr_path = os.path.join(hr_dir, img_name)

        if not os.path.exists(hr_path):
            continue

        noisy_img = Image.open(noisy_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')

        noisy_tensor = transform(noisy_img).unsqueeze(0).to(device)
        hr_tensor = transform(hr_img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(noisy_tensor)

        # Convert
        sr_img = util.tensor2uint(sr_tensor.squeeze(0).cpu(), data_range=1)
        hr_img = util.tensor2uint(hr_tensor.squeeze(0).cpu(), data_range=1)
        lr_img = util.tensor2uint(noisy_tensor.squeeze(0).cpu(), data_range=1)

        # Metrics
        psnr_val = psnr(hr_img, sr_img, data_range=255)
        ssim_val = ssim(hr_img, sr_img, channel_axis=2, data_range=255)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        csv_writer.writerow([img_name, noise_type, psnr_val, ssim_val])

        # 🔥 Save images in folders
        Image.fromarray(lr_img).save(os.path.join(lr_dir, img_name))
        Image.fromarray(sr_img).save(os.path.join(sr_dir, img_name))
        Image.fromarray(hr_img).save(os.path.join(hr_out_dir, img_name))

    # 🔥 Compute averages
    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0

    print(f"\n📊 {noise_type.upper()} AVERAGE:")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}\n")

    return avg_psnr, avg_ssim


def evaluate_all(noisy_root, hr_dir, checkpoint, csv_path, output_root):
    model, device = load_model(checkpoint)

    overall_results = {}

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "noise_type", "psnr", "ssim"])

        for noise_type in ["gaussian", "salt_pepper", "poisson"]:
            noisy_dir = os.path.join(noisy_root, noise_type)

            avg_psnr, avg_ssim = process_folder(
                model,
                device,
                noisy_dir,
                hr_dir,
                noise_type,
                writer,
                output_root
            )

            overall_results[noise_type] = (avg_psnr, avg_ssim)

    # 🔥 Final Summary
    print("\n================ FINAL SUMMARY ================")
    for noise_type, (p, s) in overall_results.items():
        print(f"{noise_type}: PSNR={p:.4f}, SSIM={s:.4f}")
    print("==============================================\n")


if __name__ == "__main__":
    noisy_root = "noisy_images"
    hr_dir = "lol_dataset/eval15/high"
    checkpoint = "checkpoints/best_model.pth"

    csv_path = "metrics_results.csv"
    output_root = "output_results"

    evaluate_all(noisy_root, hr_dir, checkpoint, csv_path, output_root)