import os
import random
from PIL import Image
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

def generate_LR_images(hr_train_dir, hr_valid_dir, output_dir, scale=4):
    # hr_images = [os.path.join(hr_dir, img) for img in os.listdir(hr_dir)
    #              if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # if len(hr_images) == 0:
    #     print("No images found in:", hr_dir)
    #     return
    
    # train_dir = os.path.join(hr_dir, "train")
    # valid_dir = os.path.join(hr_dir, "valid")
    train_dir = hr_train_dir
    valid_dir = hr_valid_dir

    train_imgs = [os.path.join(train_dir, img) for img in os.listdir(train_dir)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))]  #CHANGE
    
    val_imgs = [os.path.join(valid_dir, img) for img in os.listdir(valid_dir)
                if img.lower().endswith(('.png', '.jpg', '.jpeg'))]  #CHANGE

    if len(train_imgs) == 0 or len(val_imgs) == 0:  #CHANGE
        print("No images found in one of the directories")  #CHANGE
        return


    paths = [
        os.path.join(output_dir, 'train/HR'),
        os.path.join(output_dir, 'train/LR'),
        os.path.join(output_dir, 'valid/HR'),
        os.path.join(output_dir, 'valid/LR')
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)

    def process_images(img_list, mode):
        for img_path in tqdm(img_list, desc=f"Processing {mode} set"):
            img_name = os.path.basename(img_path)
            hr_img = Image.open(img_path).convert("RGB")
            w, h = hr_img.size
            lr_img = hr_img.resize((w // scale, h // scale), Image.BICUBIC)
            # hr_img.save(os.path.join(output_dir, f"{mode}/HR/{img_name}"))
            lr_img.save(os.path.join(output_dir, f"{mode}/LR/{img_name}"))

    process_images(train_imgs, "train")
    process_images(val_imgs, "valid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LR images and split dataset.")
    parser.add_argument("--hr_train_dir", type=str, required=True, help="Path to HR train images folder.")
    parser.add_argument("--hr_valid_dir", type=str, required=True, help="Path to HR valid images folder.")
    parser.add_argument("--output_dir", type=str, default="data", help="Output dataset folder.")
    parser.add_argument("--scale", type=int, default=4, help="Downscaling factor.")
    args = parser.parse_args()

    generate_LR_images(args.hr_train_dir, args.hr_valid_dir, args.output_dir, args.scale)
