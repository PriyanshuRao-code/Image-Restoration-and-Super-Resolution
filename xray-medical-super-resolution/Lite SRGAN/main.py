from Dataloader import DataLoader
from LiteSRGAN import LiteSRGAN, LiteSRGAN_engine
import argparse
import tensorflow as tf
import os
from tqdm import tqdm
import time
import numpy as np

parser = argparse.ArgumentParser(description='light-SRGAN training script.')

parser.add_argument('--images_dir', default=r"/content/drive/MyDrive/images_001/original_images",
                    type=str, help='provide a path containing all the data')
parser.add_argument('--img_width', default=128, type=int)
parser.add_argument('--img_height', default=128, type=int)
parser.add_argument('--upsampling_blocks', default=1, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--decay_steps', default=50000, type=int)
parser.add_argument('--decay_rate', default=0.1, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--pretraining_epochs', default=50, type=int)
parser.add_argument('--generator_weights', default=None, type=str)
parser.add_argument('--discriminator_weights', default=None, type=str)

args = parser.parse_args()

BASE_DIR = '/content/drive/MyDrive/LiteSRGAN_data'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
GEN_DIR = os.path.join(BASE_DIR, 'generatedTrails')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
PRETRAIN_DIR = os.path.join(CKPT_DIR, 'pretrain')
TRAIN_DIR = os.path.join(CKPT_DIR, 'training')

for d in [BASE_DIR, MODELS_DIR, GEN_DIR, CKPT_DIR, PRETRAIN_DIR, TRAIN_DIR]:
    os.makedirs(d, exist_ok=True)

print(f" All training data will be stored under: {BASE_DIR}")

def get_latest_checkpoint(dir_path, prefix):
    ckpts = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith('.weights.h5')]
    if not ckpts:
        return None, -1
    ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = ckpts[-1]
    epoch_num = int(latest.split('_')[-1].split('.')[0])
    return os.path.join(dir_path, latest), epoch_num


def get_latest_checkpoint_training(dir_path, prefix, model_type):
    ckpts = [f for f in os.listdir(dir_path)
             if f.startswith(prefix) and model_type in f and f.endswith('.weights.h5')]

    if not ckpts:
        return None, -1

    def extract_epoch(fname):
        parts = fname.split('_')
        for p in parts:
            if p.isdigit():
                return int(p)
        return -1

    ckpts.sort(key=extract_epoch)
    latest = ckpts[-1]
    epoch_num = extract_epoch(latest)
    return os.path.join(dir_path, latest), epoch_num


dl = DataLoader(args)
datagen = dl.dataGenerator()

lite_SRGAN = LiteSRGAN(args)
lite_SRGAN_engine = LiteSRGAN_engine(args, lite_SRGAN)
print("last layer shape of discriminator: ", lite_SRGAN.discriminator.output_shape)

latest_pre_ckpt, start_pre_epoch = get_latest_checkpoint(PRETRAIN_DIR, "pretraining_")

if latest_pre_ckpt:
    lite_SRGAN.generator.load_weights(latest_pre_ckpt)
    print(f" Resumed generator pretraining from {latest_pre_ckpt} (epoch {start_pre_epoch})")
else:
    print(" Starting generator pretraining from scratch...")
    start_pre_epoch = -1

for i in range(start_pre_epoch + 1, args.pretraining_epochs):
    print(f"------------- Pre-training epoch {i} -------------")
    lite_SRGAN_engine.generator_pretraining(datagen, i)

    save_path = os.path.join(PRETRAIN_DIR, f"pretraining_{i}.weights.h5")
    lite_SRGAN.generator.save_weights(save_path)
    print(f"💾 Saved pretraining checkpoint: {save_path}")

print("------------- End of generator pre-training -------------")

latest_train_gen, start_train_epoch = get_latest_checkpoint_training(TRAIN_DIR, "training_", "generator")
latest_train_disc, start_train_epoch_disc = get_latest_checkpoint_training(TRAIN_DIR, "training_", "discriminator")

if latest_train_gen and latest_train_disc and (start_train_epoch == start_train_epoch_disc):
    lite_SRGAN.generator.load_weights(latest_train_gen)
    lite_SRGAN.discriminator.load_weights(latest_train_disc)
    print(f" Resumed full training from epoch {start_train_epoch}")
else:
    print("⚙️ Starting full SRGAN training from scratch...")
    start_train_epoch = -1

for i in range(start_train_epoch + 1, args.epochs):
    print(f"============= SRGAN Training Epoch {i} =============")
    datagen = dl.dataGenerator()

    lite_SRGAN_engine.train(datagen, 100, i)
    lite_SRGAN_engine.saveTrails(4, i)

    gen_path = os.path.join(MODELS_DIR, f'generator_epoch_{i+1}.weights.h5')
    disc_path = os.path.join(MODELS_DIR, f'discriminator_epoch_{i+1}.weights.h5')
    lite_SRGAN.generator.save_weights(gen_path)
    lite_SRGAN.discriminator.save_weights(disc_path)

    gen_ckpt_path = os.path.join(TRAIN_DIR, f"training_{i}_generator.weights.h5")
    disc_ckpt_path = os.path.join(TRAIN_DIR, f"training_{i}_discriminator.weights.h5")
    lite_SRGAN.generator.save_weights(gen_ckpt_path)
    lite_SRGAN.discriminator.save_weights(disc_ckpt_path)

    print(f" Saved training checkpoint after epoch {i+1}")

    if (i + 1) % 7 == 0:
        print("\n Running Evaluation ...")
        psnr_scores, ssim_scores, runtime_list = [], [], []

        eval_gen = dl.dataGenerator()

        for _ in tqdm(range(50), desc="Evaluating metrics"):
            try:
                lr, hr,_= next(eval_gen)
            except StopIteration:
                break

            start = time.time()
            sr = lite_SRGAN.generator(lr, training=False)
            runtime_list.append(time.time() - start)

            sr = tf.clip_by_value(sr, 0.0, 1.0)
            hr = tf.clip_by_value(hr, 0.0, 1.0)

            min_b = min(sr.shape[0], hr.shape[0])
            sr = sr[:min_b]
            hr = hr[:min_b]

            for s, h in zip(sr, hr):
                psnr_scores.append(tf.image.psnr(s, h, max_val=1.0).numpy())
                ssim_scores.append(tf.image.ssim(s, h, max_val=1.0).numpy())

        if len(psnr_scores) > 0:
            print("\n ====== METRICS REPORT ======")
            print(f"Epoch: {i+1}")
            print(f"Average PSNR   : {np.mean(psnr_scores):.4f} dB")
            print(f"Average SSIM   : {np.mean(ssim_scores):.4f}")
            print(f"Average Runtime: {np.mean(runtime_list):.6f} sec/image\n")
        else:
            print(" Metrics skipped: No valid samples found.\n")

print(" Training completed and all data saved to Google Drive.")