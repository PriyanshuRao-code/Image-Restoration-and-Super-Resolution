# Low-Light Image Super-Resolution (DVMSR-based)

An efficient super-resolution pipeline adapted from the **NTIRE Efficient SR
Challenge DVMSR baseline** and re-purposed for **low-light image
enhancement + 4× super-resolution**. The pipeline covers dataset
preparation, augmentation, training multiple low-light-aware model
variants, evaluation, knowledge distillation, and noise-robustness testing.

---

## File Descriptions

### Data Preparation
- **`dataset.py`** — Downloads the LOL low-light dataset via `kagglehub`.
- **`generate_LR_images.py`** — Takes HR train/valid folders and bicubic-downsamples them (default scale ×4) to create paired LR images.
- **`augmentation.py`** — Expands the LR-HR training pairs with horizontal/vertical flips and 90°/180°/270° rotations, writing the augmented set to `Dataset_aug/`.

### Training
- **`train.py`** — Full fine-tuning of the base DVMSR architecture on paired LR/HR data, with checkpoint resuming and logging.
- **`train_new.py`** — Fine-tunes the DVMSR backbone (with the output layers unfrozen) into six low-light-aware variants: `baseline`, `gamma`, `illum`, `perceptual`, `full`, and `gamma_perceptual`. Trains on the augmented dataset with random patch sampling, gamma-correction preprocessing, and an optional VGG perceptual loss. Saves best-by-loss / best-by-PSNR / best-by-SSIM checkpoints per variant under `experiments/<model_type>/Checkpoint/checkpoints/`.
- **`train_student.py`** — Knowledge distillation: freezes a trained teacher DVMSR model and trains a smaller student model (`depths=[2,2]`) to match both ground truth and teacher output, for a lighter-weight deployable model.

### Inference & Evaluation
- **`inference_new.py`** — Runs every trained model variant over the validation set, computes PSNR/SSIM, and saves SR outputs plus a per-model `metrics.csv` under `inference_models/`.
- **`image_gen.py`** — Generates and saves an LR/HR/SR image triplet for a single example, useful for quick visual checks (output saved to `generated_images_lol/`).
- **`generate_test.py`** — Evaluates a trained checkpoint end-to-end, including FLOPs/parameter count reporting.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Download the dataset
```bash
python dataset.py
```

### 2. Generate LR images from HR
```bash
python generate_LR_images.py \
    --hr_train_dir <path_to_hr_train> \
    --hr_valid_dir <path_to_hr_valid> \
    --output_dir data \
    --scale 4
```

### 3. Augment the training set
```bash
python augmentation.py
```

### 4. Train model variants
```bash
python train_new.py \
    --train_dir Dataset_aug/train \
    --val_dir Dataset_aug/valid \
    --model_type all
```
(Use `--model_type <baseline|gamma|illum|perceptual|full|gamma_perceptual>` to train a single variant.)

### 5. Evaluate all trained variants
```bash
python inference_new.py
```

### 6. Generate a single LR/HR/SR sample
```bash
python image_gen.py \
    --lr_image_path <path> \
    --hr_image_path <path> \
    --model_checkpoint_path <path> \
    --save_dir generated_images_lol
```

### 7. (Optional) Distill a smaller student model
```bash
python train_student.py
```

### End-to-end pipeline
```bash
bash pipeline.sh
```

---

## Model Variants

| Variant | Description |
|---|---|
| `baseline` | Fine-tuned DVMSR without additional low-light handling |
| `gamma` | Applies gamma correction to the LR input before the forward pass |
| `illum` | Incorporates illumination-aware processing |
| `perceptual` | Adds a VGG-based perceptual loss during training |
| `full` | Combines illumination handling and perceptual loss |
| `gamma_perceptual` | Combines gamma correction and perceptual loss |

---

## From inference_models/

Best PSNR (overall)	perceptual → 18.136 dB
Best SSIM (overall)	perceptual → 0.8476

---

# Summary 

Developed a Low-Light Super Resolution (LLSR) pipeline using a pretrained DVMSR model with fine-tuning and modular enhancements.

Performed extensive ablation studies across multiple configurations including baseline, gamma correction, illumination module, perceptual loss, and hybrid combinations.

Achieved best performance using perceptual loss with:

PSNR improvement from 17.68 → 18.13 dB
SSIM improvement from 0.81 → 0.84

Demonstrated that perceptual feature-based optimization improves structural similarity and visual fidelity without increasing inference-time computational cost.

Showed that simple preprocessing (gamma correction) yields measurable gains with zero additional parameters or FLOPs, highlighting an efficient performance–complexity tradeoff.

Identified that combining illumination and perceptual modules introduces optimization conflicts under limited data conditions, leading to degraded performance.

Proposed a hybrid gamma + perceptual pipeline achieving near-optimal performance (18.09 dB PSNR, 0.84 SSIM) without any increase in model size or inference cost.