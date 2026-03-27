#!/bin/bash

# ---- Data Preparation----
HR_TRAIN_DIR="LOLv2/Synthetic/train/low"
HR_VALID_DIR="LOLv2/Synthetic/valid/low"
HIGH_TRAIN_DIR="LOLv2/Synthetic/train/high"
HIGH_VALID_DIR="LOLv2/Synthetic/valid/high"
SCALE=4


OUTPUT_DIR="LOLv2_Synthetic_data"
mkdir -p $OUTPUT_DIR

# ---- Training ----
TRAIN_DIR="${OUTPUT_DIR}/train"
VAL_DIR="${OUTPUT_DIR}/valid"
EPOCHS=1
BATCH_SIZE=8
LR=2e-4
RESUME=1
SHOW_SUMMARY=1
CHECKPOINT_TYPE="full"   # options: simple | full

# ---- Inference ----
LR_IMAGE_PATH="${OUTPUT_DIR}/valid/LR/r068812d7t.png"
HR_IMAGE_PATH="${OUTPUT_DIR}/valid/HR/r068812d7t.png"
SAVE_DIR="${OUTPUT_DIR}_image_generation"
CHECKPOINTS="checkpoints_${OUTPUT_DIR}"
MODEL_CHECKPOINT_PATH="${CHECKPOINTS}/best_model.pth"


# =========================
# STEP 1: Generate Dataset
# =========================

echo "🚀 Step 1: Generating LR-HR dataset..."

python generate_LR_images.py \
    --hr_train_dir "$HR_TRAIN_DIR" \
    --hr_valid_dir "$HR_VALID_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --scale $SCALE


echo "📁 Copying HR folder into output_dir..."

cp -r "${HIGH_TRAIN_DIR}/"* "${OUTPUT_DIR}/train/HR/"
cp -r "${HIGH_VALID_DIR}/"* "${OUTPUT_DIR}/valid/HR/"


# =========================
# STEP 2: Training
# =========================

echo "🏋️ Training model..."

python train.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --resume $RESUME \
    --show_summary $SHOW_SUMMARY \
    --checkpoint_type $CHECKPOINT_TYPE\
    --checkpoints $CHECKPOINTS

# # =========================
# # STEP 3: Inference
# # =========================

echo "🎯 Generating output images..."

python image_gen.py \
    --lr_image_path "$LR_IMAGE_PATH" \
    --hr_image_path "$HR_IMAGE_PATH" \
    --model_checkpoint_path "$MODEL_CHECKPOINT_PATH" \
    --save_dir "$SAVE_DIR"

echo "✅ Pipeline completed successfully!"