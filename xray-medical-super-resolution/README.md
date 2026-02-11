# Credits
Priyanshu Rao
Mohit Raj S
Mohit Bhagoria

# Dataset
We have taken the data from https://nihcc.app.box.com/v/ChestXray-NIHCC which is of Chest Xray
We have shown just a single image for demostrating purpose in DVMSR folder and the statistics are there in .json files

# Script Usage Guide

---

# Lite-SRGAN

To run Lite-SRGAN, simply open and execute **SRGAN.ipynb** in Google Colab.
All required files are already linked through the notebook, as they are fetched directly from the GitHub repository.
We have built upon GitHub repository https://github.com/hosamsherif/LiteSRGAN-and-LiteUNet.git

The files needed for setting up the Google Colab are 

---

# DVMSR
We have built upon https://github.com/nathan66666/DVMSR.git
## 1. Generate LR–HR Pairs
```bash
python generate_LR_images.py --hr_dir "original_images"
```
This script:
- Reads all **HR images** from the given folder
- Generates corresponding **LR images**
- Splits the dataset into **train** and **valid** folders
- Produces the following structure:

```
data/
 ├── train/
 │    ├── HR/
 │    └── LR/
 └── valid/
      ├── HR/
      └── LR/
```

---

## 2. Run Evaluation (Generate Metrics + Logs)
```bash
python generate_test.py
```
This produces:
- `dvmsr_results.json`
- `dvmsr_student_results.json`
- Log file

Switch between **full model** and **student model**:
```python
# model = DVMSR()                 # Full model
model = DVMSR(depths=[2,2])       # Student model
```

---

## 3. Generate SR Image from LR Input
```bash
python image_gen.py
```
Use this to obtain **super-resolved images** for any given LR input.

---

## 4. Train Student Model (Knowledge Distillation)
```bash
python train_student.py
```
Teacher model: **4 RSSBs**
Student model: **2 RSSBs**

Outputs best checkpoint:
```
checkpoints_distilled/student_best.pth
```

---

## 5. Train Full Model
```bash
python train.py
```
This:
- Loads pretrained model
- Freezes first **10 layers**
- Fine-tunes remaining layers on **X-ray dataset**

Outputs:
```
checkpoints/best_model.pth

```
