import torch
import os
from PIL import Image
from torchvision import transforms
from models.team07_DVMSR import DVMSR
import utils.utils_image as util 

def generate_images(model_checkpoint_path, lr_image_path, hr_image_path, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DVMSR(depths=[2,2]).to(device) # for student model
    # model = DVMSR().to(device) # for full model
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.eval()


    lr_image = Image.open(lr_image_path).convert('RGB')
    hr_image = Image.open(hr_image_path).convert('RGB')
    

    transform = transforms.Compose([transforms.ToTensor()])
    lr_tensor = transform(lr_image).unsqueeze(0).to(device)
    hr_tensor = transform(hr_image).unsqueeze(0).to(device)


    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    

    sr_image = util.tensor2uint(sr_tensor.squeeze(0).cpu(), data_range=1)
    hr_image = util.tensor2uint(hr_tensor.squeeze(0).cpu(), data_range=1)
    lr_image = util.tensor2uint(lr_tensor.squeeze(0).cpu(), data_range=1)


    os.makedirs(save_dir, exist_ok=True)
    lr_image_pth = os.path.join(save_dir, 'lr_image.png')
    hr_image_pth = os.path.join(save_dir, 'hr_image.png')
    sr_image_pth = os.path.join(save_dir, 'sr_image.png')

    Image.fromarray(lr_image).save(lr_image_pth)
    Image.fromarray(hr_image).save(hr_image_pth)
    Image.fromarray(sr_image).save(sr_image_pth)

    print(f"Images saved in {save_dir}")
    return lr_image_pth, hr_image_pth, sr_image_pth


if __name__ == "__main__":
    lr_image_path = "data/valid/LR/00000003_004.png"
    hr_image_path = "data/valid/HR/00000003_004.png"
    # model_checkpoint_path = "checkpoints/best_model.pth"  # For full model
    model_checkpoint_path = "checkpoints_distilled/student_best.pth"  # For student model 
    # save_dir = "generated_images" # For full model
    save_dir = "generated_images_student" # For student model
    
    lr_image_pth, hr_image_pth, sr_image_pth = generate_images(model_checkpoint_path, lr_image_path, hr_image_path, save_dir)
    print(f"Generated Images: \nLR: {lr_image_pth} \nHR: {hr_image_pth} \nSR: {sr_image_pth}")