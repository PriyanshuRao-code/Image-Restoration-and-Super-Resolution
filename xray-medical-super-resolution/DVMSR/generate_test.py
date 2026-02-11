import os
import glob
import torch
import json
import logging
from fvcore.nn import FlopCountAnalysis
from pprint import pprint
import sys
# sys.stdout = open("logs.txt","w") # for full model
sys.stdout = open("student_logs.txt","w") # for student model


from utils import utils_logger
from utils import utils_image as util
from models.team07_DVMSR import DVMSR


def load_dvmsr_model(weights_path, device):
    # model = DVMSR() # for full model
    model = DVMSR(depths=[2,2]) # for student model
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    return model


def forward(img_lq, model, scale=4):
    with torch.no_grad():
        return model(img_lq)


def run_evaluation(model, model_name, data_range, device, args, mode="valid"):
    sf = 4
    border = sf
    results = {f"{mode}_psnr": [], f"{mode}_ssim": [], f"{mode}_runtime": []}

    lr_dir = args.lr_dir
    hr_dir = args.hr_dir

    lr_images = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
    hr_images = sorted(glob.glob(os.path.join(hr_dir, "*.png")))

    assert len(lr_images) == len(hr_images), \
        f"Number of LR ({len(lr_images)}) and HR ({len(hr_images)}) images must match."

    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_lr_path, img_hr_path) in enumerate(zip(lr_images, hr_images)):
        img_name = os.path.basename(img_hr_path)

        img_lr = util.imread_uint(img_lr_path, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range).to(device)

        img_hr = util.imread_uint(img_hr_path, n_channels=3)
        img_hr = util.modcrop(img_hr, sf)


        start.record()
        img_sr = forward(img_lr, model)
        end.record()
        torch.cuda.synchronize()

        runtime = start.elapsed_time(end)  # milliseconds
        results[f"{mode}_runtime"].append(runtime)

        img_sr = util.tensor2uint(img_sr, data_range)



        psnr = util.calculate_psnr(img_sr, img_hr, border=border)
        ssim = util.calculate_ssim(img_sr, img_hr, border=border)

        results[f"{mode}_psnr"].append(psnr)
        results[f"{mode}_ssim"].append(ssim)

        print(f"{img_name}: PSNR = {psnr:.2f} dB | SSIM = {ssim:.4f} | Time = {runtime:.2f} ms")

        util.imsave(img_sr, os.path.join(save_path, img_name))


    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"])

    print("\n------ Evaluation Summary ------")
    print(f"Average PSNR: {results[f'{mode}_ave_psnr']:.3f} dB")
    print(f"Average SSIM: {results[f'{mode}_ave_ssim']:.4f}")
    print(f"Average Runtime: {results[f'{mode}_ave_runtime']:.3f} ms")

    return results


def main(args):
    utils_logger.logger_info("DVMSR-Evaluation", log_path="DVMSR_eval.log")
    logger = logging.getLogger("DVMSR-Evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_dvmsr_model(args.weights, device)
    model_name = "DVMSR"
    data_range = 1.0

    results = run_evaluation(model, model_name, data_range, device, args, mode=args.mode)

    input_fake = torch.rand(1, 3, 256, 256).to(device)
    flops = FlopCountAnalysis(model, input_fake).total() / 1e9
    params = sum(p.numel() for p in model.parameters()) / 1e6

    logger.info(f"FLOPs: {flops:.4f} G | Params: {params:.4f} M")

    results["flops"] = flops
    results["params"] = params

    # with open("dvmsr_results.json", "w") as f: # for full model
    with open("dvmsr_student_results.json", "w") as f: # for student model
        json.dump(results, f, indent=4)
    sys.stdout.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("DVMSR Evaluation Script")

    # parser.add_argument("--weights", type=str, default= "./checkpoints/best_model.pth", help="Path to trained DVMSR weights (.pth)") # for full model
    parser.add_argument("--weights", type=str, default= "./checkpoints_distilled/student_best.pth", help="Path to trained DVMSR weights (.pth)") # for student model
    parser.add_argument("--lr_dir", type=str, default ="./data/valid/LR", help="Path to folder containing LR images")
    parser.add_argument("--hr_dir", type=str, default ="./data/valid/HR", help="Path to folder containing HR images")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save outputs")
    parser.add_argument("--mode", type=str, default="valid", help="Mode name for saving results folder (valid/test/train)")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)