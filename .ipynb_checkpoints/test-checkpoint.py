import torch
from toolkit.pipeline_flux_inpaint import FluxInpaintPipeline
from PIL import Image
import os
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from argparse import ArgumentParser
from utils import extract_edit_instruction, extract_edit_instruction2, extract_edit_instruction_and_object
import numpy as np
import glob
import random
from utils import MaskEncoder2 as MaskEncoder

parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="../Edit-Transfer/models/newtry/")
parser.add_argument("--model_name", type=str, default="newtry.safetensors")
parser.add_argument("--prompt_file",  type=str, default="")
parser.add_argument("--img_dir", type=str, default="data/IMG/")
parser.add_argument("--img_path", type=str, default="data/IMG/2017_45509279_041.png")
parser.add_argument("--save_dir", default="results", type=str)
# parser.add_argument("--mask_sensitive",default="mydata/mask_one/00a42a80e5e8d194_029.png")

args = parser.parse_args()
img_dir = ""

pipe = FluxInpaintPipeline.from_pretrained("/root/autodl-fs/huggingface/FLUX.1-dev",  torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.load_lora_weights(args.model_dir, weight_name=args.model_name)
mask_encoder = MaskEncoder()
state = torch.load(os.path.join(args.model_dir,args.model_name.replace(".safetensors","_mask_encoder.pt")), map_location='cpu')
mask_encoder.load_state_dict(state['model_state_dict'])


mask=torch.zeros(1,1024,1024)
mask[:,mask.size(1)//2:,mask.size(2)//2:] = 1
# print(os.path.join(img_dir,"*.png"))
img_list = glob.glob(os.path.join(args.img_dir,"*.png"))


print(img_list)
for img_path in img_list:
    print(f"Processing {img_path}")
    filename = os.path.basename(img_path)
    img = Image.open(img_path).convert('RGB')
    mask_path = img_path.replace('IMG',"MASK")
    mask_sensitive = Image.open(mask_path).convert('L')
    # mask_sensitive.save("mask_sen.png")
    img_array = np.array(mask_sensitive)
    h, w = img_array.shape
    region_ratio = 0.5
    h_start = int(h * (1 - region_ratio))
    w_start = int(w * (1 - region_ratio))
    img_array[h_start:, w_start:] = 255 - img_array[h_start:, w_start:]
    mask_sensitive_inv = Image.fromarray(img_array, mode='L')
    # mask_sensitive_inv.save("mask_sensitive_inv.png")
    prompt_path = img_path.replace("png","txt")
    with open(prompt_path, 'r', encoding='utf-8') as file:
        prompt = file.readline()

    # transformation_txt = extract_edit_instruction2(prompt)
    transformation_txt,object_txt = extract_edit_instruction_and_object(prompt)
    # print(transformation_txt)
    if transformation_txt:
        transfer_prompt = f"This four-panel image grid. Apply the transformation '{transformation_txt}' from [TOP-LEFT] to [TOP-RIGHT] and from [BOTTOM-LEFT] to [BOTTOM-RIGHT]"
        restore_prompt = f"In this four-panel image grid, [BOTTOM-RIGHT] restores the '{object_txt}' region based on [BOTTOM-LEFT] while preserving the transformation shown in [TOP-RIGHT]."

    # for i in range(16):
    seed = torch.randint(low=1, high=99999, size=(1,)).item()
    out = pipe(
        prompt_transfer = transfer_prompt,
        prompt_restore = restore_prompt,
        mask_image = mask,
        mask_sensitive = mask_sensitive,
        mask_sensitive_inv = mask_sensitive_inv,
        image=img,
        guidance_scale=1.0,
        height=1024,
        width=1024,
        num_inference_steps=35,
        strength=1.0,
        generator=torch.Generator("cpu").manual_seed(seed),
        mask_encoder=mask_encoder
    ).images[0]
    
    image_save_path = os.path.join(args.save_dir, filename)
    out.save(image_save_path)
