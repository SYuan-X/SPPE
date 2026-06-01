import torch
from toolkit.pipeline_flux_inpaint import FluxInpaintPipeline
from PIL import Image
import os
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from argparse import ArgumentParser
import glob
from utils import extract_edit_instruction, extract_edit_instruction2, extract_edit_instruction_and_object,TAGEncoder
from clip_similarity import ClipSimilarity
def load_image_crop(image, quadrant="bottom_right"):
    """
    quadrant: "top_left" | "top_right" | "bottom_left" | "bottom_right"
    """
    # image = Image.open(image_path).convert("RGB")
    W, H = image.size
    half_w, half_h = W // 2, H // 2

    crop_box = {
        "top_left":     (0,      0,      half_w, half_h),
        "top_right":    (half_w, 0,      W,      half_h),
        "bottom_left":  (0,      half_h, half_w, H),
        "bottom_right": (half_w, half_h, W,      H),
    }[quadrant]

    image = image.crop(crop_box)

    transform = T.Compose([
        T.Resize((224, 224), interpolation=Image.BILINEAR),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to("cuda")  # [1,3,224,224]
parser = ArgumentParser()

parser.add_argument("--model_dir", type=str,default="models/")
parser.add_argument("--model_name", type=str,default="taskid_000006000.safetensors")
parser.add_argument("--prompt_file",  type=str)
parser.add_argument("--img_path", type=str)
parser.add_argument("--save_dir", default="results/", type=str)
parser.add_argument("--idx",type=int,default=0)
parser.add_argument("--num_parts",type=int,default=1)

args = parser.parse_args()
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

img_dir = "data/"

# from safetensors.torch import load_file


img_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

pipe = FluxInpaintPipeline.from_pretrained("FLUX.1-dev/",  torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.load_lora_weights(args.model_dir, weight_name=args.model_name)
tag_gen=TAGEncoder()
state = torch.load(os.path.join(args.model_dir,args.model_name.replace(".safetensors","_tag_gen.pt")), map_location='cpu')
# print(state['model_state_dict'])
tag_gen.load_state_dict(state['model_state_dict'])
tag_gen=tag_gen.cuda()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

mask=torch.zeros(1,1024,1024)
mask[:,mask.size(1)//2:,mask.size(2)//2:] = 1
num = 0
for img_path in img_list:
    filename = os.path.basename(img_path)
    image_save_path = os.path.join(args.save_dir, filename)
    num+=1
    if os.path.exists(image_save_path):
        continue
    prompt_path = img_path.replace("jpg","txt")
    img = Image.open(img_path).convert('RGB')
    with open(prompt_path, 'r', encoding='utf-8') as file:
        prompt = file.readline()
    transformation_txt,object_txt = extract_edit_instruction_and_object(prompt)
    print(f"[{num}/{len(selected_images)}] {filename}")
    print(transformation_txt)
    # with open(args.prompt_file, 'r', encoding='utf-8') as file:
    #     prompt = file.readline()
    
    # for i in range(16):
    seed = torch.randint(low=1, high=99999, size=(1,)).item()
    out = pipe(
        prompt=transformation_txt,
        mask_image = mask,
        image=img,
        guidance_scale=1.0,
        height=1024,
        width=1024,
        num_inference_steps=35,
        strength=1.0,
        generator=torch.Generator("cpu").manual_seed(seed),
        tag_gen=tag_gen
    ).images[0]

    src_image = load_image_crop(img,  "bottom_left" )
    src_exemplar_image = load_image_crop(img,  "top_left" )
    src_exemplar_image_edited = load_image_crop(img,  "top_right" )
    src_image_edited = load_image_crop(img )
    generated_image = load_image_crop(out)

    sim_src_keep, ssim_score,psnr_score ,sim_src_text, sim_test_text, consistency_score, direction_sim,consistency_score_gt, ssim_score_gt,psnr_score_gt  = \
        clip_similarity(src_exemplar_image,src_exemplar_image_edited,src_image,generated_image, src_image_edited,[prompt])

    image_save_path = os.path.join(args.save_dir, filename)
    out.save(image_save_path)
    # exit()
