
# import torchdata.datapipes as dp
from clip_similarity import ClipSimilarity
from PIL import Image
import torchvision.transforms as T
import torch
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
# import clip
import torchvision.transforms.functional as F
import csv
import torchvision.utils as vutils
def save_concatenated_images(images, save_path):
    # images: list of torch.Tensor with shape [1, 3, H, W]
    grid = vutils.make_grid(torch.cat(images, dim=0), nrow=len(images))  # 拼接成一行
    vutils.save_image(grid, save_path)

csv_path = "test.csv"
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'category', 'filename',
    'sim_src_keep', 'ssim', 'psnr',
    'sim_src_text', 'sim_test_text',
    'consistency_score', 'direction_sim',
    'consistency_score_gt', 'ssim_gt', 'psnr_gt'
])



code_to_prompt = {
    "001": "Let the person wear sunglasses",
    "002": "Let the person wear a cowboy hat",
    "003": "Let the person wear a cap",
    "004": "Let the person wear earrings",
    "005": "Make the person wear a scarf",
    "006": "Make the person wear a suit and tie",
    "007": "Change the person's hair color to red",
    "008": "Change the person's hair color to blonde",
    "009": "Change the person's hair color to blue",
    "010": "Make the person have curly hair",
    "011": "Make the person look like a cyborg",
    "012": "Turn the person into a cartoon character",
    "013": "Turn the person into an anime character",
    "014": "Make the person look like a superheroe",
    "015": "Turn the person into a vampire",
    "016": "Make the person look like a ghost",
    "017": "Make the person look older",
    "018": "Make the person look young",
    "019": "Turn to be vintage",
    "020": "Turn to be modern",
    "021": "Turn to be medieval",
    "022": "Turn to be high-tech",
    "023": "remove the person",
    "024": "Make it look like the 1920s",
    "025": "Make it look like the 1980s",
    "026": "Make it look futuristic",
    "027": "Make it look like the Victorian era",
    "028": "Turn to an ancient Chinese style",
    "029": "Turn to a traditional Indian style",
    "030": "Turn to an ancient Egyptian style",
    "031": "Turn to a medieval European style",
    "032": "Turn to an oil painting style",
    "033": "Turn to a pencil sketch style",
    "034": "Turn to a watercolor painting style",
    "035": "Turn to a Monet-style watercolor painting",
    "036": "Turn to a Van Gogh-style painting",
    "037": "Change the background to a night scene",
    "038": "Change the background to a sunset view",
    "039": "Change the background to a forest",
    "040": "Change the background to a bright, snowy landscape",
    "041": "Change the background to an underwater scene",
    "042": "Make it look like a luxury brand advertisement",
    "043": "Make it look like a perfume advertisement",
    "044": "Make it look like a magazine cover",
    "045": "Make it look like an old, weathered paper",
    "046": "Make it look like an ancient manuscript",
    "047": "remove text",
    "048": "Add some water stains",
    "049": "Add some coffee stain",
    "050": "Add some blue ink stains",
    "051": "Add some scratches",
    "052": "Make the credit card look like a beautiful birthday card",
    "053": "Make the credit card look like a leather credit card",
    "054": "Make the credit card look like a golden engraved credit card",
    "055": "Make the credit card look like a handwritten love letter",
    "056": "Transform the credit card into a luxurious VIP membership card",
    "057": "Make the credit card look like a marble credit card",
    "058": "Turn the credit card into an elegant wedding invitation",
    "059": "Make the credit card look like a wooden carved credit card",
    "060": "Make it look like a wooden carved ticket",
    "061": "Make it look like a golden engraved ticket",
    "062": "Make it look like a marble ticket",
    "063": "Make it look like a leather ticket",
    "064": "remove the laptop",
    "065": "remove the plate"
}
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # tmp_img = F.to_tensor(image.resize((224,224))).float().cuda()
    transform = T.Compose([
        T.Resize((224, 224),interpolation=Image.BILINEAR),
        T.ToTensor(),  # [0, 255] -> [0, 1], shape: [C, H, W]
    ])

    return transform(image).unsqueeze(0).to("cuda")  # 加 batch 维度: [1, 3, H, W]

def load_image_crop(image_path):
    image = Image.open(image_path).convert("RGB")

    # ===== 1. 裁剪右下 1/4 =====
    W, H = image.size
    crop_box = (W // 2, H // 2, W, H)
    image = image.crop(crop_box)

    # ===== 2. 再做 transform =====
    transform = T.Compose([
        T.Resize((224, 224), interpolation=Image.BILINEAR),
        T.ToTensor(),
    ])

    return transform(image).unsqueeze(0).to("cuda")  # [1,3,224,224]
def average(lst):
    return sum(lst) / len(lst) if lst else 0
data_dir = "/root/autodl-tmp/results/base_cycle_mask_cate/"
src_dir = "/root/autodl-tmp/DATASET/ORIGINAL/TEST/source"
edited_examplar_dir = "/root/autodl-tmp/DATASET/EDITED/TEST/surrogate"
edited_source_dir = '/root/autodl-tmp/DATASET/EDITED/TEST/source'
clip_similarity = ClipSimilarity().cuda()

import glob
path_list = glob.glob(os.path.join(data_dir,"*","*.jpg"))
print(len(path_list))
sim_src_keep_list = []
ssim_score_list = []
psnr_score_list = []
sim_src_text_list = []
sim_examplar_text_list = []
consistency_score_list = []
direction_sim_list = []
consistency_score_gt_list = []

ssim_score_gt_list = []
psnr_score_gt_list = []
# ssim_score_gt, psnr_score_gt
count = 0
for path in path_list:
    count+=1

    category = path.split('/')[-2]
    
    image_id_edited = path.split('/')[-1]
    image_id = image_id_edited[:-8]+'.jpg'
    baseline_res_path = path
    src_path = os.path.join(src_dir,category,image_id)
    src_examplar = src_path.replace('source','surrogate') 
    src_edited = os.path.join(edited_source_dir,category,image_id_edited)
    src_examplar_edited = src_edited.replace('source','surrogate')

    code = image_id_edited.split('.')[0][-3:]
    prompt = code_to_prompt[code]
    if os.path.exists(src_path):
        src_image = load_image(src_path)
    else:
        src_path=src_path.replace("VISPR/TEST","DIPA")
        src_image = load_image(src_path)
    if os.path.exists(src_examplar):
        src_exemplar_image = load_image(src_examplar)
    else:
        src_examplar=src_examplar.replace("VISPR/TEST","DIPA")
        src_exemplar_image = load_image(src_examplar)
    if os.path.exists(src_edited):
        src_image_edited = load_image(src_edited)
    else:
        src_edited=src_edited.replace("VISPR","DIPA")
        src_image_edited = load_image(src_edited)
    if os.path.exists(src_examplar_edited):
        src_exemplar_image_edited = load_image(src_examplar_edited)
    else:
        src_examplar_edited=src_examplar_edited.replace("VISPR","DIPA")
        src_exemplar_image_edited = load_image(src_examplar_edited)
    generated_image = load_image_crop(path)
    sim_src_keep, ssim_score,psnr_score ,sim_src_text, sim_test_text, consistency_score, direction_sim,consistency_score_gt, ssim_score_gt,psnr_score_gt  = \
        clip_similarity(src_exemplar_image,src_exemplar_image_edited,src_image,generated_image, src_image_edited,[prompt])

    # 加入列表
    sim_src_keep_list.append(sim_src_keep.item())
    ssim_score_list.append(ssim_score)
    psnr_score_list.append(psnr_score)
    sim_src_text_list.append(sim_src_text.item())
    sim_examplar_text_list.append(sim_test_text.item())
    consistency_score_list.append(consistency_score.item())
    direction_sim_list.append(direction_sim.item())
    consistency_score_gt_list.append(consistency_score_gt.item())
    ssim_score_gt_list.append(ssim_score_gt)
    psnr_score_gt_list.append(psnr_score_gt)
    csv_writer.writerow([
        category,
        image_id_edited,
        round(sim_src_keep.item(), 6),
        round(ssim_score, 6),
        round(psnr_score, 6),
        round(sim_src_text.item(), 6),
        round(sim_test_text.item(), 6),
        round(consistency_score.item(), 6),
        round(direction_sim.item(), 6),
        round(consistency_score_gt.item(), 6),
        round(ssim_score_gt, 6),
        round(psnr_score_gt, 6)
    ])
    if count==10 or count%1000==0:
        print(f"\n========={count}/{len(path_list)} AVERAGE METRICS =========")
        print(f"Average CSIM(Src): {average(sim_src_keep_list):.4f}")
        print(f"Average SSIM: {average(ssim_score_list):.4f}")
        print(f"Average PSNR: {average(psnr_score_list):.4f}")
        # print(f"Average sim_src_text: {average(sim_src_text_list):.4f}")
        # print(f"Average sim_test_text: {average(sim_examplar_text_list):.4f}")
        print(f"Average DirI: {average(consistency_score_list):.4f}")
        print(f"Average DirS: {average(direction_sim_list):.4f}")
        print(f"Average CSIM(GT): {average(consistency_score_gt_list):.4f}")
        print(f"Average GT SSIM: {average(ssim_score_gt_list):.4f}")
        print(f"Average GT PSNR: {average(psnr_score_gt_list):.4f}")

print(count)


print("\n========= AVERAGE METRICS =========")
print(f"Average CSIM(Src): {average(sim_src_keep_list):.4f}")
print(f"Average SSIM: {average(ssim_score_list):.4f}")
print(f"Average PSNR: {average(psnr_score_list):.4f}")
# print(f"Average sim_src_text: {average(sim_src_text_list):.4f}")
# print(f"Average sim_test_text: {average(sim_examplar_text_list):.4f}")
print(f"Average DirI: {average(consistency_score_list):.4f}")
print(f"Average DirS: {average(direction_sim_list):.4f}")
print(f"Average CSIM(GT): {average(consistency_score_gt_list):.4f}")
print(f"Average GT SSIM: {average(ssim_score_gt_list):.4f}")
print(f"Average GT PSNR: {average(psnr_score_gt_list):.4f}")

csv_file.close()
print(f"\nSaved all results to: {csv_path}")
