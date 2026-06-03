from __future__ import annotations

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from einops import rearrange
from torchvision.utils import save_image



class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)
        print(name)

        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text: list[str]) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def source_keep(self, img1_tensor, img2_tensor):
    
        img1_np = img1_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()  # HWC
        img2_np = img2_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()    
        # print(img1_np.max(), img1_np.min()) 
    
        ssim_score = ssim(img1_np, img2_np, data_range=1.0,channel_axis=-1)
        psnr_score = psnr(img1_np, img2_np, data_range=1.0)
        return ssim_score,psnr_score 

    def masked_image(self, img1: torch.Tensor, mask: torch.Tensor):

        masked_img1 = img1 * mask
        reverse_masked_img1 = img1 * (1- mask)
        # masked_img2 = img2 * mask

        return masked_img1,reverse_masked_img1


    def forward(self, exemplar_image_0, exemplar_image_1, image_0, test_1, image_1,text):

        exemplar_image_features_0 = self.encode_image(exemplar_image_0)
        exemplar_image_features_1 = self.encode_image(exemplar_image_1)
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        test_features_1 = self.encode_image(test_1)
        text_features = self.encode_text(text)
        # test image with text(transfer model performance)
        sim_test_text = F.cosine_similarity(test_features_1, text_features)
        # exemplar image with text(original performance of LVLM)
        sim_src_text = F.cosine_similarity(image_features_1, text_features)
        # consistency_score_fix = 1 - torch.abs(sim_src_text - sim_test_text)

        # keep source features after transfer
        # sim_src_keep = (F.cosine_similarity(test_features_1, image_features_0) + 2*masked_sim_src_keep)/3
        sim_src_keep = F.cosine_similarity(test_features_1, image_features_0)

        consistency_score_gt = F.cosine_similarity(test_features_1, image_features_1)

        consistency_score = F.cosine_similarity(image_features_1 - image_features_0, test_features_1 - image_features_0)

        # direction_sim = (F.cosine_similarity(exemplar_image_features_1 - exemplar_image_features_0, test_features_1 - image_features_0) + 2*masked_direction_sim)/3
        direction_sim = F.cosine_similarity(exemplar_image_features_1 - exemplar_image_features_0, test_features_1 - image_features_0)
        

        ssim_score, psnr_score  = self.source_keep(image_0, test_1) 
        
        ssim_score_gt, psnr_score_gt = self.source_keep(test_1, image_1) 

        return sim_src_keep, ssim_score,psnr_score ,sim_src_text, sim_test_text, consistency_score, direction_sim,consistency_score_gt, ssim_score_gt, psnr_score_gt
