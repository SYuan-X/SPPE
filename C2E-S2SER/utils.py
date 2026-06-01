import os
import re
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
import torch
import matplotlib.pyplot as plt

def decode_and_save(latent,sd, filename,output_dir="viz"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    # for latext
    latent_for_vis = latent.detach().clone() 
    latent_for_vis = latent_for_vis.to(sd.vae.dtype).to(sd.vae.device)
    with torch.no_grad():
        decoded_img = sd.vae.decode(latent_for_vis).sample  # → (1, 3, H*8, W*8)


    decoded_img = (decoded_img.clamp(-1, 1) + 1) / 2.0  # to [0,1]
    image_processor = VaeImageProcessor(vae_scale_factor=sd.vae_scale_factor * 2)
    for  i in range(decoded_img.shape[0]):
        decoded_img_res = image_processor.postprocess(decoded_img[i].unsqueeze(0))
        decoded_img_res[0].save( os.path.join(output_dir, f"{filename}-{i}.png"))


def extract_edit_instruction(text):
    # trans_text  = []
    # for text in texts:
    match = re.search(r"applying:\s*(.*?);", text)
    if match:
        return match.group(1).strip()
    else:
        return None
def extract_edit_instruction2(text):
    # trans_text  = []
    # for text in texts:
    match = re.search(r"\[TOP-RIGHT\]:.*?,\s*(.*?)\.", text)
    if match:
        return match.group(1).strip()
    else:
        return None



def extract_edit_instruction_and_object(line):
    if len(line)<1:
        return '',''
    line = line.strip() 
    
    if '[SPLIT]' in line:
        parts = line.split('[SPLIT]')
        prompt = parts[0].strip()
        category = parts[1].strip()
        return prompt, category
    else:
        print(line)
        raise ValueError("Input string does not contain [SPLIT]")

def save_mask_heatmaps_batch(mask_tensor, save_dir="viz", prefix="mask", cmap='hot'):

    os.makedirs(save_dir, exist_ok=True)

    B = mask_tensor.shape[0]
    mask_tensor = mask_tensor.detach().cpu().float()

    for i in range(B):
        mask_np = mask_tensor[i, 0].numpy()
        save_path = os.path.join(save_dir, f"{prefix}_{i}.png")

        plt.imshow(mask_np, cmap=cmap)
        plt.axis('off')
        plt.colorbar()
        plt.title(f"{prefix} #{i}")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
import torch.nn as nn



class MaskEncoder(nn.Module):
    def __init__(self, input_channels=1, embed_dim=4096):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, embed_dim)
        self.fc2 = nn.Linear(64, embed_dim)

    def forward(self, mask,flag="diff"):  # mask: [B, 1, 64, 64]

        if flag=="diff":
            x = self.encoder(mask)           # [B, 64, 1, 1]
            x = x.view(x.size(0), -1)        # [B, 64]
            x = self.fc(x)                   # [B, 4096]
            x = x.unsqueeze(1)               # [B, 1, 4096]
        elif flag=="mask":
            x = self.encoder2(mask)           # [B, 64, 1, 1]
            x = x.view(x.size(0), -1)        # [B, 64]
            x = self.fc2(x)                   # [B, 4096]
            x = x.unsqueeze(1)               # [B, 1, 4096]
        return x

    def set_transfer_grad(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.fc.parameters():
            p.requires_grad = True

    def set_restore_grad(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.encoder2.parameters():
            p.requires_grad = True
        for p in self.fc2.parameters():
            p.requires_grad = True
    def set_grad(self):
        for p in self.parameters():
            p.requires_grad = True
        # for p in self.encoder2.parameters():
        #     p.requires_grad = True
        # for p in self.fc2.parameters():
        #     p.requires_grad = True
    def force_to(self, device, dtype):
        self.to(device, dtype)
        self.encoder.to(device, dtype)
        self.encoder2.to(device, dtype)
        self.fc.to(device, dtype)
        self.fc2.to(device, dtype)

