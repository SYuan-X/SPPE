

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPEncoder(nn.Module):
    def __init__(self, clip_name="ViT-B/32"):
        super().__init__()

        import clip
        self.clip_lib = clip
        self.model, _ = clip.load(clip_name, device="cpu")

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.model.eval()
        self.image_dim = 512
        self.text_dim = 512

    @torch.no_grad()
    def encode_image(self, x):
        return self.model.encode_image(x).float()

    @torch.no_grad()
    def encode_text(self, prompts, device):
        tokens = self.clip_lib.tokenize(prompts, truncate=True).to(device)
        return self.model.encode_text(tokens).float()


class EMMA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.clip = CLIPEncoder(cfg.CLIP_MODEL_FOR_PROMPT)

        dim = 512

        in_dim = dim * 6 + 3

        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, source, surrogate, prompts):
        device = source.device

        fI = self.clip.encode_image(source)
        fS = self.clip.encode_image(surrogate)
        fT = self.clip.encode_text(prompts, device)

        fI_n = F.normalize(fI, dim=-1)
        fS_n = F.normalize(fS, dim=-1)
        fT_n = F.normalize(fT, dim=-1)

        sim_IS = (fI_n * fS_n).sum(dim=-1, keepdim=True)
        sim_IT = (fI_n * fT_n).sum(dim=-1, keepdim=True)
        sim_ST = (fS_n * fT_n).sum(dim=-1, keepdim=True)
        sim_gap = sim_ST - sim_IT

        feat = torch.cat([
            fI,
            fS,
            torch.abs(fI - fS),
            fI * fS,
            fS - fI,
            fT,
            sim_IS,
            sim_IT,
            sim_ST,
            # sim_gap,
        ], dim=-1)

        z = self.shared(feat)
        score = self.head(z).squeeze(-1)
        return score
