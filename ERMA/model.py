# # model.py
# """
# 打分模型:
#   视觉分支: ViT-B/16 (timm) 分别编码 source I 和 surrogate S
#            输出 patch tokens [B, N, D], 维度 D=EMBED_DIM
#   文本分支: CLIP text encoder 编码 prompt -> [B, PROMPT_DIM]
#            过一个 linear 投到 D 维, 作为 query
#   融合:    cross-attention 把 prompt 当 query, [I_tokens; S_tokens] 当 key/value
#            堆 NUM_CA_LAYERS 层
#   打分头:  对融合后的 prompt token 接 MLP -> scalar
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ---------- 视觉主干 ----------
# def _build_vit(embed_dim: int):
#     """返回一个 timm ViT-B/16, forward_features 给 patch tokens"""
#     import timm
#     # 经典选择: vit_base_patch16_224, ImageNet-1k 预训练, hidden=768
#     model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
#     actual_dim = model.embed_dim
#     assert actual_dim == embed_dim, \
#         f"timm ViT 的 embed_dim={actual_dim}, 但 config 写的是 {embed_dim}"
#     return model


# class ViTPatchEncoder(nn.Module):
#     """对 timm ViT 取出 patch tokens (含 CLS)"""
#     def __init__(self, embed_dim: int):
#         super().__init__()
#         self.vit = _build_vit(embed_dim)

#     def forward(self, x):
#         # forward_features 返回 [B, 1+N, D] (CLS + patch)
#         feats = self.vit.forward_features(x)
#         if feats.dim() == 2:                  # 极少数 timm 版本会返回 pooled
#             feats = feats.unsqueeze(1)
#         return feats                          # [B, T, D]


# # ---------- CLIP text encoder ----------
# class PromptEncoder(nn.Module):
#     """
#     用 CLIP 文本编码器给 prompt 出一个 [B, PROMPT_DIM] 的向量。
#     冻结参数 (省显存, 训练只调融合部分和打分头)
#     """
#     def __init__(self, clip_name: str, prompt_dim: int):
#         super().__init__()
#         self.backend = None
#         self.clip_name = clip_name
#         self.prompt_dim = prompt_dim
#         self._load()

#     def _load(self):
#         try:
#             import open_clip
#             oc_name = self.clip_name.replace("/", "-")
#             self.model, _, _ = open_clip.create_model_and_transforms(
#                 oc_name, pretrained="openai"
#             )
#             self.tokenizer = open_clip.get_tokenizer(oc_name)
#             self.backend = "open_clip"
#         except Exception:
#             import clip
#             self.model, _ = clip.load(self.clip_name, device="cpu")
#             self.tokenizer = clip.tokenize
#             self.backend = "openai_clip"

#         for p in self.model.parameters():
#             p.requires_grad_(False)
#         self.model.eval()

#     @torch.no_grad()
#     def forward(self, prompts):
#         """prompts: list[str], 长度 B -> [B, prompt_dim]"""
#         device = next(self.model.parameters()).device
#         if self.backend == "open_clip":
#             tokens = self.tokenizer(prompts).to(device)
#             feat = self.model.encode_text(tokens)
#         else:
#             tokens = self.tokenizer(prompts).to(device)
#             feat = self.model.encode_text(tokens)
#         return feat.float()


# # ---------- Cross-attention 融合 ----------
# class CrossAttentionBlock(nn.Module):
#     """标准 transformer decoder block 的简化: SA + CA + FFN"""
#     def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
#                  dropout: float = 0.1):
#         super().__init__()
#         self.norm_q1 = nn.LayerNorm(dim)
#         self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
#                                                batch_first=True)
#         self.norm_q2 = nn.LayerNorm(dim)
#         self.norm_kv = nn.LayerNorm(dim)
#         self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
#                                                 batch_first=True)
#         self.norm_ff = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, q, kv):
#         # q: [B, Tq, D], kv: [B, Tkv, D]
#         h = self.norm_q1(q)
#         sa, _ = self.self_attn(h, h, h, need_weights=False)
#         q = q + sa

#         h = self.norm_q2(q)
#         kvn = self.norm_kv(kv)
#         ca, _ = self.cross_attn(h, kvn, kvn, need_weights=False)
#         q = q + ca

#         q = q + self.mlp(self.norm_ff(q))
#         return q


# # ---------- 完整模型 ----------
# class IQAModel(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg

#         self.encoder = ViTPatchEncoder(cfg.EMBED_DIM)

#         self.prompt_enc = PromptEncoder(cfg.CLIP_MODEL_FOR_PROMPT, cfg.PROMPT_DIM)
#         self.prompt_proj = nn.Sequential(
#             nn.Linear(cfg.PROMPT_DIM, cfg.EMBED_DIM),
#             nn.GELU(),
#             nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
#         )

#         # 给 source / surrogate 的 token 各加一个可学习的 type embedding,
#         # 让 cross-attn 能区分出来自哪一支
#         self.type_emb = nn.Parameter(torch.zeros(2, cfg.EMBED_DIM))
#         nn.init.trunc_normal_(self.type_emb, std=0.02)

#         self.ca_layers = nn.ModuleList([
#             CrossAttentionBlock(cfg.EMBED_DIM, cfg.NUM_HEADS)
#             for _ in range(cfg.NUM_CA_LAYERS)
#         ])

#         # 打分头: prompt token -> scalar; tanh 限制到 [-1,1] 与 cos label 对齐
#         self.head = nn.Sequential(
#             nn.LayerNorm(cfg.EMBED_DIM),
#             nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM // 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(cfg.EMBED_DIM // 2, 1),
#             nn.Tanh(),
#         )

#     def forward(self, source, surrogate, prompts):
#         # 视觉
#         tok_I = self.encoder(source)       # [B, T, D]
#         tok_S = self.encoder(surrogate)    # [B, T, D]

#         # 加 type embedding
#         tok_I = tok_I + self.type_emb[0].view(1, 1, -1)
#         tok_S = tok_S + self.type_emb[1].view(1, 1, -1)
#         kv = torch.cat([tok_I, tok_S], dim=1)   # [B, 2T, D]

#         # prompt -> query
#         with torch.no_grad():
#             p = self.prompt_enc(prompts)   # [B, PROMPT_DIM] on whatever device
#         p = p.to(kv.device).to(kv.dtype)
#         q = self.prompt_proj(p).unsqueeze(1)   # [B, 1, D]

#         for blk in self.ca_layers:
#             q = blk(q, kv)

#         score = self.head(q.squeeze(1)).squeeze(-1)   # [B]
#         return score


# model.py
"""
EMMA: Editability-aware Multi-modal Multi-task Assessment

Input:
  source image I
  surrogate image S
  prompt T

Output:
  score in [0, 1]

Important:
  Training label can be constructed from edited images:
  q = (cos(f(I') - f(I), f(S') - f(S)) + 1) / 2

  But inference only uses I, S, T.
"""

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

        # image features:
        # fI, fS, |fI-fS|, fI*fS, fS-fI  => 5 * 512
        # text feature:
        # fT => 512
        # scalar similarities:
        # cos(I,S), cos(I,T), cos(S,T), gap => 4
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