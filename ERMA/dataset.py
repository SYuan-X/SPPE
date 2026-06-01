#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from prompts import get_prompt_by_code


def build_transform(img_size, train):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class ERMADataset(Dataset):
    """
    Label JSON format (one record per image pair):
      {
        "category": "a105_face_all",
        "stem":     "2017_52966836",
        "code":     "018",
        "source":   "VISPR/TEST/source/a105_face_all/2017_52966836.jpg",
        "surrogate":"VISPR/TEST/surrogate/a105_face_all/2017_52966836.jpg",
        "label":    0.763
      }

    Args:
        label_json: path to *_labels.json
        data_root:  root directory; source/surrogate paths are joined to it
        img_size:   resize target (default 224)
        train:      if True, applies random horizontal flip augmentation
    """
    def __init__(self, label_json, data_root, img_size=224, train=True):
        self.data_root = Path(data_root)
        self.records   = json.loads(Path(label_json).read_text())
        self.tf        = build_transform(img_size, train)

    def __len__(self):
        return len(self.records)

    def _load(self, rel_path):
        return self.tf(Image.open(self.data_root / rel_path).convert("RGB"))

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "source":    self._load(r["source"]),
            "surrogate": self._load(r["surrogate"]),
            "prompt":    get_prompt_by_code(r["code"]),
            "label":     torch.tensor(float(r["label"]), dtype=torch.float32),
            "meta": {
                "category": r["category"],
                "stem":     r["stem"],
                "code":     r["code"],
            },
        }


def collate_fn(batch):
    return {
        "source":    torch.stack([b["source"]    for b in batch]),
        "surrogate": torch.stack([b["surrogate"] for b in batch]),
        "prompt":    [b["prompt"] for b in batch],
        "label":     torch.stack([b["label"]     for b in batch]),
        "meta":      [b["meta"]  for b in batch],
    }
