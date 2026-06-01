# ERMA: Editability-aware Relational Multi-modal Assessment

ERMA is an image quality assessment model for evaluating AI-generated image edits. Given a source image, a surrogate (reference) image, and a text prompt describing the desired edit, ERMA predicts a quality score in [0, 1].

## Requirements

```bash
pip install torch torchvision numpy scipy pillow tqdm ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

## Dataset

Labels are in `labels/` (train / val / test splits). Each entry contains:

```json
{
  "category": "a105_face_all",
  "stem":     "2017_52966836",
  "code":     "018",
  "source":   "VISPR/TEST/source/a105_face_all/2017_52966836.jpg",
  "surrogate":"VISPR/TEST/surrogate/a105_face_all/2017_52966836.jpg",
  "label":    0.763
}
```

Image paths are **relative to a `--data_root` directory**. Organize your data as:

```
<data_root>/
└── VISPR/
    ├── TRAIN/source/<category>/<stem>.jpg
    ├── TRAIN/surrogate/<category>/<stem>.jpg
    ├── VAL/...
    └── TEST/...
```

The `code` field maps to a natural language edit prompt — see `prompts.py` for all 65 edit types.

## Training

```bash
python train.py \
  --data_root /path/to/data \
  --train_labels labels/train_labels.json \
  --val_labels labels/val_labels.json \
  --ckpt_dir checkpoints
```

The model is evaluated on the validation set periodically. Best checkpoint (best on validation set) is saved to `checkpoints/best.pt`. Our pretrain model is [Link](https://drive.google.com/file/d/1ocSo-u6m2040Nv2t_H85WFIJBbZk7mJV/view?usp=drive_link)


## Evaluation

```bash
python test.py \
  --data_root /path/to/data \
  --test_labels labels/test_labels.json \
  --checkpoint pretrained/erma.pt
```

The script prints SRCC and PLCC on the test set.
