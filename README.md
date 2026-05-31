# SPPE dataset and SOER baseline

This repository contains the official data preparation and implementation code for the paper:

> **When Privacy Meets Recovery: The Overlooked Half of Surrogate-Driven Privacy Preservation for MLLM Editing**  
> *AAAI 2025*  


---

## 1. Dataset Download
The raw images and annotations used to construct our benchmark are derived from publicly available source datasets. These original datasets should be obtained from their official sources, including [Link](https://github.com/tribhuvanesh/vpa.git), [Link](https://github.com/tribhuvanesh/visual_redactions.git) and [Link](https://anranxu.github.io/DIPA2_VIS/)

Our processed benchmark dataset, including the source/surrogate before-and-after editing pairs, is available at:
[Link](https://drive.google.com/drive/folders/1Kl2dR6vuakLvN8JqzTV3vdpVS0lW6SDJ?usp=drive_link)

Each sample contains **two editing pairs**:

1. `source_ori` & `source_edited`  
2. `surrogate_ori` & `surrogate_edited`

---

## 2. Dataset Preparation

After downloading the dataset:

### 2.1 Construct training samples  
For each sample, horizontally concatenate each pair and then vertically stack the two pairs:

source_ori | source_edited
surrogate_ori | surrogate_edited


This 4-grid image is used as the model input/output.

Ensure that files with the same name are **renamed** to avoid overwriting during saving. Example Python snippet (load & concatenate four images):

```python
import os
from PIL import Image

def load_and_concat(source_ori, source_edit, sur_ori, sur_edit, save_path):
    img1 = Image.open(source_ori).convert("RGB").resize((512,512))
    img2 = Image.open(source_edit).convert("RGB").resize((512,512))
    img3 = Image.open(sur_ori).convert("RGB").resize((512,512))
    img4 = Image.open(sur_edit).convert("RGB").resize((512,512))

    w, h = img1.size

    top = Image.new("RGB", (w * 2, h))
    top.paste(img1, (0, 0))
    top.paste(img2, (w, 0))

    bottom = Image.new("RGB", (w * 2, h))
    bottom.paste(img3, (0, 0))
    bottom.paste(img4, (w, 0))

    full = Image.new("RGB", (w * 2, h * 2))
    full.paste(top, (0, 0))
    full.paste(bottom, (0, h))

    full.save(save_path)
```

### 2.3 Mask processing

Corresponding mask generation should set all non-target regions to black, with the bottom-right quadrant kept as the mask.

### 2.4 Prompts

You can find the editing instructions for each sample in data/prompts.txt. Each instruction is matched to a sample based on the numeric ID at the end of the image filename. The category is taken from the image parent folder name (for example face, text, plate, etc.)— this folder name is used as the surrogate category when generating surrogate images.

An example image sample and its corresponding prompt file are provided in `data/IMG`. An example mask sampleis provided in `data/MASK`.


## Training

Run:
```python
python run.py config/recovery.yml
```

Parameters can be modified inside config/recovery.yml.



## Testing

Pretrained models can be downloaded from:

[Link](https://drive.google.com/drive/folders/1ToouonPPU7wI_Ox9LanF6QYsvK3hAJJr?usp=drive_link)  


Run:
```python
python test.py
```
The final restored image is obtained by cropping the bottom-right quadrant.

## Benchmark metrics 

Benchmark metrics can be computed using:

```python
python analysis.py
```


Our code is built on **ai-toolkit**, we gratefully acknowledge their excellent work [Link](https://github.com/ostris/ai-toolkit.git). 


## Citation

If you find our work useful, please cite:
```python
@inproceedings{xu2026privacy,
  title={When Privacy Meets Recovery: The Overlooked Half of Surrogate-Driven Privacy Preservation for MLLM Editing},
  author={Xu, Siyuan and Liu, Yibing and Chen, Peilin and Li, Yung-Hui and Wang, Shiqi and Kwong, Sam},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={42},
  pages={35958--35966},
  year={2026}
}
```
