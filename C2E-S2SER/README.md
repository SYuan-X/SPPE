# C2E-S2SER

This repository contains the official implementation of **C2E-S2SER** for surrogate-to-source edit recovery in privacy-preserving MLLM image editing.


## Training

Run:
```python
python run.py config/recovery.yml
```

Parameters can be modified inside config/recovery.yml.



## Testing

Pretrained models can be downloaded from:

[Link](https://drive.google.com/drive/folders/1Jun4OFLIwEyHGR179tdVvF55CTSAQjAx?usp=drive_link) 


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
