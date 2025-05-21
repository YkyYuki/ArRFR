# ArRFR

This repository contains the official implementaion for ArRFR

## Environment
- Python 3.8
- PyTorch 2.0.0
- TensorboardX
- os, numpy, monai, pyyaml, tqdm, gc, random, matplotlib, math

## Preparations
### Dataset
Create the path `./paired_dataset/`. Then download the dataset [here](https://drive.google.com/drive/folders/1q_MtKQCnfC7s1HjO8xP6Afhn2fjkxY3T?usp=drive_link).
### Pretrained Checkpoints
Create the path `./checkpoints/pretrained/`. Then download the pretrained checkpoints [here](https://drive.google.com/drive/folders/1Gzng2dUsj8cFyx3tYCyk0uSMNx7HB5Ch?usp=drive_link). `vgg.pth` is the pretrained VGG-16 checkpoint. `ArRFR_sna=0_st2=10_fna=6_ft2=4.pth` is the pretrained ArRFR checkpoint.

## Quick Start
### Reproducing Experiment
Use the pretrained `ArRFR_sna=0_st2=10_fna=6_ft2=4.pth` for reproducing experiment. Run the following command:
~~~python
python test.py
~~~
Results will be saved in `./results`.
### Train Your Own Model
Run the following command:
~~~python
python train.py
~~~
Checkpoints will be saved every 10 epochs in `./checkpoints/`

