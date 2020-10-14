# skin-lesions
The skin-lesions detection


## Features
1. Pretrained resnet model 101
2. Focal loss

## Requirements

tensorboard>=2.0.0
torch>=1.3.1,<1.6.0
torchvision>=0.4.1,<0.7.0
opencv-python
pandas
Pillow>=6.2.1
scikit-learn>=0.21.3
tqdm>=4.36.1
scikit-image>=0.16.2
matplotlib==3.1.0

## Usage:
Download the open dataset from kaggle.

https://www.kaggle.com/wanderdust/skin-lesion-analysis-toward-melanoma-detection 

1. Unzip the dataset and put it with preprocessing.py

`$ python preprocessing.py --data ./skin-lesions`

2. Training

`$ python train.py --out ./result/`

3. Infering


`$ python infer.py --input test.png --weight ./result/net_100.pth`