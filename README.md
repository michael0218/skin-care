# skin-lesions
The skin-lesions detection


## Features
1. Pretrained resnet model 101
2. Focal loss

## Usage:
Download the open dataset from kaggle.

https://www.kaggle.com/wanderdust/skin-lesion-analysis-toward-melanoma-detection 

1. Unzip the dataset and put it with preprocessing.py

`$ python preprocessing.py --data ./skin-lesions`

2. Training

`$ python train.py --out ./result/`

3. Infering


`$ python infer.py --input test.png --weight ./result/net_100.pth`