#%%
#import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
#import argparse
import torchvision.models as models
from PIL import Image
#from skin_API import test_model

def test_model(pathImg, pathWeight):
    #%%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(pathImg)

    transform_test = transforms.Compose([
            transforms.Resize((280,280)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    #%%
    Num2Cata = {0:"nevus",1:"melanoma",2:"seborrheic_keratosis"}

    resnet101 = models.resnet101(pretrained=False)
    resnet101.fc = nn.Linear(2048, 3)
    net = resnet101.to(device)

    #weightPath = r'E:\Claire\AiInTaiwan\skin_resnet101_centercrop\weights\weights.pth'
    # weightPath = '/mnt/data-home/mike/modelweight/skin/skin_resnet101_centercrop_normalization_pretrained_focalloss/net_084.pth'
    net.load_state_dict(torch.load(pathWeight, map_location='cpu'))

    print("Start Testing Your Skincare!!\n")

    print("Processing... Please Wait...\n")

    with torch.no_grad():
        net.eval()
        images= transform_test(img).unsqueeze(0)
        images = images.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        #print(predicted[0])

        print("Testing Finished!\n")
        print("Your Skincare level is {0}!!\n".format(Num2Cata[predicted[0]]))

    return Num2Cata[predicted[0]]
    # %%
