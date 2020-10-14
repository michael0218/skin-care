#%%

import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Skin-lesions Training')
parser.add_argument('--data', default='./skin-lesions/', help='folder to images')
args = parser.parse_args()

fileList = []
for pathName, dirName, fileNames in os.walk(args.data):
    for fileName in fileNames:
        if fileName.endswith('.jpg'):
            pathImg = os.path.join(pathName, fileName)
            #print(pathImg)
            category = pathImg.split('/')[-2]
            phase = pathImg.split('/')[-3]
            fileList.append([fileName, pathImg, category, phase])

df = pd.DataFrame(fileList, columns=["fileName", "pathImg", 'category', "phase"])
df.to_csv('./data.csv')