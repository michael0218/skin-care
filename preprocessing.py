#%%

import os
import pandas as pd
dataRoot = '/mnt/data-home/mike/dataset/skin-lesions'

fileList = []
for pathName, dirName, fileNames in os.walk(dataRoot):
    for fileName in fileNames:
        if fileName.endswith('.jpg'):
            pathImg = os.path.join(pathName, fileName)
            #print(pathImg)
            category = pathImg.split('/')[-2]
            phase = pathImg.split('/')[-3]
            fileList.append([fileName, pathImg, category, phase])

df = pd.DataFrame(fileList, columns=["fileName", "pathImg", 'category', "phase"])
#/mnt/data-home/mike/dataset/skin-lesions/valid/seborrheic_keratosis/ISIC_0014568.jpg
# %%
df.to_csv(os.path.join(dataRoot,'df_phase.csv'))