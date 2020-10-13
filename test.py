#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from skin_dataset_loader import SkinLesion
import numpy as np
import pandas as pd
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_class = 3
pre_epoch = 0
BATCH_SIZE = 8 




transform_test = transforms.Compose([
        transforms.Resize((224,224)),

        # transforms.Resize((250,250)),
        # transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


num2num_label_converting = {1:1,2:2,3:3,4:4,5:5,6:6,8:0,17:7}
Num2Cata = {0:"nevus",1:"melanoma",2:"seborrheic_keratosis"}
Cata2Num = {"nevus":0,"melanoma":1,"seborrheic_keratosis":2}
classes = ['nevus','melanoma','seborrheic_keratosis']
df = pd.read_csv('/mnt/data-home/mike/dataset/skin-lesions/df_phase.csv')
df_valid = df[df.phase == 'test'].reset_index(drop=True)
valid_set = SkinLesion(df_valid, transforms=transform_test)
testloader = torch.utils.data.DataLoader(valid_set, batch_size=10, shuffle=False, num_workers=2)

resnet101 = models.resnet101(pretrained=False)
resnet101.fc = nn.Linear(2048, n_class)
net = resnet101.to(device)
PATH = '/mnt/data-home/mike/modelweight/skin/skin_resnet101_centercrop_normalization_pretrained_focalloss/net_021.pth'
net.load_state_dict(torch.load(PATH))


#%%
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
#%%
if __name__ == "__main__":


    print("Waiting Test!")
    predicteds = np.array([])
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            #print(predicted)
            predicteds = np.concatenate((predicteds,predicted.cpu().numpy()))
    print(predicteds)
# #%%
# data = testloader.dataset[0]
# images, pslice = data
# images = images.to(device)
# outputs = net(images)
# # %%

# %%
gt = [Cata2Num[i] for i in df_valid.category.tolist()]
pred = predicteds.astype(int).tolist()



# %%
from sklearn.metrics import classification_report
print(classification_report(gt,pred,target_names = classes))
report = classification_report(gt, pred, target_names = classes, output_dict=True,digits=4)
df_classification_report = pd.DataFrame(report).transpose()
df_classification_report.to_csv('./metrics.csv')
# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(gt, pred)
# %%
plot_confusion_matrix(confusion_matrix(gt, pred),
                      normalize    = False,
                      target_names = classes,
                      title        = "Confusion Matrix")

# %%
plot_confusion_matrix(confusion_matrix(gt, pred),
                      normalize    = True,
                      target_names = classes,
                      title        = "Confusion Matrix")
# %%

# %%
