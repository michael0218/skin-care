#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.transforms as transforms
import argparse
import os
from skin_dataset_loader import SkinLesion
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from focalLoss import FocalLoss

parser = argparse.ArgumentParser(description='Skin-lesions Training')
parser.add_argument('--out', default='./result/', help='folder to output images and model checkpoints') 
args = parser.parse_args()

num_workers = 8
n_class = 3

# check cpu or cuda torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet101 = models.resnet101(pretrained=True)
resnet101.fc = nn.Linear(2048, n_class)
net = resnet101.to(device)

print(net)

#%%
# Hyperparameter
EPOCH = 200
pre_epoch = 0  
BATCH_SIZE = 36
LR = 0.001   # learning rate 


df = pd.read_csv('./data.csv')
df_train = df[df.phase == 'train'].reset_index(drop=True)
df_valid = df[df.phase == 'valid'].reset_index(drop=True)


writer = SummaryWriter(os.path.join(args.out,'tensorboard'))



transform_train = transforms.Compose([
        transforms.Resize((280,280)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.CenterCrop((224,224)),
        transforms.RandomAffine(5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
        transforms.Resize((280,280)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

train_set = SkinLesion(df_train, transforms=transform_train)
valid_set = SkinLesion(df_valid, transforms=transform_test)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers) 
testloader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)




criterion = FocalLoss(n_class) 
optimizer = optim.Adam(net.parameters(), lr=LR)
graph_create_flag = 0



if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 0  
    best_epoch = 0
    print("Start Training") 

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
        
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # calculate loss and accuracy every batch
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            writer.add_scalar('train/Loss', sum_loss / (i + 1), i + 1 + epoch * length)
            writer.add_scalar('train/Accuracy', (correct / total).item(), i + 1 + epoch * length)
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))


        print("Calculate Train/Valid accuracy!")
        with torch.no_grad():

            correctT = 0
            totalT = 0
            sum_lossT = 0
            for data in trainloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                sum_lossT += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                totalT += labels.size(0)
                correctT += (predicted == labels).sum().item()
            print('Train accuracy: %.3f%%' % (100 * (correctT/ totalT)))
            accT = 100. * correctT / totalT

            correct = 0
            total = 0
            sum_loss = 0

            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1) # similar to agrmax
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('valid accuracy: %.3f%%' % (100 * (correct/ total)))
            acc = 100. * correct / total


            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                print('Saving model......')
                torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))

            writer.add_scalars('Best',{'best_acc': best_acc, 'best_epoch':best_epoch}, epoch)
            
    print("Training Finished, TotalEPOCH=%d" % EPOCH)
# %%
