import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
# resnet from https://www.kaggle.com/code/jiayuwang1472/cifar10-resnet-90-accuracy-less-than-5-min/edit

# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
test_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

batch_size = 32


data_dir = '/home/ec2-user/imageAI/image_resource/pix2pix/pack2'
train_ds = ImageFolder(data_dir+'/train', train_tfms)
valid_ds = ImageFolder(data_dir+'/test', valid_tfms)
test_ds = ImageFolder(data_dir+'/test', test_tfms)

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
test_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
dataloaders = {'train': train_dl, 'val': valid_dl, 'test': test_dl}


num_epochs = 75
max_lr = 0.0001
steps_per_epoch = len(dataloaders['train'])
weight_decay = 1e-4
momentum=0.9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet_model = models.resnet152(pretrained=True)
num_classes = 2  
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model = resnet_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet_model.parameters(), lr=max_lr, weight_decay = weight_decay, momentum=momentum)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=num_epochs, steps_per_epoch=steps_per_epoch)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, file_path):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)


        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0


            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Phase', leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)


                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            torch.save(model.state_dict(), file_path)


    return model

model_ft = train_model(resnet_model, dataloaders, criterion, optimizer,sched, num_epochs, f'resnet_model_resource/lr{max_lr}_momen{momentum}_decay{weight_decay}_resnet152.pth')


def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model
def evaluate_model(model, dataloader, criterion):
    model.eval()  
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)


            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return epoch_acc


file_path = f'resnet_model_resource/lr{max_lr}_momen{momentum}_decay{weight_decay}_resnet152.pth'
model_ft = load_model(resnet_model, file_path)
test_accuracy = evaluate_model(model_ft, test_dl, criterion)
