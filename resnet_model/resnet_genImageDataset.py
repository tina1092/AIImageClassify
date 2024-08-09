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
train_tfms = tt.Compose([tt.Resize((400, 400)),
                         tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([
    tt.Resize((400,400)),  
    tt.ToTensor(), 
    tt.Normalize(*stats)
])
test_tfms = tt.Compose([
    tt.Resize((400,400)),  
    tt.ToTensor(), 
    tt.Normalize(*stats)
])

batch_size = 32

# PyTorch datasets

data_dir = '/home/ec2-user/imageAI/image_resource/GenImage/distribution4'
train_ds = ImageFolder(data_dir+'/train', train_tfms)
valid_ds = ImageFolder(data_dir+'/val', valid_tfms)
test_ds = ImageFolder(data_dir+'/test', test_tfms)
# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size*2, num_workers=3, pin_memory=True)
dataloaders = {'train': train_dl, 'val': valid_dl, 'test': test_dl}


num_epochs = 300
max_lr = 0.001
steps_per_epoch = len(dataloaders['train'])
weight_decay = 1e-4
momentum=0.9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet_model = models.resnet50(pretrained=True)
num_classes = 2  
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model = resnet_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet_model.parameters(), lr=max_lr, weight_decay = weight_decay, momentum=momentum)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=num_epochs, steps_per_epoch=steps_per_epoch)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, file_path):
    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Phase', leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  # Update learning rate

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = model.state_dict()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), file_path)
    print(f'Best val Acc: {best_val_acc:.4f}')

    # Save training history
    history_path = file_path.replace('.pth', '_history.pt')
    torch.save(history, history_path)

    return model, history

# Assuming the model and other variables are already defined

model_ft, history = train_model(resnet_model, dataloaders, criterion, optimizer, sched, num_epochs, f'resnet_model_resource/genImage/sdv4/lr{max_lr}_momen{momentum}_decay{weight_decay}_resnet50.pth')

# Function to load the model
def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

# Function to compute accuracy on the test dataset
def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0

    # Disable gradient computation
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return epoch_acc

# Load the saved model weights
file_path = f'resnet_model_resource/genImage/sdv4/lr{max_lr}_momen{momentum}_decay{weight_decay}_resnet50.pth'
model_ft = load_model(resnet_model, file_path)

# Evaluate the model using the test dataset
test_accuracy = evaluate_model(model_ft, test_dl, criterion)

import torch
import matplotlib.pyplot as plt
file_path = f'resnet_model_resource/genImage/sdv4/lr{max_lr}_momen{momentum}_decay{weight_decay}_resnet50_history.pt'

model_data = torch.load(file_path)

# Plot training and validation result
def plot_training_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)
save_path = f'resnet_model_resource/genImage/sdv4/lr{max_lr}_momen{momentum}_decay{weight_decay}_resnet50_loss&Accuracy.jpg'

plot_training_history(model_data, save_path)

