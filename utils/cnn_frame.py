

from utils import model_root,plot

import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.optim
import torch.nn as nn
import torchvision.transforms as transf
from torch.utils.data import random_split



def my_model():
    
    
    # Augmentation of dataset

    data_transform = transf.Compose([
                    transf.Resize((224,224)),
                    transf.ToTensor()
    ])


    # Hyperparamter tuning 

    epoch_n = 20
    lr_rate = 0.0001
    batch_size_tr = 100
    batch_size_val = 100


    # Read files from specified folders

    ds = MNIST(root='data/', download=True,train=True,transform = data_transform)
    train_ds, val_ds = random_split(ds,[50000,10000])
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size_tr, shuffle=True,drop_last=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size_val, shuffle=True,drop_last=True)

    # Check for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    # Feed forward and back-propogation
    train_loss_list = []
    val_loss_list = []
    def train(model, criterion, optim, epoch_n):
    
        best_acc= 0.0  
        for epoch in range(epoch_n):
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            for images,labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optim.zero_grad()
                with torch.set_grad_enabled(True):
                    output = model(images)
                    _,preds = torch.max(output,1)
                    loss = criterion(output,labels)
                    loss.backward()
                    optim.step()
                running_loss += loss.item()*batch_size_tr 
                running_acc += torch.sum(preds==labels)
            running_val_loss, running_val_acc = eval(model, criterion)
            epoch_train_loss = running_loss/len(train_ds)
            epoch_train_acc = running_acc.double()/len(train_ds)
            print("Epoch {}".format(epoch+1))
            print('-'*10)
            print("Train Loss: {:.4f}   Train Acc: {:.4f}".format(epoch_train_loss, epoch_train_acc))
            epoch_val_loss = running_val_loss/len(val_ds)
            epoch_val_acc = running_val_acc.double()/len(val_ds)
            print("Val Loss: {:.4f}   Val Acc: {:.4f}".format(epoch_val_loss, epoch_val_acc))
            train_loss_list.append(epoch_train_loss)
            val_loss_list.append(epoch_val_loss)
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
        print("The best instance of the model has an accuracy of: {:.4f}".format(best_acc))
        print()


    def eval(model, criterion):
        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _,preds = torch.max(output,1)
            loss = criterion(output, labels)
            running_val_loss += loss.item()*batch_size_val
            running_val_acc += torch.sum(preds==labels)
        return running_val_loss, running_val_acc


    # Define the modules
    model = model_root.SimpleConv()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optim = torch.optim.Adam(model.parameters(),lr = lr_rate)


    # Call the framework
    train(model, criterion, optim, epoch_n)


    # Plot the train-validation curve
    plot.train_val_graph(epoch_n, train_loss_list, val_loss_list)


