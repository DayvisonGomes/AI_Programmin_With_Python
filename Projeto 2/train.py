import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import PIL as pil
import json
import argparse
import data
import model_nn

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default="./flowers/", help='Directory of the data')
parser.add_argument('--save_dir', type=str, default="./checkpoint.pth", help='Save the checkpoint of the nn')
parser.add_argument('--arch', type=str, default="resnet50", help='The arch of the nn')
parser.add_argument('--learning_rate', type=float, default=0.003, help='The learning rate')
parser.add_argument('--hidden_units', type=int, default=2048, help='The hidden units')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')

args = parser.parse_args()
data_dir = args.data_dir
path = args.save_dir
lr = args.learning_rate
structure = args.arch
hidden_units = args.hidden_units
epochs = args.epochs
dropout = args.dropout

def main():

    trainloader, validloader, testloader, train_data = data.load_data(data_dir)
    model, criterion, optimizer, device = model_nn.setup_network(structure, dropout, hidden_units, lr)
    
    epochs = 5
    steps = 0
    running_loss = 0
    train_losses, valid_losses = [], []
    
    print("--Training starting--")
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy_valid = 0

            with torch.no_grad():
                model.eval()

                for images, labels in validloader:

                    images, labels = images.to(device), labels.to(device)

                    log_ps = model(images)
                    valid_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy_valid += torch.mean(equals.type(torch.FloatTensor))

            model.train()      
            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))

            print("Epochs: {}/{}..".format(e+1, epochs),
                 "Training loss: {:.3f}..".format(running_loss/len(trainloader)),
                 "Valid loss: {:.3f}..".format(valid_loss/len(validloader)),
                 "Valid accuracy: {:.3f}".format(accuracy_valid/len(validloader)))
    
    model.class_to_idx =  train_data.class_to_idx
    
    model_nn.save_checkpoint(train_data, model=model, path=path, structure=structure, hidden_units=hidden_units, dropout=dropout, lr=lr, epochs=epochs)

    print("Checkpoint created")


main()
