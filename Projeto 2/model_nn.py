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

def setup_network(structure='resnet50', dropout=0.2, hidden_units=512, lr=0.003):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if structure == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif structure == 'vgg16':
        model = models.vgg16(pretrained=True)
   
    for para in model.parameters():
        para.requires_grad = False   

    classifier = nn.Sequential(nn.Linear(2048 , hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    
    model.fc = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr)
    model.to(device)

    return model, criterion, optimizer, device

def save_checkpoint(train_data, model=None, path='checkpoint.pth', structure='resnet50', hidden_units=2048, dropout=0.2, lr=0.003, epochs=5):
   
    torch.save({'structure' :structure,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    
def load_checkpoint(path='checkpoint.pth'):
    
    checkpoint = torch.load(path)
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['no_of_epochs']
    structure = checkpoint['structure']

    model, _, _, _ = setup_network(structure, dropout, hidden_units, lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, topk=5):
    
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        output = model(img.cuda())
        
    probs = torch.exp(output).data
    
    return probs.topk(topk)


def process_image(image):

    img_pil = pil.Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)

    return image
