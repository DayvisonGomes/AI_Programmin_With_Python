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

parser.add_argument('--input', type = str, default='./flowers/test/10/image_07090.jpg')
parser.add_argument('--dir', type=str, default="./flowers/")
parser.add_argument('--checkpoint', type = str, default='./checkpoint.pth' )
parser.add_argument('--top_k', type=int, default=5 )
parser.add_argument('--category_names', default='cat_to_name.json', type=str)

args = parser.parse_args()
path_image = args.input
top_k = args.top_k
category_names = args.category_names
path = args.checkpoint

def main():
    model = model_nn.load_checkpoint(path)
    
    with open(category_names, 'r') as json_file:
        names = json.load(json_file)
        
    probabilities = model_nn.predict(path_image, model, top_k)
    probability = np.array(probabilities[0][0])
    labels = [names[str(i + 1)] for i in np.array(probabilities[1][0])]
    
    print(f'\nThe Top_k:{top_k} probabilities: \n')
    for i in range(top_k):
        print("{} with a probability of {}\n".format(labels[i], probability[i]))
    
    print("Finish")
    
main()
