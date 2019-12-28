import matplotlib.pyplot as plt
import os
import numpy as np
import time

import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

import copy

import argparse
import utils

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', help='Files directory path', default='./flowers')

parser.add_argument('--arch', action='store',
                    dest='arch',
                    default='vgg16',
                    help='Model architecture')
parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='Saved model Directory', default='checkpoint2.pth')
parser.add_argument('--hidden_units', action='append',
                    dest='hidden_units',
                    default=256,
                    help='Model Hidden units')
parser.add_argument('--ephocs', action='store',
                    dest='epochs',
                    default=20,
                    help='Epochs model training')
parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    default=0.001,
                    help='Model Learning rate')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='GPU device')

results = parser.parse_args()

data_dir = results.data_dir
structure = results.arch
save_dir = results.save_dir
learning_rate = results.learning_rate
hidden_units = results.hidden_units
epochs = results.epochs
gpu = results.gpu

device = torch.device("cuda:0" if gpu else "cpu")

#Load data
dataloaders, image_datasets = utils.load_data(data_dir)

#Setup model parameters
model, criterion, optimizer = utils.pretrained_model(structure, hidden_units, learning_rate)
model = model.to(device)

#train model
model = utils.train_model(model, criterion, optimizer, dataloaders, device, epochs)

#save model
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'epoch': epochs, 
              'class_to_idx': model.class_to_idx,
              'layers': results.hidden_units,
              'optimizer_state_dict': optimizer.state_dict(), 
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint2.pth')
