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

import json
import argparse
import utils

parser = argparse.ArgumentParser()


parser.add_argument('image_path',
                    help='Image directory', default='./flowers/test/102/image_08012.jpg')
parser.add_argument('--arch', action='store',
                    dest='arch',
                    default='vgg16',
                    help='Model architecture')
parser.add_argument('checkpoint', action='store',
                    default='/home/workspace/paind-project/checkpoint2.pth',
                    help='Saved model Directory checkpoint')
parser.add_argument('--top_k', action='store',
                    dest='top_k',
                    default=5,
                    help='Top k prediction prob')
parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    default='cat_to_name.json',
                    help='classes and names association Json')
parser.add_argument('--gpu', action='store',
                    default='gpu',
                    dest='gpu',
                    help='GPU mode')

results = parser.parse_args()

image_path = results.image_path
arch =  results.arch
checkpoint = results.checkpoint
category_names = results.category_names

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# Prediction
model = utils.load_checkpoint(checkpoint)
model.eval()

device = torch.device("cuda:0" if gpu else "cpu")
probs, classes = utils.predict(image_path, model)
print('probabilities: {}'.format(probs))
print('classes: {}'.format(classes))