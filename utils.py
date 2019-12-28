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

arch = {"vgg16":25088, "alexnet":9216}

def load_data(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(train_dir), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(valid_dir), data_transforms['val']),
        'test': datasets.ImageFolder(os.path.join(test_dir), data_transforms['test'])
    }
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }
    return dataloaders, image_datasets

def pretrained_model(structure, hidden_units, learning_rate):

    #Model VGG
    if structure == 'vgg16':
        
        model = models.vgg16(pretrained=True)
        
    #Model alexnet
    elif structure == 'alexnet':

        model = models.resnet50(pretrained=True)

    else:
        print('Model entered not available, please enter other model.')

    for param in model.parameters():
        param.requires_grad = False


    n_inputs = arch[structure]
    n_classes=102
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(n_inputs, hidden_units)), 
        ('relu1', nn.ReLU()), 
        ('droupout1', nn.Dropout(0.2)),
        ('hidden_units1', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('droupout2', nn.Dropout(0.3)),
        ('hidden_units2', nn.Linear(90, 80)),
        ('relu3', nn.ReLU()),
        ('hidden_units3', nn.Linear(80, n_classes)), 
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    return model, criterion, optimizer

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=20):
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_accuracy(testloader):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy of the Model VGG16: %d %%' % (100 * correct / total))
    
def load_checkpoint(PATH, arch):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)

    elif arch == 'alexnet':
        mode = models.alexnet(pretrained=True)
    else:
        print('Model not available. Enter new model.')
    
    #optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(PATH, map_location=map_location)
    
    n_inputs = arch['structure']
    n_classes=102
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(n_inputs, hidden_units)), 
        ('relu1', nn.ReLU()), 
        ('droupout1', nn.Dropout(0.2)),
        ('hidden_units1', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('droupout2', nn.Dropout(0.3)),
        ('hidden_units2', nn.Linear(90, 80)),
        ('relu3', nn.ReLU()),
        ('hidden_units3', nn.Linear(80, n_classes)), 
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    model.class_to_idx = checkpoint['class_to_idx']
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_im = Image.open(image)
    transform_image = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    np_image = transform_image(pil_im)
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    top_classes = []
    
    model.to(device)
    
    image = process_image(image_path).to(device)
    np_image = image.unsqueeze_(0)
    
    model.eval()
    
    with torch.no_grad():
        logps = model.forward(np_image)
    
    ps = torch.exp(logps)
    top_k, top_classes_idx = ps.topk(topk, dim=1)
    top_k, top_classes_idx = np.array(top_k.to(device)[0]), np.array(top_classes_idx.to(device)[0])
    
    # Invert dictionary
    idx_to_class = {u: v for v, u in model.class_to_idx.items()}
    
    for index in top_classes_idx:
        top_classes.append(idx_to_class[index])
        
    l_top_k = list(top_k)
    l_top_classes = list(top_classes)
    
    return l_top_k, l_top_classes