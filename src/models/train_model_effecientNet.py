# Import 
import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pdb

# settings
torch.manual_seed(1234)
with_augs = False
lr = 0.001
with_norm = False
num_epochs = 10

# Res
# False, 0.001, False, 10: Accuracy train: 94.8%    test: 93.3%
# False, 0.001, True, 10: Accuracy train: 94.4%    test: 92.9%
# True, 0.001, True, 10: Accuracy train: 96.1%    test: 92.9%
# True, 0.001, True, 10: Accuracy train: 95.7%    test: 93.2%



# Get CUDA
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define Hotdog data class
class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02514/hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


# Load data
size = 224
train_test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5226-0.5226*(1.0*with_norm), 0.4412-0.4412*(1.0*with_norm), 0.3585-0.3585*(1.0*with_norm)), 
                                                         (0.0036+(1-0.0036)*(1.0*with_norm), 0.0036+(1-0.0036)*(1.0*with_norm), 0.0050+(1-0.0050)*(1.0*with_norm)))
                                    ])
train_transform_aug = transforms.Compose([transforms.Resize((size, size)), 
                                          transforms.RandomRotation(degrees=30),
                                          transforms.RandomCrop(200, padding=(12,12)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5226-0.5226*(1.0*with_norm), 0.4412-0.4412*(1.0*with_norm), 0.3585-0.3585*(1.0*with_norm)), 
                                                         (0.0036+(1-0.0036)*(1.0*with_norm), 0.0036+(1-0.0036)*(1.0*with_norm), 0.0050+(1-0.0050)*(1.0*with_norm)))
                                          ])

batch_size = 64
trainset = Hotdog_NotHotdog(train=True, transform=train_test_transform)
trainset_aug = Hotdog_NotHotdog(train=True, transform=train_test_transform)
if with_augs:
    trainset= torch.utils.data.ConcatDataset([trainset, trainset_aug])
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)

testset = Hotdog_NotHotdog(train=False, transform=train_test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

"""
means = torch.zeros((3,))
stds = torch.zeros((3,))
for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
    means += torch.mean(torch.mean(torch.mean(data,dim=0),dim=1),dim=1)
    stds += torch.std(torch.std(torch.std(data,dim=0),dim=1),dim=1)

means = means/len(train_loader)
stds = stds/len(train_loader)
pdb.set_trace()
"""

# Get model
efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# Freeze parameters for pretrained model (to avoid overfitting)
for param in efficientnet.parameters():
    param.requires_grad = False
efficientnet.to(device)

# Change model classifier to hotdog/notdog (don't freeze params of last layer)
num_ftrs = efficientnet.classifier.fc.in_features
efficientnet.classifier.fc = nn.Linear(num_ftrs, 2)

# Set optimizer and model
model = efficientnet.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

def loss_fun(output, target):
    crit = nn.CrossEntropyLoss()
    return crit(output, target)

out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}

for epoch in tqdm(range(num_epochs), unit='epoch'):
    #For each epoch
    train_correct = 0
    train_loss = []
    model.train()
    for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        #Zero the gradients computed for each weight
        optimizer.zero_grad()
        #Forward pass your image through the network
        output = model(data)
        #Compute the loss
        loss = loss_fun(output, target)
        #Backward pass through the network
        loss.backward()
        #Update the weights
        optimizer.step()
        
        train_loss.append(loss.cpu().item())
        #Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct += (target==predicted).sum().cpu().item()
    #Comput the test accuracy
    model.eval()
    test_loss = []
    test_correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss.append(loss_fun(output, target).cpu().item())
        predicted = output.argmax(1)
        test_correct += (target==predicted).sum().cpu().item()
    train_acc = train_correct/len(trainset)
    test_acc = test_correct/len(testset)

    out_dict['train_acc'].append(train_correct/len(trainset))
    out_dict['test_acc'].append(test_correct/len(testset))
    out_dict['train_loss'].append(np.mean(train_loss))
    out_dict['test_loss'].append(np.mean(test_loss))
    print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))

# Save model
name = "EfficientNet_WithAugs-"+str(with_augs)+"_WithNorm-"+str(with_norm)+"_LR-"+str(lr)+"_NumEpochs-"+str(num_epochs)
torch.save(efficientnet.state_dict(), os.getcwd()+"/models/"+name+".pt")

# Make plot of training curves
fig, ax = plt.subplots(1,2,figsize=(15,7))
ax[0].plot(out_dict['train_acc'])
ax[0].plot(out_dict['test_acc'])
ax[0].legend(('Train accuracy','Test accuracy'))
ax[0].set_xlabel('Epoch number')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Training and test accuracy')
ax[1].plot(out_dict['train_loss'])
ax[1].plot(out_dict['test_loss'])
ax[1].legend(('Train loss','Test loss'))
ax[1].set_xlabel('Epoch number')
ax[1].set_ylabel('Loss')
ax[1].set_title('Training and test loss')
fig.savefig(os.getcwd()+'/reports/figures/training_curves/'+name+'.png')