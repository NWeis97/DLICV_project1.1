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

name = "EfficientNet_WithAugs-"+str(with_augs)+"_WithNorm-"+str(with_norm)+"_LR-"+str(lr)+"_NumEpochs-"+str(num_epochs)


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

batch_size = 64
testset = Hotdog_NotHotdog(train=False, transform=train_test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=3)

# Get model
efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',pretrained=False)
# Change model classifier to hotdog/notdog (don't freeze params of last layer)
num_ftrs = efficientnet.classifier.fc.in_features
efficientnet.classifier.fc = nn.Linear(num_ftrs, 2)
# Freeze parameters for pretrained model (to avoid overfitting)
efficientnet.load_state_dict(torch.load(os.getcwd()+"/models/"+name+".pt"))
efficientnet.to(device)
efficientnet.eval()


# Get accuracy of each class
test_correct = {i:0 for i in range(2)}
test_labels = {i:0 for i in range(2)}
for data, target in test_loader:
    data = data.to(device)
    with torch.no_grad():
        output = efficientnet(data)
    predicted = output.argmax(1).cpu()
    for i in range(2):
        test_correct[i] += (target[target==i]==predicted[target==i]).sum().item()
        test_labels[i] += (target==i).sum().item()

for i in range(2):
    test_correct[i] = test_correct[i]/test_labels[i]

print(test_correct)

# For wrong classifications, view the misclassifications
test_count = 0
plt.figure(figsize=(20,10))
transform = transforms.ToPILImage()

for data, target in test_loader:
    data = data.to(device)
    with torch.no_grad():
        output = efficientnet(data)
    predicted = output.argmax(1).cpu()
    for i in range(len(target)):
        if target[i]!=predicted[i]:
            plt.subplot(5,4,test_count+1)
            plt.imshow(data[i].cpu().numpy()[0], 'gray')
            plt.title(f'target: {"hotdog" if target[i].item() == 0 else "notdog"}, pred: {"hotdog" if predicted[i].item() == 0 else "notdog"}')
            plt.axis('off')
            
            test_count += 1
    
        if test_count == 8:
            break;
    if test_count == 8:
        break;

test_count += 4
for data, target in test_loader:
    data = data.to(device)
    with torch.no_grad():
        output = efficientnet(data)
    predicted = output.argmax(1).cpu()
    for i in range(len(target)):
        if target[i]==predicted[i]:
            plt.subplot(5,4,test_count+1)
            #convert image back to Height,Width,Channels
            #img = np.transpose(data[i].cpu().numpy(), (1,2,0))
            #plt.imshow(img, 'gray')
            plt.imshow(data[i].cpu().numpy()[0], 'gray')
            plt.title(f'target: {"hotdog" if target[i].item() == 0 else "notdog"}, pred: {"hotdog" if predicted[i].item() == 0 else "notdog"}')
            plt.axis('off')
            
            test_count += 1
    
        if test_count == 20:
            break;
    if test_count == 20:
        break;
plt.tight_layout()
plt.savefig(os.getcwd()+'/reports/figures/test_images_classified/'+name+'.png')








# Saliency for same images
test_count = 0
fig, axs = plt.subplots(5,4,figsize=(12,10), gridspec_kw={'height_ratios': [1, 1, 0.3, 1, 1]})

for data, target in test_loader:
    data = data.to(device)
    data.requires_grad_()
    output = efficientnet(data)
    predicted = output.argmax(1).cpu()
    
    # Catch the output
    output_idx = output.argmax(dim=1)

    for i in range(len(target)):
        if target[i]!=predicted[i]:
            # Do backpropagation to get the derivative of the output based on the image
            output_max = output[i,output_idx[i]]
            output_max.backward(retain_graph=True)
            saliency, _ = torch.max(data.grad.data[i].abs(), dim=0) 
            saliency = saliency.reshape(size, size)
            saliency = (saliency-torch.min(saliency))/(torch.max(saliency)-torch.min(saliency))

            axs.flatten()[test_count+4].imshow(saliency.cpu().numpy(), 'gnuplot2')
            axs.flatten()[test_count+4].set_title(f'Saliency of prediction')
            axs.flatten()[test_count+4].axis('off')

            axs.flatten()[test_count].imshow(data[i].detach().cpu().numpy()[0], 'gray')
            axs.flatten()[test_count].set_title(f'target: {"hotdog" if target[i].item() == 0 else "notdog"}, pred: {"hotdog" if predicted[i].item() == 0 else "notdog"}')
            axs.flatten()[test_count].axis('off')

            
            test_count += 1
    
        if test_count == 4:
            break;
    if test_count == 4:
        break;

test_count += 8
axs.flatten()[9].axis('off')
axs.flatten()[10].axis('off')
axs.flatten()[11].axis('off')
axs.flatten()[8].axis('off')
for data, target in test_loader:
    data = data.to(device)
    data.requires_grad_()
    output = efficientnet(data)
    predicted = output.argmax(1).cpu()

    # Catch the output
    output_idx = output.argmax(dim=1)

    for i in range(len(target)):
        if target[i]==predicted[i]:
            # Do backpropagation to get the derivative of the output based on the image
            output_max = output[i,output_idx[i]]
            output_max.backward(retain_graph=True)
            saliency, _ = torch.max(data.grad.data[i].abs(), dim=0) 
            saliency = saliency.reshape(size, size)
            saliency = (saliency-torch.min(saliency))/(torch.max(saliency)-torch.min(saliency))
            axs.flatten()[test_count+4].imshow(saliency.cpu().numpy(), 'gnuplot2')
            axs.flatten()[test_count+4].set_title(f'Saliency of prediction')
            axs.flatten()[test_count+4].axis('off')

            axs.flatten()[test_count].imshow(data[i].detach().cpu().numpy()[0], 'gray')
            axs.flatten()[test_count].set_title(f'target: {"hotdog" if target[i].item() == 0 else "notdog"}, pred: {"hotdog" if predicted[i].item() == 0 else "notdog"}')
            axs.flatten()[test_count].axis('off')
            
            test_count += 1
    
        if test_count == 16:
            break;
    if test_count == 16:
        break;
#plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.2, wspace=0.02)
plt.tight_layout()
plt.savefig(os.getcwd()+'/reports/figures/saliency_maps/'+name+'.png')