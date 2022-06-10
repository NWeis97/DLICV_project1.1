# Import 
import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pdb
import re

import cv2 as cv
import numpy as np
import sys

# Get CUDA
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



dataset_path = '../../../../../../../dtu/datasets1/02514/data_wastedetection'
anns_file_path = dataset_path + '/' + 'annotations.json'
batch_size = 4
seed = 1234

# Set seeds
np.random.seed(seed)
torch.manual_seed(seed)
cv.setRNGSeed(seed)

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)

# Load categories and super categories
cat_names = []
super_cat_names = []
super_cat_ids = {}
super_cat_last_name = ''
nr_super_cats = 0
for cat_it in categories:
    cat_names.append(cat_it['name'])
    super_cat_name = cat_it['supercategory']
    # Adding new supercat
    if super_cat_name != super_cat_last_name:
        super_cat_names.append(super_cat_name)
        super_cat_ids[super_cat_name] = nr_super_cats
        super_cat_last_name = super_cat_name
        nr_super_cats += 1


print('Number of super categories:', nr_super_cats)
print('Number of categories:', nr_cats)
print('Number of annotations:', nr_annotations)
print('Number of images:', nr_images)

map_category_to_super = {}
for i in range(nr_cats):
    super_cat = dataset['categories'][i]['supercategory']
    map_category_to_super[i] = super_cat_names.index(super_cat)-1


# Define TACO data class (IMPORTANT: USE SAME SEED FOR TRAIN, VAL, AND TEST)
class TACO(torch.utils.data.Dataset):
    def __init__(self, data, data_path=dataset_path, seed=1234):
        'Initialization'
        self.seed = seed
        np.random.seed(self.seed)

        # Read annotations
        with open(dataset_path + '/' + 'annotations.json', 'r') as f:
            dataset = json.loads(f.read())

        # Extract annotations and image ids
        anns = dataset['annotations']
        imgs = dataset['images']

        # Extract images with image_id ordered [0,1,2,...,1500]
        self.image_paths = [os.path.join(data_path, imgs[0]['file_name'])]
        for i in range(1,len(imgs)):
            self.image_paths.append(os.path.join(data_path, imgs[i]['file_name']))

        # Extract categories, ids, and boxes
        self.categories = {i:[] for i in range(len(imgs))}
        self.object_ids = {i:[] for i in range(len(imgs))}
        self.image_ids = {i:[] for i in range(len(imgs))}
        self.bboxs = {i:[] for i in range(len(imgs))}

        for i in range(len(anns)):
            self.categories[anns[i]['image_id']].append(map_category_to_super[anns[i]['category_id']])
            self.object_ids[anns[i]['image_id']].append(anns[i]['id'])
            self.image_ids[anns[i]['image_id']].append(anns[i]['image_id'])
            self.bboxs[anns[i]['image_id']].append(np.array(anns[i]['bbox']).astype(int))
        
        # Select train, val, or test
        rand_perm = np.random.permutation(range(len(imgs)))
        if data == 'train':
            data_idx = rand_perm[0:int(len(imgs)*0.6)]
        elif data == 'val':
            data_idx = rand_perm[int(len(imgs)*0.6):int(len(imgs)*0.75)]
        else:
            data_idx = rand_perm[int(len(imgs)*0.75):]

        # Extract train/val/test data
        self.image_paths = [self.image_paths[index] for index in data_idx]
        self.categories = [self.categories[index] for index in data_idx]
        self.image_ids = [self.image_ids[index] for index in data_idx]
        self.object_ids = [self.object_ids[index] for index in data_idx]
        self.bboxs = [self.bboxs[index] for index in data_idx]
        


    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        y = self.categories[idx]
        image_ids = self.image_ids[idx]
        object_ids = self.object_ids[idx]
        bbox = self.bboxs[idx]
        return image_path, y, object_ids, image_ids, bbox

def edge_boxes(model,image_path, max_boxes=100):
    #image = Image.open(image_path)
    im = cv.imread(image_path)

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(max_boxes)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    return boxes


def IoU(a, b):
    a = np.array(a)
    b = np.array(b)

    a_xmin = a[0]
    a_ymin = a[1]
    a_xmax = a_xmin + a[2]
    a_ymax = a_ymin + a[3]

    b_xmin = b[0]
    b_ymin = b[1]
    b_xmax = b_xmin + b[2]
    b_ymax = b_ymin + b[3]

    dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
    dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)

    if (dx>=0) and (dy>=0):
        area_I = dx*dy
        area_U = a[2]*a[3]+b[2]*b[3]-area_I
        return area_I/area_U
    else:
        return 0

def assign_to_box(list_of_GT_boxes,box):
    box_indx = -1
    max_IoU = 0
    for indx,GT_box in enumerate(list_of_GT_boxes):
         GT_box = np.array(GT_box).ravel()
         IoU_box = IoU(GT_box, box)
         if IoU_box > max_IoU:
             box_indx = indx
             max_IoU = IoU_box

    return box_indx, max_IoU


# Define TACO data class (IMPORTANT: USE SAME SEED FOR TRAIN, VAL, AND TEST)
class Proposals(torch.utils.data.Dataset):
    def __init__(self, data_loader, transform, model_edge_boxes, data_path=dataset_path, max_boxes=100, num_bg_examples=None):
        'Initialization'
        self.transform = transform
        self.data_path = data_path

        self.proposal_image_paths = [] #path to image
        self.proposal_image_ids = [] #image id
        self.proposal_object_ids = [] #object id (assigned object if IoU above 0.7)
        self.proposal_box = [] #proposal box
        self.proposal_class = [] #29 for background
        self.proposal_IoU = [] #set to 0 if background 


        for enum, (image_path, target, object_ids, image_ids, bboxs) in enumerate(data_loader):
            # Get boxes
            proposal_boxes = edge_boxes(model_edge_boxes,image_path[0],max_boxes=max_boxes)

            # Init lists
            indx_bg = []
            indx_save = []
            max_IoU_list = []
            proposal_object_ids_list = []
            proposal_class_list = []
            
            # Run through each proposal and calc IoU to nearest GT
            for indx_bo,box in enumerate(proposal_boxes):
                box_indx, max_IoU = assign_to_box(bboxs,box)

                # append to list
                max_IoU_list.append(max_IoU)
                proposal_object_ids_list.append(object_ids[box_indx].item())

                if max_IoU >= 0.5:
                    proposal_class_list.append(target[box_indx].item())
                    indx_save.append(indx_bo)
                else:
                    proposal_class_list.append(29)
                    indx_bg.append(indx_bo)


            path_splits = image_path[0].split('/')

            # Remove most bg proposals or not
            if num_bg_examples is None:
                self.proposal_image_paths.extend([path_splits[-2]+'/'+path_splits[-1]]*max_boxes)
                self.proposal_image_ids.extend([image_ids[0].item()]*max_boxes)
                self.proposal_box.extend(proposal_boxes)
                self.proposal_IoU.extend(max_IoU_list)
                self.proposal_object_ids.extend(proposal_object_ids_list)
                self.proposal_class.extend(proposal_class_list)
            else:
                indx_bg = np.random.permutation(indx_bg)
                indx_bg = indx_bg[:num_bg_examples]
                indx_save.extend(indx_bg) 
                self.proposal_image_paths.extend([path_splits[-2]+'/'+path_splits[-1]]*len(indx_save))
                self.proposal_image_ids.extend([image_ids[0].item()]*len(indx_save))
                self.proposal_box.extend([proposal_boxes[i] for i in indx_save])
                self.proposal_IoU.extend([max_IoU_list[i] for i in indx_save])
                self.proposal_object_ids.extend([proposal_object_ids_list[i] for i in indx_save])
                self.proposal_class.extend([proposal_class_list[i] for i in indx_save])

            # Print progress
            print(f'{enum}/{len(data_loader)}')


    def __len__(self):
        'Returns the total number of samples'
        return len(self.proposal_image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = os.path.join(self.data_path, self.proposal_image_paths[idx])

        # Crop image
        image = Image.open(image_path)
        left = self.proposal_box[idx][0]
        top = self.proposal_box[idx][1]
        right = self.proposal_box[idx][0]+self.proposal_box[idx][2]
        bottom = self.proposal_box[idx][1]+self.proposal_box[idx][3]
        image = image.crop((left, top, right, bottom))

        # Extract image and meta data
        X = self.transform(image)
        y = self.proposal_class[idx]
        img_index = self.proposal_image_ids[idx]
        obj_index = self.proposal_object_ids[idx]
        box = self.proposal_box[idx]
        IoU = self.proposal_IoU[idx]
        return X, y, img_index, obj_index, box, IoU



# Load data
trainset_taco = TACO(data = 'train', seed=seed)
train_loader_taco = DataLoader(trainset_taco, batch_size=1, shuffle=False, num_workers=0)
valset_taco = TACO(data = 'val', seed=seed)
val_loader_taco = DataLoader(valset_taco, batch_size=1, shuffle=False, num_workers=0)
testset_taco = TACO(data = 'test', seed=seed)
test_loader_taco = DataLoader(testset_taco, batch_size=1, shuffle=False, num_workers=0)

size = 224
train_test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                           transforms.ToTensor()])


# save datasets
train_dataset = Proposals(train_loader_taco, train_test_transform, "models/model.yml.gz", data_path=dataset_path,max_boxes=500, num_bg_examples=40)
torch.save(train_dataset,'models/datasets/train_dataset_taco.pt')
print('Saved train dataset')
val_dataset = Proposals(val_loader_taco, train_test_transform, "models/model.yml.gz", data_path=dataset_path,max_boxes=100)
torch.save(val_dataset,'models/datasets/val_dataset_taco.pt')
print('Saved val dataset')
test_dataset = Proposals(test_loader_taco, train_test_transform, "models/model.yml.gz", data_path=dataset_path,max_boxes=100)
torch.save(test_dataset,'models/datasets/test_dataset_taco.pt')
print('Saved test dataset')



# Load train, val, and test datasets
train_dataset = torch.load('models/datasets/train_dataset_taco.pt')
val_dataset = torch.load('models/datasets/val_dataset_taco.pt')
test_dataset = torch.load('models/datasets/test_dataset_taco.pt')


# Include in data loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)




# Get model
efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# Freeze parameters for pretrained model (to avoid overfitting)
for param in efficientnet.parameters():
    param.requires_grad = False
efficientnet.to(device)

# Change model classifier to hotdog/notdog (don't freeze params of last layer)
num_ftrs = efficientnet.classifier.fc.in_features
efficientnet.classifier.fc = nn.Linear(num_ftrs, 30)

# Set optimizer and model
model = efficientnet.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

def loss_fun(output, target):
    crit = nn.CrossEntropyLoss()
    return crit(output, target)

out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}

num_epochs = 2
for epoch in tqdm(range(num_epochs), unit='epoch'):
    #For each epoch
    train_correct = 0
    train_loss = []
    model.train()
    for minibatch_no, (data, target, img_index, obj_index, box, IoU_val) in tqdm(enumerate(train_loader), total=len(train_loader)):

        data, target = data.to(device), target.to(device)
        print(f'{minibatch_no+1}/{len(train_loader)}')
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
    for minibatch_no, (data, target, img_index, obj_index, box, IoU_val) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss.append(loss_fun(output, target).cpu().item())
        predicted = output.argmax(1)
        test_correct += (target==predicted).sum().cpu().item()
        print(f'{minibatch_no+1}/{len(val_loader)}')
    train_acc = train_correct/len(train_dataset)
    test_acc = test_correct/len(test_dataset)

    out_dict['train_acc'].append(train_correct/len(train_dataset))
    out_dict['test_acc'].append(test_correct/len(test_dataset))
    out_dict['train_loss'].append(np.mean(train_loss))
    out_dict['test_loss'].append(np.mean(test_loss))
    print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))