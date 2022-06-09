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


import re
# Define TACO data class (IMPORTANT: USE SAME SEED FOR TRAIN, VAL, AND TEST)
class TACO(torch.utils.data.Dataset):
    def __init__(self, data, transform, data_path=dataset_path, seed=1234):
        'Initialization'
        self.seed = seed
        np.random.seed(self.seed)
        self.transform = transform

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
        self.ids = {i:[] for i in range(len(imgs))}
        self.bboxs = {i:[] for i in range(len(imgs))}

        for i in range(len(anns)):
            self.categories[anns[i]['image_id']].append(anns[i]['category_id'])
            self.ids[anns[i]['image_id']].append(anns[i]['id'])
            self.bboxs[anns[i]['image_id']].append(np.array(anns[i]['bbox']).astype(int))
        """
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
        self.ids = [self.ids[index] for index in data_idx]
        self.bboxs = [self.bboxs[index] for index in data_idx]
        """


    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        X = self.transform(image)
        y = self.categories[idx]
        index = self.ids[idx]
        bbox = self.bboxs[idx]
        return image_path, y, index, bbox

def edge_boxes(model,image_path, max_boxes=100):
    #image = Image.open(image_path)
    im = cv.imread(image_path)
        #transforms(image))

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(max_boxes)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    return boxes



# Load data
train_test_transform = transforms.Compose([transforms.ToTensor()])
trainset_taco = TACO(data = 'train', transform=train_test_transform, seed=seed)
train_loader_taco = DataLoader(trainset_taco, batch_size=1, shuffle=False, num_workers=0)



test,_,_,_ = trainset_taco.__getitem__(0)
boxes = edge_boxes("models/model.yml.gz",test )
pdb.set_trace()