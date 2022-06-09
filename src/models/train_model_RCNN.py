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


import re
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
            self.categories[anns[i]['image_id']].append(anns[i]['category_id'])
            self.object_ids[anns[i]['image_id']].append(anns[i]['id'])
            self.image_ids[anns[i]['image_id']].append(anns[i]['image_id'])
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


# Load data
trainset_taco = TACO(data = 'train', seed=seed)
train_loader_taco = DataLoader(trainset_taco, batch_size=1, shuffle=False, num_workers=0)


# Define TACO data class (IMPORTANT: USE SAME SEED FOR TRAIN, VAL, AND TEST)
class Proposals(torch.utils.data.Dataset):
    def __init__(self, data_loader, transform, model_edge_boxes, data_path=dataset_path, max_boxes=100):
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
            path_splits = image_path[0].split('/')
            self.proposal_image_paths.extend([path_splits[-2]+'/'+path_splits[-1]]*max_boxes)
            self.proposal_image_ids.extend([image_ids[0].item()]*max_boxes)

            proposal_boxes = edge_boxes(model_edge_boxes,image_path[0],max_boxes=max_boxes)
            self.proposal_box.extend(proposal_boxes)

            max_IoU_list = []
            proposal_object_ids_list = []
            proposal_class_list = []
            for indx_bo,box in enumerate(proposal_boxes):
                box_indx, max_IoU = assign_to_box(bboxs,box)
                max_IoU_list.append(max_IoU)

                if max_IoU >= 0.6:
                    print(f'hit: {max_IoU} and {indx_bo} and {image_ids[0].item()}')
                    proposal_object_ids_list.append(object_ids[box_indx].item())
                    proposal_class_list.append(target[box_indx].item())
                else:
                    proposal_object_ids_list.append(-1)
                    proposal_class_list.append(29)

                self.proposal_IoU.extend(max_IoU_list)
                self.proposal_object_ids.extend(proposal_object_ids_list)
                self.proposal_class.extend(proposal_class_list)

            print(enum)
            if enum == 1:
                break;

    def __len__(self):
        'Returns the total number of samples'
        return len(self.proposal_image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = os.path.join(self.data_path, self.proposal_image_paths[idx])

        image = Image.open(image_path)
        image_size = image.size
        left = self.proposal_box[idx][0]
        top = self.proposal_box[idx][1]
        right = self.proposal_box[idx][0]+self.proposal_box[idx][2]
        bottom = self.proposal_box[idx][1]+self.proposal_box[idx][3]
        image = image.crop((left, top, right, bottom))

        X = self.transform(image)
        y = self.proposal_class[idx]
        img_index = self.proposal_image_ids[idx]
        obj_index = self.proposal_object_ids[idx]
        box = self.proposal_box[idx]
        IoU = self.proposal_IoU[idx]
        return X, y, img_index, obj_index, box, IoU


size = 224
train_test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                           transforms.ToTensor()])
test = Proposals(train_loader_taco, train_test_transform, "models/model.yml.gz", data_path=dataset_path,max_boxes=2000)
X, y, img_index, obj_index, box, IoU = test.__getitem__(5)
plt.imshow(X[0])
plt.savefig('hello1.png')
image_path, y, object_ids, image_ids, bbox = trainset_taco.__getitem__(img_index)

transf = transforms.ToTensor()
img = Image.open(image_path)
img = transf(img)
plt.imshow(img[0])
plt.savefig('hello1_full.png')

