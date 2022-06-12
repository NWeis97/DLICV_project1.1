# Import 
from operator import index
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
import torchvision.models as models
import matplotlib.pyplot as plt
import pdb
import re
import pandas as pd

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
    map_category_to_super[i] = super_cat_names.index(super_cat)


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
    def __init__(self, data_loader,  model_edge_boxes, data_path=dataset_path, max_boxes=2000, train_val=True):
        'Initialization'
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
            
            proposal_boxes = [box.tolist() for box in proposal_boxes]
            path_splits = image_path[0].split('/')

            if train_val is True: #Include GT for training
                self.proposal_image_paths.extend([path_splits[-2]+'/'+path_splits[-1]]*len(target))
                self.proposal_image_ids.extend([image_ids[0].item()]*len(target))
                self.proposal_object_ids.extend(obj.numpy().item() for obj in object_ids)
                self.proposal_box.extend([box.numpy().tolist()[0] for box in bboxs])
                self.proposal_class.extend([tar.numpy().item() for tar in target])
                self.proposal_IoU.extend([1]*len(target))
            
            
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

                if max_IoU >= 0.7:
                    proposal_class_list.append(target[box_indx].item())
                    indx_save.append(indx_bo)
                elif max_IoU < 0.3:
                    proposal_class_list.append(28)
                    indx_bg.append(indx_bo)
                else:
                    proposal_class_list.append(28)
            

            # Remove most bg proposals or not
            #if train_val is False:
            self.proposal_image_paths.extend([path_splits[-2]+'/'+path_splits[-1]]*len(proposal_boxes))
            self.proposal_image_ids.extend([image_ids[0].item()]*len(proposal_boxes))
            self.proposal_box.extend(proposal_boxes)
            self.proposal_IoU.extend(max_IoU_list)
            self.proposal_object_ids.extend(proposal_object_ids_list)
            self.proposal_class.extend(proposal_class_list)
            """
            else:
                num_pos = len(target[0]) + len(indx_save)
                indx_bg = np.random.permutation(indx_bg)
                indx_bg = indx_bg[:num_pos*3]
                indx_save.extend(indx_bg) 
                self.proposal_image_paths.extend([path_splits[-2]+'/'+path_splits[-1]]*len(indx_save))
                self.proposal_image_ids.extend([image_ids[0].item()]*len(indx_save))
                self.proposal_box.extend([proposal_boxes[i] for i in indx_save])
                self.proposal_IoU.extend([max_IoU_list[i] for i in indx_save])
                self.proposal_object_ids.extend([proposal_object_ids_list[i] for i in indx_save])
                self.proposal_class.extend([proposal_class_list[i] for i in indx_save])
            """
            # Print progress
            print(f'{enum}/{len(data_loader)}')
            #if enum == 7:
             #   break;
                


    def __len__(self):
        'Returns the total number of samples'
        return len(self.proposal_image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = os.path.join(self.data_path, self.proposal_image_paths[idx])

        # Extract image and meta data
        y = self.proposal_class[idx]
        img_index = self.proposal_image_ids[idx]
        obj_index = self.proposal_object_ids[idx]
        box = self.proposal_box[idx]
        IoU = self.proposal_IoU[idx]
        return image_path, y, img_index, obj_index, box, IoU



# Load data
trainset_taco = TACO(data = 'train', seed=seed)
train_loader_taco = DataLoader(trainset_taco, batch_size=1, shuffle=False, num_workers=0)
valset_taco = TACO(data = 'val', seed=seed)
val_loader_taco = DataLoader(valset_taco, batch_size=1, shuffle=False, num_workers=0)
testset_taco = TACO(data = 'test', seed=seed)
test_loader_taco = DataLoader(testset_taco, batch_size=1, shuffle=False, num_workers=0)


"""
# save datasets
train_dataset = Proposals(train_loader_taco,  "models/model.yml.gz", data_path=dataset_path,max_boxes=2000, train_val=True)
torch.save(train_dataset,'models/datasets/train_dataset_taco_v2.pt')
print('Saved train dataset')
val_dataset = Proposals(val_loader_taco, "models/model.yml.gz", data_path=dataset_path,max_boxes=2000, train_val=True)
torch.save(val_dataset,'models/datasets/val_dataset_taco_v2.pt')
print('Saved val dataset')

test_dataset = Proposals(test_loader_taco,  "models/model.yml.gz", data_path=dataset_path,max_boxes=2000, train_val=False)
torch.save(test_dataset,'models/datasets/test_dataset_taco_v2.pt')
print('Saved test dataset')
"""


# Load train, val, and test datasets
train_dataset = torch.load('models/datasets/train_dataset_taco_v2.pt')
val_dataset = torch.load('models/datasets/val_dataset_taco_v2.pt')
test_dataset = torch.load('models/datasets/test_dataset_taco_v2.pt')


# Include in data loader
torch.manual_seed(98765)
train_loader = DataLoader(train_dataset, batch_size=3000, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=3000, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)






size = 224
train_test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                           transforms.ToTensor()])

model = models.resnet18(pretrained=True)
#for param in model.parameters():
#    param.requires_grad = False

num_ftrs = model.fc.in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 29)

out_dict = {'train_acc': [],
              'val_acc': [],
              'train_loss': [],
              'val_loss': []}

lowest_val_loss=1e10



###### Load already trained model #######
print('Loading existing model')
model_state_dict = torch.load(os.path.join(os.getcwd(),'models/RCNN_model_state_dict_v2.pt'))
model.load_state_dict(model_state_dict)
out_dict = pd.read_csv(os.path.join(os.getcwd(),'reports/training_RCNN_v2.csv'),index_col=0)
lowest_val_loss = out_dict['val_loss'].to_numpy()[-1]
out_dict = out_dict.to_dict()
for key in out_dict.keys():
    new_list = []
    for val in out_dict[key].values():
        new_list.append(val)
    
    out_dict[key] = new_list

#New seed for training since the
np.random.seed(98765)
#########################################




model = model.to(device)

# Observe that all parameters are being optimized
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

def loss_fun(output, target):
    crit = nn.CrossEntropyLoss()
    return crit(output, target)



num_epochs = 25



for epoch in tqdm(range(num_epochs), unit='epoch'):
    #For each epoch
    train_correct = 0
    train_correct_class = {i:0 for i in range(29)}
    train_num_class = {i:0 for i in range(29)}
    train_loss = []
    model.train()
    
    for minibatch_no, (image_path, target, img_index, obj_index, box, IoU_val) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        ################## Downsample bg images ###################
        pos = np.arange(0,len(target),1)[target!=28]
        if len(pos) == 0:
            continue;
        neg = np.arange(0,len(target),1)[target==28]

        neg = np.random.permutation(neg)
        neg = neg[:len(pos)*3].tolist()
        pos = pos.tolist()
        pos.extend(neg) 

        target = target[pos]
        img_index = img_index[pos]
        obj_index = obj_index[pos]
        box = [torch.tensor([box[0][p].item(),box[1][p].item(),box[2][p].item(),box[3][p].item()]) for p in pos]
        IoU_val = IoU_val[pos]

        # BELOW CODE SHOULD BE USED FOR MAKING "DATA" OBJECT FROM IMAGE_PATHS
        data = []
        for i in range(len(target)):
            # Crop image
            image = Image.open(image_path[pos[i]])
            left = box[i][0].item()
            top = box[i][1].item()
            right = box[i][0].item()+box[i][2].item()
            bottom = box[i][1].item()+box[i][3].item()
            image = image.crop((left, top, right, bottom))
            X = train_test_transform(image)
            data.append(X)

        data = torch.stack(data,dim=0)
        #############################################################

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
        #exp_lr_scheduler.step()
        
        train_loss.append(loss.cpu().item())
        #Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct += (target==predicted).sum().cpu().item()

        for i in range(29):
            train_correct_class[i] += (target[target==i]==predicted[target==i]).sum().cpu().item()
            train_num_class[i] += (target==i).sum().cpu().item()


    #Comput the test accuracy
    model.eval()
    val_loss = []
    val_correct = 0
    val_correct_class = {i:0 for i in range(29)}
    val_num_class = {i:0 for i in range(29)}

    for minibatch_no, (image_path, target, img_index, obj_index, box, IoU_val) in enumerate(val_loader):
        # Downsample bg images
        pos = np.arange(0,len(target),1)[target!=28]
        if len(pos) == 0:
            continue;
        neg = np.arange(0,len(target),1)[target==28]

        neg = np.random.permutation(neg)
        neg = neg[:len(pos)*3].tolist()
        pos = pos.tolist()
        pos.extend(neg) 

        target = target[pos]
        img_index = img_index[pos]
        obj_index = obj_index[pos]
        box = [torch.tensor([box[0][p].item(),box[1][p].item(),box[2][p].item(),box[3][p].item()]) for p in pos]
        IoU_val = IoU_val[pos]

        data = []
        for i in range(len(pos)):
            # Crop image
            image = Image.open(image_path[pos[i]])
            left = box[i][0].item()
            top = box[i][1].item()
            right = box[i][0].item()+box[i][2].item()
            bottom = box[i][1].item()+box[i][3].item()
            image = image.crop((left, top, right, bottom))
            X = train_test_transform(image)
            data.append(X)

        data = torch.stack(data,dim=0)

        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)
        val_loss.append(loss_fun(output, target).cpu().item())
        predicted = output.argmax(1)
        val_correct += (target==predicted).sum().cpu().item()
        for i in range(29):
            val_correct_class[i] += (target[target==i]==predicted[target==i]).sum().cpu().item()
            val_num_class[i] += (target==i).sum().cpu().item()
        
        print(f'{minibatch_no+1}/{len(val_loader)}')
    

    train_acc = train_correct/(sum(train_num_class.values()))
    val_acc = val_correct/(sum(val_num_class.values()))
    for i in range(29):
        if (train_num_class[i] != 0) & (val_num_class[i] != 0):
            print(f'Class: {i} - Accuracy train: {train_correct_class[i]/train_num_class[i]*100:.1f}%\t test: {val_correct_class[i]/val_num_class[i]*100:.1f}%')


    out_dict['train_acc'].append(train_correct/(sum(train_num_class.values())))
    out_dict['val_acc'].append(val_correct/(sum(val_num_class.values())))
    out_dict['train_loss'].append(np.mean(train_loss))
    out_dict['val_loss'].append(np.mean(val_loss))

    # Save training data
    pd_save = pd.DataFrame.from_dict(out_dict)
    pd_save.to_csv(os.path.join(os.getcwd(),'reports/training_RCNN_v2.csv'))
    print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*val_acc, train=100*train_acc))

    # Save model if validation loss is lower than current best
    if np.mean(val_loss) < lowest_val_loss:
        torch.save(model.state_dict(),os.path.join(os.getcwd(),'models/RCNN_model_state_dict_v2.pt'))
        lowest_val_loss = np.mean(val_loss)
        print(f'New best model - Model saved')



