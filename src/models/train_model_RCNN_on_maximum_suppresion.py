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
from torchvision.ops import batched_nms

import cv2 as cv
import numpy as np
import sys


dataset_path = '../../../../../../../dtu/datasets1/02514/data_wastedetection'

print('test0')

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



# Get CUDA
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test data
test_dataset = torch.load('models/datasets/test_dataset_taco_v2.pt')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


print('test1')
# Load model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 29)
model_state_dict = torch.load(os.path.join(os.getcwd(),'models/RCNN_model_state_dict.pt'))
model.load_state_dict(model_state_dict)
model.to(device)


size = 224
train_test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                           transforms.ToTensor()])


results_list_2 = {
    "target": [],
    "img_index": [],
    "obj_index": [],
    "box": [],
    "pred" : [],
    "probs" : [],
    "IoU_to_GT": []
}     

print('test2')
# Go through data and extract probs
for minibatch_no, (image_path, target, img_index, obj_index, box, IoU_val) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
    # BELOW CODE SHOULD BE USED FOR MAKING "DATA" OBJECT FROM IMAGE_PATHS
    data = []
    box = [torch.tensor([box[0][p].item(),box[1][p].item(),box[2][p].item(),box[3][p].item()]) for p in range(len(target))]

    for i in range(len(target)):
        # Crop image
        image = Image.open(image_path[i])
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

    #Forward pass your image through the network

    with torch.no_grad():
        output = model(data)
    output = F.softmax(output,dim=1)
    pred = output.argmax(dim=1)
    probs = output[np.arange(0,output.shape[0],1),pred]

    results_list_2['target'].extend(target.cpu().numpy().tolist())
    results_list_2['img_index'].extend(img_index.cpu().numpy().tolist())
    results_list_2['obj_index'].extend(obj_index.cpu().numpy().tolist())
    results_list_2['box'].extend([b.cpu().numpy().tolist() for b in box])
    results_list_2['probs'].extend(probs.cpu().detach().numpy().tolist())
    results_list_2['pred'].extend(pred.cpu().detach().numpy().tolist())
    results_list_2['IoU_to_GT'].extend(IoU_val.cpu().detach().numpy().tolist())

    print(f'{minibatch_no}/{len(test_loader)}')
    if (minibatch_no+1) % 10 == 0:
        print(f'Saving updated version of results')
        results = pd.DataFrame.from_dict(results_list_2).sort_values(['img_index'])
        results_dict = {}
        j_count = 0
        prob_min = 0.5
        pdb.set_trace()
        for j in results['img_index'].unique():
            if results[(results['img_index']==j) & (((results['pred']!=28) & (results['probs']>prob_min)))].empty == False:
                results_dict[j_count]= results[(results['img_index']==j) & (((results['pred']!=28) & (results['probs']>prob_min)))].reset_index().drop(columns='index')
                for i in range(len(results_dict[j_count])):
                    results_dict[j_count]['box'][i][2] = results_dict[j_count]['box'][i][0]+results_dict[j_count]['box'][i][2]
                    results_dict[j_count]['box'][i][3] = results_dict[j_count]['box'][i][1]+results_dict[j_count]['box'][i][3]
                
                j_count += 1



        iou_threshold = 0.5
        for ix in range(len(results_dict.keys())):
            boxes = torch.tensor(results_dict[ix]['box']).float()
            scores = torch.tensor(results_dict[ix]['probs']).float()
            idxs = torch.tensor(results_dict[ix]['pred']).int()
            out = batched_nms(boxes, scores, idxs,iou_threshold)

            results_dict[ix] = results_dict[ix].iloc[out.numpy()].reset_index().drop(columns='index')



        results_df = pd.concat(results_dict,axis=0).reset_index().drop(columns=['level_0','level_1'])
        results_df.to_csv('reports/RCCN_res_after_nms.csv')





