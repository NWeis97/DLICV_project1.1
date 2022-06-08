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


# Get CUDA
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



dataset_path = '../../../../../../../dtu/datasets1/02514/data_wastedetection'
anns_file_path = dataset_path + '/' + 'annotations.json'

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



# Define TACO data class
class TACO(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path=dataset_path):
        'Initialization'
        self.transform = transform
        self.batches = np.arange(1,13,1) if train == True else np.arange(13,16,1)
        data_paths = [os.path.join(data_path, 'batch_' + str(x)) for x in self.batches]
        pdb.set_trace()
        self.image_paths = glob.glob(data_paths + '/*/*.jpg')
        
        # Read annotations
        with open(anns_file_path, 'r') as f:
            dataset = json.loads(f.read())

        self.categories = {}
        self.ids = {}
        self.bboxs = {}
        for i in len(anns):
            if type(self.categories[anns[i]['image_id']]) == list:
                self.categories[anns[i]['image_id']].append(anns[i]['category_id'])
            else:
                self.categories[anns[i]['image_id']] = [anns[i]['category_id']]

            if type(self.ids[anns[i]['image_id']]) == list:
                self.ids[anns[i]['image_id']].append(anns[i]['id'])
            else:
                self.ids[anns[i]['image_id']] = [anns[i]['id']]

            if type(self.bboxs[anns[i]['image_id']]) == list:
                self.bboxs[anns[i]['image_id']].append(anns[i]['bbox'])
            else:
                self.bboxs[anns[i]['image_id']] = [anns[i]['bbox']]

        
        
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
                                    transforms.ToTensor()])
trainset = TACO(train=True, transform=train_test_transform)
pdb.set_trace()