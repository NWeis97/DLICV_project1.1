# from torchvision import datasets, transforms
import torch
import os
import glob
import random
from random import sample
import numpy as np
import PIL.Image as Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02514/hotdog_nothotdog', pics=None):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        if pics is not None:
            self.image_paths = pics
        else:
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

def get_data(data_path, transform=None):
    random.seed(30)

    input_size = 224
    batch_size = 12
    N_val = 200

    # All train pics
    train_pics = glob.glob('/dtu/datasets1/02514/hotdog_nothotdog/train/*/*.jpg')
    # Sample N_val validation pics
    val_pics = sample(train_pics, N_val)
    # Train on the other pics
    train_pics = np.setdiff1d(train_pics, val_pics)

    # Apply data augmentation to trainset only
    train_transform = transforms.Compose([transforms.Resize((input_size, input_size)), 
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(45),
                                      transforms.ColorJitter(),
                                      transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                     transforms.ToTensor()])

    train_data = Hotdog_NotHotdog(train=True, transform=train_transform, pics=train_pics)
    val_data = Hotdog_NotHotdog(train=False, transform=test_transform, pics=val_pics)
    test_data = Hotdog_NotHotdog(train=False, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, train_data, val_data, test_data



if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    # main()
    data_path = '/dtu/datasets1/02514/hotdog_nothotdog'
    train_loader, test_loader, trainset, testset = get_data(data_path)
