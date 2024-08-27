import os
import numpy as np
import json
import random
from itertools import groupby
from PIL import Image
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def shuffle_by_prefix_and_split(lst, random_seed, train=True, split_percentage=0.8):
    grouped = [list(group) for key, group in groupby(lst, key=lambda i: i.split('_')[0])]
    random.seed(random_seed)
    random.shuffle(grouped)
    split_pos = int(len(grouped)*split_percentage)
    if train:
        return [item for group in grouped[:split_pos] for item in group]
    else:
        return [item for group in grouped[split_pos:] for item in group]


def transform(r_size=128,p=0.5):
    transform = A.Compose([
        A.CropAndPad(percent=(-0.2, 0.2), p=p, keep_size=False),
        A.Rotate(limit=10, p=p), 
        A.GridDistortion(border_mode= cv2.BORDER_CONSTANT, p=p),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, border_mode= cv2.BORDER_CONSTANT, p=p),
        A.Affine(p=p),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=p),
        A.Resize(r_size, r_size),
        A.Normalize(mean=0.5, std=0.25), # (x/255.0-mean)/std
        ToTensorV2() # transfer to tensor
        ],is_check_shapes=False)
    return transform


class CAMUS(Dataset):
    def __init__(self, train=True, view="4CH", position="LV", r_size=128, random_seed=0):
        '''
        view: "2CH", "4CH"
        position: "LV", "LVmyo", "LA"
        '''
        assert view in ["2CH", "4CH"] and position in ["LV", "LVmyo", "LA"]
        root_dir = os.path.join("Echodata/CAMUS", view)     
        image_dir = os.path.join(root_dir,"Original")
        mask_dir = os.path.join(root_dir,f"Mask_{position}")
        filenames = os.listdir(image_dir)
        if train is not None:
            if train==True:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=True, split_percentage=0.9)
                self.transform = transform(r_size,p=0.5)
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
            elif train==False:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=False, split_percentage=0.9)
                self.transform = transform(r_size,p=0)    
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
        else:
            self.transform = transform(r_size,p=0)    
            self.image_paths = [os.path.join(image_dir,i) for i in filenames]
            self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('L'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))/255.0
        #image: [H,W]
        if self.transform:
            # [H,W,1]
            transformed = self.transform(image=np.expand_dims(image,axis=-1), mask=mask.astype(np.int32))  

        return transformed['image'],transformed['mask']

class CardiacUDA(Dataset):
    def __init__(self, train=True, view="4CH", position="LV", r_size=128, random_seed=0):
        '''
        view: "4CH"
        position: "LV", "LA", "RV", "RA"
        '''
        assert view in ["4CH"] and position in ["LV", "LA", "RV", "RA"]
        root_dir = os.path.join("Echodata/CardiacUDA", view)     
        image_dir = os.path.join(root_dir,"Original")
        mask_dir = os.path.join(root_dir,f"Mask_{position}")
        filenames = os.listdir(image_dir)
        if train is not None:
            if train==True:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=True, split_percentage=0.9)
                self.transform = transform(r_size,p=0.5)
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
            elif train==False:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=False, split_percentage=0.9)
                self.transform = transform(r_size,p=0)    
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
        else:
            self.transform = transform(r_size,p=0)    
            self.image_paths = [os.path.join(image_dir,i) for i in filenames]
            self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('L'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))/255.0
        #image: [H,W]
        if self.transform:
            # [H,W,1]
            transformed = self.transform(image=np.expand_dims(image,axis=-1), mask=mask.astype(np.int32))  

        return transformed['image'],transformed['mask']
    

class HMC_QU(Dataset):
    def __init__(self, train=True, view="4CH", position="LVmyo", r_size=128, random_seed=0):
        '''
        view: "4CH"
        position: "LVmyo"
        '''
        assert view in ["4CH"] and position in ["LVmyo"]
        root_dir = os.path.join("Echodata/HMC_QU", view)     
        image_dir = os.path.join(root_dir,"Original")
        mask_dir = os.path.join(root_dir,f"Mask_{position}")
        filenames = os.listdir(image_dir)
        if train is not None:
            if train==True:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=True, split_percentage=0.9)
                self.transform = transform(r_size,p=0.5)
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
            elif train==False:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=False, split_percentage=0.9)
                self.transform = transform(r_size,p=0)    
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
        else:
            self.transform = transform(r_size,p=0)    
            self.image_paths = [os.path.join(image_dir,i) for i in filenames]
            self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('L'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))/255.0
        #image: [H,W]
        if self.transform:
            # [H,W,1]
            transformed = self.transform(image=np.expand_dims(image,axis=-1), mask=mask.astype(np.int32))  

        return transformed['image'],transformed['mask']
    

class EchoNet_Dynamic(Dataset):
    def __init__(self, train=True, view="4CH", position="LV", r_size=128, random_seed=0):
        '''
        view: "4CH"
        position: "LV"
        '''
        assert view in ["4CH"] and position in ["LV"]
        root_dir = os.path.join("Echodata/EchoNet_Dynamic", view)     
        image_dir = os.path.join(root_dir,"Original")
        mask_dir = os.path.join(root_dir,f"Mask_{position}")
        filenames = os.listdir(image_dir)
        if train is not None:
            if train==True:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=True, split_percentage=0.9)
                self.transform = transform(r_size,p=0.5)
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
            elif train==False:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=False, split_percentage=0.9)
                self.transform = transform(r_size,p=0)    
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
        else:
            self.transform = transform(r_size,p=0)    
            self.image_paths = [os.path.join(image_dir,i) for i in filenames]
            self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('L'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))/255.0
        #image: [H,W]
        if self.transform:
            # [H,W,1]
            transformed = self.transform(image=np.expand_dims(image,axis=-1), mask=mask.astype(np.int32))
            # transformed = self.transform(image=image, mask=mask.astype(np.int32))  

        return transformed['image'],transformed['mask']


class EchoNet_Pediatric(Dataset):
    def __init__(self, train=True, view="4CH", position="LV", r_size=128, random_seed=0):
        '''
        view: "4CH", "PSAX"
        position: "LV"
        '''
        assert view in ["4CH", "PSAX"] and position in ["LV"]
        root_dir = os.path.join("Echodata/EchoNet_Pediatric",view)  
        image_dir = os.path.join(root_dir,"Original")
        mask_dir = os.path.join(root_dir,f"Mask_{position}")
        filenames = os.listdir(image_dir)
        if train is not None:
            if train==True:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=True, split_percentage=0.9)
                self.transform = transform(r_size,p=0.5)
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
            elif train==False:
                filenames = shuffle_by_prefix_and_split(filenames, random_seed, train=False, split_percentage=0.9)
                self.transform = transform(r_size,p=0)    
                self.image_paths = [os.path.join(image_dir,i) for i in filenames]
                self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]
        else:
            self.transform = transform(r_size,p=0)    
            self.image_paths = [os.path.join(image_dir,i) for i in filenames]
            self.mask_paths = [os.path.join(mask_dir,i) for i in filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('L'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))/255.0
        #image: [H,W]
        if self.transform:
            # [H,W,1]
            transformed = self.transform(image=np.expand_dims(image,axis=-1), mask=mask.astype(np.int32))  

        return transformed['image'],transformed['mask']