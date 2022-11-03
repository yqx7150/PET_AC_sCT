from __future__ import print_function, division
import os, random, time

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils

from glob import glob
from PIL import Image as PILImage
import numbers
from imageio import imread
import scipy.io as io
import cv2

import matplotlib.pyplot as plt


def generate_mask(img_height,img_width,radius,center_x,center_y):
 
    y,x=np.ogrid[0:img_height,0:img_width]
 
    # circle mask
 
    mask = (x-center_x)**2+(y-center_y)**2<=radius**2
 
    return mask
    

def create_mask(height_down, height_up, width_left, width_right):
    mask = np.zeros((128, 128))
    mask[height_down:height_up, width_left:width_right] = 1
    return mask

class mriDataset(Dataset):
    def __init__(self, opt,root1,root2,root3): 
    
        self.task = opt.task
        input_2 = np.array([root2 +"/"+ x  for x in os.listdir(root2)])
        target_forward = np.array([root1 +"/"+ x  for x in os.listdir(root1)])
        input_3 = np.array([root3 +"/"+ x  for x in os.listdir(root3)])
        
        assert len(input_2) == len(target_forward) == len(input_3)

        self.data = {'input_2':input_2, 'target_forward':target_forward,'input_3':input_3}
            
    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)

    def __len__(self):
        return len(self.data['target_forward'])

    def __getitem__(self, idx):
        self.data['input_2'].sort()
        self.data['target_forward'].sort()
        self.data['input_3'].sort()
        
        input_2_path = self.data['input_2'][idx]
        target_forward_path = self.data['target_forward'][idx]
        input_3_path = self.data['input_3'][idx]
        

        
        assert (input_2_path.split('/')[-1]) == (target_forward_path.split('/')[-1]) == (input_3_path.split('/')[-1])
                     
        input_2_data = io.loadmat(input_2_path)['data']  # (250, 250)
        target_forward_data = io.loadmat(target_forward_path)['data']  # (512, 512)
        input_3_data = io.loadmat(input_3_path)['data']  # (250, 250)
        
        num_of_ct = int(target_forward_path.split("/")[-1].split(".")[0])
        
        
        mask = generate_mask(128, 128, 64, 64, 64)
        
        if target_forward_data.shape == (512, 512):
        
            target_forward_data = cv2.resize(target_forward_data, (250, 250))
        
        target_forward_data_ori = target_forward_data
        target_forward_data = target_forward_data[61:189, 61:189]
        input_2_data_ori = input_2_data
        input_2_data = input_2_data[61:189, 61:189]
        input_3_data_ori = input_3_data
        input_3_data = input_3_data[61:189, 61:189]
        

        target_forward_data = target_forward_data * mask
        
        input_2_data = input_2_data * mask
        
        input_3_data = input_3_data * mask
        
        
        if num_of_ct>=641 and num_of_ct<=680:
            target_forward_data = target_forward_data_ori
            input_2_data = input_2_data_ori
            input_3_data = input_3_data_ori
            target_forward_data = target_forward_data[61:189, 61:189]
            input_2_data = input_2_data[61:189, 61:189]
            input_3_data = input_3_data[61:189, 61:189]
            
        
        
        h,w = input_2_data.shape

        target_forward_img = np.expand_dims(target_forward_data, 2) 
        target_forward_img = np.concatenate((target_forward_img,target_forward_img,target_forward_img),axis=2)

        
        input_2_img = np.expand_dims(input_2_data, 2) 
        input_3_img = np.expand_dims(input_3_data, 2)
        
        input_img = np.zeros((h,w,3))
                           
        if  self.task== '1to1':                      
            
            input_img[:,:,0] = input_2_img[:,:,0]
            input_img[:,:,1] = input_2_img[:,:,0]
            input_img[:,:,2] = input_2_img[:,:,0]
        else:
            assert 0
            



        input_target_img = input_img.copy()

        input_img = self.np2tensor(input_img).float()
        target_forward_img = self.np2tensor(target_forward_img).float()
        input_target_img = self.np2tensor(input_target_img).float()

        sample = {'input_img':input_img, 'target_forward_img':target_forward_img, 'input_target_img':input_target_img,
                    'input2_name':input_2_path.split("/")[-1].split(".")[0],'input3_name':input_3_path.split("/")[-1].split(".")[0],'target_forward_name':target_forward_path.split("/")[-1].split(".")[0]}
        return sample


