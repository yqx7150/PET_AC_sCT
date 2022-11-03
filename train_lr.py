
import numpy as np
import os, time, random
import argparse
import json
import cv2

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torchvision
import torch.nn as nn

from model.model import InvISPNet
from dataset.mri_dataset import mriDataset
from config.config import get_arguments

from tensorboardX import SummaryWriter
from skimage.measure import compare_psnr
import scipy.io as sio
import matplotlib.pyplot as plt
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.system('rm tmp')

parser = get_arguments()
parser.add_argument("--out_path", type=str, default="./results/", help="Path to save checkpoint. ")
parser.add_argument("--root1", type=str, default="./data/T2", help="Output images. ")
parser.add_argument("--root2", type=str, default="./data/T1", help="Input images. ")
parser.add_argument("--root3", type=str, default="./data/PD", help="Another input images. ")
parser.add_argument("--resume", dest='resume', action='store_true',  help="Resume training. ")
parser.add_argument("--loss", type=str, default="L2", choices=["L1", "L2"], help="Choose which loss function to use. ")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))

os.makedirs(args.out_path, exist_ok=True)
os.makedirs(args.out_path+"%s"%args.task, exist_ok=True)
os.makedirs(args.out_path+"%s/checkpoint"%args.task, exist_ok=True)

with open(args.out_path+"%s/commandline_args.yaml"%args.task , 'w') as f:
    json.dump(args.__dict__, f, indent=2)

def save_img(img, img_path):
    img = np.clip(img*255,0,255)
    cv2.imwrite(img_path, img)


def main(args):
    # ======================================define the model======================================
    writer = SummaryWriter(args.out_path+"%s"%args.task)
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    net.cuda()
    # load the pretrained weight if there exists one
    if args.resume:
        net.load_state_dict(torch.load(args.out_path+"%s/checkpoint/latest.pth"%args.task))
        print("[INFO] loaded " + args.out_path+"%s/checkpoint/latest.pth"%args.task)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250], gamma=0.5)

    print("[INFO] Start data loading and preprocessing")
    Dataset = mriDataset(opt=args,root1=args.root1,root2=args.root2,root3=args.root3)
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    print("[INFO] Start to train")
    step = 0
    loss_all = np.zeros((300), dtype='float')
    
    for epoch in range(0, 300):
        epoch_time = time.time()             
        PSNR = []
        loss_this_time = 0
        for i_batch, sample_batched in enumerate(dataloader):
            step_time = time.time()

            input, target_forward, input_target = sample_batched['input_img'].cuda(), sample_batched['target_forward_img'].cuda(), \
                                        sample_batched['input_target_img'].cuda()
            file_name = sample_batched['target_forward_name'][0]
            
            

            reconstruct_for = net(input) 
            reconstruct_for = torch.clamp(reconstruct_for, 0, 1)  

            #reconstruct_for_m = ((reconstruct_for[:,0,:,:]+reconstruct_for[:,1,:,:])/2).squeeze()
            
            

            #forward_loss = F.mse_loss(reconstruct_for_m, target_forward.squeeze())    # 
            forward_loss = F.l1_loss(reconstruct_for, target_forward)
            
            writer.add_scalar('forward_loss',forward_loss.item(),global_step=step)

            reconstruct_rev = net(reconstruct_for, rev=True)

            reconstruct_rev = torch.clamp(reconstruct_rev, 0, 1)  

            #rev_loss = F.mse_loss(reconstruct_rev, input_target)
            rev_loss = F.l1_loss(reconstruct_rev, input_target)
            
            writer.add_scalar('rev_loss',rev_loss.item(),global_step=step)
            
            
            
            loss =  args.weight * forward_loss + rev_loss
            writer.add_scalar('loss',loss.item(),global_step=step)
            print('epoch: ' + str(epoch) + ' iter: ' + str(i_batch) +' loss: ' + str(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_this_time = loss_this_time + loss
            
            step += 1
            
        loss_this_time = loss_this_time / 1440.0
        loss_all[epoch] = loss_this_time
        
        torch.save(net.state_dict(), args.out_path+"%s/checkpoint/latest.pth"%args.task)
        if epoch % 1 == 0:
            # os.makedirs(args.out_path+"%s/checkpoint/%04d"%(args.task,epoch), exist_ok=True)
            torch.save(net.state_dict(), args.out_path+"%s/checkpoint/%04d.pth"%(args.task,epoch))
            print("[INFO] Successfully saved "+args.out_path+"%s/checkpoint/%04d.pth"%(args.task,epoch))
            
        if epoch % 10 == 0:    
            print("task: %s Epoch: %d Step: %d || loss: %.5f rev_loss: %.10f forward_loss: %.5f  || lr: %f time: %f"%(
                args.task, epoch, step, loss.detach().cpu().numpy(), rev_loss.detach().cpu().numpy(),
                forward_loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time()-step_time
            ))

        scheduler.step()   
        
        print("[INFO] Epoch time: ", time.time()-epoch_time, "task: ", args.task)    

if __name__ == '__main__':

    torch.set_num_threads(4)
    main(args)
