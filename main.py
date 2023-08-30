import cv2
import opticalflow
from options import Options
import torch
import torch.nn as nn
import os
import timm
import numpy as np
import types
from PIL import Image
from torchvision import transforms
import math
import torch.nn.functional as F
import networks

options = Options()
opts = options.parse()

'''
transform = transforms.Compose([
        transforms.Resize((192,640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
img1 = Image.open('./samples/kitti/rgb/0000000009.png')
img2 = Image.open('./samples/kitti/rgb/0000000010.png')
img1.save("./original.jpg")
img1 = transform(img1).unsqueeze(0).cuda()
img2 = transform(img2).unsqueeze(0).cuda()


#img1 = cv2.imread('./samples/kitti/rgb/0000000054.png')
#img2 = cv2.imread('./samples/kitti/rgb/0000000055.png')
img1 = cv2.imread('./samples/kitti/rgb/0000000009.png')
img2 = cv2.imread('./samples/kitti/rgb/0000000010.png')
img1 = cv2.imread('./samples/nyu/rgb/rgb_00008.jpg')
img2 = cv2.imread('./samples/nyu/rgb/rgb_00009.jpg')
cv2.imwrite(os.path.join('./', "original.jpg") , img1)
img1 = torch.from_numpy(img1).permute(2, 0, 1).float()  # NumPy 배열을 Torch Tensor로 변환
img1 = img1.unsqueeze(0).to('cuda')
img2 = torch.from_numpy(img2).permute(2, 0, 1).float()  # NumPy 배열을 Torch Tensor로 변환
img2 = img2.unsqueeze(0).to('cuda')

seq = [img1, img2]
RAFT = opticalflow.OpticalFlow(opts)
RAFT(seq)
'''

#'''
img = Image.open('./samples/kitti/rgb/0000000050.png')
#img = Image.open('./samples/nyu/rgb/rgb_00005.jpg')
input_size = (192,640)
transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


RGB = networks.rgb_encoder(pretrained=True, input_size = input_size)
#RGB.load_state_dict(torch.load("rgb.pth"))
cv2.imwrite("./mask.png", RGB.show_mask_on_image(np.array(img)[:, :, ::-1], RGB(transform(img).unsqueeze(0).cuda())))
img.resize((640,192)).save("nomask.png")
torch.save(RGB.state_dict(), "rgb.pth")
#'''



