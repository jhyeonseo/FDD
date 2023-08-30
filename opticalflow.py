import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
from core.utils.flow_viz import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.colors as mcolors


class OpticalFlow():
    def __init__(self, args):
        self.args = args
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.flow_model))
        self.model = self.model.module
        self.model.to('cuda')
        self.model.eval()
        self.batch_size = self.args.batch_size
        self.channel = 3
        self.height = self.args.height
        self.width = self.args.width
        self.padder = InputPadder((self.batch_size, self.channel, self.height, self.width), mode='kitti')
        
        self.colorwheel = torch.from_numpy(make_colorwheel())
        self.mean = 0
        self.var = 0
        self.num = 0
        self.max = 0
        
        '''
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).cuda()
        self.sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).cuda()
        self.sobel_y = sobel_y.view(1, 1, 3, 3)
        '''
        
    def __call__(self, seq, iters=20):
        flow = []
        with torch.no_grad():
            for i in range(len(seq)-1):  
                img1 = seq[i] * 255
                img2 = seq[i+1] * 255
                
                img1, img2 = self.padder.pad(img1, img2)
                flow_low, flow_up = self.model(img1, img2, iters=iters, test_mode=True)  # flow map 찾아내기
                flow_up = self.convert_flow(flow_up)                                     # flow map을 mag, normalized flow map으로 변환
                flow.append(flow_up)
                
        return flow
    
        
    def create_flow(self, seq, iters=20):   # pytorch tensor image sequence
        flow = []
        with torch.no_grad():
            for i in range(len(seq)-1):  
                img1 = seq[i] * 255
                img2 = seq[i+1] * 255
                img1, img2 = self.padder.pad(img1, img2)
                flow_low, flow_up = self.model(img1, img2, iters=iters, test_mode=True)
                flow.append(flow_up)
                
        return flow
        
               
    def save_flow(self, rgb_path, save_path, option='black'):
        seq = []
        # 디렉토리 내의 파일들을 순회하면서 이미지 파일을 찾아 리스트에 추가
        for filename in sorted(os.listdir(rgb_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(rgb_path, filename)
                img = cv2.imread(img_path)
                img = torch.from_numpy(img).permute(2, 0, 1).float()  # NumPy 배열을 Torch Tensor로 변환
                img = img.unsqueeze(0).to('cuda')
                seq.append(img)
                
        flow = self.create_flow(seq)
        # 이미지들을 저장할 디렉토리 내에 순서대로 저장
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        for i, flo in enumerate(flow):
            filename = f"flow_{i}.jpg" 
            
            if option == 'black':
                cv2.imwrite(os.path.join(save_path, filename) , self.viz_withflow(flo))
            else:
                cv2.imwrite(os.path.join(save_path, filename) , self.viz_withflow(flo,seq[i]))

    
    def viz_withrgb(self, img1, img2, iters=20, option='rgb'):
        with torch.no_grad():
            img1 = img1 * 255
            img2 = img2 * 255
            img1, img2 = self.padder.pad(img1, img2)
            flow_low, flow_up = self.model(img1, img2, iters=iters, test_mode=True)
        
        img1 = img1[0].permute(1,2,0).cpu().numpy()
        flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
        
        height, width = img1.shape[:2]
        # 이미지와 flow map를 함께 그리기 위한 캔버스 생성
        if option == 'black':
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            canvas = cv2.cvtColor(img1/255, cv2.COLOR_RGB2BGR)
        
        # 화살표 그리기
        for y in range(0, height, int(height//16)):
            for x in range(0, width, int(width//48)):
                dx, dy = flow_up[y, x]
                cv2.arrowedLine(canvas, (x, y), (round(x + dx), round(y + dy)), (0, 255, 0), 1)

        return canvas
    
    
    def viz_withflow(self, flow, img=None):
        flow = flow[0].permute(1,2,0).cpu().numpy()
        
        height, width = flow.shape[:2]
        if img == None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            img = img[0].permute(1,2,0).cpu().numpy()
            canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            
        for y in range(0, height, int(height//32)):
            for x in range(0, width, int(width//96)):
                dx, dy = flow[y, x]
                cv2.arrowedLine(canvas, (x, y), (round(x + dx), round(y + dy)), (0, 255, 0), 1)
                
        return canvas
        
    
    def convert_flow(self, flow, max_flow=669):
        flow_x, flow_y = flow[:, 0, :, :], flow[:, 1, :, :]
        if self.max < torch.max(flow):
            self.max = torch.max(flow)
        flow_x = flow_x / max_flow
        flow_y = flow_y / max_flow
        
        return self.flow_uv_to_colors(flow_x, flow_y, flow.size(0))
        
        
    def flow_uv_to_colors(self, u, v, B):
        """
        Applies the flow color wheel to (possibly clipped) flow components u and v.

        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun

        Args:
            u (torch.Tensor): Input horizontal flow of shape [B, H, W]
            v (torch.Tensor): Input vertical flow of shape [B, H, W]
            convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

        Returns:
           torch.Tensor: Flow visualization image of shape [B, H, W, 3]
        """
        H = self.height
        W = self.width
        flow_image = torch.zeros((B, 3, H, W), dtype=torch.float32).cuda()
        ncols = self.colorwheel.shape[0]
        rad = torch.sqrt(torch.square(u) + torch.square(v))
        
        a = torch.atan2(-v, -u) / torch.tensor(np.pi)
        fk = (a + 1) / 2 * (ncols - 1)
        k0 = fk.floor().long()
        k1 = k0 + 1
        k1[k1 == ncols] = 0
        f = fk - k0
        for i in range(self.colorwheel.shape[1]):
            tmp = self.colorwheel[:, i].cuda()
            col0 = tmp[k0] / 255.0
            col1 = tmp[k1] / 255.0
            col = (1 - f) * col0 + f * col1
            idx = (rad <= 1)
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            col[~idx] = col[~idx] * 0.75   # out of range
            # Note the 2-i => BGR instead of RGB
            #ch_idx = 2 - i if convert_to_bgr else i
            ch_idx = i

            flow_image[:, ch_idx, :, :] = col
            
        self.mean += torch.mean(flow_image)
        self.var += torch.var(flow_image)
        self.num += 1
        
            
        return flow_image
    
        
'''   
    def convert_flow(self, flow, max_flow=64):
            flow_x, flow_y = flow[:, 0, :, :], flow[:, 1, :, :]
            magnitude = torch.sqrt(flow_x ** 2 + flow_y ** 2)
            angle = torch.atan2(-flow_y, -flow_x)

            # Magnitude를 0에서 1 사이로 정규화
            #magnitude = torch.clamp(magnitude / max_flow, 1e-7, 1)
            magnitude = torch.clamp(magnitude / magnitude.max(), 1e-7, 1)

            # Angle을 [-π, π] 범위로 조정
            angle = (angle * (180.0 / torch.tensor(np.pi)) + 180.0) / 360


            # Hue 채널로 변환 (0에서 1 사이의 값)
            hue = angle

            # Saturation 채널로 변환 (1로 고정)
            saturation = torch.ones_like(hue)

            # Value 채널로 변환 (정규화된 magnitude 값)
            value = magnitude

            # HSV 컬러맵으로 변환 (Hue, Saturation, Value)
            hsv = torch.stack((hue, saturation, value), dim=1)

            # HSV에서 RGB로 변환
            hsv_np = hsv.permute(0, 2, 3, 1).cpu().numpy()
            rgb_np = np.clip(mcolors.hsv_to_rgb(hsv_np), 0, 1)
            rgb = torch.tensor(rgb_np).permute(0, 3, 1, 2).to(hsv.device)
            
            self.mean += torch.mean(rgb)
            self.var += torch.var(rgb)
            self.num += 1
            
            return rgb
#'''

''' Magnitude and Normal vectors
    def convert_flow(self, optical_flow):
        flow_x, flow_y = optical_flow[:, 0, :, :], optical_flow[:, 1, :, :]

        magnitude = torch.sqrt(flow_x**2 + flow_y**2)
        magnitude = torch.clamp(magnitude, min=1e-7)
        
        flow_x /= magnitude
        flow_y /= magnitude
    
        min = magnitude.view(magnitude.size(0),-1).min(dim=1)[0][:, None, None]
        max = magnitude.view(magnitude.size(0),-1).max(dim=1)[0][:, None, None]
        magnitude = (magnitude - min) / (max - min)                         # 0과 1사이의 범위에 오도록 정규화
        print(max)
                        
        
        optical_flow = torch.stack([magnitude, flow_x, flow_y], dim=1)

        return optical_flow
'''


''' Divergence and Curl
    def convert_flow(self, optical_flow):
        Vx = optical_flow[:, 0, :, :].unsqueeze(1)
        Vy = optical_flow[:, 1, :, :].unsqueeze(1)

        Vx_dx = self.x_derivative(Vx)
        Vy_dx = self.x_derivative(Vy)
        Vx_dy = self.y_derivative(Vx)
        Vy_dy = self.y_derivative(Vx)
        
        divergence = (Vx_dx + Vy_dy).squeeze(1)
        curl = (Vy_dx - Vx_dy).squeeze(1)
          
        min = divergence.view(divergence.size(0),-1).min(dim=1)[0][:, None, None]
        max = divergence.view(divergence.size(0),-1).max(dim=1)[0][:, None, None]
        divergence = (divergence - min) / (max - min)                       
    
        min = curl.view(curl.size(0),-1).min(dim=1)[0][:, None, None]
        max = curl.view(curl.size(0),-1).max(dim=1)[0][:, None, None]
        curl = (curl - min) / (max - min)                     
        # 0과 1사이의 범위에 오도록 정규화
        
        optical_flow = torch.stack([divergence, curl], dim=1)
        
        return optical_flow   


    def x_derivative(self, tensor):
        return torch.nn.functional.conv2d(tensor, self.sobel_x, padding=1)


    def y_derivative(self, tensor):
        return torch.nn.functional.conv2d(tensor, self.sobel_y, padding=1)
'''