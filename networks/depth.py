import cv2
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
from collections import OrderedDict
from layers import *


'''
class DepthDecoder2(nn.Module):
    def __init__(self):
        super(DepthDecoder2, self).__init__()
       
        #self.conv1 = ConvBlock(768, 384)
        #self.conv2 = ConvBlock(384, 192)
        #self.conv3 = ConvBlock(192, 96)
        #self.conv4 = ConvBlock(96, 48)
        #self.conv5 = Conv3x3(48, 1)
    
        self.conv1 = ConvBlock(512, 256)
        self.conv2 = ConvBlock(256, 128)
        self.conv3 = ConvBlock(128, 64)
        self.conv4 = ConvBlock(64, 32)
        self.conv5 = Conv3x3(32, 1)
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        self.outputs = {}
        x = x[3]
        x = upsample(x)
        #x = x.reshape(-1, x.size(1), 12, 40)
        
        x = self.conv1(x)
        x = upsample(x)
        
        x = self.conv2(x)
        x = upsample(x)
        
        x = self.conv3(x)
        x = upsample(x)

        x = self.conv4(x)
        x = upsample(x)
        
        x = self.conv5(x)
        x = self.sigmoid(x)
            
        self.outputs[("disp",0)] = x


        return self.outputs
'''
class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec, num_layers=5, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec
        self.num_layers = num_layers - 1

        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_layers, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_layers else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, frame_id=0):
        self.outputs = {}
        x = input_features[-1]

        for i in range(self.num_layers, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", frame_id, i)] = self.sigmoid(self.convs[("dispconv", i)](x))


        return self.outputs


