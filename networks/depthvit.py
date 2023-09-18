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


class DEPTHVIT(nn.Module):
    def __init__(self, model_type='vit_base_patch16_384', pretrained=True, num_classes=0, input_size=(192,640), patch_size=[16,16], start_index=1, num_input_channels=6):  # 16patch로 실험 
        super(DEPTHVIT, self).__init__()
        
        self.patch_size = patch_size
        self.start_index = start_index
        self.h, self.w = input_size
        self.patch_h = self.h // self.patch_size[1]
        self.patch_w = self.w // self.patch_size[0]
        self.attentions = []
        self.masked_attentions = []
        self.transformer_layer_num = 12  # 24 for large model
        self.dropout_prob = 0.75
        self.num_input_channels = num_input_channels - 3                      
        
        # tuning timm model and copy it to class instance
        encoder = timm.create_model(model_type, pretrained=pretrained, num_classes = num_classes)                   # pretrained model 받아옴
        encoder.pos_embed = self._resize_pos_embed(encoder.pos_embed, self.patch_h, self.patch_w)                   # pos_embed의 사이즈를 변경
        
        param_shape = (encoder.patch_embed.proj.out_channels, self.num_input_channels, patch_size[1], patch_size[0])
        new_parameter = torch.nn.Parameter(torch.randn(*param_shape))
        weight = nn.Parameter(torch.cat([encoder.patch_embed.proj.weight, new_parameter],dim=1))
        encoder.patch_embed.proj = nn.Conv2d(6, encoder.patch_embed.proj.out_channels, (patch_size[1], patch_size[0]), stride=(patch_size[1], patch_size[0]))
        encoder.patch_embed.proj.weight = weight
        
        self.dim = 768                                                # small model 384

        self.encoder = encoder
        self.to_kv = nn.Linear(self.dim, self.dim * 2, bias = False)
        nn.init.xavier_uniform_(self.to_kv.weight)
        self.to_q = nn.Linear(self.dim, self.dim, bias = False)
        nn.init.xavier_uniform_(self.to_q.weight)
        
        self.proj = nn.Linear(self.dim, self.dim)
        # Use attention rollout
        '''
        for name, module in self.encoder.named_modules():
            if 'attn_drop' in name:
                module.register_forward_hook(self.get_attention)
        '''
        
    def forward(self, input):
        self.attentions.clear()
        self.masked_attentions.clear()
        #print(input.shape)                                                     # input shape = []
        if self.training:
            self.drop = torch.rand(1) < self.dropout_prob
        else:
            self.drop = False

        x = self.encoder.patch_embed.proj(input).flatten(2).transpose(1, 2)     # 이미지를 패치 단위로 쪼개고 D 차원으로 projection 수행 -> 모든 패치를 1차원으로 flatten 한 후 (B,P,D) shape로 변경 # [1, 1920, 384]
        
        #print(x.shape)
        
        if getattr(self.encoder, "dist_token", None) is not None:
            cls_tokens = self.encoder.cls_token.expand(x.shape[0], -1, -1)      # stole cls_tokens impl from Phil Wang, thanks
            dist_token = self.encoder.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.encoder.cls_token.expand(x.shape[0], -1, -1)      # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.encoder.pos_embed                                          # positional embedding 추가
        x = self.encoder.pos_drop(x)
        
        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.norm(x)                                               # [1, 1921, 384]

        return self.make_depth(x)
        

    def make_depth(self, input):
        #print(input.shape)                                          # input shape [1, 1921, 384]
        q = input[:,0,:].unsqueeze(1)                                # cls_token 으로 쿼리 
        x = input[:,1:,:]
        B,N,C = x.shape
        kv = self.to_kv(x).reshape(B, N, 2, C).permute(2, 0, 1, 3)  # 2, B, N, C
        k, v = kv.unbind(0)                                         # B, N, C
        q = self.to_q(q)                                            # B, 1, C
        q = q * 0.05103                                             # /sqrt(384)
        
        num_indices = C // 8                                        # 96개의 패치
        attn = q @ k.transpose(-2, -1)                              # B, 1, N
        random_indices = torch.randperm(N)
        row_indices = torch.arange(0, N-1)                          # partition을 가로 기준으로 
        chunks = torch.chunk(random_indices, num_indices)           # N개의 수를 C//8개의 chunk로 나눔 row, random 중에
        row_chunks = torch.chunk(row_indices, 80)
        masked = torch.zeros(attn.shape).cuda()
        
        if self.drop:                                                              # training시에만 partition
            for idx in chunks:
                part = attn[:,:,idx].clone()
                part = part.softmax(dim=-1)                                        # B, 1, num_indices
                median = torch.median(part, dim=-1)[0].unsqueeze(-1).cuda()        # B, 1, 1   # softmax값들의 median  median말고 더 낮은 수 고려      
                mask_index = (part < median).cuda()                                # B, 1, num_indices part > median -> part < median 고려
                attn[:,:,idx] = part
                mask = part.clone()
                mask[mask_index] *= 0
                masked[:,:,idx] = mask
        '''else:
            attn = attn.softmax(dim=-1)'''
            
        masked /= (num_indices/2)    ## ??
        
        #training시에만 mask 적용
        if self.drop:
            x = masked @ v
        else:
            x = attn @ v
            
        x = x.transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.encoder.norm(x)
        
        print(x.shape)                                                           # depth decoder용 
        
        return x, (attn.reshape(B,self.h//self.patch_size[1], self.w//self.patch_size[0]), masked.reshape(B,self.h//self.patch_size[1], self.w//self.patch_size[0]))  #pose_inputs, attn_map, masked_attn_map
        
        
    def _resize_pos_embed(self, pos_emb, gs_h, gs_w):
        posemb_tok, posemb_grid = (pos_emb[:, : self.start_index], pos_emb[0, self.start_index :],)
        gs_old = int(math.sqrt(len(posemb_grid)))                                                          # 기존 이미지 크기에서 나타나는 패치의 총 개수
        
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)                       # 기존 pos_embed를 2차원화
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=True)   # 기존 pos_embed를 bilinear interpolation을 통해 새로운 크기로 변경
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)                          # 기존 pos_embed를 1차원화
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)                                               # 기존 pos_embed를 사용해 interpolation을 거친 pos_embed를 생성, 학습을 위해 requires_grad_ 추가
        
        return nn.Parameter(posemb)
    
        
    def get_attention(self, module, input, output):
        self.attentions.append(output)
        new = output.clone().cuda()
        copy = output[:, :, 0 , 1 :].clone().cuda().detach()
        median = torch.median(copy, dim=-1)[0].unsqueeze(-1).cuda()
        copy[(copy > median)] = 0
        copy[(copy > 0)] = 1
        new[:, :, 0, 1 :] *= copy
        self.masked_attentions.append(new)
            
        if self.drop:
            return new
        else:
            return output