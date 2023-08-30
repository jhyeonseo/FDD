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


class Encoder(nn.Module):
    def __init__(self, model_type='vit_base_patch16_384', pretrained=True, num_classes=0, input_size=(192,640), patch_size=[16,16], start_index=1):
        super(Encoder, self).__init__()
        
        self.patch_size = patch_size
        self.start_index = start_index
        self.h, self.w = input_size
        self.patch_h = self.h // self.patch_size[1]
        self.patch_w = self.w // self.patch_size[0]
        self.attentions = []
        self.transformer_layer_num = 12  # 24 for large model
        
        # tuning timm model and copy it to class instance
        encoder = timm.create_model(model_type, pretrained=pretrained, num_classes = num_classes).cuda()             # pretrained model 받아옴
        encoder.pos_embed = self._resize_pos_embed(encoder.pos_embed, self.patch_h, self.patch_w)                    # pos_embed의 사이즈를 변경
        
        ##### magnitude는 고정하는것이 최선이라 분석 끝남, orientation은 어떤 식으로 포현해야 하는지 생각해야함 #####
        ##### patch_embed.proj 함수가 channel 5의 input을 받을 수 있도록 변경되어야 함 #####
        ##### pos_embed를 좀 더 창의적으로 넣어주면 더 좋은 성능을 낼 수 있을 것 같음 #####
        
        self.encoder = encoder
        self.fus_attnmap = nn.Conv2d(self.transformer_layer_num, 1, kernel_size=1, stride=1, padding=0, bias=False).cuda() # 여러 layer의 attention map들을 fusion하기 위한 코드
        nn.init.constant_(self.fus_attnmap.weight, 1.0 / self.transformer_layer_num)                                       # 여러 layer의 attention map들을 fusion하기 위한 코드
        self.norm = nn.LayerNorm(self.patch_w * self.patch_h).cuda()                                                       # attention score의 분산을 늘려주기 위한 코드
        self.softmax = nn.Softmax(dim=-1).cuda()                                                                           # attention score의 총합을 1로 만들어 주기 위한 코드

    
        # store attention map in each layer
        for name, module in self.encoder.named_modules():
            if 'attn_drop' in name:
                module.register_forward_hook(self.get_attention)
        
        
    def forward(self, input):
        self.attentions.clear()
        
        x = self.encoder.patch_embed.proj(input).flatten(2).transpose(1, 2)     # 이미지를 패치 단위로 쪼개고 D 차원으로 projection 수행 -> 모든 패치를 1차원으로 flatten 한 후 (B,P,D) shape로 변경
        
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
            
        x = self.encoder.norm(x)
        x = x[:,1:,:].permute(0,2,1)
        
        return x, self.rollout((self.h//self.patch_size[1], self.w//self.patch_size[0]))     # features, attention map
        
        
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
    
    
    def rollout(self, input_shape):
        result = []
        with torch.no_grad():
            for attention in self.attentions:
                #attention_heads_fused = ((attention + attention.permute(0,1,3,2))/2.0).mean(axis=1)
                attention_heads_fused = attention.mean(axis=1)
                result.append(attention_heads_fused)         

        
        result = torch.stack(result)
        result = result[:,:,1:,1:]
        result = result.permute(1, 0, 2, 3)
        
        result = self.fus_attnmap(result).squeeze(dim=1)         # 여러 layer들의 attention map을 fusion
        result = (result + result.permute(0,2,1))/2.0            # 대각 원소들을 참고하여 새로운 score 생성
        result = self.norm(result)                               # attention value에 normalization 적용
        result = self.softmax(result)                            # attention map을 다시 확률화

        result = result.reshape(-1, input_shape[0], input_shape[1], input_shape[0], input_shape[1])   # 1, 12, 40, 12, 40

        return result 
    
    
    def show_mask_on_image(self, img, mask):
        mask = mask[0][0][12].cpu()
        min_value = torch.min(mask)
        max_value = torch.max(mask)
        mask = (mask - min_value) / (max_value - min_value)
        mask = mask.detach().numpy()
        mask = cv2.resize(mask, (self.w, self.h))
        img = cv2.resize(img, (self.w, self.h))
        img = np.float32(img) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)