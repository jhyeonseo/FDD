# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy, math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable


class Transformer_CA(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.3,
                 activation="relu", normalize_before=False):
        super().__init__()
        CA_layer = TransformerCALayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        CA_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerCA(CA_layer, num_encoder_layers, CA_norm)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src1, src2, key_padding_mask=None, attn_mask=None):
        print(src1.shape)
        src1=src1.permute(1,0,2)
        src2=src2.permute(1,0,2)

        memory,list_attn_maps = self.encoder(src1, src2, src_attn_mask=attn_mask, src_key_padding_mask=key_padding_mask)

        memory=memory.permute(1,0,2)
        return memory, list_attn_maps



class TransformerCA(nn.Module):
    def __init__(self, ca_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(ca_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src1, src2,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        list_attn_maps=[]

        for layer in self.layers:
            output,attn_map = layer(src1, src2, src_attn_mask=src_attn_mask, src_key_padding_mask=src_key_padding_mask)
            list_attn_maps.append(attn_map)
        if self.norm is not None:
            output = self.norm(output)

        return output,list_attn_maps



class TransformerCALayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3, activation="relu", normalize_before=False):
        super().__init__()
        self.batch_first = False
        self.self_attn_src1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first = self.batch_first)
        # Implementation of Feedforward model
        self.linear1_src1 = nn.Linear(d_model, dim_feedforward)
        self.linear2_src1 = nn.Linear(dim_feedforward, d_model)

        self.norm1_src1 = nn.LayerNorm(d_model)
        self.norm2_src1 = nn.LayerNorm(d_model)
        
        self.dropout0_src1 = nn.Dropout(dropout)
        self.dropout1_src1 = nn.Dropout(dropout)
        self.dropout2_src1 = nn.Dropout(dropout)

        self.activation_src1 = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        
        self.self_attn_src2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first = self.batch_first)
        # Implementation of Feedforward model
        self.linear1_src2 = nn.Linear(d_model, dim_feedforward)
        self.linear2_src2 = nn.Linear(dim_feedforward, d_model)

        self.norm1_src2 = nn.LayerNorm(d_model)
        self.norm2_src2 = nn.LayerNorm(d_model)
        
        self.dropout0_src2 = nn.Dropout(dropout)
        self.dropout1_src2 = nn.Dropout(dropout)
        self.dropout2_src2 = nn.Dropout(dropout)

        self.activation_src2 = _get_activation_fn(activation)


    def forward(self,src1, src2,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None):
        
        src1_ma, attn_map1 = self.self_attn_src1(src2, src1, value=src1, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)
        src1 = src1 + self.dropout0_src1(src1_ma)
        src1 = self.norm1_src1(src1)
        src1_ma = self.linear2_src1(self.dropout1_src1(self.activation_src1(self.linear1_src1(src1))))
        src1 = src1 + self.dropout2_src1(src1_ma)
        src1 = self.norm2_src1(src1)
        
        
        src2_ma, attn_map2 = self.self_attn_src2(src1, src2, value=src2, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)
        src2 = src2 + self.dropout0_src2(src2_ma)
        src2 = self.norm1_src2(src2)
        src2_ma = self.linear2_src2(self.dropout1_src2(self.activation_src2(self.linear1_src2(src2))))
        src2 = src2 + self.dropout2_src2(src2_ma)
        src2 = self.norm2_src2(src2)
        
        fus = src1 + src2
        return fus, attn_map1


class Transformer_Encoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.3,
                 activation="relu", normalize_before=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_pos, key_padding_mask=None, attn_mask=None):
        src=src.permute(1,0,2)
        src_pos=src_pos.permute(1,0,2)

        memory,list_attn_maps = self.encoder(src, src_attn_mask=attn_mask, src_key_padding_mask=key_padding_mask, src_pos=src_pos)

        memory=memory.permute(1,0,2)
        return memory, list_attn_maps
        

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None):
        output = src
        list_attn_maps=[]
        
        for layer in self.layers:
            output,attn_map = layer(output, src_attn_mask=src_attn_mask,
                           src_key_padding_mask=src_key_padding_mask, src_pos=src_pos)
            list_attn_maps.append(attn_map)
        if self.norm is not None:
            output = self.norm(output)

        return output,list_attn_maps

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.batch_first = False
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first = self.batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_attn_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, src_pos)
        src2, attn_map = self.self_attn(q, k, value=src, attn_mask=src_attn_mask, key_padding_mask=src_key_padding_mask)

        src = src + self.dropout0(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attn_map


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


#Below is the position encoding
#Attention is all you need
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=5000):#dropout, 
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x_pe = Variable(self.pe[:, :x.size(1)],requires_grad=False)
        x_pe = x_pe.repeat(x.size(0),1,1)

        return x_pe
