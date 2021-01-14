
"""
Created on Wed Oct 21 21:08:41 2020

@author: xuegeeker
@email: xuegeeker@163.com
"""

import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, Sequential, Conv2d, Conv3d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class PAM_Module(Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        
    def forward(self, x):
        x = x.squeeze(-1)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = (self.gamma*out + x).unsqueeze(-1)
        return out


class CAM_Module(Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        
    def forward(self, x):
        m_batchsize, C, height, width, channle = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        out = self.gamma*out + x 
        return out


class HResNetAM(nn.Module):
    def __init__(self, band, classes):
        super(HResNetAM, self).__init__()
        self.name = 'HResNetAM'
        

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, 5), stride=(1, 1, 2))
        self.batch_norm11 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv122 = nn.Conv3d(in_channels=6, out_channels=6, padding=(0, 0, 2), kernel_size=(1, 1, 5), stride=(1, 1, 1))
        self.batch_norm122 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        self.conv123 = nn.Conv3d(in_channels=12, out_channels=6, padding=(0, 0, 2), kernel_size=(1, 1, 5), stride=(1, 1, 1))
        self.batch_norm123 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        self.conv124 = nn.Conv3d(in_channels=12, out_channels=6, padding=(0, 0, 2), kernel_size=(1, 1, 5), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())
        kernel_3d = math.floor((band - 4) / 2)
        self.conv13 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, band), stride=(1, 1, 1))
        self.batch_norm21 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())
        
        self.conv222 = nn.Conv3d(in_channels=6, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm222 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        
        self.conv223 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm223 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        
        self.conv224 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())
        
        self.attention_spectral = CAM_Module(24)
        self.attention_spatial = PAM_Module(24)
        
        self.batch_norm_spectral = nn.Sequential(nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), mish(), nn.Dropout(p=0.5))
        self.batch_norm_spatial = nn.Sequential(nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), mish(), nn.Dropout(p=0.5))
        
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(nn.Linear(48, classes))
        
    def forward(self, X):
        
        X11 = self.conv11(X)
        X11 = self.batch_norm11(X11)

        XS1 = torch.chunk(X11,4,dim=1)
        X121 = XS1[0]
        X122 = self.conv122(XS1[1])
        X122 = self.batch_norm122(X122)
        X123 = torch.cat((X122, XS1[2]), dim=1)
        X123 = self.conv123(X123)
        X123 = self.batch_norm123(X123)
        X124 = torch.cat((X123, XS1[3]), dim=1)
        X124 = self.conv124(X124)
        X12 = torch.cat((X121, X122, X123, X124), dim=1)
        X12 = self.batch_norm12(X12)
        X13 = self.conv13(X12)
        
        X1 = self.attention_spectral(X13)
        X1 = torch.mul(X1, X13)
        

        X21 = self.conv21(X)
        X21 = self.batch_norm21(X21)

        XS2 = torch.chunk(X21,4,dim=1)
        X221 = XS2[0]
        X222 = self.conv222(XS2[1])
        X222 = self.batch_norm222(X222)
        
        X223 = torch.cat((X222, XS2[2]), dim=1)
        X223 = self.conv223(X223)
        X223 = self.batch_norm223(X223)
        
        X224 = torch.cat((X223, XS2[3]), dim=1)
        X224 = self.conv224(X224)
        X22 = torch.cat((X221, X222, X223, X224), dim=1)
        X22 = self.batch_norm22(X22)
        
        X2 = self.attention_spatial(X22)
        X2 = torch.mul(X2, X22)
        
        X1 = self.batch_norm_spectral(X1)
        X1 = self.global_pooling(X1)
        X1 = X1.squeeze(-1).squeeze(-1).squeeze(-1)
        X2 = self.batch_norm_spatial(X2)
        X2= self.global_pooling(X2)
        X2 = X2.squeeze(-1).squeeze(-1).squeeze(-1)

        X_pre = torch.cat((X1, X2), dim=1)
        
        output = self.full_connection(X_pre)
        return output














