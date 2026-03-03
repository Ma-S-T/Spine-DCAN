import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch.nn as nn
import torch

from utils.SpatialContrastNorm import SpatialContrastNorm


from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmcv.cnn.bricks import DropPath

""" 
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,trunc_normal_init)
"""


# %%

class DWConv(nn.Module):
    def __init__(self, hidden_features):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(hidden_features,hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
    def forward(self, x):
        x = self.dwconv(x)
        return x
class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        in_channels=in_features
        out_channels = out_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.res_path = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        shorcut=self.res_path(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        out = shorcut + x
        return out

class AttentionModule(BaseModule):
    """ MSCA模块
    """
    def __init__(self, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv0 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding,groups=out_channels)
        self.conv0_1 = nn.Conv2d(out_channels,out_channels,kernel_size=(1, 7), padding=(0, 3), groups=out_channels)
        self.conv0_2 = nn.Conv2d(out_channels,out_channels,kernel_size= (7, 1), padding=(3, 0), groups=out_channels)

        self.conv1_1 = nn.Conv2d(out_channels,out_channels, kernel_size=(1, 11), padding=(0, 5), groups=out_channels)
        self.conv1_2 = nn.Conv2d(out_channels,out_channels, kernel_size=(11, 1), padding=(5, 0), groups=out_channels)

        self.conv2_1 = nn.Conv2d(out_channels,out_channels,kernel_size=(1, 21), padding=(0, 10), groups=out_channels)
        self.conv2_2 = nn.Conv2d(out_channels,out_channels,kernel_size=(21, 1), padding=(10, 0), groups=out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels,kernel_size=1)

    def forward(self, x):
        u = self.conv0(x)
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)
        out = attn * u
        return out


class SpatialAttention(BaseModule):
    """ Attention模块
    """
    def __init__(self, in_channels,out_channels,kernel_size, stride, padding):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels = out_channels

        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.proj_1 = nn.Conv2d(in_channels,out_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(out_channels, kernel_size, stride, padding)
        self.proj_2 = nn.Conv2d(out_channels,in_channels, kernel_size=1)

    def forward(self, x):
        y = self.conv0(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        out = x + y
        return out


class AttentionBlock(BaseModule):
    """ MSCAN模块
    """
    def __init__(self,in_channels,out_channels,kernel_size, stride, padding,mlp_ratio,drop,drop_path,act_layer):
        super().__init__()

        self.norm1 = nn.GroupNorm(1, out_channels, affine=False)
        self.attn = SpatialAttention(in_channels,out_channels,kernel_size, stride, padding)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.GroupNorm(1, out_channels, affine=False)
        mlp_hidden_dim = int(out_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim,out_features=out_channels,act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        # x.shape (B,C,H,W), C=out_channels
        x = self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)* self.attn(self.norm1(x)))
        x = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)* self.mlp(self.norm2(x)))
        return x
        # (B,out_channels,H_new,W_new)

# %%
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        hid_dim = 2 * max([in_channels, out_channels])
        self.conv = nn.Sequential(nn.Conv2d(in_channels, hid_dim, kernel_size, stride, padding),
                                  nn.GroupNorm(1, hid_dim, affine=False),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(hid_dim, hid_dim, kernel_size=1, stride=1, padding=0),
                                  nn.GroupNorm(1, hid_dim, affine=False),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(hid_dim, out_channels, kernel_size=3, stride=1, padding=1))
        self.res_path = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # x.shape (B,C,H,W), C=in_channels
        y = self.res_path(x) + self.conv(x)
        return y
        # (B,out_channels,H_new,W_new)


# %%
class Block(nn.Module):
    def __init__(self, H_in, W_in, C_in,C_out,
                 kernel_size, stride, padding,  # conv parameters
                 mlp_ratio,drop,drop_path,act_layer  # attn parameters
                 ):
        super().__init__()
        if C_in == 1:
            self.norm0 = SpatialContrastNorm(kernel_size=(H_in // 8, W_in // 8))
        else:
            self.norm0 = nn.Identity()
        self.cnn_block = CNNBlock(C_in, C_out, kernel_size, stride, padding)
        self.attn_block = AttentionBlock(C_in, C_out,kernel_size, stride, padding,mlp_ratio,drop,drop_path,act_layer)
        self.out = nn.Sequential(nn.GroupNorm(1, C_out, affine=False),nn.LeakyReLU(inplace=True))

    def forward(self, x,a1=1, a2=1):
        x = self.norm0(x)
        y1 = 0
        if a1 != 0:
            y1 = self.cnn_block(x)
        y2 = 0
        if a2 != 0:
            y2 = self.attn_block(x)
        y = a1 * y1 + a2 * y2
        y = self.out(y)
        return y


# %%
class MergeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, scale_factor):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.norm1 = nn.GroupNorm(1, in_channels, affine=False)
        self.norm2 = nn.GroupNorm(1, skip_channels, affine=False)
        self.proj1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.proj2 = nn.Conv2d(skip_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.out = nn.Sequential(nn.GroupNorm(1, out_channels, affine=False),
                                 nn.LeakyReLU(inplace=True))

    def forward(self, skip, x, a1=1, a2=1):
        # x.shape (B, C1, H/4, W/4), C1=in_channels
        # skip.shape (B, C2, H, W), C2=skip_channels
        x = self.up(x)  # (B, C1, H, W)
        x = self.norm1(x)
        skip = self.norm2(skip)
        x = self.proj1(x)
        skip = self.proj2(skip)
        y = a1 * x + a2 * skip
        y = self.out(y)
        # y.shape (B,C,H,W), C=out_channels
        return y


# %%
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling):
        super().__init__()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.conv = nn.Sequential(nn.GroupNorm(1, in_channels, affine=False),
                                  nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=5, stride=1, padding=2),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        y = self.conv(self.upsampling(x))
        return y
    # %%


class SpineDcan(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self.use_cnn = args.use_cnn
        self.use_attn = args.use_attn
        self.T0a = Block(H_in=512, W_in=512, C_in=1, C_out=32,
                         kernel_size=5, stride=1, padding=2,#new_h=h,new_w=w,layer_name
                         mlp_ratio=4.,drop=0.,drop_path=0.,act_layer=nn.GELU)
        #H_out=512, W_out=512,
        self.T1a = Block(H_in=512, W_in=512, C_in=32,C_out=128,
                         kernel_size=5, stride=4, padding=2,#new_h=1/4h,new_w=1/4w
                         mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU)
        # H_out=128, W_out=128,
        self.T2a = Block(H_in=128, W_in=128, C_in=128,C_out=512,
                         kernel_size=5, stride=4, padding=2,#new_h=1/4h,new_w=1/4w
                         mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU)
        # H_out=32, W_out=32,
        self.T3a = Block(H_in=32, W_in=32, C_in=512,C_out=512,
                         kernel_size=3, stride=1, padding=1,#new_h=h,new_w=w
                         mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU)
        # H_out=32, W_out=32,
        self.T3b = nn.Identity()
        # nn.Conv2d(in_channels=1024, out_channels=512,
        #          kernel_size=1, stride=1, padding=0)

        self.M2 = MergeLayer(in_channels=512, out_channels=512,
                             skip_channels=512, scale_factor=1)

        self.T2b = Block(H_in=32, W_in=32, C_in=512, C_out=128,
                         kernel_size=3, stride=1, padding=1,#new_h=h,new_w=w
                         mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU)
        # H_out=32, W_out=32,
        self.M1 = MergeLayer(in_channels=128, out_channels=128,
                             skip_channels=128, scale_factor=4)

        self.T1b = Block(H_in=128, W_in=128, C_in=128,C_out=32,
                         kernel_size=5, stride=1, padding=2,#new_h=h,new_w=w
                         mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU)
        # H_out=128, W_out=128,
        self.M0 = MergeLayer(in_channels=32, out_channels=32,
                             skip_channels=32, scale_factor=4)

        self.T0b = Block(H_in=512, W_in=512, C_in=32,C_out=32,
                         kernel_size=5, stride=1, padding=2,#new_h=h,new_w=w
                         mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU)
        # H_out=512, W_out=512,
        self.seghead = SegmentationHead(32, self.num_classes, upsampling=1)

    def forward(self, x, return_attn_weight=False):

        a1 = 1
        a2 = 1
        if self.use_cnn != 1:
            a1 = 0
        if self.use_attn != 1:
            a2 = 0
            # print('x', x.shape)
        t0a = self.T0a(x, a1=a1, a2=a2)
        # print("t0a.shape", t0a.shape)

        t1a = self.T1a(t0a, a1=a1, a2=a2)
        # print("t1a.shape", t1a.shape)
        t2a = self.T2a(t1a, a1=a1, a2=a2)
        # print("t2a.shape", t2a.shape)
        t3a = self.T3a(t2a, a1=a1, a2=a2)
        # print("t3a.shape", t3a.shape)
        # t3a (B,C,H,W)
        # B, C, H, W = t3a.shape
        # t3a=t3a.permute(0,2,3,1).view(B,H*W,C).reshape(B,H*W,32,32)
        t3b = self.T3b(t3a)
        # print("t3b.shape", t3b.shape)
        m2 = self.M2(t2a, t3b)
        # print("m2.shape", m2.shape)
        t2b = self.T2b(m2, a1=a1, a2=a2)
        # print("t2b.shape", t2b.shape)
        m1 = self.M1(t1a, t2b)
        # print("m1.shape", m1.shape)
        t1b = self.T1b(m1, a1=a1, a2=a2)
        # print("t1b.shape", t1b.shape)
        m0 = self.M0(t0a, t1b)
        # print("m0.shape", m0.shape)
        t0b = self.T0b(m0, a1=a1, a2=a2)
        # print("t0b.shape", t0b.shape)
        out = self.seghead(t0b)
        # print("out",out.shape)
        return out



