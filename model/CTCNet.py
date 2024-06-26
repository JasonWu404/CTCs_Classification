#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Hefei Institute of Intelligent Machinery, Chinese Academy of Sciences All Rights Reserved.
#
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List,Tuple,Optional

from timm.models import register_model
from timm.models.layers import DropPath, trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from models.modules.mobileone import MobileOneBlock
from models.modules.replknet import ReparamLargeKernelConv
try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    """
    Split the feature map into non-overlapping windows according to window_size.
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_partition2(x, window_size):
    """
    Split the feature map into non-overlapping windows according to window_size.
    Args:
        x: (B, C, H, W)
        window_size (tuple[int]): window size(M)
    Returns:
        windows: (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    # view: -> [B, C, H//Wh, Wh, W//Ww, Ww]
    x = x.view(B, C, H // window_size[0], window_size[1], W // window_size[0], window_size[1])
    # permute: -> [B, H//Wh, W//Ww, Wh, Ww, C]
    # view: -> [B*num_windows, Wh, Ww, C]
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0] * window_size[1], C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Reducing a window to a feature map
    num_windows = H//Wh * W//Ww
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Wh, Ww, C] -> [B, H//Wh, W//Ww, Wh, Ww, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, H//Wh, Wh, W//Ww, Ww, C]
    # view: [B, H//Wh, Wh, W//Ww, Ww, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_reverse2(windows, window_size, H: int, W: int):
    """
    Windows reverse to feature map.
    [B * H // win * W // win , win*win , C] --> [B, C, H, W]
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # view: [B*num_windows, N, C] -> [B, H//window_size, W//window_size, window_size, window_size, C]
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, C, H//Wh, Wh, W//Ww, Ww]
    # view: [B, C, H//Wh, Wh, W//Ww, Ww] -> [B, C, H, W]
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x

def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,
            padding=1,groups=1,inference_mode=inference_mode, use_se=False,num_conv_branches=1,),
        MobileOneBlock(
            in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=2,
            padding=1,groups=out_channels,inference_mode=inference_mode,use_se=False, num_conv_branches=1,),
        MobileOneBlock(
            in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,
            padding=0,groups=1,inference_mode=inference_mode,use_se=False,num_conv_branches=1,),
    )

class PatchEmbed(nn.Module):
    """
    Convolutional patch embedding layer.
    """

    def __init__(
        self,patch_size: int,stride: int,
        in_channels: int,embed_dim: int,inference_mode: bool = False,
    ) -> None:
        """
        Build patch embedding layer.
        Args:
            patch_size: Patch size for embedding computation.
            stride: Stride for convolutional embedding layer.
            in_channels: Number of channels of input tensor.
            embed_dim: Number of embedding dimensions.
            inference_mode: Flag to instantiate model in inference mode. Default: ``False``
        """
        super().__init__()
        block = list()
        block.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,
                stride=stride,groups=in_channels,small_kernel=3,inference_mode=inference_mode,)
        )
        block.append(
            MobileOneBlock(
                in_channels=embed_dim,out_channels=embed_dim,kernel_size=1,stride=1,
                padding=0,groups=1,inference_mode=inference_mode,use_se=False,num_conv_branches=1,)
        )
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class paralleltokenmixer(nn.Module):
    """
    paralleltokenmixer.
    Modified from Window self attention (W-MSA) module
    with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        dwconv_kernel_size (int): The kernel size for dw-conv
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale
            of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, dwconv_kernel_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        attn_dim = dim // 2
        self.window_size = window_size  # Wh, Ww
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Definition Relative Position Offset
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        # get pair-wise relative position index for each token inside the window
        relative_coords = self._get_rel_pos()
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw] get the final relative position bias
        self.register_buffer("relative_position_index", relative_position_index)

        # prev proj layer
        self.proj_attn = nn.Linear(dim, dim // 2)  # in the Attention branch, the number of channels is halved.
        self.proj_attn_norm = nn.LayerNorm(dim // 2)
        self.proj_cnn = nn.Linear(dim, dim)
        self.proj_cnn_norm = nn.LayerNorm(dim)

        # conv branch
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=self.dwconv_kernel_size,
                padding=self.dwconv_kernel_size // 2,
                groups=dim
            ),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 2, kernel_size=1),  # halve the number of channels in the Attention branch
        )
        self.projection = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm2d(dim // 2)

        # window-attention branch
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)  # halve the number of channels in the Attention branch
        self.attn_drop = nn.Dropout(attn_drop)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)  # final spatial information output channel 1
        )
        self.attn_norm = nn.LayerNorm(dim // 2)
        # final projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos(self):
        """
            Get pair-wise relative position index for each token inside the window.
            Args:
                window_size (tuple[int]): window size
        """
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw] an absolute position matrix was created
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        # broadcasting mechanism - the former replicates the 3rd dimension Mh*Mw times, the latter replicates the 2nd dimension Mh*Mw times
        # subtract the absolute position index of other pixels from the absolute position index of the current pixel to get the relative position index of this pixel
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2] get the final relative position index
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        return relative_coords

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            H: the height of the feature map
            W: the width of the feature map
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww)
                or None
        """
        # proj_attn(): -> [B*num_windows, N, C/2]  the fully connected layer expects an input tensor of the form: [B, *, C]
        x_atten = self.proj_attn_norm(self.proj_attn(x))
        # proj_cnn(): -> [B*num_windows, N, C]
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x))
        # window_reverse2(): -> [B, C, H, W]
        x_cnn = window_reverse2(x_cnn, self.window_size, H, W)

        # conv branch
        # dwconv3×3(): -> [B, C, H, W]
        x_cnn = self.dwconv3x3(x_cnn)
        # AvgPool2d(1): -> [B, C, 1, 1]  input data format requirements[B, C, H, W]
        # conv(): -> [B, C/8, 1, 1]
        # conv(): -> [B, C/2, 1, 1]  corresponds to halving the number of channels in the Attention branch.
        channel_interaction = self.channel_interaction(x_cnn)
        # projection(): -> [B, C/2, H, W]
        x_cnn = self.projection(x_cnn)

        # attention branch
        # B_: B*num_windows;  N: Window_size ** 2;  C: C/2  corresponds to halving the number of channels in the Attention branch.
        B_, N, C = x_atten.shape
        # qkv(): -> [B*num_windows, N, 3*C] --- C: C/2
        # reshape: -> [B*num_windows, N, 3, num_heads, C/num_heads] --- C: C/2
        # permute: -> [3, B*num_windows, num_heads, N, C/num_heads] --- C: C/2
        qkv = self.qkv(x_atten).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # unbind(): -> [B*num_windows, num_heads, N, C/num_heads] --- C: C/2
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple) 分别取出Q、K、V

        # channel interaction
        # reshape -> [B, 1, num_heads, 1, C/num_heads] --- C: C/2
        x_cnn2v = torch.sigmoid(channel_interaction).reshape(-1, 1, self.num_heads, 1, C // self.num_heads)
        # reshape: -> [B, num_heads, num_heads, N, C/num_heads] --- C: C/2
        v = v.reshape(x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads)
        # *: -> [B, num_heads, num_heads, N, C/num_heads] --- C: C/2
        v = v * x_cnn2v
        # reshape: -> [B*num_windows, num_heads, N, C/num_heads] --- C: C/2
        v = v.reshape(-1, self.num_heads, N, C // self.num_heads)

        # transpose: -> [B*num_windows, num_heads, C/num_heads, N] --- C: C/2
        # @: multiply -> [B*num_windows, num_heads, N, N]
        q = q * self.scale  # Q/sqrt(dk)
        attn = (q @ k.transpose(-2, -1))  # Q*K^{T} / sqrt(dk)

        # relative_position_bias_table.view: [win*win*win*win,num_heads] -> [win*win*win*win,num_heads]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
        # +: -> [B*num_windows, num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)  # attention + relative positional bias

        #  if mask is available, mask the q*k value of the discontinuous region by directly adding the value of mask to the corresponding part of the attn result
        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            # softmax: -> [B*num_windows, num_heads, N, N]
            attn = self.softmax(attn)

        # [B*num_windows, num_heads, N, N]
        attn = self.attn_drop(attn)

        # @: multiply -> [B*num_windows, num_heads, N, C/num_heads] --- C: C/2
        # transpose: -> [B*num_windows, N, num_heads, C/num_heads] --- C: C/2
        # reshape: -> [B*num_windows, N, C] --- C: C/2 corresponding attention to the halving of the number of branch channels.
        x_atten = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # spatial interaction
        # window_reverse2: -> [B, C, H, W] --- C: C/2 corresponding attention to the halving of the number of branch channels.
        x_spatial = window_reverse2(x_atten, self.window_size, H, W)
        # conv: -> [B, C/8, H, W] --- C: C/2
        # conv: -> [B, 1, H, W]
        spatial_interaction = self.spatial_interaction(x_spatial)
        # sigmoid: -> [B, 1, H, W]
        # * -> [B, C, H, W] --- C: C/2
        x_cnn = torch.sigmoid(spatial_interaction) * x_cnn
        x_cnn = self.conv_norm(x_cnn)
        # [B, C, H, W] --> [num_windows*B, N, C] --- C: C/2
        x_cnn = window_partition2(x_cnn, self.window_size)

        # concat
        x_atten = self.attn_norm(x_atten)
        # cat(): -> [num_windows*B, N, C] --- C: C
        x = torch.cat([x_cnn, x_atten], dim=2)
        # proj: -> [num_windows*B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DWConvFFN(nn.Module):
    """
    DWConvFFN module
    """

    def __init__(
        self,in_channels: int,hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,act_layer: nn.Module = nn.GELU, drop: float = 0.0,
    ) -> None:
        """
        Build DWConvFFN module.
        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=5,
                      padding=3,groups=in_channels,bias=False,),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ParallelMixBlock(nn.Module):
    r"""
    ParallelMixBlock in CTCNet.
    Modified from  MetaFormer and VAN.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        shift_size (int): Shift size for SW-SA.
            We do not use shift in CTCNet. Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.
    """

    def __init__(self, dim, num_heads, window_size=7, dwconv_kernel_size=3, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert self.shift_size == 0,"No shift in CTCNet"

        self.norm1 = norm_layer(dim)
        self.attn = paralleltokenmixer(
            dim, window_size=(self.window_size, self.window_size), dwconv_kernel_size=dwconv_kernel_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.convffn = DWConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
    def forward(self, x, mask_matrix):
        """
        Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W,"input feature has wrong size"

        # [B, L, C] --- L: H * W
        shortcut = x
        x = self.norm1(x)
        # reshape(): -> [B, H, W, C]
        x = x.reshape(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            # moves the input data in the height and width directions by specified rows and columns.
            # roll(): -> [B, H', W', C]
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            # [B, Hp, Wp, C]
            shifted_x = x
            attn_mask = None

        # window_partition: -> [num_windows*B, window_size, window_size, C]
        x_windows = window_partition(shifted_x, self.window_size)
        # view: -> [num_windows*B, N, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)


        # attn(): -> [num_windows*B, N, C]
        attn_windows = self.attn(x_windows, Hp, Wp, mask=attn_mask)

        # merge windows  calculation complete, change from window back to data
        # view(): -> [num_windows*B, window_size, window_size, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # window_reverse(): -> [B, Hp, Wp, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            # [B, H', W', C] -> [B, Hp, Wp, C]
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            # [B, Hp, Wp, C]
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            # remove the data from the front pad: -> [B, H, W, C]
            x = x[:, :H, :W, :].contiguous()
        # view(): -> [B, H*W, C]
        x = x.view(B, H * W, C)
        # DWConvFFN
        # [B, H*W, C]
        x = shortcut + self.drop_path(x)
        # mlp: -> [B, H*W, C]
        # +: -> [B, H*W, C]
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,out_channels=2 * kernel_size[0] * kernel_size[1],kernel_size=kernel_size,
                                    padding=padding,stride=stride,dilation=dilation,bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,
                                                        padding=padding,groups=groups,stride=stride,dilation=dilation,bias=False)
    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

class deformable_LKA(nn.Module):
    """
    Build deformable_LKA in DLKA Block.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5,5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7,7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class DeformableLargeKernelAttention(nn.Module):
    """
    Build DLKAttention in DLKA Block.
    """
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class DLKAttentionBlock(nn.Module):
    """
    Implementation of DLKAttentionBlock with DeformableLargeKernelAttention as token mixer.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        """
        Build DLKAttentionBlock.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = DeformableLargeKernelAttention(dim=dim)

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = DWConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x

def basic_blocks(
    dim: int,
    token_mixer_type: str,
    block_index: int,
    num_blocks: List[int],
    num_heads: int = 4,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    qkv_bias: bool = True,
    qk_scale: float = None,
    use_layer_scale:bool = True,
    layer_scale_init_value: float = 1e-5,
) -> nn.Sequential:
    """Build ParallelMix Block within a stage.

    Args:
        dim: Number of embedding dimensions.
        token_mixer_type: token_mixer_type
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        num_heads: num_heads default:4.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        qkv_bias: bool = True.
        qk_scale: float = None.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.

    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "paralleltokenmixer":
            blocks.append(
                ParallelMixBlock(
                    dim,num_heads = num_heads,mlp_ratio=mlp_ratio,act_layer=act_layer,
                    drop=drop_rate,drop_path=block_dpr,qkv_bias=qkv_bias,qk_scale=qk_scale,)
            )
        elif token_mixer_type == "DeformableLargeKernelAttention":
            blocks.append(
                DLKAttentionBlock(
                    dim,mlp_ratio=mlp_ratio,act_layer=act_layer,norm_layer=norm_layer,
                    drop=drop_rate,drop_path=block_dpr,use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,)
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )
    blocks = nn.Sequential(*blocks)

    return blocks

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }

default_cfgs = {
    "ctcnet_tiny": _cfg(crop_pct=0.9),
}

class CTCNet(nn.Module):

    def __init__(
        self,layers,dim,token_mixers: Tuple[str, ...],embed_dims=4,mlp_ratios=None,downsamples=None,
        num_heads=4,norm_layer: nn.Module = nn.BatchNorm2d,act_layer: nn.Module = nn.GELU,num_classes=4,
        pos_embs=None,down_patch_size=7,down_stride=2,drop_rate=0.0,drop_path_rate=0.0,
        qkv_bias=True,qk_scale=None,fork_feat=False,init_cfg=None, pretrained=None,
        cls_ratio=2.0,inference_mode=False,**kwargs,
    ) -> None:

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        if pos_embs is None:
            pos_embs = [None] * len(layers)

        # convolutional_stem
        self.patch_embed = convolutional_stem(3, embed_dims, inference_mode)

        # Build the main stages of the network architecture
        network = []
        for i in range(len(layers)):
            # Add position embeddings if requested
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](
                        embed_dims, embed_dims, inference_mode=inference_mode
                    )
                )
            stage = basic_blocks(
                embed_dims,i,layers,dim,token_mixers[i],num_heads,
                mlp_ratios[i],act_layer,norm_layer,drop_rate,drop_path_rate,qkv_bias,qk_scale,)
            network.append(stage)
            if i >= len(layers) - 1:
                break

            # Patch merging/downsampling between stages.
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,stride=down_stride,
                        in_channels=embed_dims[i],embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,)
                )

        self.network = nn.ModuleList(network)

        # For segmentation and detection, extract intermediate output
        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.conv_exp = MobileOneBlock(
                in_channels=embed_dims[-1],out_channels=int(embed_dims[-1] * cls_ratio),kernel_size=3,
                stride=1,padding=1,groups=embed_dims[-1],inference_mode=inference_mode,use_se=True,num_conv_branches=1,)
            self.head = (
                nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

        # load pre-trained model
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m: nn.Module) -> None:
        """Init. for classification"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _scrub_checkpoint(checkpoint, model):
        sterile_dict = {}
        for k1, v1 in checkpoint.items():
            if k1 not in model.state_dict():
                continue
            if v1.shape == model.state_dict()[k1].shape:
                sterile_dict[k1] = v1
        return sterile_dict

    def init_weights(self, pretrained: str = None) -> None:
        """Init. for mmdetection or mmsegmentation by loading
        ImageNet pre-trained weights.
        """
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warning(
                f"No pre-trained weights for "
                f"{self.__class__.__name__}, "
                f"training start from scratch"
            )
            pass
        else:
            assert "checkpoint" in self.init_cfg, (
                f"Only support "
                f"specify `Pretrained` in "
                f"`init_cfg` in "
                f"{self.__class__.__name__} "
            )
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg["checkpoint"]
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location="cpu")
            if "state_dict" in ckpt:
                _state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                _state_dict = ckpt["model"]
            else:
                _state_dict = ckpt

            sterile_dict = CTCNet._scrub_checkpoint(_state_dict, self)
            state_dict = sterile_dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # output features of four stages for dense prediction
            return x
        # for image classification
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        cls_out = self.head(x)
        return cls_out

@register_model
def ctcnet_tiny(pretrained=False, **kwargs):
    """Instantiate CTCNet_Tiny model variant."""
    layers = [2, 2, 2, 2]
    dim = 3
    embed_dims = [48, 96, 192, 384]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    num_heads = 4
    token_mixers = ("paralleltokenmixer", "paralleltokenmixer", "paralleltokenmixer", "DeformableLargeKernelAttention")
    model = CTCNet(
        layers,dim,token_mixers,num_heads,
        embed_dims,mlp_ratios,downsamples,**kwargs,)
    model.default_cfg = default_cfgs["ctcnet_tiny"]
    if pretrained:
        url ='pre-training weight'
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint["state_dict"])
    return model