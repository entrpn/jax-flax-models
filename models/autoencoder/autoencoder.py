import numpy as np
import jax.numpy as jnp
import jax
import flax
import flax.linen as nn

## PyTorch
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

from jax import random

class Upsample(nn.Module):
    in_channels : int
    with_conv : bool

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(features=self.in_channels, kernel_size=(3,3), strides=(1,1), padding=((1,1),(1,1)))

    def __call__(self, x):
        b,h,w,c = x.shape
        x = jax.image.resize(x,shape=(b, h*2, w*2, c), method="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class DownSample(nn.Module):
    in_channels : int
    with_conv : bool

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(features=self.in_channels, kernel_size=(3,3), strides=2, padding=0)
    
    def __call__(self, x):
        if self.with_conv:
            pad = ((0,0),(0, 1), (0, 1), (0, 0))
            x = jnp.pad(x, pad_width=pad)
            x = self.conv(x)
        else:
            x = nn.avg_pool(x, shape_window=(2,2), stride=2)
        return x

class AttnBlock(nn.Module):
    in_channels : int
    num_heads : int = 8

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, group_size=None, epsilon=1e-6)
        
        self.q = nn.Conv(features=self.in_channels, kernel_size=(1,1), strides=1, padding=0)
        self.kv = nn.Conv(features=self.in_channels, kernel_size=(1,1), strides=1, padding=0)

        self.mha = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)

        self.proj_out = nn.Conv(features=self.in_channels, kernel_size=(1,1), strides=1, padding=0)
    
    def __call__(self, x, train=True):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        kv = self.kv(h_)

        h_ = self.mha(q,kv, deterministic=not train)

        h_ = self.proj_out(h_)
        return x+h_

class ResnetBlock(nn.Module):
    in_channels: int
    out_channels : int
    use_conv_shortcut = False
    temb_channels : int = 512
    dropout_prob : float = 0.0

    def setup(self):
        self.norm1 = nn.GroupNorm(num_groups=32, group_size=None, epsilon=1e-6)
        self.conv1 = nn.Conv(features=self.out_channels,kernel_size=(3,3), strides=1,padding=1)

        if self.temb_channels > 0:
            self.temb_proj = nn.Dense(self.out_channels)
        
        self.norm2 = nn.GroupNorm(num_groups=32, group_size=None)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.conv2 = nn.Conv(features=self.out_channels,kernel_size=(3,3),strides=1,padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv(self.out_channels,kernel_size=(3,3),strides=1,padding=1)
            else:
                self.nin_shortcut = nn.Conv(self.out_channels, kernel_size=(1,1),strides=1,padding=0)
    
    def __call__(self,x,temb=None,train=True):
        h = x
        h = self.norm1(h)
        h = nn.activation.swish(h)
        h = self.conv1(h)

        if temb is not None:
            temb = nn.activation.swish(temb)
            h = h + self.temb_proj(temb)[:,:,None,None]
        
        h = self.norm2(h)
        h = nn.activation.swish(h)
        h = self.dropout(h,deterministic=not train)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        
        return x+h

class Encoder(nn.Module):
    ch : int
    ch_mult : list
    num_res_blocks : int
    attn_resolutions : list
    resolution : int
    z_channels : int
    dropout : float = 0.0
    temb_ch : int = 0
    resamp_with_conv : bool = True
    double_z : bool = True
    
    def setup(self):
        self.num_resolutions = len(self.ch_mult)
        self.conv_in = nn.Conv(features=self.ch, kernel_size=(3,3), strides=1, padding=1)

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(self.ch_mult)


        down_blocks = []

        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = self.ch*in_ch_mult[i_level]
            block_out = self.ch*self.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,temb_channels=self.temb_ch, dropout_prob=self.dropout))
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            dn = {}
            dn['block'] = block
            dn['attn'] = attn
            if i_level != self.num_resolutions-1:
                dn['downsample'] = DownSample(block_in, self.resamp_with_conv)
                curr_res = curr_res // 2
            down_blocks.append(dn)
        self.down_blocks = down_blocks
        
        self.mid_block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,temb_channels=self.temb_ch,dropout_prob=self.dropout)
        self.mid_attn_1 = AttnBlock(block_in)
        self.mid_block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,temb_channels=self.temb_ch,dropout_prob=self.dropout)

        self.norm_out = nn.GroupNorm(num_groups=32, group_size=None, epsilon=1e-6)
        self.conv_out = nn.Conv(features=2*self.z_channels if self.double_z else self.z_channels, kernel_size=(3,3), strides=1, padding=1)
    
    def __call__(self,x,train=True):
        temb = None

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down_blocks[i_level]['block'][i_block](hs[-1], temb,train)
                if len(self.down_blocks[i_level]['attn']) > 0:
                    h = self.down_blocks[i_level]['attn'][i_block](h,train)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down_blocks[i_level]['downsample'](hs[-1]))
        
        h = hs[-1]
        h = self.mid_block_1(h, temb)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h, temb)

        h = self.norm_out(h)
        h = nn.activation.swish(h)
        h = self.conv_out(h)
        return h
        

class Decoder(nn.Module):
    ch : int
    out_ch : int
    num_res_blocks : int
    attn_resolutions : list
    ch_mult : list
    resolution : int
    z_channels : int
    temb_ch : int = 0
    give_pre_end : bool = False
    tanh_out : bool = False
    dropout : float = 0.0
    resamp_with_conv : bool = True

    def setup(self):
        self.num_resolutions = len(self.ch_mult)
        
        in_ch_mult = (1,) + tuple(self.ch_mult)
        block_in = self.ch*self.ch_mult[self.num_resolutions-1]
        curr_res = self.resolution // 2 **(self.num_resolutions-1)
        self.z_shape = (1, curr_res, curr_res, self.z_channels)
        print("working with z of shape {} = {} dimensions".format(self.z_shape, np.prod(self.z_shape)))

        self.conv_in = nn.Conv(features=block_in,kernel_size=(3,3), strides=1, padding=1)

        self.mid_block1 = ResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_ch,dropout_prob=self.dropout)
        self.mid_attn_1 = AttnBlock(in_channels=block_in)
        self.mid_block_2 = ResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_ch,dropout_prob=self.dropout)

        up_blocks = []

        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = self.ch*self.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                temb_channels=self.temb_ch,
                dropout_prob=self.dropout))
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = {}
            up['block'] = block
            up['attn'] = attn
            if i_level != 0:
                up['upsample'] = Upsample(block_in, self.resamp_with_conv)
                curr_res = curr_res * 2
            
            up_blocks.insert(0, up)
        self.up_blocks = up_blocks

        self.norm_out = nn.GroupNorm(num_groups=32, group_size=None, epsilon=1e-6)
        self.conv_out = nn.Conv(features=self.out_ch, kernel_size=(3,3), strides=1, padding=1)
    
    def __call__(self, z, train=True):
        temb = None
        print('z',z.shape)
        h = self.conv_in(z)

        h = self.mid_block1(h,temb,train)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h, temb, train)
        print('after mid block',h.shape)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up_blocks[i_level]['block'][i_block](h, temb)
                if len(self.up_blocks[i_level]['attn']) > 0:
                    h = self.up_blocks[i_level]['attn'][i_block](h)
            if i_level != 0:
                h = self.up_blocks[i_level]['upsample'](h)
            print('upsample.shape',h.shape)
        
        if self.give_pre_end:
            return h
        
        h = self.norm_out(h)
        h = nn.activation.swish(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = nn.activation.tanh(h)
        return h

class Autoencoder(nn.Module):
    ch : int
    ch_mult : int
    num_res_blocks : int
    attn_resolutions : list
    resolution : int
    z_channels : int
    out_ch : int
    temb_ch : int = 0
    resamp_with_conv : bool = True
    double_z : bool = True
    give_pre_end : bool = False
    dropout : float = 0.0
    tanh_out : bool = False

    @nn.compact
    def __call__(self, x, train=True):
        x = Encoder(
            ch=self.ch,
            ch_mult=self.ch_mult,
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_resolutions,
            resolution=self.resolution,
            z_channels=self.z_channels,
            dropout=self.dropout,
            temb_ch=self.temb_ch,
            double_z=self.double_z,
            resamp_with_conv=self.resamp_with_conv
        )(x,train)

        return Decoder(
            ch=self.ch,
            out_ch=self.out_ch,
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_resolutions,
            ch_mult=self.ch_mult,
            resolution=self.resolution,
            z_channels=self.z_channels,
            temb_ch=self.temb_ch,
            give_pre_end=self.give_pre_end,
            tanh_out=self.tanh_out,
            dropout=self.dropout,
            resamp_with_conv=self.resamp_with_conv
        )(x,train)