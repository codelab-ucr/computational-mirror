#!/usr/bin/env python

import os
import sys
import tqdm
import importlib
import pdb

import numpy as np
from scipy import io

import cv2
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Parameter

import matplotlib.pyplot as plt

import models

models = importlib.reload(models)

class KiloNeRF(nn.Module):
    '''
        KiloNeRF model for INR representation

        Inputs:
            imsize: (H, W) size of image
            config: configuration namespace
    '''
    def __init__(self, imsize, config):
        super().__init__()

        H, W = imsize

        self.unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
        self.fold = torch.nn.Fold(output_size=imsize, kernel_size=config.ksize,
                                   stride=config.stride)
        
        # To find out number of chunks
        weight = torch.ones(1, 1, H, W)
        self.nchunks = self.unfold(weight).shape[-1]

        # Get model
        self.model = models.get_model(config, self.nchunks)

        # Create a window for overlap add approach
        self.window = get_window(config.ksize, config.stride, 'triangle')
        self.window = self.window.reshape(1, -1, 1).cuda()

        # Get coordinates
        self.coords = get_coords([H, W], config.ksize, 'local', self.unfold)

        self.indices = torch.arange(self.nchunks)

    def forward(self, coords):
        im_estim_chunked = self.model(coords, self.indices).permute(2, 1, 0)

        im_estim = self.fold(im_estim_chunked).squeeze()[None, ...]

        return im_estim

def get_coords(imsize, ksize, coordstype, unfold):
    '''
        Generate coordinates for MINER training
        
        Inputs:
            imsize: (H, W) image size
            ksize: Kernel size
            coordstype: 'global' or 'local'
            unfold: Unfold operator
    '''
    ndim = len(imsize)
    if ndim == 2:
        H, W = imsize    
        
        #nchunks = int(H*W/(ksize**ndim))
        nchunks = unfold(torch.ones(1, 1, H, W)).shape[-1]
        
        # Create inputs
        if coordstype == 'global':
            X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                                  torch.linspace(-1, 1, H))
            coords = torch.cat((X[None, None, ...], Y[None, None, ...]), 0)
            coords_chunked = unfold(coords).permute(2, 1, 0)
        elif coordstype == 'local':
            Ysub, Xsub = torch.meshgrid(
                torch.linspace(-1, 1, ksize[0]),
                torch.linspace(-1, 1, ksize[1]))
            coords_sub = torch.cat((Xsub[None, None, ...],
                                    Ysub[None, None, ...]), 0)
            coords_chunked_sub = unfold(coords_sub).permute(2, 1, 0)
            coords_chunked = torch.repeat_interleave(coords_chunked_sub,
                                                     nchunks, 0)
        else:
            raise AttributeError('Coordinate type not understood')
    else:
        H, W, T = imsize    
        nchunks = int(H*W*T/(ksize**ndim))
        # Create inputs
        if coordstype == 'global':
            X, Y, Z = torch.meshgrid(torch.linspace(-1, 1, W),
                                     torch.linspace(-1, 1, H),
                                     torch.linspace(-1, 1, T))
            coords = torch.cat((X[None, None, ...],
                                Y[None, None, ...],
                                Z[None, None, ...]), 0)
            coords_chunked = unfold(coords).permute(2, 1, 0)
        elif coordstype == 'local':
            Xsub, Ysub, Zsub = torch.meshgrid(torch.linspace(-1, 1, ksize),
                                              torch.linspace(-1, 1, ksize),
                                              torch.linspace(-1, 1, ksize))
            coords_sub = torch.cat((Xsub[None, None, ...],
                                    Ysub[None, None, ...],
                                    Zsub[None, None, ...]), 0)
            coords_chunked_sub = unfold(coords_sub).permute(2, 1, 0)
            coords_chunked = torch.repeat_interleave(coords_chunked_sub,
                                                     nchunks, 0)
        else:
            raise AttributeError('Coordinate type not understood')
    
    return coords_chunked

def get_window(ksize, stride, wintype):
    '''
        Get window for overlap add approach
    '''
    kh, kw = ksize
    sh, sw = stride

    sh, sw = sh//4, sw//4

    dh = kh - sh
    dw = kw - sw

    row_win = torch.ones(1, kw)
    col_win = torch.ones(kh, 1)
    
    if wintype == 'triangle':
        row_trans = torch.linspace(0, 1, sw)
        col_trans = torch.linspace(0, 1, sh)

    elif wintype == 'flat':
        row_trans = 0.5*torch.ones(sw)
        col_trans = 0.5*torch.ones(sh)

    elif wintype == 'cosine':
        row_trans = 0.5 + 0.5*torch.cos(torch.linspace(1, 0, sw)*np.pi)
        col_trans = 0.5 + 0.5*torch.cos(torch.linspace(1, 0, sh)*np.pi)

    row_win[0, :sw] = row_trans
    row_win[0, -sw:] = row_trans.flip(0)

    col_win[:sh, 0] = col_trans
    col_win[-sh:, 0] = col_trans.flip(0)

    window = (col_win*row_win)[None, ..., None]

    return 0.5 + 0.5*window