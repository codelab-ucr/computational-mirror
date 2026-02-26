#!/usr/bin/env python

import os
import sys
from torch.functional import align_tensors
import tqdm
import pdb
import math

import numpy as np
from skimage.metrics import structural_similarity as ssim_func

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import kornia
#from pytorch_wavelets import DWTForward

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
#from pytorch_wavelets import DWTForward, DWTInverse

import skimage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2

import utils

class TVNorm(nn.Module):
    def __init__(self, mode='l1'):
        super().__init__()
        self.mode = mode

        kx = torch.Tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])
        ky = torch.Tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])
        
        kx = torch.Tensor([[1, -1],
                           [1, -1]])
        ky = torch.Tensor([[1, 1],
                           [-1, -1]])
    
        kx = kx.view(1, 1, 2, 2).float()
        ky = ky.view(1, 1, 2, 2).float()

        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
    def __call__(self, img):
        _, C, _, _ = img.shape
        #grad_x = torch.nn.functional.conv2d(img, self.kx, padding='same', groups=C)
        #grad_y = torch.nn.functional.conv2d(img, self.ky, padding='same', groups=C)

        gradients = kornia.filters.spatial_gradient(img)
        grad_x = gradients[:, :, 0, ...]
        grad_y = gradients[:, :, 1, ...]
        
        if self.mode == 'isotropic':
            return torch.sqrt(grad_x.abs()**2 + grad_y.abs()**2).mean()
        elif self.mode == 'l1':
            return abs(grad_x).mean() + abs(grad_y).mean()
        else:
            return (grad_x.pow(2) + grad_y.pow(2)).mean()   
        
class SpecNorm():
    def __init__(self, mode='l1'):
        self.mode = mode
    def __call__(self, cube, penalize_final=False):   
        diff_spec = cube[:, 1:, ...] - cube[:, :-1, ...]

        if self.mode == 'l2':
            loss_spec = diff_spec.square().mean() 
        elif self.mode == 'l1':
            loss_spec = diff_spec.abs().mean()
        elif self.mode == 'sparse':
            loss_spec = cube.abs().mean()
                    
        return loss_spec
    
class WaveletNorm(torch.nn.Module):
    def __init__(self, J, wave='db6', mode='per'):
        super().__init__()
        
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
        
    def __call__(self, img):
        _, Yh = self.dwt(img)
        
        loss = torch.zeros(1).cuda()
        
        for Yh_sub in Yh:
            loss = loss + abs(Yh_sub).mean()
            
        return loss
    
class WaveletGroupNorm(torch.nn.Module):
    def __init__(self, J, wave='db6', mode='per'):
        super().__init__()
        
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
        
    def __call__(self, img):
        Yl, Yh = self.dwt(img)
        
        B, nchan, H, W = img.shape
        
        coefs = Yl.reshape(B, nchan, -1)
        
        for Yh_sub in Yh:
            coefs_sub = Yh_sub.reshape(B, nchan, -1)
            coefs = torch.cat((coefs, coefs_sub), dim=-1)
            
        loss = ((coefs.square().mean(1)).sqrt()).mean()
        
        return loss
            
    
class HessianNorm():
    def __init__(self):
        pass
    def __call__(self, img):
        # Compute Individual derivatives
        fxx = img[..., 1:-1, :-2] + img[..., 1:-1, 2:] - 2*img[..., 1:-1, 1:-1]
        fyy = img[..., :-2, 1:-1] + img[..., 2:, 1:-1] - 2*img[..., 1:-1, 1:-1]
        fxy = img[..., :-1, :-1] + img[..., 1:, 1:] - \
              img[..., 1:, :-1] - img[..., :-1, 1:]
          
        return torch.sqrt(fxx.abs().pow(2) +\
                          2*fxy[..., :-1, :-1].abs().pow(2) +\
                          fyy.abs().pow(2)).mean()
    
class L1Norm():
    def __init__(self):
        pass
    def __call__(self, x):
        return abs(x).mean()        

class L2Norm():
    def __init__(self):
        pass
    def __call__(self, x):
        return (x.pow(2)).mean()    

@torch.no_grad()
def get_metrics(gt, estim, lpip_func=None, pad=True):
    '''
        Compute SNR, PSNR, SSIM, and LPIP between two images.
        
        Inputs:
            gt: Ground truth image
            estim: Estimated image
            lpip_func: CUDA function for computing lpip value
            pad: if True, remove boundaries when computing metrics
            
        Outputs:
            metrics: dictionary with following fields:
                snrval: SNR of reconstruction
                psnrval: Peak SNR 
                ssimval: SSIM
                lpipval: VGG perceptual metrics
    '''
    if min(gt.shape) < 50:
        pad = False
    if pad:
        gt = gt[20:-20, 20:-20]
        estim = estim[20:-20, 20:-20]
        
    snrval = utils.asnr(gt, estim)
    psnrval = utils.asnr(gt, estim, compute_psnr=True)
    ssimval = ssim_func(gt, estim, multichannel=True)
    
    # Need to convert images to tensors for computing lpips
    gt_ten = torch.tensor(gt)[None, None, ...]
    estim_ten = torch.tensor(estim)[None, None, ...]
    
    # For some reason, the image values should be [-1, 1]
    if lpip_func is not None:
        lpipval = lpip_func(2*gt_ten-1, 2*estim_ten-1).item()
    else:
        lpipval = 0
    
    metrics = {'snrval': snrval,
               'psnrval': psnrval,
               'ssimval': ssimval,
               'lpipval': lpipval}
    return metrics