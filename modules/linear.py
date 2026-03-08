#!/usr/bin/env python

import os
import sys
import tqdm
import pdb

import numpy as np
import torch
from scipy import interpolate
from skimage import color
from deepali.core.flow import expv
from deepali.core import Grid  # Corrected import
from deepali.core.bspline import evaluate_cubic_bspline
from deepali.spatial.bspline import FreeFormDeformation
from deepali.core import Grid
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn
from utils import set_kernel
import kornia
import sys
sys.path.append("/home/akayabasi/computational-mirror/")
from modules.inr_models import INR
from scipy import linalg

import torchvision
import cv2

def create_grid(cps,imsize,nterms):
        k = set_kernel([cps, cps]) # 2D
        padding = [(len(k_i) - 1) // 2 for k_i in k]
        padding_2d = (padding[1], padding[1], padding[0], padding[0])
        padded_sh1 = imsize[0] + padding_2d[2] + padding_2d[3]
        padded_sh2 = imsize[1] + padding_2d[0] + padding_2d[1]
        sh1 = len(torch.arange(cps//2, padded_sh1, step=cps))
        sh2 = len(torch.arange(cps//2, padded_sh2, step=cps))
        return torch.zeros((nterms, 2, sh1, sh2)) 

class ZoomConv(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            nfocus: Number of focus settings. If None, 
                set it to nterms
    '''
    def __init__(self, nterms, dmax, rmax, imsize, nfocus=None):
        super().__init__()

        # Learnable parameters are rfactors, kernels, and weights

        H, W, nchan = imsize
        self.nterms = nterms

        if nfocus is None:
            nfocus = nterms

        disk_sizes = torch.linspace(1, dmax, nterms)
        self.rfactors = torch.linspace(1, rmax, nterms)

        # What if we give independence to both axes?
        #self.rfactors = torch.repeat_interleave(self.rfactors[:, None, None, None],
        #                                        2, -1)
        self.rfactors = self.rfactors[:, None, None, None]

        # So we will have weights that are rank-1, which means we need
        # kernel weights
        self.kweights = torch.zeros(nfocus, nterms, 1, 1, 1)

        # Let us try initializing with stepped values
        kstep = nterms//nfocus
        for k_idx in range(0, nfocus):
            self.kweights[k_idx, k_idx*kstep:(k_idx+1)*kstep] = 1

        # Create convolution kernels
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax),
                                torch.linspace(-1, 1, dmax),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms, 0)
        
        kernels = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/dmax))
        kernels = kernels.float()
        
        kernsum = kernels.sum(-1, keepdim=True).sum(-2, keepdim=True)
        
        self.kernels = kernels/kernsum
                
        # Create expansion grids
        X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                            torch.linspace(-1, 1, H),
                            indexing='xy')
        grid = torch.cat((X[None, ..., None], Y[None, ..., None]), -1)
        grid_rep = torch.repeat_interleave(grid, nterms, 0)
                
        # Create weights for each term
        R = torch.hypot(X, Y)[None, ...]
        cR = torch.linspace(np.sqrt(2), 0, nfocus)[:, None, None]
        cS = (2/nfocus)**2
        
        self.weights = torch.exp(-((R - cR)**2)/(4*cS))[:, None, ...]

        # Make parameters learnable
        self.rfactors = ten2param(self.rfactors)
        self.kernels = ten2param(self.kernels)
        self.weights = ten2param(self.weights)
        self.kweights = ten2param(self.kweights)

        # Register grid as a parameter but not learnable
        self.register_buffer('grid_rep', grid_rep)
    
    def forward(self, imten, f_idx=[0]):
        # Create grid
        grid_stretched = self.grid_rep/self.rfactors

        # Replicate image tensor
        imten_rep = torch.repeat_interleave(imten, self.nterms, 0)

        # Interpolate
        imten_stretch = torch.nn.functional.grid_sample(
            imten_rep, grid_stretched, mode='bilinear',
            align_corners=False
        )
        
        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms,
            padding='same').permute(1, 0, 2, 3)
        
        imten_blur_weighed = self.kweights*imten_blur_stack[None, ...]
                
        imten_focus = imten_blur_weighed[f_idx, 0, ...]*self.weights[f_idx, ...]
        imten_defocus = imten_blur_weighed[f_idx, 1:, ...].mean(1)*(1-self.weights[f_idx, ...])

        imten_blur = imten_focus + imten_defocus

        return imten_blur
    
class AffineConvRS(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image. This is the radially symmetric version where the
        weights are rank-1 in the polar domain.

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            nfocus: Number of focus settings. If None, 
                set it to nterms
    '''
    def __init__(self, nterms, dmax, rmax, imsize, nfocus=None):
        super().__init__()

        # Learnable parameters are rfactors, kernels, and weights

        H, W, nchan = imsize
        self.nterms = nterms
        self.imsize = imsize

        if nfocus is None:
            nfocus = nterms

        disk_sizes = torch.linspace(1, dmax, nterms)
        self.rfactors = torch.linspace(1, rmax, nterms)

        self.rfactors = self.rfactors[:, None, None, None]

        self.rfactors = torch.repeat_interleave(self.rfactors, 2, -1)

        # So we will have weights that are rank-1, which means we need
        # kernel weights
        self.kweights = torch.zeros(nfocus, nterms, 1, 1, 1)

        # Let us try initializing with stepped values
        kstep = nterms//nfocus
        for k_idx in range(0, nfocus):
            self.kweights[k_idx, k_idx*kstep:(k_idx+1)*kstep] = 1

        # Create convolution kernels
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax),
                                torch.linspace(-1, 1, dmax),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms, 0)
        
        kernels = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/dmax))
        kernels = kernels.float()

        with torch.no_grad():
            kernels[nterms//2:, ...].random_(0, 1000)/1000
        
        kernsum = kernels.sum(-1, keepdim=True).sum(-2, keepdim=True)
        
        self.kernels = kernels/kernsum
                
        # Create expansion grids
        X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                            torch.linspace(-1, 1, H),
                            indexing='xy')
        grid = torch.cat((X[None, ..., None], Y[None, ..., None]), -1)
        grid_rep = torch.repeat_interleave(grid, nterms, 0)
                
        # Create weights for each term
        Rmax = np.ceil(np.hypot(H, W)).astype(int)
        R = torch.linspace(0, np.sqrt(2), Rmax)[None, :]
        cR = torch.linspace(np.sqrt(2), 0, nfocus)[:, None]
        cS = (2/nfocus)**2
        
        self.pol_weights = 0.8 + 0.2*torch.exp(-((R - cR)**2)/(cS))
        self.pol_weights = self.pol_weights[..., None]

        with torch.no_grad():
            self.pol_weights = self.pol_weights +\
                  0.1*torch.rand_like(self.pol_weights)
        
        # Initiate polar to cartesian conversion function
        self.init_pol2cart(imsize, Rmax)

        # Make parameters learnable
        self.rfactors = ten2param(self.rfactors)
        self.kernels = ten2param(self.kernels)
        self.pol_weights = ten2param(self.pol_weights)
        self.kweights = ten2param(self.kweights)

        # Register grid as a parameter but not learnable
        self.register_buffer('grid_rep', grid_rep)
        self.register_buffer('pol2cart_grid',
                              self._pol2cart_grid)
        
    def init_pol2cart(self, imsize, Rmax):
        height, width, _ = imsize

        # Center is center of the image, but in future
        # we can make this learnable
        center = (width/2, height/2)
        
        # We will assume width is larger than or equal to
        # height
        r_max = Rmax/2
        theta_max = 2*torch.pi
        
        # Create a grid of Cartesian coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height),
            torch.arange(width)
        )

        # Shift coordinates relative to the center
        x_shifted = x_coords - center[0]
        y_shifted = y_coords - center[1]
        
        # Convert to polar coordinates
        r = torch.sqrt(x_shifted**2 + y_shifted**2)
        theta = torch.atan2(y_shifted, x_shifted)
        
        theta_normalized = ((theta + torch.pi)/theta_max)
        
        # Create grid for F.grid_sample (normalized to [-1, 1])
        grid_x = 2*theta_normalized - 1
        grid_y = 2* (r/r_max) - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        self._pol2cart_grid = grid
        
    def pol2cart(self, pol_weights):
        # Small wrapper to convert polar weights to 
        # cartesian weights
        pol_weights = pol_weights.unsqueeze(0)
        
        cart_weights = torch.nn.functional.grid_sample(
            pol_weights, self.pol2cart_grid,
            'bilinear', padding_mode='zeros',
            align_corners=True
        )
        
        return cart_weights.permute(1, 0, 2, 3)
    
    def forward(self, imten, f_idx=[0], eval=False):
        # Convert polar weights to Cartesian weights
        self.weights = self.pol2cart(self.pol_weights)

        # Create grid
        grid_stretched = self.grid_rep/self.rfactors

        # Replicate image tensor
        imten_rep = torch.repeat_interleave(imten, self.nterms, 0)

        # Interpolate
        imten_stretch = torch.nn.functional.grid_sample(
            imten_rep, grid_stretched, mode='bilinear',
            align_corners=True
        )
        
        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms,
            padding='same').permute(1, 0, 2, 3)

        imten_blur_weighed = self.kweights*imten_blur_stack[None, ...]
                
        imten_focus = imten_blur_weighed[f_idx, 0, ...]*self.weights[f_idx, ...]
        imten_defocus = imten_blur_weighed[f_idx, 1:, ...].mean(1)*(1-self.weights[f_idx, ...])

        imten_blur = imten_focus + imten_defocus

        return imten_blur
    
class AffineConvRSAdapt(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image. This is the radially symmetric version where the
        weights are rank-1 in the polar domain.

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            nfocus: Number of focus settings. If None, 
                set it to nterms
    '''
    def __init__(self, nterms, dmax, rmax, imsize, nfocus=None):
        super().__init__()

        # Learnable parameters are rfactors, kernels, and weights

        H, W, nchan = imsize
        self.nterms = nterms
        self.imsize = imsize

        nterms_full = nterms + nterms//2 + nterms//4

        if nfocus is None:
            nfocus = nterms

        disk_sizes = torch.linspace(1, dmax, nterms)
        self.rfactors = torch.linspace(1, rmax, nterms_full)

        self.rfactors = self.rfactors[:, None, None, None]

        self.rfactors = torch.repeat_interleave(self.rfactors, 2, -1)

        # So we will have weights that are rank-1, which means we need
        # kernel weights
        self.kweights = torch.zeros(nfocus, nterms_full, 1, 1, 1)

        # Let us try initializing with stepped values
        kstep = nterms_full//nfocus
        for k_idx in range(0, nfocus):
            self.kweights[k_idx, k_idx*kstep:(k_idx+1)*kstep] = 1

        # Create convolution kernels -- smallest
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax),
                                torch.linspace(-1, 1, dmax),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms, 0)
        
        kernels1 = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/dmax))
        kernels1 = kernels1.float()
        
        kernsum = kernels1.sum(-1, keepdim=True).sum(-2, keepdim=True)
        
        self.kernels1 = kernels1/kernsum

        # Medium
        disk_sizes = torch.linspace(1, dmax*2, nterms//2)
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax*2 + 1),
                                torch.linspace(-1, 1, dmax*2 + 1),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms//2, 0)
        
        kernels2 = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/(2*dmax)))
        kernels2 = kernels2.float()
        
        kernsum = kernels2.sum(-1, keepdim=True).sum(-2, keepdim=True)
        
        self.kernels2 = kernels2/kernsum

        # Large
        disk_sizes = torch.linspace(1, dmax*4, nterms//4)
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax*4 + 1),
                                torch.linspace(-1, 1, dmax*4 + 1),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms//4, 0)
        
        kernels3 = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/(4*dmax)))
        kernels3 = kernels3.float()
        
        kernsum = kernels3.sum(-1, keepdim=True).sum(-2, keepdim=True)
        
        self.kernels3 = kernels3/kernsum
                
        # Create expansion grids
        X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                            torch.linspace(-1, 1, H),
                            indexing='xy')
        grid = torch.cat((X[None, ..., None], Y[None, ..., None]), -1)
        grid_rep = torch.repeat_interleave(grid, nterms_full, 0)
                
        # Create weights for each term
        Rmax = np.ceil(np.hypot(H, W)).astype(int)
        R = torch.linspace(0, np.sqrt(2), Rmax)[None, :]
        cR = torch.linspace(np.sqrt(2), 0, nfocus)[:, None]
        cS = (1/nfocus)**2
        
        self.pol_weights = torch.exp(-((R - cR)**2)/(cS))
        self.pol_weights = self.pol_weights[..., None]

        with torch.no_grad():
            self.pol_weights = self.pol_weights +\
                  0.1*torch.rand_like(self.pol_weights)
        
        # Initiate polar to cartesian conversion function
        self.init_pol2cart(imsize, Rmax)

        kernels1.random_(0, 1000)/1000
        kernels2.random_(0, 1000)/1000
        kernels3.random_(0, 1000)/1000

        # Make parameters learnable
        self.rfactors = ten2param(self.rfactors)
        self.kernels1 = ten2param(self.kernels1)
        self.kernels2 = ten2param(self.kernels2)
        self.kernels3 = ten2param(self.kernels3)
        self.pol_weights = ten2param(self.pol_weights)
        self.kweights = ten2param(self.kweights)

        # Register grid as a parameter but not learnable
        self.register_buffer('grid_rep', grid_rep)
        self.register_buffer('pol2cart_grid',
                              self._pol2cart_grid)
        
        
    def init_pol2cart(self, imsize, Rmax):
        height, width, _ = imsize

        # Center is center of the image, but in future
        # we can make this learnable
        center = (width/2, height/2)
        
        # We will assume width is larger than or equal to
        # height
        r_max = Rmax/2
        theta_max = 2*torch.pi
        
        # Create a grid of Cartesian coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height),
            torch.arange(width)
        )

        # Shift coordinates relative to the center
        x_shifted = x_coords - center[0]
        y_shifted = y_coords - center[1]
        
        # Convert to polar coordinates
        r = torch.sqrt(x_shifted**2 + y_shifted**2)
        theta = torch.atan2(y_shifted, x_shifted)
        
        theta_normalized = ((theta + torch.pi)/theta_max)
        
        # Create grid for F.grid_sample (normalized to [-1, 1])
        grid_x = 2*theta_normalized - 1
        grid_y = 2* (r/r_max) - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        self._pol2cart_grid = grid
        
    def pol2cart(self, pol_weights):
        # Small wrapper to convert polar weights to 
        # cartesian weights
        pol_weights = pol_weights.unsqueeze(0)
        
        cart_weights = torch.nn.functional.grid_sample(
            pol_weights, self.pol2cart_grid,
            'bilinear', padding_mode='zeros',
            align_corners=True
        )
        
        return cart_weights.permute(1, 0, 2, 3)
    
    def forward(self, imten, f_idx=[0]):
        # Convert polar weights to Cartesian weights
        self.weights = self.pol2cart(self.pol_weights)

        # Create grid
        grid_stretched = self.grid_rep/self.rfactors

        # Replicate image tensor
        nt1 = self.nterms
        nt2 = self.nterms//2
        nt3 = self.nterms//4

        imten_rep = torch.repeat_interleave(imten, nt1+nt2+nt3, 0)

        # Interpolate
        imten_stretch = torch.nn.functional.grid_sample(
            imten_rep, grid_stretched, mode='bilinear',
            align_corners=True
        )

        # Convolve
        imten_blur_stack1 = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3)[:, :nt1, ...],
            self.kernels1, groups=self.nterms,
            padding='same').permute(1, 0, 2, 3)
        
        imten_blur_stack2 = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3)[:, nt1:nt1+nt2, ...],
            self.kernels2, groups=self.nterms//2,
            padding='same').permute(1, 0, 2, 3)
        
        imten_blur_stack3 = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3)[:, nt1+nt2:, ...],
            self.kernels3, groups=self.nterms//4,
            padding='same').permute(1, 0, 2, 3)
        
        imten_blur_stack = torch.cat((imten_blur_stack1,
                                      imten_blur_stack2,
                                      imten_blur_stack3), 0)

        imten_blur_weighed = self.kweights*imten_blur_stack[None, ...]
                
        imten_focus = imten_blur_weighed[f_idx, 0, ...]*self.weights[f_idx, ...]
        imten_defocus = imten_blur_weighed[f_idx, 1:, ...].mean(1)*(1-self.weights[f_idx, ...])

        imten_blur = imten_focus + imten_defocus

        return imten_blur

class AffineConvSingle(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            R_focus: Where the image is focused. 0 means at the center of the image
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32):
        super().__init__()

        # Learnable parameters are rfactors, kernels, and weights

        H, W, nchan = imsize
        self.nterms = nterms
        self.ds_factor = ds_factor

        disk_sizes = torch.linspace(1, dmax, nterms)
        self.pad =  dmax//2
        
        self.affine = torch.zeros(nterms, 2, 3)
        self.affine[:, 0, 0] = torch.linspace(1, rmax, nterms)
        self.affine[:, 1, 1] = torch.linspace(1, rmax, nterms)

        # Create convolution kernels
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax),
                                torch.linspace(-1, 1, dmax),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms, 0)
        
        kernels = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/dmax))
        kernels = kernels.float()

        kernsum = kernels.sum(-1, keepdim=True).sum(-2, keepdim=True)
        self.kernels = kernels/kernsum
                
        # Create expansion grids
        X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                            torch.linspace(-1, 1, H),
                            indexing='xy')
        grid = torch.cat((X[None, ..., None], Y[None, ..., None]), -1)
        grid_rep = torch.repeat_interleave(grid, nterms, 0)

        X, Y = torch.meshgrid(torch.linspace(-1, 1, W//ds_factor),
                            torch.linspace(-1, 1, H//ds_factor),
                            indexing='xy')
                
        # Create weights for each term
        R = torch.hypot(X, Y)[None, ...]
        cR = torch.linspace(0, np.sqrt(2), nterms)[:, None, None]
        cS = (4/nterms)**2
        
        self.weights = torch.exp(-((R - cR)**2)/(cS))[:, None, ...]

        # Make parameters learnable
        self.affine = ten2param(self.affine)
        self.kernels = ten2param(self.kernels)
        self.weights = ten2param(self.weights)

        self.register_buffer('grid_rep', grid_rep)
    
    def forward(self, imten, eval=False):
        # Create grid
        nterms, H, W, _ = self.grid_rep.shape

        grid_mat = self.grid_rep.reshape(nterms, H*W, 2)
        grid_mul_only = torch.bmm(grid_mat, self.affine[:, :, :2])
        grid_stretched = grid_mul_only + self.affine[:, :, [2]].permute(0, 2, 1)
        grid_stretched = grid_stretched.reshape(nterms, H, W, 2)

        # Replicate image tensor
        imten_rep = torch.repeat_interleave(imten, self.nterms, 0)

        # Interpolate
        if eval:
            padding_mode = 'border'
        else:
            padding_mode = 'zeros'
        imten_stretch = torch.nn.functional.grid_sample(
            imten_rep, grid_stretched, mode='bilinear',
            align_corners=False, padding_mode=padding_mode
        )
        self.imten_stretch = imten_stretch

        imten_padded = torch.nn.functional.pad(
            imten_stretch.permute(1, 0, 2, 3),
            [self.pad]*4,
            mode='replicate'
        )
        
        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms)
        imten_blur_stack = imten_blur_stack.permute(1, 0, 2, 3)

        _, _, Hs, Ws = imten_blur_stack.shape

        # Upsample weights
        weights_up = torch.nn.functional.interpolate(self.weights,
                                                     #scale_factor=self.ds_factor,
                                                     size=(Hs, Ws),
                                                     mode='bilinear')
        
        imten_blur = (imten_blur_stack*weights_up).mean(0)

        return imten_blur
    
class DefINRConvSingle(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            R_focus: Where the image is focused. 0 means at the center of the image
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32, cps=32,use_checkpoint=False):
        super().__init__()

        # Learnable parameters are rfactors, kernels, and weights

        H, W, nchan = imsize
        self.nterms = nterms
        self.ds_factor = ds_factor
        self.use_checkpoint = use_checkpoint

        disk_sizes = torch.linspace(1, dmax, nterms)
        self.pad =  dmax//2
        self.im_size= torch.Size((H, W))

        self.grid_deformable = Grid(shape=self.im_size)
        self.bspline_cps = cps
        bspline_field =FreeFormDeformation(self.grid_deformable, stride=cps)
        bspline_kernel = bspline_field.kernel()
        bspline_grid = torch.zeros((nterms,*bspline_field.data_shape))
        self.bspline_interpolator = evaluate_cubic_bspline
        self.ch, self.cw = bspline_field.data_shape[-2:]
        contr_x, contr_y = torch.meshgrid(torch.linspace(-1, 1,self.cw),
                    torch.linspace(-1, 1, self.ch),
                    indexing='xy')
        control_grid = torch.cat((contr_x[None, ..., None], contr_y[None, ..., None]), -1)
        control_grid = control_grid.view(self.cw*self.ch,-1)


        # Create deformation field (real part -> x, imaginary part -> y)
        self.deformation_network = INR(in_features=2,hidden_features=512,hidden_layers=1,out_features=nterms,nonlinearity="gabor2d",scale=4,zero_init=True,first_omega_0=1, hidden_omega_0=1)
        
        # Create convolution kernels
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax),
                                torch.linspace(-1, 1, dmax),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms, 0)
        
        kernels = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/dmax))
        kernels = kernels.float()

        kernsum = kernels.sum(-1, keepdim=True).sum(-2, keepdim=True)
        self.kernels = kernels/kernsum

 
        # Create expansion grids
        X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                            torch.linspace(-1, 1, H),
                            indexing='xy')
        grid = torch.cat((X[None, ..., None], Y[None, ..., None]), -1)
        grid_rep = torch.repeat_interleave(grid, nterms, 0)

        X, Y = torch.meshgrid(torch.linspace(-1, 1, W//ds_factor),
                            torch.linspace(-1, 1, H//ds_factor),
                            indexing='xy')
                
        # Create weights for each term
        R = torch.hypot(X, Y)[None, ...]
        cR = torch.linspace(0, np.sqrt(2), nterms)[:, None, None]
        cS = (4/nterms)**2
        
        self.weights = torch.exp(-((R - cR)**2)/(cS))[:, None, ...]

        # Make parameters learnable
        #self.affine = ten2param(self.affine)
        self.kernels = ten2param(self.kernels)
        self.weights = ten2param(self.weights)
        self.register_buffer('grid_rep', grid_rep)
        self.register_buffer('bspline_kernel0',  bspline_kernel[0])
        self.register_buffer('bspline_kernel1',  bspline_kernel[1])
        self.register_buffer('control_grid', control_grid)


    
    def forward(self, imten, eval=False):
        # Create grid
        nterms, H, W, _ = self.grid_rep.shape

        grid_mat = self.grid_rep.reshape(nterms, H*W, 2)
        #grid_mul_only = torch.bmm(grid_mat, self.affine[:, :, :2])
        #grid_stretched = grid_mul_only + self.affine[:, :, [2]].permute(0, 2, 1)
        #grid_stretched = grid_stretched.reshape(nterms, H, W, 2)

        # Replicate image tensor
        imten_rep = torch.repeat_interleave(imten, self.nterms, 0)
        if self.use_checkpoint:
            deformation_x = checkpoint(self.deformation_network, self.control_grid, use_reentrant=False)
        else:
            deformation_x = self.deformation_network(self.control_grid)
        deformation_y = self.deformation_network.imag_output

        bspline_kernel = (self.bspline_kernel0,self.bspline_kernel1)
        deformation_field = torch.stack([deformation_x,deformation_y],-1)# ch*cw,32,2
        deformation_field = 0.4*torch.tanh(deformation_field)
        deformation_field = deformation_field.permute(1,2,0).view(self.nterms,2,self.ch,self.cw)
        self.deformation_field = deformation_field
        deformation_field = self.bspline_interpolator(deformation_field,stride=self.bspline_cps,shape= self.im_size,kernel=bspline_kernel)
        deformation_field = expv(deformation_field, steps=7)
        deformation_field = deformation_field.permute(0,2,3,1)
        grid_stretched = deformation_field + grid_mat.view(nterms,H,W,2)



        # Interpolate
        if eval:
            padding_mode = 'border'
        else:
            padding_mode = 'zeros'
        imten_stretch = torch.nn.functional.grid_sample(
            imten_rep, grid_stretched, mode='bilinear',
            align_corners=False, padding_mode=padding_mode
        )
        self.imten_stretch = imten_stretch

        imten_padded = torch.nn.functional.pad(
            imten_stretch.permute(1, 0, 2, 3),
            [self.pad]*4,
            mode='replicate'
        )
        
        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms)
        imten_blur_stack = imten_blur_stack.permute(1, 0, 2, 3)

        _, _, Hs, Ws = imten_blur_stack.shape

        # Upsample weights
        weights_up = torch.nn.functional.interpolate(self.weights,
                                                     #scale_factor=self.ds_factor,
                                                     size=(Hs, Ws),
                                                     mode='bilinear')
        
        imten_blur = (imten_blur_stack*weights_up).mean(0)

        return imten_blur




class RadialConvSingle(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            R_focus: Where the image is focused. 0 means at the center of the image
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32):
        super().__init__()

        # Learnable parameters are rfactors, kernels, and weights

        H, W, nchan = imsize
        self.nterms = nterms
        self.ds_factor = ds_factor

        disk_sizes = torch.linspace(1, dmax, nterms)
        self.pad =  dmax//2
        
        self.affine = torch.zeros(nterms, 2, 3)
        self.affine[:, 0, 0] = torch.linspace(1, rmax, nterms)
        self.affine[:, 1, 1] = torch.linspace(1, rmax, nterms)
        self.k1 = torch.zeros(nterms,1,1)
        self.center = torch.zeros(1,1,2)
        # Create convolution kernels
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax),
                                torch.linspace(-1, 1, dmax),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms, 0)
        
        kernels = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/dmax))
        kernels = kernels.float()

        kernsum = kernels.sum(-1, keepdim=True).sum(-2, keepdim=True)
        self.kernels = kernels/kernsum
                
        # Create expansion grids
        X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                            torch.linspace(-1, 1, H),
                            indexing='xy')
        grid = torch.cat((X[None, ..., None], Y[None, ..., None]), -1)
        grid_rep = torch.repeat_interleave(grid, nterms, 0)

        X, Y = torch.meshgrid(torch.linspace(-1, 1, W//ds_factor),
                            torch.linspace(-1, 1, H//ds_factor),
                            indexing='xy')
                
        # Create weights for each term
        R = torch.hypot(X, Y)[None, ...]
        cR = torch.linspace(0, np.sqrt(2), nterms)[:, None, None]
        cS = (4/nterms)**2
        
        self.weights = torch.exp(-((R - cR)**2)/(cS))[:, None, ...]

        # Make parameters learnable
        self.affine = ten2param(self.affine)
        self.kernels = ten2param(self.kernels)
        self.weights = ten2param(self.weights)
        self.center = ten2param(self.center)
        self.k1 = ten2param(self.k1)
        self.register_buffer('grid_rep', grid_rep)
    
    def forward(self, imten, eval=False):
        # Create grid
        nterms, H, W, _ = self.grid_rep.shape

        grid_mat = self.grid_rep.reshape(nterms, H*W, 2)
        grid_mul_only = torch.bmm(grid_mat, self.affine[:, :, :2])
        grid_stretched = grid_mul_only + self.affine[:, :, [2]].permute(0, 2, 1)
        # Add radial distortion model with assumption that optical axis align with image center
        radial_distance = torch.sum((grid_stretched - self.center)**2,dim=-1,keepdim=True)
        grid_stretched =self.center + grid_stretched * (1.0 + self.k1 * radial_distance)
        grid_stretched = grid_stretched.reshape(nterms, H, W, 2)

        # Replicate image tensor
        imten_rep = torch.repeat_interleave(imten, self.nterms, 0)

        # Interpolate
        if eval:
            padding_mode = 'border'
        else:
            padding_mode = 'zeros'
        imten_stretch = torch.nn.functional.grid_sample(
            imten_rep, grid_stretched, mode='bilinear',
            align_corners=False, padding_mode=padding_mode
        )
        self.imten_stretch = imten_stretch

        imten_padded = torch.nn.functional.pad(
            imten_stretch.permute(1, 0, 2, 3),
            [self.pad]*4,
            mode='replicate'
        )
        
        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms)
        imten_blur_stack = imten_blur_stack.permute(1, 0, 2, 3)

        _, _, Hs, Ws = imten_blur_stack.shape

        # Upsample weights
        weights_up = torch.nn.functional.interpolate(self.weights,
                                                     #scale_factor=self.ds_factor,
                                                     size=(Hs, Ws),
                                                     mode='bilinear')
        
        imten_blur = (imten_blur_stack*weights_up).mean(0)

        return imten_blur


class BSplineConvSingle(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            R_focus: Where the image is focused. 0 means at the center of the image
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32,cps_coarse=128,cps_middle=64,cps_dense=32):
        super().__init__()

        # Learnable parameters are rfactors, kernels, and weights

        H, W, nchan = imsize
        self.nterms = nterms
        self.ds_factor = ds_factor

        disk_sizes = torch.linspace(1, dmax, nterms)
        self.pad =  dmax//2
        self.cps_coarse = cps_coarse
        self.cps_middle = cps_middle
        self.cps_dense = cps_dense
        self.im_size= torch.Size((H, W))
        #coarse_grid = create_grid(cps_coarse,imsize,nterms)
        #dense_grid = create_grid(cps_dense,imsize,nterms)

        self.grid_deformable = Grid(shape=self.im_size)
        coarse_field =FreeFormDeformation(self.grid_deformable, stride=cps_coarse)
        middle_field = FreeFormDeformation(self.grid_deformable,stride=cps_middle)
        dense_field=FreeFormDeformation(self.grid_deformable, stride=cps_dense)
        kernel_coarse = coarse_field.kernel()
        kernel_middle= middle_field.kernel()
        kernel_dense =  dense_field.kernel()
        coarse_grid = torch.zeros((nterms,*coarse_field.data_shape))
        middle_grid = torch.zeros((nterms,*middle_field.data_shape))
        dense_grid = torch.zeros((nterms,*dense_field.data_shape))
        self.transformation = evaluate_cubic_bspline

        #self.affine = torch.zeros(nterms, 2, 3)
        #self.affine[:, 0, 0] = torch.linspace(1, rmax, nterms)
        #self.affine[:, 1, 1] = torch.linspace(1, rmax, nterms)

        # Create convolution kernels
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax),
                                torch.linspace(-1, 1, dmax),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms, 0)
        
        kernels = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/dmax))
        kernels = kernels.float()

        kernsum = kernels.sum(-1, keepdim=True).sum(-2, keepdim=True)
        self.kernels = kernels/kernsum
                
        # Create expansion grids
        X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                            torch.linspace(-1, 1, H),
                            indexing='xy')
        grid = torch.cat((X[None, ..., None], Y[None, ..., None]), -1)
        grid_rep = torch.repeat_interleave(grid, nterms, 0)

        X, Y = torch.meshgrid(torch.linspace(-1, 1, W//ds_factor),
                            torch.linspace(-1, 1, H//ds_factor),
                            indexing='xy')
                
        # Create weights for each term
        R = torch.hypot(X, Y)[None, ...]
        cR = torch.linspace(0, np.sqrt(2), nterms)[:, None, None]
        cS = (4/nterms)**2
        
        self.weights = torch.exp(-((R - cR)**2)/(cS))[:, None, ...]

        # Make parameters learnable
        #self.affine = ten2param(self.affine)
        self.kernels = ten2param(self.kernels)
        self.weights = ten2param(self.weights)
        self.coarse_grid = ten2param(coarse_grid)
        self.middle_grid = ten2param(middle_grid)
        self.dense_grid= ten2param(dense_grid)


        self.register_buffer('grid_rep', grid_rep)
        self.register_buffer('kernel_coarse0', kernel_coarse[0])
        self.register_buffer('kernel_coarse1', kernel_coarse[1])
        self.register_buffer("kernel_middle0", kernel_middle[0])
        self.register_buffer("kernel_middle1", kernel_middle[1])
        self.register_buffer('kernel_dense0',  kernel_dense[0])
        self.register_buffer('kernel_dense1',  kernel_dense[1])
    
    def forward(self, imten, eval=False):
        # Create grid
        nterms, H, W, _ = self.grid_rep.shape

        grid_mat = self.grid_rep.reshape(nterms, H*W, 2)
        #grid_mul_only = torch.bmm(grid_mat, self.affine[:, :, :2])
        #grid_stretched = grid_mul_only + self.affine[:, :, [2]].permute(0, 2, 1)
        #grid_stretched = grid_stretched.reshape(nterms, H, W, 2)

        # Replicate image tensor
        imten_rep = torch.repeat_interleave(imten, self.nterms, 0)

        bspline_coarse = (self.kernel_coarse0,self.kernel_coarse1)
        bspline_middle = (self.kernel_middle0,self.kernel_middle1)
        bspline_dense = (self.kernel_dense0,self.kernel_dense1)

        coarse_sparse_velocities = 0.5 * torch.tanh(self.coarse_grid)
        coarse_velocities = self.transformation(coarse_sparse_velocities, stride=self.cps_coarse, shape=self.im_size, kernel=bspline_coarse)
        coarse_displacement = expv(coarse_velocities,steps=7)

        middle_sparse_velocities = 0.2*torch.tanh(self.middle_grid)
        middle_velocities =  self.transformation(middle_sparse_velocities,stride=self.cps_middle, shape=self.im_size, kernel=bspline_middle)
        middle_displacement = expv(middle_velocities,steps=7)

        dense_sparse_velocities = 0.1*torch.tanh(self.dense_grid)
        dense_velocities = self.transformation(dense_sparse_velocities, stride=self.cps_dense, shape=self.im_size, kernel=bspline_dense)
        dense_displacement = expv(dense_velocities,steps=7)



        self.total_displacement = coarse_displacement + dense_displacement + middle_displacement
        grid_stretched = self.total_displacement.permute(0, 2, 3, 1) + grid_mat.view(nterms,H,W,2)



        # Interpolate
        if eval:
            padding_mode = 'border'
        else:
            padding_mode = 'zeros'
        imten_stretch = torch.nn.functional.grid_sample(
            imten_rep, grid_stretched, mode='bilinear',
            align_corners=False, padding_mode=padding_mode
        )
        self.imten_stretch = imten_stretch

        imten_padded = torch.nn.functional.pad(
            imten_stretch.permute(1, 0, 2, 3),
            [self.pad]*4,
            mode='replicate'
        )
        
        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms)
        imten_blur_stack = imten_blur_stack.permute(1, 0, 2, 3)

        _, _, Hs, Ws = imten_blur_stack.shape

        # Upsample weights
        weights_up = torch.nn.functional.interpolate(self.weights,
                                                     #scale_factor=self.ds_factor,
                                                     size=(Hs, Ws),
                                                     mode='bilinear')
        
        imten_blur = (imten_blur_stack*weights_up).mean(0)

        return imten_blur


class AffineConvMulti(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
                NOTE: Here, nchan is the number of focus images, and not wavelength
                    channels!
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32):
        super().__init__()
        nchan = imsize[2]
        self.conv_ops = nn.ModuleList([AffineConvSingle(nterms, dmax,
                                                        rmax, imsize,
                                                        ds_factor) for _ in range(nchan)])
    
    def forward(self, image, eval=False):
        output_list = [conv_op(image, eval)[None, ...] for conv_op in self.conv_ops]

        klist = [conv_op.kernels for conv_op in self.conv_ops]
        self.kernels = torch.cat(klist, dim=0)

        wlist = [conv_op.weights for conv_op in self.conv_ops]
        self.weights = torch.cat(wlist, dim=0)

        return torch.cat(output_list, dim=0)
    

class RadialConvMulti(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
                NOTE: Here, nchan is the number of focus images, and not wavelength
                    channels!
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32):
        super().__init__()
        nchan = imsize[2]
        self.conv_ops = nn.ModuleList([RadialConvSingle(nterms, dmax,
                                                        rmax, imsize,
                                                        ds_factor) for _ in range(nchan)])
    
    def forward(self, image, eval=False):
        output_list = [conv_op(image, eval)[None, ...] for conv_op in self.conv_ops]

        klist = [conv_op.kernels for conv_op in self.conv_ops]
        self.kernels = torch.cat(klist, dim=0)

        wlist = [conv_op.weights for conv_op in self.conv_ops]
        self.weights = torch.cat(wlist, dim=0)

        return torch.cat(output_list, dim=0)
    
class BSplineConvMulti(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
                NOTE: Here, nchan is the number of focus images, and not wavelength
                    channels!
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize,stride_coarse=256,stride_middle=128,stride_dense=64, ds_factor=32):
        super().__init__()
        nchan = imsize[2]
        self.conv_ops = nn.ModuleList([BSplineConvSingle(nterms, dmax,
                                                        rmax, imsize,cps_coarse=stride_coarse,cps_middle=stride_middle,cps_dense=stride_dense,
                                                        ds_factor=ds_factor) for _ in range(nchan)])
    
    def forward(self, image, eval=False):
        output_list = [conv_op(image, eval)[None, ...] for conv_op in self.conv_ops]

        klist = [conv_op.kernels for conv_op in self.conv_ops]
        self.kernels = torch.cat(klist, dim=0)

        wlist = [conv_op.weights for conv_op in self.conv_ops]
        self.weights = torch.cat(wlist, dim=0)

        return torch.cat(output_list, dim=0)
    
class DefINRConvMulti(torch.nn.Module):
    '''
        Spatially varying convolution with Affine transform of image

        Parameters:
            nterms: Number of stretches/affine transforms to perform on images
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
                NOTE: Here, nchan is the number of focus images, and not wavelength
                    channels!
            ds_factor: Downsampling factor for weights, since they should be slowly varying
            use_checkpoint: If True, use gradient checkpointing on INR to save memory
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32, use_checkpoint=False,cps=32):
        super().__init__()
        nchan = imsize[2]
        self.conv_ops = nn.ModuleList([DefINRConvSingle(nterms, dmax, rmax, imsize,ds_factor=ds_factor, cps=cps,use_checkpoint=use_checkpoint) for _ in range(nchan)])
    
    def forward(self, image, eval=False):
        output_list = [conv_op(image, eval)[None, ...] for conv_op in self.conv_ops]

        klist = [conv_op.kernels for conv_op in self.conv_ops]
        self.kernels = torch.cat(klist, dim=0)

        wlist = [conv_op.weights for conv_op in self.conv_ops]
        self.weights = torch.cat(wlist, dim=0)

        return torch.cat(output_list, dim=0)
    

    


class EigenPSF(torch.nn.Module):
    '''
        Spatially varying convolution with Eigen-ish PSF model

        Parameters:
            nterms: Number of terms in the decomposition
            dmax: Kernel size
            rmax: Not functional. (compatibility term)
            imsize: 3-tuple of image size (H, W, nchan)
            R_focus: Not functional. (compatibility term)
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32):
        super().__init__()

        # Learnable parameters are rfactors, kernels, and weights

        H, W, nchan = imsize
        self.nterms = nterms
        self.ds_factor = ds_factor

        disk_sizes = torch.linspace(1, dmax, nterms)

        self.pad = dmax//2

        # Create convolution kernels
        kX, kY = torch.meshgrid(torch.linspace(-1, 1, dmax),
                                torch.linspace(-1, 1, dmax),
                                indexing='xy')
        kR = torch.hypot(kX, kY)[None, None, ...]
        kR = torch.repeat_interleave(kR, nterms, 0)
        
        kernels = 1 - torch.sigmoid(100*(kR - disk_sizes[:, None, None, None]/dmax))
        kernels = kernels.float()
        
        kernsum = kernels.sum(-1, keepdim=True).sum(-2, keepdim=True)
        
        self.kernels = kernels/kernsum
                
        X, Y = torch.meshgrid(torch.linspace(-1, 1, W//ds_factor),
                            torch.linspace(-1, 1, H//ds_factor),
                            indexing='xy')
                
        # Create weights for each term
        R = torch.hypot(X, Y)[None, ...]
        cR = torch.linspace(np.sqrt(2), 0, nterms)[:, None, None]
        cS = (1/nterms)**2
        
        self.weights = torch.exp(-((R - cR)**2)/(cS))[:, None, ...]

        # Make parameters learnable
        self.kernels = ten2param(self.kernels)
        self.weights = ten2param(self.weights)

    def forward(self, imten):
        # Replicate image tensor
        imten_rep = torch.repeat_interleave(imten, self.nterms, 0)

        # Convolve
        imten_padded = torch.nn.functional.pad(
            imten_rep.permute(1, 0, 2, 3),
            [self.pad]*4,
            mode='replicate'
        )
        
        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_rep.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms,)
        imten_blur_stack = imten_blur_stack.permute(1, 0, 2, 3)

        _, _, Hs, Ws = imten_blur_stack.shape
        
        # Upsample weights
        weights_up = torch.nn.functional.interpolate(self.weights,
                                                     #scale_factor=self.ds_factor,
                                                     size=(Hs, Ws),
                                                     mode='bilinear')
        
        imten_blur = (imten_blur_stack*weights_up).mean(0)

        return imten_blur
    
class EigenPSFMulti(torch.nn.Module):
    '''
        Spatially varying convolution with Eigen-ish PSF model

        Parameters:
            nterms: Number of terms in the decomposition
            dmax: Kernel size
            rmax: Not functional. (compatibility term)
            imsize: 3-tuple of image size (H, W, nchan)
            R_focus: Not functional. (compatibility term)
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, imsize, ds_factor=32):
        super().__init__()
        rmax = dmax
        nchan = imsize[2]
        self.conv_ops = nn.ModuleList([EigenPSF(nterms, dmax,
                                                rmax, imsize,
                                                ds_factor=ds_factor) for _ in range(nchan)])
    
    def forward(self, image, eval=False):
        output_list = [conv_op(image)[None, ...] for conv_op in self.conv_ops]

        klist = [conv_op.kernels for conv_op in self.conv_ops]
        self.kernels = torch.cat(klist, dim=0)

        wlist = [conv_op.weights for conv_op in self.conv_ops]
        self.weights = torch.cat(wlist, dim=0)

        return torch.cat(output_list, dim=0)
    
class PatchWiseConv(nn.Module):
    def __init__(self, img_shape=(512, 512), kernel_size=31, patch_size=64, stride=32):
        """
        Args:
            img_shape (tuple): Fixed (Height, Width) of the input image.
            kernel_size (int): Size of the learnable kernel (must be odd).
            patch_size (int): Size of the output patch.
            stride (int): Step size for the sliding window.
        """
        super().__init__()
        self.img_h, self.img_w = img_shape
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        
        # 1. Calculate Padding and Grid Dimensions
        # Source patch must be larger than target patch to allow valid convolution
        self.src_size = patch_size + kernel_size - 1
        self.pad_size = self.src_size // 2
        
        # Calculate how many patches fit in the height and width
        # (This logic mirrors how F.unfold calculates output size)
        # Padded dims
        pad_h = self.img_h + 2 * self.pad_size
        pad_w = self.img_w + 2 * self.pad_size
        
        self.n_h = (pad_h - self.src_size) // stride + 1
        self.n_w = (pad_w - self.src_size) // stride + 1
        self.n_patches = self.n_h * self.n_w
        
        # 2. Define Learnable Kernels
        # Shape: (Num_Patches, 1, K, K)
        # We assume the same blur applies to all channels (RGB), so 1 input channel.
        # If you want channel-specific blur, change dim 1 to 'channels'.
        self.kernels = nn.Parameter(torch.randn(self.n_patches, 1, kernel_size, kernel_size))
        
        # Initialize kernels to something reasonable (e.g., Identity/Delta spike)
        # so training starts from a sharp image, not random noise.
        nn.init.zeros_(self.kernels)
        center = kernel_size // 2
        with torch.no_grad():
            self.kernels[:, :, center, center] = 0.1
        
        # 3. Precompute Fixed Tensors (Window and Normalization Map)
        self.register_buffer('window', self._create_window(patch_size))
        self.register_buffer('norm_map', self._precompute_norm_map())

    def _create_window(self, size):
        """Creates a 2D Hanning window."""
        x = torch.hann_window(size)
        window_2d = torch.outer(x, x)
        return window_2d.view(1, 1, size, size)

    def _precompute_norm_map(self):
        """
        Precomputes the overlap-add normalization divisor.
        Since grid and window are fixed, this map is constant.
        """
        # Create a dummy "all ones" tensor to represent weights
        # Shape: (1, 1, n_patches, p*p)
        ones_flat = torch.ones(1, 1, self.n_patches, self.patch_size**2)
        
        # Apply window weights
        ones_flat = ones_flat * self.window.view(1, -1)

        # Reshape for folding: (1, p*p, n_patches)
        ones_flat = ones_flat[0].permute(0, 2, 1)
        
        # Calculate Fold Output Size
        out_h = (self.n_h - 1) * self.stride + self.patch_size
        out_w = (self.n_w - 1) * self.stride + self.patch_size
        
        # Fold to create the weight map
        norm_map = torch.nn.functional.fold(
            ones_flat, 
            output_size=(out_h, out_w), 
            kernel_size=self.patch_size, 
            stride=self.stride
        )
        
        # Crop to actual image size
        ch, cw = norm_map.shape[2]//2, norm_map.shape[3]//2
        norm_map = norm_map[:, :, 
                            ch - self.img_h//2 : ch + self.img_h//2, 
                            cw - self.img_w//2 : cw + self.img_w//2]
        
        # Add epsilon to prevent division by zero
        return norm_map + 1e-8

    def forward(self, image, eval=True):
        """
        Args:
            image: (B, C, H, W)
        """
        B, C, H, W = image.shape
        
        # 1. Pad and Unfold
        img_pad = F.pad(image, (self.pad_size,)*4, mode='reflect')
        patches = F.unfold(img_pad, kernel_size=self.src_size, stride=self.stride)
        # patches shape: (B, C * src_size^2, n_patches)
        
        # 2. Reshape for Grouped Conv
        # Target: (1, B * n_patches * C, src_size, src_size)
        patches = patches.view(B, C, self.src_size, self.src_size, self.n_patches)
        patches = patches.permute(0, 4, 1, 2, 3) # (B, n_patches, C, src, src)
        patches = patches.reshape(1, B * self.n_patches * C, self.src_size, self.src_size)
        
        # 3. Prepare Kernels
        # self.kernels shape: (n_patches, 1, k, k)
        # We need to repeat them for Batch size and Channels
        # Target dim 0 must match groups: B * n_patches * C
        
        # (n_patches, 1, k, k) -> (B, n_patches, C, 1, k, k)
        k_rep = self.kernels.unsqueeze(0).unsqueeze(2) # Add B and C dims
        k_rep = k_rep.expand(B, -1, C, -1, -1, -1)     # Expand
        k_rep = k_rep.reshape(B * self.n_patches * C, 1, self.kernel_size, self.kernel_size)
        
        # 4. Perform Convolution
        # Groups = B * n_patches * C ensures every patch gets its specific kernel
        convolved = F.conv2d(patches, k_rep, groups=B * self.n_patches * C)
        
        # 5. Window and Fold
        convolved = convolved * self.window
        
        # Reshape: (1, B*N*C, p, p) -> (B, C, p*p, N) -> (B, C*p*p, N)
        convolved_flat = convolved.view(B, self.n_patches, C, -1)
        convolved_flat = convolved_flat.permute(0, 2, 3, 1) # (B, C, p*p, N)
        convolved_flat = convolved_flat.reshape(B, C * self.patch_size**2, self.n_patches)
        
        # Fold size matches the precomputed norm map size
        out_h = (self.n_h - 1) * self.stride + self.patch_size
        out_w = (self.n_w - 1) * self.stride + self.patch_size
        
        out_image = F.fold(
            convolved_flat, 
            output_size=(out_h, out_w), 
            kernel_size=self.patch_size, 
            stride=self.stride
        )
        
        # 6. Crop and Normalize
        ch, cw = out_image.shape[2]//2, out_image.shape[3]//2
        out_crop = out_image[:, :, 
                             ch - self.img_h//2 : ch + self.img_h//2, 
                             cw - self.img_w//2 : cw + self.img_w//2]
        
        return out_crop / self.norm_map
    
class PatchWiseConvMulti(nn.Module):
    def __init__(self, img_shape=(512, 512), kernel_size=31, patch_size=64, stride=32, nchan=1):
        """
        Args:
            img_shape (tuple): Fixed (Height, Width) of the input image.
            kernel_size (int): Size of the learnable kernel (must be odd).
            patch_size (int): Size of the output patch.
            stride (int): Step size for the sliding window.
            nchan (int): Number of focus stack images
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_ops = nn.ModuleList([PatchWiseConv(img_shape,
                                                     kernel_size,
                                                     patch_size,
                                                     stride) for _ in range(nchan)])
    
    def forward(self, image, eval=True):
        pad = self.kernel_size//2
        output_list = [conv_op(image)[..., pad:-pad, pad:-pad] for conv_op in self.conv_ops]

        klist = [conv_op.kernels for conv_op in self.conv_ops]
        self.kernels = torch.cat(klist, dim=0)

        return torch.cat(output_list, dim=0)

def ten2param(ten):
    ten = torch.autograd.Variable(ten, requires_grad=True)
    ten = torch.nn.Parameter(ten)

    return ten

@torch.no_grad()
def edgetaper(img, ksize, conv_op):
    """
    Refines image boundaries to reduce ringing in FFT-based deconvolution.
    Port of MATLAB's edgetaper function.
    
    Args:
        img (Tensor): Image (B, C, H, W)
        kernel (Tensor): Blur Kernel (H_k, W_k)
    """
    h, w = img.shape[-2:]
    n_taper = ksize

    # 1. Create Weighting Mask (alpha)
    # Ramps up from 0 at edge to 1 in center
    alpha_h = torch.ones(h, device=img.device)
    alpha_w = torch.ones(w, device=img.device)
    
    # Create 1D Hanning-like windows
    taper_h = torch.linspace(0, 1, n_taper, device=img.device)
    taper_w = torch.linspace(0, 1, n_taper, device=img.device)
    
    alpha_h[:n_taper] = taper_h
    alpha_h[-n_taper:] = taper_h.flip(0)
    alpha_w[:n_taper] = taper_w
    alpha_w[-n_taper:] = taper_w.flip(0)
    
    # Outer Product to create 2D mask
    alpha = alpha_h.unsqueeze(1) * alpha_w.unsqueeze(0) # (H, W)
    alpha = alpha.view(1, 1, h, w) # (1, 1, H, W)

    # 2. Blur the image with the known kernel
    blurred_img = conv_op(img)/conv_op(torch.ones_like(img))

    # 3. Blend: Edges come from Blurred Image, Center from Original
    # This matches the frequency content at the boundaries to the kernel
    return alpha * img + (1 - alpha) * blurred_img

@torch.no_grad()
def get_contrast_map(conv_op, imsize, ksize, device='cpu'):
    '''
    Get contrast across the full field of view
    
    :param conv_op: Description
    :param imsize: (H, W) image size
    :param ksize: scalar kernel size for smoothening
    '''
    imones = torch.ones((1, 1, imsize[0], imsize[1]), device=device)
    
    weights = conv_op(imones)

    focus_mag = 0

    for idx in range(100):
        focus_op = conv_op(torch.rand_like(imones))

        focus_op = focus_op/weights
        if focus_op.ndim == 3:
            focus_op = focus_op[None, ...]
        gradients = kornia.filters.spatial_gradient(focus_op)
        grad_x = gradients[:, :, 0, :, :]
        grad_y = gradients[:, :, 1, :, :]

        grad_mag = torch.sqrt(grad_x.abs()**2 + grad_y.abs()**2)
        focus_mag = focus_mag + kornia.filters.gaussian_blur2d(grad_mag,
                                                               2*(ksize//2) + 1,
                                                               (ksize/3, ksize/3))

    contrast_map = (focus_mag - focus_mag.min())/(focus_mag.max() - focus_mag.min())

    return contrast_map