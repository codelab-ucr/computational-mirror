#!/usr/bin/env python

'''
    Spatially varying PSF models: SeidelConv (affine warps), the B-spline model (smooth B-spline
    deformations), DefINR (deformations from an implicit neural network), EigenPSF and PatchWiseConv. The
    *Multi classes hold one model per image of the focal stack.
'''

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from deepali.core import Grid
from deepali.core.bspline import evaluate_cubic_bspline
from deepali.core.flow import expv
from deepali.spatial.bspline import FreeFormDeformation

from inr_models import INR


class SeidelConvSingle(torch.nn.Module):
    '''
        SeidelConv: spatially varying convolution as a sum of terms, each an affine warp of the image, a
        convolution kernel and a smooth weight map

        Parameters:
            nterms: Number of terms (each a warp, a kernel and a weight map)
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32):
        super().__init__()

        # Learnable parameters: affine matrices, kernels and weight maps

        H, W, _ = imsize
        self.nterms = nterms

        disk_sizes = torch.linspace(1, dmax, nterms)
        
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

        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms)
        imten_blur_stack = imten_blur_stack.permute(1, 0, 2, 3)

        _, _, Hs, Ws = imten_blur_stack.shape

        # Upsample weights
        weights_up = torch.nn.functional.interpolate(self.weights,
                                                     size=(Hs, Ws),
                                                     mode='bilinear')
        
        imten_blur = (imten_blur_stack*weights_up).mean(0)

        return imten_blur
    
class DefINRConvSingle(torch.nn.Module):
    '''
        DefINR: spatially varying convolution as a sum of terms, each warped by a deformation that an implicit
        neural network (Gabor INR) gives on a B-spline control grid, a convolution kernel and a smooth weight map

        Parameters:
            nterms: Number of terms (each a warp, a kernel and a weight map)
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32, cps=32,use_checkpoint=False):
        super().__init__()

        # Learnable parameters: the deformation network, kernels and weight maps

        H, W, _ = imsize
        self.nterms = nterms
        self.use_checkpoint = use_checkpoint

        disk_sizes = torch.linspace(1, dmax, nterms)
        self.im_size= torch.Size((H, W))

        self.grid_deformable = Grid(shape=self.im_size)
        self.bspline_cps = cps
        bspline_field =FreeFormDeformation(self.grid_deformable, stride=cps)
        bspline_kernel = bspline_field.kernel()
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

        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms)
        imten_blur_stack = imten_blur_stack.permute(1, 0, 2, 3)

        _, _, Hs, Ws = imten_blur_stack.shape

        # Upsample weights
        weights_up = torch.nn.functional.interpolate(self.weights,
                                                     size=(Hs, Ws),
                                                     mode='bilinear')
        
        imten_blur = (imten_blur_stack*weights_up).mean(0)

        return imten_blur


class BSplineConvSingle(torch.nn.Module):
    '''
        B-spline model: spatially varying convolution as a sum of terms, each a smooth B-spline deformation of the
        image (coarse, middle and dense control grids), a convolution kernel and a smooth weight map

        Parameters:
            nterms: Number of terms (each a warp, a kernel and a weight map)
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32, cps_coarse=None, cps_middle=None, cps_dense=None):
        super().__init__()

        # Learnable parameters: coarse, middle and dense B-spline control grids, kernels and weight maps

        H, W, _ = imsize
        self.nterms = nterms

        # Control-point spacing (x, y) in pixels: 2 x 2, 8 x 8 and 32 x 32 grid cells over the frame, as in the
        # published model ((320, 256), (80, 64), (20, 16) for 640 x 512 images)
        cps_coarse = (W//2, H//2) if cps_coarse is None else cps_coarse
        cps_middle = (W//8, H//8) if cps_middle is None else cps_middle
        cps_dense = (W//32, H//32) if cps_dense is None else cps_dense

        disk_sizes = torch.linspace(1, dmax, nterms)
        self.cps_coarse = cps_coarse
        self.cps_middle = cps_middle
        self.cps_dense = cps_dense
        self.im_size= torch.Size((H, W))

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


        total_displacement = coarse_displacement + dense_displacement + middle_displacement
        grid_stretched = total_displacement.permute(0, 2, 3, 1) + grid_mat.view(nterms,H,W,2)


        # Interpolate
        if eval:
            padding_mode = 'border'
        else:
            padding_mode = 'zeros'
        imten_stretch = torch.nn.functional.grid_sample(
            imten_rep, grid_stretched, mode='bilinear',
            align_corners=False, padding_mode=padding_mode
        )

        # Convolve
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_stretch.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms)
        imten_blur_stack = imten_blur_stack.permute(1, 0, 2, 3)

        _, _, Hs, Ws = imten_blur_stack.shape

        # Upsample weights
        weights_up = torch.nn.functional.interpolate(self.weights,
                                                     size=(Hs, Ws),
                                                     mode='bilinear')
        
        imten_blur = (imten_blur_stack*weights_up).mean(0)

        return imten_blur


class SeidelConvMulti(torch.nn.Module):
    '''
        SeidelConv for a focal stack: one SeidelConvSingle per focus image

        Parameters:
            nterms: Number of terms (each a warp, a kernel and a weight map)
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
        self.conv_ops = nn.ModuleList([SeidelConvSingle(nterms, dmax,
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
        B-spline model for a focal stack: one BSplineConvSingle per focus image

        Parameters:
            nterms: Number of terms (each a warp, a kernel and a weight map)
            dmax: Maximum size of PSF disk
            rmax: Maximum rescaling factor
            imsize: 3-tuple of image size (H, W, nchan)
                NOTE: Here, nchan is the number of focus images, and not wavelength
                    channels!
            stride_coarse, stride_middle, stride_dense: control-point spacing (x, y) in pixels of the three
                B-spline grids; default (W/2, H/2), (W/8, H/8), (W/32, H/32) as in the published model
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, stride_coarse=None, stride_middle=None, stride_dense=None, ds_factor=32):
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
        DefINR for a focal stack: one DefINRConvSingle per focus image

        Parameters:
            nterms: Number of terms (each a warp, a kernel and a weight map)
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
            ds_factor: Downsampling factor for weights, since they should be slowly varying
    '''
    def __init__(self, nterms, dmax, rmax, imsize, ds_factor=32):
        super().__init__()

        # Learnable parameters: kernels and weight maps

        H, W, _ = imsize
        self.nterms = nterms

        disk_sizes = torch.linspace(1, dmax, nterms)

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
        imten_blur_stack = torch.nn.functional.conv2d(
            imten_rep.permute(1, 0, 2, 3),
            self.kernels, groups=self.nterms,)
        imten_blur_stack = imten_blur_stack.permute(1, 0, 2, 3)

        _, _, Hs, Ws = imten_blur_stack.shape
        
        # Upsample weights
        weights_up = torch.nn.functional.interpolate(self.weights,
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
        
        pad = self.kernel_size//2
        return (out_crop / self.norm_map)[..., pad:-pad, pad:-pad]
    
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
        output_list = [conv_op(image) for conv_op in self.conv_ops]

        klist = [conv_op.kernels for conv_op in self.conv_ops]
        self.kernels = torch.cat(klist, dim=0)

        return torch.cat(output_list, dim=0)

def ten2param(ten):
    ten = torch.autograd.Variable(ten, requires_grad=True)
    ten = torch.nn.Parameter(ten)

    return ten