#!/usr/bin/env python

import os
import sys
import tqdm
import pdb
import importlib
import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import numpy as np
import torch
from scipy import interpolate
from skimage import color
from torch import nn

from scipy.sparse import linalg
from scipy import io
from scipy import ndimage

import torchvision
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import cv2

plt.gray()

sys.path.append('../modules')

import losses
import utils
import linear
import pnpsolver

utils = importlib.reload(utils)
losses = importlib.reload(losses)
linear = importlib.reload(linear)
pnpsolver = importlib.reload(pnpsolver)

if __name__ == '__main__':
    expname = 'iso_reschart'
    expfolder = '2.17.validation'
    calibfolder = '2.19.calib'
    root = '/home/akayabasi/broadband_mirror/real/pyprocessed'
    mode = 'multi'
    psfmodel = 'bspline'
    nterms = 31
    dmax = 11
    nfocus = 3
    rmax = 1.1
    ds_factor = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Reconstruction method
    rec_method = 'pnp'
    pnp_weights_path = '../modules/drunet_gray.pth'
    n_iters = 100
    cg_iters = 10
    lamb = 1e-5
    step_size = 1.9e-3
    sigma_min = 1e-4
    sigma_max = 5e-2

    params_path = 'models/%s/%s%d_%s_%d_%d_params.pth'%(calibfolder, mode, nfocus,
                                                        psfmodel, nterms, dmax)

    # If we are loading existing weights, we need image size
    params = torch.load(params_path)
    
    data = io.loadmat('%s/%s/%s_multi%d.mat'%(root,
                                              expfolder, 
                                              expname,
                                              nfocus))
    im = data['imgt'].astype(np.float32)[..., np.newaxis]**2
    meas = data['imraw'].astype(np.float32)

    meas -= meas.min()

    H, W, nchan = im.shape
            
    imten = torch.tensor(im).permute([2, 0, 1])[None, ...].to(device)

    H, W, nchan = im.shape
    if psfmodel == 'affineplus':
        conv_op = linear.AffineConvMulti(nterms, dmax, rmax, (H, W, nfocus),
                                         ds_factor=ds_factor).to(device)
    elif psfmodel == "bspline":
         conv_op = linear.BSplineConvMulti(nterms,dmax,rmax,(H,W,nfocus),stride_coarse=256,stride_dense=128,ds_factor=ds_factor).to(device)      
    elif psfmodel == 'radial':
        conv_op = linear.RadialConvMulti(nterms, dmax, rmax,
                                         (H, W, nfocus), ds_factor).to(device)   
    elif psfmodel == 'eigen':
        conv_op = linear.EigenPSFMulti(nterms, dmax, (H, W, nfocus), ds_factor).to(device)
    elif psfmodel == 'patchwise':
        conv_op = linear.PatchWiseConvMulti((H, W), dmax, patch_size=64,
                                            stride=16, nchan=nfocus).to(device)

    conv_op.load_state_dict(torch.load(params_path))

    # Hack 2 -- we need to weight the edges for the deconvolution to work
    with torch.no_grad():
        conv_weight = conv_op(torch.ones(1, 1, H, W, device=device), eval=False)
        conv_weight = conv_weight/conv_weight.max()

    # Hack -- scaling factor
    pad = (dmax-1)//2
    sfact = meas.max()/conv_op(imten, eval=False).max().item()
    imblur_ten = torch.tensor(meas).to(device).permute([2, 0, 1])[:, None, ...]
    
    # Generate initialization by weighted sum
    X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                            torch.linspace(-1, 1, H),
                            indexing='xy')
    
    R = torch.hypot(X, Y)[None, ...]
    cR = torch.linspace(0, np.sqrt(2), nfocus)[:, None, None]
    cS = (1/nfocus)**2
    
    weights = torch.exp(-((R - cR)**2)/(cS))[:, None, ...].cuda()

    x_init = (imblur_ten*weights).sum(0, keepdim=True)/weights.sum(0, keepdim=True)

    imblur_ten = imblur_ten[..., pad:-pad, pad:-pad]*conv_weight

    # PnP stuff
    prior = pnpsolver.GrayscaleDPIR(pnp_weights_path,
                                    device='cuda')

    cv2.imshow('Ground truth', im[..., ::-1])
    cv2.imshow('Blurred image', meas[..., 0]/meas.max())
    cv2.waitKey(1)

    forward_op = lambda x : conv_op(x, eval=False)
    best_imten = pnpsolver.pnp_hqs_blackbox(
        y=imblur_ten,
        forward_op=conv_op,
        denoiser_fn=prior.forward,
        x_init=x_init,
        n_iters=n_iters,
        cg_iters=cg_iters,
        lamb=lamb,
        device='cuda',
        sigma_0=sigma_max,
        sigma_end=sigma_min
    )
    
    # Compute metrics
    os.makedirs('results/%s/%s'%(expfolder, expname), exist_ok=True)
    im_rec = best_imten.squeeze().cpu().numpy().astype(np.float32)
    io.savemat('results/%s/%s/%s_%s_%d_%d_%s.mat'%(expfolder,
                                               expname,
                                               mode,
                                               psfmodel,
                                               nterms, dmax,
                                               rec_method),
                {'rec': im_rec})
    
    im_rec = im_rec/im_rec[2*dmax:-2*dmax, 2*dmax:-2*dmax].max()
    imsave = np.clip(im_rec, 0, 1)
    
    plt.imsave('results/%s/%s/%s%d_%s_%d_%d_%s.png'%(expfolder,
                                               expname,
                                               mode,
                                               nfocus,
                                               psfmodel,
                                               nterms, dmax,
                                               rec_method), imsave)
    
    cv2.imshow('Compare', utils.normalize(np.hstack((meas[..., 0]/meas.max(), imsave))))