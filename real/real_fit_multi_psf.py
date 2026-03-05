#!/usr/bin/env python

import os
import sys
import tqdm
import pdb
import importlib

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import numpy as np
import torch
from scipy import io
from deepali.losses.functional import bspline_bending_loss, bending_loss
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR,LambdaLR
import matplotlib.pyplot as plt
import cv2

plt.gray()

sys.path.append('../modules')

import losses
import utils
import linear

utils = importlib.reload(utils)
losses = importlib.reload(losses)
linear = importlib.reload(linear)

# 31 and 30 terms are 2.3.calib
# 32 is 2.19.calib

if __name__ == '__main__':
    session_name = '2.19.calib'
    root = '/mnt/mydisk/alper/broadband_imaging/data'
    mode = 'multi'
    nfocus = 1
    psfmodel = 'definr'      # affine or eigen
    nterms = 31                  # Number of decomposition terms
    dmax = 11                     # Maximum disk size (initialization)
    rmax = 1.02                  # Maximum stretch factor (initialization)
    R_focus = 0               # Initial focus position (0 for on axis)
    device = 'cuda'
    ds_factor = 32              # Downsampling factor for weights
    bending_loss_on = True
    # Learning constants
    spline_level =3
    if spline_level==3:
        epochs = 3000
    elif spline_level==2:
        epochs=2000
    learning_rate = 5e-3
    lambda_kern = 1e-6
    lambda_weights = 1e-1
    bsize = 10                   # Batch size
    warmup_steps = 500
    # Load data
    data = io.loadmat('%s/%s/pytraining_%s%d.mat'%(root, session_name, mode, nfocus))
    gt = data['gt_stack'][..., :10]
    meas = data['meas_stack'][..., :10]

    meas -= meas[..., :10].min(3, keepdims=True)
    meas /= meas.max()

    H, W, nchan, nimg = meas.shape

    gt_ten = torch.tensor(gt)[None, ...].to(device).permute(0, 3, 1, 2)
    meas_ten = torch.tensor(meas).to(device).permute(2, 3, 0, 1)
            
    if psfmodel == 'affineplus':
        conv_op = linear.AffineConvMulti(nterms, dmax, rmax,
                                         (H, W, nchan), ds_factor).to(device)
    elif psfmodel == "bspline":
         conv_op = linear.BSplineConvMulti(nterms,dmax,rmax,(H,W,nfocus),stride_coarse=256,stride_middle=128,stride_dense=80,ds_factor=ds_factor).to(device)    
    elif psfmodel == "definr":
         conv_op = linear.DefINRConvMulti(nterms,dmax,rmax,(H,W,nfocus),ds_factor=ds_factor,cps=32).to(device)  
    elif psfmodel == 'radial':
        conv_op = linear.RadialConvMulti(nterms, dmax, rmax,
                                         (H, W, nchan), ds_factor).to(device)    
    elif psfmodel == 'eigen':
        conv_op = linear.EigenPSFMulti(nterms, dmax, (H, W, nchan), ds_factor).to(device)
    elif psfmodel == 'patchwise':
        conv_op = linear.PatchWiseConvMulti((H, W), dmax, patch_size=64,
                                       stride=16, nchan=nchan).to(device)

    # Fix weights for now.
    params = list(conv_op.parameters())

    for m_idx in range(nfocus):
        conv_op_sub = conv_op.conv_ops[m_idx]
        params_sub = list(conv_op_sub.parameters())

        meas_ten_sub = meas_ten[m_idx, ...]

        # Create optimizer
        optimizer = torch.optim.Adam(lr=learning_rate, params=params_sub)

        # Scheduler
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)

        # Cosine: Decays from base LR down to 0 over the remaining steps
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(epochs - 500), eta_min=0.1*learning_rate)

        # 4. Chain them together
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_steps]
        )
        
        criterion_data = losses.L2Norm()
        criterion_l1 = losses.L1Norm()
        criterion_tv = losses.TVNorm(mode='l2').cuda()
        
        loss_array = np.zeros(epochs)
        mse_array = np.zeros(epochs)
        
        best_loss = float('inf')
        best_epoch = 0

        #cv2.imshow('Measurement', meas[..., m_idx, 0]/meas.max())
        #cv2.waitKey(1)

        pad = (dmax-1)//2

        tbar = tqdm.tqdm(range(epochs))

        for idx in tbar:  


        
            imblur_estim = conv_op_sub(gt_ten)

            loss_l1 = criterion_data(imblur_estim - meas_ten_sub[:, pad:-pad, pad:-pad])
            loss_kern = criterion_l1(conv_op_sub.kernels)
            
            loss = loss_l1 + lambda_kern*loss_kern
            if bending_loss_on:
                bending_energy = 1e-3*bspline_bending_loss(conv_op_sub.deformation_field,stride=conv_op_sub.bspline_cps)
                loss = loss + bending_energy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            
            loss_array[idx] = loss_l1.item()
            
            if loss_array[idx] < best_loss:
                best_loss = loss_array[idx]
                best_epoch = idx

            tbar.set_description('Loss: %.3e dB'%loss_l1.item())
            tbar.refresh()
            
            with torch.no_grad():
                imblur_test = conv_op_sub(gt_ten[:, [0], ...])

                img = imblur_test[0, ...].detach().cpu().numpy()
                err = abs(img - meas[pad:-pad, pad:-pad, m_idx, 0])
                
                #cv2.imshow('Estimate', img/img.max())
                #cv2.imshow('Error', 10*err/err.max())
                #cv2.waitKey(1)

    # To get kernels
    with torch.no_grad():
        tmp = conv_op(gt_ten[:, [0], ...])

    # Save state dictionary
    os.makedirs('models/%s'%session_name, exist_ok=True)
    torch.save(conv_op.cpu().state_dict(),
               'models/%s/%s%d_%s_%d_%d_params.pth'%(session_name,
                                                 mode,
                                                 nfocus,
                                                 psfmodel,
                                                 nterms,
                                                 dmax))

    plt.figure(0)
    utils.grid_plot(conv_op.kernels.detach().squeeze().cpu())

    if psfmodel != 'patchwise':
        plt.figure(1)
        utils.grid_plot(conv_op.weights.detach().squeeze().cpu())
    plt.show()
            