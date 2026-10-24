#!/usr/bin/env python

'''
    Fits a spatially varying PSF model to the training data of real_gen_training.py, one focus image at a time.
    Writes models/<session>/<mode><nfocus>_<psfmodel>_<nterms>_<dmax>_params.pth (the model's state dictionary).
'''

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import tqdm
import numpy as np
import torch
from scipy import io
from deepali.losses.functional import bspline_bending_loss
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR,LambdaLR
import matplotlib.pyplot as plt

plt.gray()

sys.path.append('../modules')

import losses
import utils
import linear

if __name__ == '__main__':
    session_name = '2.19.calib'
    root = '/mnt/mydisk/alper/broadband_imaging/data'
    mode = 'multi'
    nfocus = 1
    psfmodel = 'bspline'         # 'bspline' (ours), 'seidelconv' (baseline); experimental: 'definr', 'eigen', 'patchwise'
    nterms = 31                  # Number of decomposition terms
    dmax = 11                     # Maximum disk size (initialization)
    rmax = 1.02                  # Maximum stretch factor (initialization)
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
    scheduler_type = "cosine_with_warmup" # exponential or cosine_with_warmup
    warmup_steps = 50
    # Load data
    data = io.loadmat('%s/%s/pytraining_%s%d.mat'%(root, session_name, mode, nfocus))
    gt = data['gt_stack'][..., :10]
    meas = data['meas_stack'][..., :10]

    meas -= meas[..., :10].min(3, keepdims=True)
    meas /= meas.max()

    H, W, nchan, _ = meas.shape

    gt_ten = torch.tensor(gt)[None, ...].to(device).permute(0, 3, 1, 2)
    meas_ten = torch.tensor(meas).to(device).permute(2, 3, 0, 1)

    if psfmodel == 'seidelconv':
        conv_op = linear.SeidelConvMulti(nterms, dmax, rmax,
                                         (H, W, nchan), ds_factor).to(device)
    elif psfmodel == "bspline":
         conv_op = linear.BSplineConvMulti(nterms,dmax,rmax,(H,W,nfocus),stride_coarse=256,stride_middle=128,stride_dense=80,ds_factor=ds_factor).to(device)
    elif psfmodel == "definr":
         conv_op = linear.DefINRConvMulti(nterms,dmax,rmax,(H,W,nfocus),ds_factor=ds_factor,cps=32).to(device)
    elif psfmodel == 'eigen':
        conv_op = linear.EigenPSFMulti(nterms, dmax, (H, W, nchan), ds_factor).to(device)
    elif psfmodel == 'patchwise':
        conv_op = linear.PatchWiseConvMulti((H, W), dmax, patch_size=64,
                                       stride=16, nchan=nchan).to(device)

    for m_idx in range(nfocus):
        conv_op_sub = conv_op.conv_ops[m_idx]
        params_sub = list(conv_op_sub.parameters())
        if psfmodel == "bspline":
            for name,params in conv_op_sub.named_parameters():
                if "dense_grid" in name:
                    params.requires_grad= False
                elif "middle_grid" in name:
                    params.requires_grad= False
        meas_ten_sub = meas_ten[m_idx, ...]

        # Create optimizer
        optimizer = torch.optim.Adam(lr=learning_rate, params=params_sub)


        if scheduler_type == "cosine_with_warmup":
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(epochs - warmup_steps), eta_min=0.1*learning_rate)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        elif scheduler_type=="exponential":
            scheduler = LambdaLR(optimizer, lambda x: 0.1**(min(x/epochs, 1)))

        criterion_data = losses.L2Norm()
        criterion_l1 = losses.L1Norm()

        loss_array = np.zeros(epochs)

        pad = (dmax-1)//2

        tbar = tqdm.tqdm(range(epochs))

        for idx in tbar:

            if (idx==1000) and (psfmodel=="bspliine"):
                for name,params in conv_op_sub.named_parameters():
                    if "dense_grid" in name:
                        params.requires_grad= False
                    elif "middle_grid" in name:
                        params.requires_grad = True
                    elif "coarse_grid" in name:
                        params.requires_grad = False
            if (idx==2000) and (psfmodel=="bspline"):
                 for name,params in conv_op_sub.named_parameters():
                    if "dense_grid" in name:
                        params.requires_grad= True
                    elif "middle_grid" in name:
                        params.requires_grad = False
                    elif "coarse_grid" in name:
                        params.requires_grad = False

            imblur_estim = conv_op_sub(gt_ten)

            loss_data = criterion_data(imblur_estim - meas_ten_sub[:, pad:-pad, pad:-pad])
            loss_kern = criterion_l1(conv_op_sub.kernels)

            loss = loss_data + lambda_kern*loss_kern
            if bending_loss_on:
                if psfmodel == "definr":
                    bending_energy = 1e-3*bspline_bending_loss(conv_op_sub.deformation_field,stride=conv_op_sub.bspline_cps)
                elif psfmodel == "bspline":
                    if idx <1000:
                        bending_energy= 1e-4*bspline_bending_loss(conv_op_sub.coarse_grid,stride=conv_op_sub.cps_coarse)
                    elif (idx)<2000:
                        bending_energy = 1e-5*bspline_bending_loss(conv_op_sub.middle_grid,stride=conv_op_sub.cps_middle)
                    else:
                        bending_energy = 1e-6*bspline_bending_loss(conv_op_sub.dense_grid,stride=conv_op_sub.cps_dense)

                loss = loss + bending_energy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            loss_array[idx] = loss_data.item()

            tbar.set_description('Loss: %.3e'%loss_data.item())
            tbar.refresh()

    # One forward pass sets conv_op.kernels and conv_op.weights for the plots below
    with torch.no_grad():
        conv_op(gt_ten[:, [0], ...])

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