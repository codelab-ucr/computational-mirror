#!/usr/bin/env python

import os
import sys
import pdb
import tqdm
import importlib

import numpy as np
import torch
from scipy import interpolate
from scipy import io

import matplotlib.pyplot as plt
import cv2

import losses
import utils
import spectral
import deep_prior

utils = importlib.reload(utils)
spectral = importlib.reload(spectral)
losses = importlib.reload(losses)

def downsample_A(calib_spectra, calib_scale, calib_wvl, wavelengths):
    '''
        Downsample calibration information
    '''
    dlambda = (wavelengths[1] - wavelengths[0])/2
    
    nrows, nwvl = calib_spectra.shape
    nwvl2 = wavelengths.size
    calib_spectra_ds = np.zeros((nrows, nwvl2))
    calib_scale_ds = np.zeros(nwvl2)
    
    for idx in range(nwvl2):
        lmb = wavelengths[idx]
        valid = (calib_wvl > lmb - dlambda)*(calib_wvl < lmb + dlambda)
        calib_spectra_ds[:, idx] = calib_spectra[:, valid].mean(1)
        calib_scale_ds[idx] = calib_scale[valid].mean()
        
    return calib_spectra_ds, calib_scale_ds

def hyper2rgb(cube, wavelengths, camspec):
    '''
        Compute linear hyperspectral to RGB image
    '''
    response = np.zeros((wavelengths.size, 3))
    
    for idx in range(3):
        x = camspec['wvl'].ravel().astype(np.float32)
        y = camspec['resp'][:, idx]
        
        func = interpolate.interp1d(x, y, fill_value=0, bounds_error=False)
        response[:, idx] = func(wavelengths)
        
    H, W, nwvl = cube.shape
    hsmat = cube.reshape(H*W, nwvl)
    rgb_vec = hsmat.dot(response)
    
    imrgb = rgb_vec.reshape(H, W, 3)
    
    return imrgb/imrgb.max()

def load(expname, root, restored_root, calibroot):
    '''
        Load data from real samples
    '''
    # Load data
    full_data = io.loadmat('../%s/%s/data.mat'%(root, expname))
    calib_data = io.loadmat('%s/VS_inv_linear.mat'%calibroot)
    restored_data = io.loadmat('../%s/%s.mat'%(restored_root, expname))
    crop_data = io.loadmat('../../results/CropValues/%s.mat'%expname)
    
    return full_data, calib_data, restored_data, crop_data

def get_calib_info(calib_data, wavelengths):
    '''
        Get calibration info
    '''
    calib_wvl = calib_data['wvl'].ravel()
    calib_spectra = calib_data['spectra_norm']
    calib_scale = calib_data['wvl_scale'].ravel()
    
    # 8/10 -- wavelengths is irrelevant, just use calib_wvl
    wavelengths = calib_wvl
        
    camspec = io.loadmat('../../resources/camspec2.mat')
    rgb_response = utils.interp1(camspec['resp'].T,
                                 camspec['wvl'].ravel().astype(np.float32),
                                 wavelengths).astype(np.float32)
    rgb_response = torch.tensor(rgb_response.T).cuda()[None, ...]
            
    # Interpolate matrix to desired wavelengths
    #calib_spectra, calib_scale = downsample_A(calib_spectra,
    #                                          calib_scale,
    #                                          calib_wvl,
    #                                          wavelengths)
    calib_scale /= calib_scale.max()
    
    return rgb_response, camspec, calib_spectra, calib_scale
    
def prepare(data_list, crop_rows, crop_cols, pat_idx, data_type, get_fullscan):
    '''
        Prepare data for reconstruction
    '''
    full_data, _, restored_data, crop_data = data_list

    if get_fullscan:
        full_scan = full_data['full_scan'].astype(np.float32)
        full_scan = full_scan/65535
        full_scan = utils.crop_center_array(full_scan, crop_rows, crop_cols)
    else:
        full_scan = None
    
    # Grab the RGB image to use as guide
    guide = full_data['guide'].astype(np.float32)
    guide /= guide.max()
    guide = utils.crop_center_array(guide, crop_rows, crop_cols)
    
    # Create sampling pattern
    pattern = full_data['assort_index'][..., pat_idx].astype(np.uint8)
    pattern = utils.crop_center_array(pattern, crop_rows, crop_cols)
    
     # Generate measurement
    if data_type == 'real':
        gt = full_data['assort_meas'][..., pat_idx].astype(np.float32)/65535
        gt = utils.crop_center_array(gt, crop_rows, crop_cols)
    elif data_type == 'restored':
        # Now load the restored data a la Vijay
        gt = restored_data['assort_restored'][pat_idx, ...]
        gt = gt.astype(np.float32)
    else:
        gt = full_data['assort_sim'][..., pat_idx].astype(np.float32)/65535
        gt = utils.crop_center_array(gt, crop_rows, crop_cols)
        
    # Crop
    Xcrop = crop_data['Y'].ravel()
    Ycrop = crop_data['X'].ravel()
    
    if full_scan is not None:
        full_scan = full_scan[Xcrop[0]+1:Xcrop[1], Ycrop[0]+1:Ycrop[1], :]
        
    pattern = pattern[Xcrop[0]+1:Xcrop[1], Ycrop[0]+1:Ycrop[1]]
    guide = guide[Xcrop[0]+1:Xcrop[1], Ycrop[0]+1:Ycrop[1], :]
    gt = gt[Xcrop[0]+1:Xcrop[1], Ycrop[0]+1:Ycrop[1]]
    
    return full_scan, pattern, guide, gt

def get_metrics(cube_gt, cube_rec):
    '''
        Compute reconstruction metrics
    '''
    psnrval = utils.psnr(cube_gt/cube_gt.max(), cube_rec/cube_gt.max())
    rsnrval = utils.rsnr(cube_gt/cube_gt.max(), cube_rec/cube_gt.max())
    sam_map = utils.SAM_3d(cube_gt/cube_gt.max(), cube_rec/cube_gt.max())
    
    sam_mean = sam_map.mean()
    sam_median = np.median(sam_map)
    
    return rsnrval, psnrval, sam_mean, sam_median

def rec_TV_snapshot(pattern, gt, calib_spectra, params, method='tv'):
    '''
        Recover from snapshot image using TV norm or deep image prior
        
        Inputs:
            pattern: (H, W) 2D pattern
            gt: (H, W) measurements
            calib_spectra: (256, nwvl) calibration matrix
            params: Dictionary with learning parameters
                lr: learning rate
                epochs: Number of epochs
                lambda_tv: Penalty for tv/wavelet loss
                lambda_spec: Penalty for spectral smoothness
                losstype: 'tv' or 'wavelet'
            method: 'tv' or 'dip'
        
        Outputs:
            cube: (H, W, nwvl) hyperspectral cube
    '''
    H, W = pattern.shape
    _, nwvl = calib_spectra.shape
    
    init_nconv = 128
    num_channels_up = 5
    
    if method == 'tv':
        hsten = torch.rand(1, H*W, nwvl).cuda()/10.0
        hsten = torch.autograd.Variable(hsten, requires_grad=True).cuda()
        hsten = torch.nn.Parameter(hsten)
        
        optim_params = [hsten]
    elif method == 'dip':
        hs_net = deep_prior.get_net(nwvl, 'skip', 'reflections',
                                    upsample_mode='bilinear',
                                    skip_n33d=init_nconv,
                                    skip_n33u=init_nconv,
                                    num_scales=num_channels_up,
                                    n_channels=nwvl).cuda()
        hs_inp = deep_prior.get_noise(nwvl, 'noise', [H, W]).cuda().detach()
                
        optim_params = hs_net.parameters()
        
        params['lambda_tv'] = 1e-6
        
    # Generate spectra at each pixel -- can't believe we did not do this
    calib_spectra_per_pix = calib_spectra[pattern.ravel(), :].astype(np.float32)
    calib_ten = torch.tensor(calib_spectra_per_pix)[None, ...].cuda()
        
    # Create optimizer
    optimizer = torch.optim.Adam(lr=params['lr'], params=optim_params)
    
    gt_ten = torch.tensor(gt).cuda().reshape(1, -1)
    
    criterion_l1 = losses.L2Norm()
    criterion_spec = losses.SpecNorm()
    
    if params['losstype'] == 'tv':
        criterion_tv = losses.TVNorm(mode='isotropic')
    elif params['losstype'] == 'wavelet':
        J = int(np.log2(min(H, W))) - 5
        criterion_tv = losses.WaveletNorm(J=J, wave='db6').cuda()
    else:
        J = int(np.log2(min(H, W))) - 5
        criterion_tv = losses.WaveletGroupNorm(J=J, wave='db6').cuda()
    
    loss_array = np.zeros(params['epochs'])
    
    best_loss = float('inf')
    hsmat = None
    
    for idx in tqdm.tqdm(range(params['epochs'])):
        if method == 'dip':
            hypercube_ten = hs_net(hs_inp)
            hsten = hypercube_ten.permute(0, 2, 3, 1).reshape(1, H*W, nwvl)
        else:
            hypercube_ten = hsten.view(1, H, W, nwvl).permute(0, 3, 1, 2)
            
        output = (hsten*calib_ten).sum(2)
    
        cube_ten = hsten.permute(0, 2, 1).reshape(1, nwvl, H, W)
        
        loss_l1 = criterion_l1(output - gt_ten)
        loss_tv = criterion_tv(cube_ten)
        loss_spec = criterion_spec(cube_ten)
        
        loss = loss_l1 +\
               params['lambda_tv']*loss_tv +\
               params['lambda_spec']*loss_spec
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss_l1 < best_loss:
            hsmat = hsten.detach().squeeze().cpu().numpy()
            best_loss = loss_l1.item()

        loss_array[idx] = loss_l1.item()
        
        im_cpu = cube_ten[0, idx%nwvl, :, :].detach().cpu().numpy()
        
        if sys.platform == 'win32':
            cv2.imshow('Rec', im_cpu)
            cv2.waitKey(1)
                    
    cube = hsmat.reshape(H, W, nwvl)
    cube[cube < 0] = 0
    
    return cube, loss_array

def rec_TV_LR_snapshot(pattern, gt, calib_spectra, rank, params):
    '''
        Recover from snapshot image using TV norm and low rank decomposition
        
        Inputs:
            pattern: (H, W) 2D pattern
            gt: (H, W) measurements
            calib_spectra: (256, nwvl) calibration matrix
            rank: Rank of decomposition
            params: Dictionary with learning parameters
                lr: learning rate
                epochs: Number of epochs
                lambda_tv: Penalty for tv/wavelet loss
                lambda_spec: Penalty for spectral smoothness
                losstype: 'tv' or 'wavelet'
        
        Outputs:
            cube: (H, W, nwvl) hyperspectral cube
    '''
    H, W = pattern.shape
    _, nwvl = calib_spectra.shape
    
    u_ten = torch.rand(1, H*W, rank).cuda()/10.0
    u_ten = torch.autograd.Variable(u_ten, requires_grad=True)
    u_ten = torch.nn.Parameter(u_ten)
    
    v_ten = torch.rand(1, rank, nwvl).cuda()/10.0
    v_ten = torch.autograd.Variable(v_ten, requires_grad=True)
    v_ten = torch.nn.Parameter(v_ten)
        
    # Generate spectra at each pixel -- can't believe we did not do this
    calib_spectra_per_pix = calib_spectra[pattern.ravel(), :].astype(np.float32)
    calib_ten = torch.tensor(calib_spectra_per_pix)[None, ...].cuda()
        
    # Create optimizer
    optim_params = [u_ten] + [v_ten]
    optimizer = torch.optim.Adam(lr=params['lr'], params=optim_params)
    
    gt_ten = torch.tensor(gt).cuda().reshape(1, -1)
    
    criterion_l1 = losses.L2Norm()
    criterion_spec = losses.SpecNorm()
    
    if params['losstype'] == 'tv':
        criterion_tv = losses.TVNorm(mode='isotropic')
    elif params['losstype'] == 'wavelet':
        J = int(np.log2(min(H, W))) - 5
        criterion_tv = losses.WaveletNorm(J=J, wave='db6').cuda()
    else:
        J = int(np.log2(min(H, W))) - 5
        criterion_tv = losses.WaveletGroupNorm(J=J, wave='db6').cuda()
    
    loss_array = np.zeros(params['epochs'])
    
    best_loss = float('inf')
    hsmat = None
        
    for idx in tqdm.tqdm(range(params['epochs'])):
        hsten = torch.bmm(u_ten, v_ten)
        
        output = (hsten*calib_ten).sum(2)
    
        cube_ten = hsten.permute(0, 2, 1).reshape(1, nwvl, H, W)
        u_img_ten = u_ten.permute(0, 2, 1).reshape(1, rank, H, W)
        
        loss_l1 = criterion_l1(output - gt_ten)
        loss_tv = criterion_tv(u_img_ten)
        loss_spec = criterion_spec(v_ten.permute(0, 2, 1))
        
        loss = loss_l1 +\
               params['lambda_tv']*loss_tv +\
               params['lambda_spec']*loss_spec
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss_l1 < best_loss:
            hsmat = hsten.detach().squeeze().cpu().numpy()
            best_loss = loss.item()

        loss_array[idx] = loss_l1.item()
        
        if params['showimg']:
            im_cpu = cube_ten[0, nwvl//2, :, :].detach().cpu().numpy()
            cv2.imshow('Rec', im_cpu/(1e-12+im_cpu.max()))
            cv2.waitKey(1)
                    
    cube = hsmat.reshape(H, W, nwvl)
    cube[cube < 0] = 0
    
    return cube, loss_array