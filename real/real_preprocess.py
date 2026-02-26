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
from scipy import io

from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import cv2

plt.gray()

sys.path.append('../modules')

import calibutils
import visutils
import utils
import linear

utils = importlib.reload(utils)
calibutils = importlib.reload(calibutils)
visutils = importlib.reload(visutils)
linear = importlib.reload(linear)

# This script preprocesses data for deconvolution
#
# Assumes that 'real_calibration.py' is run and 'pycalib.mat'
# is already available

# Chexmex is cfocus=27

if __name__ == '__main__':
    root = '/home/akayabasi/broadband_mirror/real/data'     # Root of your data
    saveroot = '/home/akayabasi/broadband_mirror/real/pyprocessed'
    mode = 'multi'                # 'raw', 'petzval', 'multi'
    nfocus = 3                  # If mode is multi
    ksize = 32                  # Size of center patch to find optimal focus
    datamode = 'monitor'        # 'monitor' or 'wild'
    bypass_cfocus = None         # Use this to bypass automatic estimation of focused idx
    multi_idx_min = 20

    # For homography
    session = '2.17.validation'
    calibsession = '2.19.calib'

    filenames = glob.glob('%s/%s/*.mat'%(root, session))
    filenames = [os.path.split(filename)[-1].replace('.mat', '') for filename in filenames]
    filenames = [filename for filename in filenames if 'black' not in filename]
    filenames = [filename for filename in filenames if 'vignetting' not in filename]

    #imlist = filenames

    #imlist = ['apples_%s'%wvl for wvl in ['blue', 'green', 'red', 'nir', 'swir']]
    #imlist = ['iso_reschart']
    imlist = ['0810', '0886', '0872', '0900', 'grid_32', 'iso_reschart', 'sector', 'usaf_grid']
    #imlist = ['chexmex']
    
    # Load data
    ds = io.loadmat('%s/%s/black.mat'%(root, session))
    offset = ds['imstack'] #/ds['exposure']

    ds = io.loadmat('%s/%s/vignetting.mat'%(root, session))
    vignetting = ds['imstack']/ds['exposure'] - offset/ds['exposure']
    
    vignetting = cv2.GaussianBlur(vignetting, [99, 99], 0)

    # Load calibration information
    ds = io.loadmat('%s/%s/pycalib.mat'%(root, calibsession))
    hmat = ds['hmat']
    cfocus_idx = ds['cfocus_idx'].flatten()[0]
    fcenter = ds['fcenter'].flatten()
    
    H, W, _ = vignetting.shape
    
    # Multifocus indices. This is heuristic
    multifocus_idx = np.linspace(cfocus_idx, cfocus_idx-multi_idx_min,
                                 nfocus).astype(int)
    
    os.makedirs('%s/%s'%(saveroot, session), exist_ok=True)
        
    tbar = tqdm.tqdm(imlist)
    for imname in tbar:
        ds = io.loadmat('%s/%s/%s.mat'%(root, session, imname))
        
        imstack = ds['imstack']/ds['exposure']
        imstack -= offset/ds['exposure']

        imstack[imstack < 0] = 0
        imstack /= vignetting

        # If datamode is 'wild', we need to restimate cfocus_idx
        if datamode == 'wild':
            imstack_sub = imstack[H//2-ksize:H//2+ksize,
                                  W//2-ksize:W//2+ksize, :]
            cfocus_idx, _, _ = calibutils.find_sharpest_from_stack(
                imstack_sub.transpose(2, 0, 1).astype(np.float64)
            )

            # Bypass
            if bypass_cfocus is not None:
                cfocus_idx = bypass_cfocus

            multifocus_idx = np.linspace(cfocus_idx,
                                         cfocus_idx-multi_idx_min,
                                         nfocus).astype(int)
            
        if bypass_cfocus is not None:
            cfocus_idx = bypass_cfocus

        multifocus_idx = np.linspace(cfocus_idx,
                                         cfocus_idx-multi_idx_min,
                                         nfocus).astype(int)
            
        tbar.set_description('%s | focus = %d'%(imname, cfocus_idx))
        tbar.refresh()
        
        if mode == 'raw':
            meas = imstack[..., cfocus_idx]
        elif mode == 'petzval':
            raise ValueError('Not yet implemented')
        else:
            meas = imstack[..., multifocus_idx]

        if datamode == 'monitor': 
            gt = ds['gt'][:, ::-1, 1]

            # Temporary hack, do not use this afterwards
            #gt = cv2.resize(gt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            gt_reg = cv2.warpPerspective(gt, hmat, [W, H])
        else:
            gt_reg = imstack[..., cfocus_idx]

        mdict = {
            'imraw': meas,
            'imgt': gt_reg,
            'hmat': hmat,
            'cfocus_idx': cfocus_idx,
            'fcenter': fcenter
        }
        if mode == 'multi':
            mname = 'multi%d'%nfocus
            mdict['multifocus_idx'] = multifocus_idx
        else:
            mname = mode
        io.savemat('%s/%s/%s_%s.mat'%(saveroot,
                                      session,
                                      imname,
                                      mname),
                mdict)
        