#!/usr/bin/env python

'''
    Test captures for deconvolution: offset- and vignetting-corrected focal-stack images (mode 'multi') and, for
    captures of the monitor, the displayed image registered to the sensor. Needs pycalib.mat of the calibration session.
    Writes <saveroot>/<session>/<name>_<mode>.mat ('multi<nfocus>' in mode 'multi').
'''

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import tqdm
import numpy as np
from scipy import io
import cv2

sys.path.append('../modules')

import calibutils

if __name__ == '__main__':
    root = '/home/akayabasi/broadband_mirror/real/data'     # Root of your data
    saveroot = '/home/akayabasi/broadband_mirror/real/pyprocessed'
    mode = 'multi'                # 'raw' (the centre-focused image only) or 'multi'
    nfocus = 3                  # If mode is multi
    ksize = 32                  # Size of center patch to find optimal focus
    datamode = 'monitor'        # 'monitor' or 'wild'
    bypass_cfocus = None         # Use this to bypass automatic estimation of focused idx
    multi_idx_min = 20

    # For homography
    session = '2.17.validation'
    calibsession = '2.19.calib'

    imlist = ['0810', '0886', '0872', '0900', 'grid_32', 'iso_reschart', 'sector', 'usaf_grid']

    # Load data
    ds = io.loadmat('%s/%s/black.mat'%(root, session))
    offset = ds['imstack']

    ds = io.loadmat('%s/%s/vignetting.mat'%(root, session))
    vignetting = ds['imstack']/ds['exposure'] - offset/ds['exposure']

    vignetting = cv2.GaussianBlur(vignetting, [99, 99], 0)

    # Load calibration information
    ds = io.loadmat('%s/%s/pycalib.mat'%(root, calibsession))
    hmat = ds['hmat']
    cfocus_idx = ds['cfocus_idx'].flatten()[0]
    fcenter = ds['fcenter'].flatten()

    H, W, _ = vignetting.shape

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

        if bypass_cfocus is not None:
            cfocus_idx = bypass_cfocus

        # Multifocus indices. This is heuristic
        multifocus_idx = np.linspace(cfocus_idx,
                                     cfocus_idx-multi_idx_min,
                                     nfocus).astype(int)

        tbar.set_description('%s | focus = %d'%(imname, cfocus_idx))
        tbar.refresh()

        if mode == 'raw':
            meas = imstack[..., cfocus_idx]
        else:
            meas = imstack[..., multifocus_idx]

        if datamode == 'monitor':
            gt = ds['gt'][:, ::-1, 1]
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