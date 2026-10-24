#!/usr/bin/env python

'''
    Training data for fitting the forward model: the random-disk captures of the calibration session, offset- and
    vignetting-corrected (mode 'multi': the focal-stack images used for the PSF model), and the displayed patterns
    registered to the sensor. Needs <root>/<session>/pycalib.mat from real_calibration.py.
    Writes <root>/<session>/pytraining_<mode>.mat ('multi<nfocus>' in mode 'multi').
'''

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import tqdm
import numpy as np
from scipy import io
import cv2

if __name__ == '__main__':
    root = '/home/akayabasi/broadband_mirror/real/data'     # Root of your data
    mode = 'multi'                # 'raw' (the centre-focused image only) or 'multi'

    ntrain = 10                 # Number of training images
    nfocus = 3                  # If mode is multi
    multi_idx_min = 20          # 20 for 50mm F/1, 10 for 100mm F/2

    # For homography
    session = '2.19.calib'

    # Load data
    ds = io.loadmat('%s/%s/vignetting.mat'%(root, session))
    vignetting = ds['imstack'] - ds['offset_single'][..., np.newaxis]

    vignetting = cv2.GaussianBlur(vignetting, [99, 99], 0)

    # Load calibration information
    ds = io.loadmat('%s/%s/pycalib.mat'%(root, session))
    hmat = ds['hmat']
    cfocus_idx = ds['cfocus_idx'].flatten()[0]
    fcenter = ds['fcenter'].flatten()

    H, W, _ = vignetting.shape

    # Multifocus indices. This is heuristic
    multifocus_idx = np.linspace(cfocus_idx, cfocus_idx-multi_idx_min,
                                 nfocus).astype(int)

    if mode == 'multi':
        meas_stack = np.zeros((H, W, nfocus, ntrain),
                              dtype=np.float32)
    else:
        meas_stack = np.zeros((H, W, ntrain),
                              dtype=np.float32)

    gt_stack = np.zeros((H, W, ntrain), dtype=np.float32)

    for idx in tqdm.tqdm(range(ntrain)):
        ds = io.loadmat('%s/%s/pdisk_16_%d.mat'%(root, session, idx+1))

        imstack = ds['imstack']
        imstack -= ds['offset_single'][..., np.newaxis]

        imstack[imstack < 0] = 0
        imstack /= vignetting

        if mode == 'raw':
            meas_stack[..., idx] = imstack[..., cfocus_idx]
        else:
            imstack_sub = imstack[..., multifocus_idx]
            meas_stack[..., idx] = imstack_sub

        gt = ds['gt'][:, ::-1, 1]
        gt_reg = cv2.warpPerspective(gt, hmat, [W, H])

        gt_stack[..., idx] = gt_reg.astype(np.float32)/255

    mdict = {
        'meas_stack': meas_stack,
        'gt_stack': gt_stack,
        'hmat': hmat,
        'cfocus_idx': cfocus_idx,
        'fcenter': fcenter
    }
    if mode == 'multi':
        mname = 'multi%d'%nfocus
        mdict['multifocus_idx'] = multifocus_idx
    else:
        mname = mode
    io.savemat('%s/%s/pytraining_%s.mat'%(root, session, mname),
               mdict)