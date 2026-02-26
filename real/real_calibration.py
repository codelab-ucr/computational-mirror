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

# This script estimates the following:
# 1. Homography between the monitor
# 2. Index for which the central part of the measured stack is in focus
# 
#
# The generated calibration file is void if you change the following
# 1. Position of sensor or mirror
# 2. Position of monitor
# 3. Range of positions ('pos_array' in capture_focal_stack.py)
#
if __name__ == '__main__':
    root = '/home/akayabasi/broadband_mirror/real/data'

    # For homography
    session = '2.19.calib'
    calibname = 'aruco25'

    # For finding focus center
    centercalibname  = 'autofocus4'

    # For testing calibration
    testsession = '2.19.calib'
    testname = 'aruco25'

    # Load data
    ds = io.loadmat('%s/%s/vignetting.mat'%(root, session))
    vignetting = ds['imstack']/ds['exposure']

    ds = io.loadmat('%s/%s/black.mat'%(root, session))
    offset = ds['imstack']/ds['exposure']

    ds = io.loadmat('%s/%s/%s.mat'%(root, session, calibname))
    imstack = ds['imstack']/ds['exposure']
    gt = ds['gt'][:, ::-1, 1]

    ds = io.loadmat('%s/%s/%s.mat'%(root, session, centercalibname))
    center_imstack = ds['imstack']/ds['exposure']

    ds = io.loadmat('%s/%s/%s.mat'%(root, testsession, testname))
    test_imstack = ds['imstack']/ds['exposure']
    test_gt = ds['gt'][:, ::-1, 1]

    # Post processing
    imstack -= offset
    imstack[imstack < 0] = 0

    test_imstack -= ds['offset_single'][..., np.newaxis]/ds['exposure']
    test_imstack[test_imstack < 0] = 0

    # Step 1: Find most focused image
    print('Finding center focused image index')
    H, W, nf = imstack.shape
    imstack_sub = imstack[H//2-32:H//2+32, W//2-32:W//2+32, :].transpose(2, 0, 1)
    cfocus_idx, best_img, _ = calibutils.find_sharpest_from_stack(imstack_sub)

    # Step 2: region where the image is focused
    print('Estimating region of focus center')
    imfocused = center_imstack[..., cfocus_idx]
    grad_x = cv2.Sobel(imfocused, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(imfocused, cv2.CV_64F, 0, 1, ksize=3)
    imfocused_grad = np.hypot(grad_x, grad_y)
    imfocused_grad = cv2.GaussianBlur(imfocused_grad, (99, 99), 0)
    max_idx = imfocused_grad.argmax()
    center = np.unravel_index(max_idx, imfocused_grad.shape)

    # Step 3: Homography estimation
    print('Estimating homography matrix')
    imdst = np.uint8(utils.normalize(imstack[..., cfocus_idx])*255)
    imsrc = gt

    hmat, imsrc_warped = calibutils.sift_reg(imdst, imsrc)

    visutils.imshowpair(imdst, imsrc_warped, wname='train')

    # Step 4: Testing calibration
    imdst_test = np.uint8(utils.normalize(test_imstack[..., cfocus_idx])*255)
    imsrc_test = test_gt

    imsrc_test_warped = cv2.warpPerspective(imsrc_test, hmat, (W, H))

    visutils.imshowpair(imdst_test, imsrc_test_warped, wname='test')

    mdict = {
        'hmat': hmat,
        'cfocus_idx': cfocus_idx,
        'fcenter': center
    }
    io.savemat('%s/%s/pycalib.mat'%(root, session), mdict)