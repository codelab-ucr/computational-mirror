#!/usr/bin/env python

'''
    Calibration of the camera from captures of one session:
    1. the focus index at which the centre of the frame is sharpest,
    2. the region of the frame that is in focus,
    3. the homography from the displayed pattern to the sensor.
    Writes <root>/<session>/pycalib.mat (hmat, cfocus_idx, fcenter).

    The calibration is void if the sensor, the mirrors or the monitor move, or if the focal-stack positions change.
'''

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import numpy as np
from scipy import io
import cv2

sys.path.append('../modules')

import calibutils
import visutils
import utils

if __name__ == '__main__':
    root = '<DATA>/visible'                                 # your data folder (README: Data)

    # For homography
    session = '2.19.calib'
    calibname = 'aruco25'

    # For finding focus center
    centercalibname  = 'autofocus4'

    # For testing calibration
    testsession = '2.19.calib'
    testname = 'aruco25'

    # Load data
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
    H, W, _ = imstack.shape
    imstack_sub = imstack[H//2-32:H//2+32, W//2-32:W//2+32, :].transpose(2, 0, 1)
    cfocus_idx, _, _ = calibutils.find_sharpest_from_stack(imstack_sub)

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