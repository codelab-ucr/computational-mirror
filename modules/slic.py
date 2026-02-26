#!/usr/bin/env python

import os
import sys

import numpy as np
import cv2

import cassi_cp

def slic_update(imrgb, mask, compactness=10.0):
    '''
        Performs one single step of SLIC to update membership

        Inputs:
            imrgb: RGB image
            mask: Sparse sampling mask / centroids of superpixels after
                sanitizing
            compactness: SLIC compactness parameter

        Outputs:
            L: superpixel membership map
            N: Total number of super pixels
    '''

    H, W, _ = imrgb.shape
    ch, cw = np.where(mask == 1)

    # Create LabXY image
    [Y, X] = np.mgrid[:H, :W]

    imlabxy = np.zeros((H, W, 5), dtype=np.float32)

    imlabxy[:, :, :3] = cv2.cvtColor(imrgb, cv2.COLOR_RGB2Lab)
    imlabxy[:, :, 3] = X
    imlabxy[:, :, 4] = Y

    # Reshape to a matrix
    imlabxymat = imlabxy.reshape(H*W, 5).astype(np.float32)

    centroids_labxy = imlabxy[ch, cw, :].astype(np.float32)
    N = ch.size
    S = int(np.sqrt(H*W/N))

    nmembers = np.zeros(N)

    dist_matrix = np.ones((H, W), dtype=np.float32)*float('inf')
    L = np.ones((H, W), dtype=np.uint16)

    # Inefficient, but just do it
    for idx in range(N):
        hmin = max(0, ch[idx] - 2*S); hmax = min(H, ch[idx] + 2*S)
        wmin = max(0, cw[idx] - 2*S); wmax = min(W, cw[idx] + 2*S)

        imlabxy_patch = imlabxy[hmin:hmax, wmin:wmax, :]
        dist_patch_old = dist_matrix[hmin:hmax, wmin:wmax]
        dist_patch = cassi_cp._get_dist_cp(centroids_labxy[idx, :], imlabxy_patch,
                                           np.float32(compactness), S)

        L_patch = L[hmin:hmax, wmin:wmax]
        L_patch[dist_patch < dist_patch_old] = idx

        L[hmin:hmax, wmin:wmax] = L_patch

        dist_matrix[hmin:hmax, wmin:wmax] = np.minimum(dist_patch,
                                                       dist_patch_old)

    return L.astype(np.uint16), N