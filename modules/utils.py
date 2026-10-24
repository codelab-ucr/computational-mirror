#!/usr/bin/env python

'''
    Small helpers for the scripts: normalisation and plotting
'''

import numpy as np
import matplotlib.pyplot as plt


def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized


def grid_plot(imdata):
    '''
        Plot 3D set of images into a 2D grid using subplots.

        Inputs:
            imdata: N x H x W image stack

        Outputs:
            None
    '''

    N, H, W = imdata.shape

    nrows = int(np.sqrt(N))
    ncols = int(np.ceil(N/nrows))

    for idx in range(N):
        plt.subplot(nrows, ncols, idx+1)
        plt.imshow(imdata[idx, :, :], cmap='gray')
        plt.xticks([], [])
        plt.yticks([], [])