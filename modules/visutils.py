#!/usr/bin/env python

'''
    Visualization utilities
'''

import cv2
import numpy as np


def imshowpair(im1, im2, wname='Pair Comparison'):
    H, W = im1.shape

    imcat = np.zeros((H, W, 3))

    imcat[..., 0] = im1/im1.max()
    imcat[..., 1] = imcat[..., 0]
    imcat[..., 2] = im2/im2.max()

    cv2.imshow(wname, imcat)
    cv2.waitKey(1)