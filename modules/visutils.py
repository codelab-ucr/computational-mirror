#!/usr/bin/env python

'''
    Visualization utilities
'''

# System imports
import os
import sys
import time
import pickle
import pdb
import glob
import yaml
import argparse

import torch

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.ndimage as ndim
from scipy import io
from scipy.sparse.linalg import svds
from scipy import signal
from scipy import interpolate

from skimage.metrics import structural_similarity as ssim_func

# Plotting
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cv2
import numpy as np

def implay(video_volume, fps=30):
    """
    Simulates MATLAB's implay for a 3D or 4D numpy array.
    
    Args:
        video_volume: Array of shape (Frames, Height, Width) or (F, H, W, Color).
        fps: Approximate frames per second for playback.
    """
    # 1. Validation and Normalization
    if video_volume.ndim not in [3, 4]:
        raise ValueError("Input must be 3D (grayscale) or 4D (color) array.")
    
    num_frames = video_volume.shape[0]
    window_name = "Python Implay"
    
    # 2. Setup the Window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 3. Create a helper for the Trackbar (Slider)
    def nothing(x):
        pass
    
    # Create slider: 'Frame' is the label, 0 is start, num_frames-1 is max
    cv2.createTrackbar('Frame', window_name, 0, num_frames - 1, nothing)
    
    # 4. Playback Loop
    is_playing = True
    delay = int(1000 / fps)
    
    current_frame_idx = 0
    
    print("Controls:")
    print(" [SPACE] - Pause/Play")
    print(" [ESC]   - Exit")
    print(" [Slider]- Scrub through frames manually")

    while True:
        # Get the current position of the trackbar
        # If user moves slider, this updates. If loop updates, we set this.
        if is_playing:
            current_frame_idx += 1
            if current_frame_idx >= num_frames:
                current_frame_idx = 0
            # Update slider position visually
            cv2.setTrackbarPos('Frame', window_name, current_frame_idx)
        else:
            # If paused, read from slider
            current_frame_idx = cv2.getTrackbarPos('Frame', window_name)

        # Retrieve frame
        frame = video_volume[current_frame_idx]

        # Convert to BGR if it's grayscale (for consistency in display)
        # Note: If your data is float (0..1), convert to uint8 (0..255)
        if frame.dtype != np.uint8:
             # Simple normalization for display
             frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        cv2.imshow(window_name, frame)

        # Handle User Input
        key = cv2.waitKey(delay) & 0xFF
        
        if key == 27: # ESC to quit
            break
        elif key == 32: # SPACE to pause/play
            is_playing = not is_playing

    cv2.destroyAllWindows()

def imshowpair(im1, im2, wname='Pair Comparison'):
    H, W = im1.shape

    imcat = np.zeros((H, W, 3))

    imcat[..., 0] = im1/im1.max()
    imcat[..., 1] = imcat[..., 0]
    imcat[..., 2] = im2/im2.max()

    cv2.imshow(wname, imcat)
    cv2.waitKey(1)