#!/usr/bin/env python

''' Script to test display class'''

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import sys
import cv2
import time
import pdb
import signal
import queue
from threading import Thread

import numpy as np
from scipy import io

import toupcam

import matplotlib.pyplot as plt

class TCam(object):
    '''
        Class to control camera from a Touptek camera.
        
        The camera relies on a callback method to get image as and when the camera sends it.
        
        This class is a thin wrapper to mimic the behavior of spinnaker sdk.
        
        Inputs:
            id: Camera id (0, 1, .. etc.)
            exposure: Exposure in ms
            gain: Gain in dB
        
    '''
    def __init__(self, id=0, exposure=10, gain=0, roi=None):
        self.id = id
        self.exposure = int(exposure*1000)
        self.gain = int(100*pow(10, gain/10))
        
        self.buffer = None
        self.image = None
        self.im_available = False
        self.roi = roi
        
        # Queue to store images
        self.queue = queue.LifoQueue()
        
        self.connect()

    def connect(self):
        self.camlist = toupcam.Toupcam.EnumV2()
        
        if len(self.camlist) < self.id+1:
            raise AttributeError('Could not find the required number of cameras')
        
        self.cam = toupcam.Toupcam.Open(self.camlist[self.id].id)
        
        self.W, self.H = self.cam.get_Size()
        
        self.cam.put_AutoExpoEnable(0)
        self.cam.put_ExpoTime(self.exposure)
        self.cam.put_ExpoAGain(self.gain)
        
        if self.roi is not None:
            xOffset, yOffset, xWidth, yWidth = self.roi
            self.cam.put_Roi(xOffset, yOffset,
                             xWidth, yWidth)
            
            self.H = yWidth
            self.W = xWidth
        
        # We will use 12 bit RAW measurements
        self.cam.put_Option(toupcam.TOUPCAM_OPTION_RAW, 1)
        self.cam.put_Option(toupcam.TOUPCAM_OPTION_BITDEPTH, 1)
    
        self.cam.put_RealTime(1)
        self.cam.put_Speed(9)
        
        self.cam.put_Option(toupcam.TOUPCAM_OPTION_DENOISE, 0)
    
        # Although measurements are 12 bit, they are aligned 16 bits, and hence this buffer configuration
        bufsize = toupcam.TDIBWIDTHBYTES(self.W * 16) * self.H
        
        self.buffer = bytes(bufsize)
        
        # This function starts the callback mechanism, where the camera informs
        # the software that an image is available.
        self.cam.StartPullModeWithCallback(self.callback, self)
        
        self.init_time = time.time()
        self.last_snapshot_time = 0
        
    @staticmethod
    def callback(event, ctx):
        # I am not sure why we need two callback functions
        if event == toupcam.TOUPCAM_EVENT_IMAGE:
            ctx._callback(event)
      
    def _callback(self, event):
        if event == toupcam.TOUPCAM_EVENT_IMAGE:
            self.cam.PullImageV3(self.buffer, 0, 16, 0, None)
            
            data = np.frombuffer(self.buffer, np.uint16)
            image = data.reshape(self.H, self.W)
            
            # This is for the capture function
            self.queue.put_nowait(image)

    def capture(self, nimg=1):
        image = self.queue.get()
        self.queue.task_done()
        return image
    
    def close(self):
        # Windows causes forrtl error when closing with ctrl+c.
        # You will need a SIGINT handler to ensure the camera is closed
        # correctly.
        #
        # See this link for further info: https://github.com/ContinuumIO/anaconda-issues/issues/905#issuecomment-232498034
        self.cam.Close()
        self.cam = None
        self.buffer = None

def exit_handler(signum, frame):
    print('Closing camera before exiting')
    global cam
    _ = cam.capture()
    cam.close()
    sys.exit(0)
    
def imshow(im_queue):
    while True:
        im = im_queue.get()
        im_norm = (im - im.min())/(im.max() - im.min())
        im_norm = cv2.resize(im_norm, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_AREA)
        cv2.imshow('Image', im_norm)
        cv2.waitKey(1)
        im_queue.task_done()

if __name__ == '__main__':
    # Constants
    camname = 'tcam'
    exposure = 33.33            # Exposure time in milliseconds
    camera_id = 0           # Hardware ID of the camera
    gain = 0                # Gain in dB. 
    
    # xoffset, yoffset, xwidth, ywidth. X = 640, Y = 512
    roi_size = None
    
    im_queue = queue.LifoQueue()

    cam = TCam(camera_id, exposure, gain, roi_size)
    
    discard = cam.capture()
    print(discard.shape)
    
    t1 = time.time()
    t2 = t1

    nframes = 0
    total_frames = 0
    
    signal.signal(signal.SIGINT, exit_handler)
    
    thread = Thread(target=imshow, args=[im_queue])
    thread.daemon = True
    thread.start()

    H, W = discard.shape
    imstack = np.zeros((H, W, 100), dtype=np.uint16)
    temp_stack = np.zeros(100)
    stack_saved = False
    
    cam.cam.put_Temperature(int(10))
    
    #while True:
    while total_frames < 10000:
        im = cam.capture(1)
        nframes += 1
        total_frames += 1
        
        im_queue.put_nowait(im)
        
        if nframes == 100:
            t2 = time.time()
            
            fps = 100/(t2 - t1 + 1e-12)
            temp = cam.cam.get_Temperature(0)/10.0
            print('Frame rate: %.2f | max: %.1f | mean: %.1f | std: %.1f | Temp: %.1f'%(
                fps, im.max(), im.mean(), im.std(), temp))
            
            t1 = t2
            nframes = 0
            
            imstack[..., (total_frames-1)//100] = im
            temp_stack[(total_frames-1)//100] = temp
            
    cam.close()