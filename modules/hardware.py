#!/usr/bin/env python

'''
    Module for controlling hardware devices such as display, spectrometer
    and (soon) Hamamatsu camera
'''


# System imports
from threading import Thread
import queue
import os
import sys
import pdb
import ctypes
from tqdm import tqdm
import csv
import time
    
try:
    from modules import toupcam
except OSError:
    print('SWIR camera will not work')
except ModuleNotFoundError:
    print('Probably running script from this folder')
    
# Numpy-ish imports
import numpy as np
import matplotlib.pyplot as plt

# Import Spinnaker module
try:
    import PySpin
except ImportError:
    print('PySpin not found. Spinnaker camera will not work')

# Hacky way of getting SDL2 path
from modules import this_path
SDL_HOME = this_path.__file__.replace('this_path.py', 'libsdl/')
    
# Set the SDL home only if in windows
if sys.platform == 'win32':
    os.environ['PYSDL2_DLL_PATH'] = SDL_HOME
    
# NOTE: It is very important to set this variable to zero, else only the
#       last created window works.
os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'

# SDL imports
import sdl2
import sdl2.ext

class Display(object):
    '''
        Display class for displaying fullscreen images on (second) screen.
        Relies on SDL2 library
    '''
    def __init__(self, screen_id=0, window_name='New window', initialize=True,
                 delay=5):
        '''
            Initializer for display class.

            Inputs:
                screen_id: Which screen to display images on. Default is 0
                window_name: Name of the window. Default is 'New window'
                initialize: If True, initialize video system.
                delay: Time delay after rendering in milliseconds

            Outputs:
                None
        '''
        self.id = screen_id
        self.window_name = window_name
        self.delay = delay

        # Use this for saving created surfaces.
        self._counter = 0

        # First, initialize SDL Video interface
        if initialize:
            sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)

        # Find number of displays
        ndisplays = sdl2.SDL_GetNumVideoDisplays()

        if ndisplays < screen_id+1:
            raise RuntimeError('Fewer displays than required')

        # Get screen bounds
        self.bounds = sdl2.SDL_Rect()
        ret = sdl2.SDL_GetDisplayBounds(self.id, self.bounds)

        # Add bounds size as a member variable
        self.shape = [self.bounds.h, self.bounds.w]

        # Create a new window
        flags = (sdl2.SDL_WINDOW_FULLSCREEN | sdl2.SDL_WINDOW_OPENGL |
                 sdl2.SDL_WINDOW_SHOWN)

        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 0)

        self._window = sdl2.SDL_CreateWindow(window_name.encode(),
                                             self.bounds.x,
                                             self.bounds.y,
                                             self.bounds.w,
                                             self.bounds.h,
                                             flags)

        # Create a GL Context for VSYNC
        self._glcontext = sdl2.SDL_GL_CreateContext(self._window)

        # Now create a renderer
        flags = (sdl2.SDL_RENDERER_ACCELERATED |
                 sdl2.SDL_RENDERER_PRESENTVSYNC |
                 sdl2.SDL_RENDERER_TARGETTEXTURE)
        self._renderer = sdl2.SDL_CreateRenderer(self._window, -1, flags)

        # Set swap interval to 1 for update synced with VSYNC
        retval = sdl2.SDL_GL_SetSwapInterval(1)
        if retval == -1:
            print('Adaptive VSYNC not supported')

        # Now time to create the data buffer for screen image.
        # XXX: Make sure the data is contiguous by setting order='c'
        h = self.bounds.h
        w = self.bounds.w
        self._data = np.zeros((h, w, 3), dtype=np.uint8, order='C')

        # Create the surface
        self._create_surface()

        # Start off by displaying the blank image
        self.render()

    def _create_surface(self):
        '''
            Convert RGB image to SDL2 surface
        '''
        h = self.bounds.h
        w = self.bounds.w

        # Masks for RGB channels
        rmask = 0x000000ff
        gmask = 0x0000ff00
        bmask = 0x00ff0000
        amask = 0x00000000

        # Compute the stride
        depth = 24
        pitch = 3*w

        # Create a data pointer out of member data buffer
        data_ptr = ctypes.c_void_p(self._data.ctypes.data)

        # Now create the surface
        self._surface = sdl2.SDL_CreateRGBSurfaceFrom(data_ptr,
                                                      ctypes.c_int(w),
                                                      ctypes.c_int(h),
                                                      ctypes.c_int(depth),
                                                      ctypes.c_int(pitch),
                                                      ctypes.c_uint32(rmask),
                                                      ctypes.c_uint32(gmask),
                                                      ctypes.c_uint32(bmask),
                                                      ctypes.c_uint32(amask))

    def create_textures_from_stack(self, imstack):
        '''
            Create a list of textures from a 3D numpy stack.

            Inputs:
                imstack: Numpy stack of arrays (H x W x nimg). Values must
                    be normalized between 0, 1 before passing.

            Outputs:
                textures: List of texture for each image
        '''
        textures = []

        if imstack.dtype == np.uint8:
            pass
        else:
            imstack = (imstack*255).astype(np.uint8)

        for idx in tqdm(range(imstack.shape[2])):
            textures.append(self.create_texture_from_numpy(imstack[:, :, idx]))

        return textures

    def create_texture_from_numpy(self, im):
        '''
            Create texture from a numpy array by saving and loading image.

            Inputs:
                im: Numpy array (uint8)

            Outputs:
                texture: Created texture.
        '''
        if np.ndim(im) == 2:
            im = np.tile(im[:, :, np.newaxis], [1, 1, 3])

        # Make sure the size of image matches size of screen
        [H, W, _] = im.shape

        if (H != self.bounds.h) or (W != self.bounds.w):
            raise ValueError('Image must be same size as screen')

        # Copy image data to our own buffer
        self._data[:, :, :] = im[:, :, :]

        # Now create texture
        im_texture = sdl2.SDL_CreateTextureFromSurface(self._renderer,
                                                       self._surface)

        return im_texture

    def destroy_textures(self, textures):
        for texture in textures:
            sdl2.SDL_DestroyTexture(texture)

    def render(self, texture=None):
        '''
            Render the image to GPU buffer and onto screen for a given texture.

            If texture is None, self._surface is used to create one.
        '''
        # https://stackoverflow.com/questions/22227811/ ...
        # ... sdl2-and-opengl-functions-with-two-windows
        #
        # If handling multiple windows, you need to set the current GL
        # context
        sdl2.SDL_GL_MakeCurrent(self._window, self._glcontext)
        
        if texture is None:
            texture = sdl2.SDL_CreateTextureFromSurface(self._renderer,
                                                        self._surface)

        #sdl2.SDL_RenderClear(self._renderer)
        sdl2.SDL_RenderCopy(self._renderer, texture, None, None)
        sdl2.SDL_RenderPresent(self._renderer)

    def showData(self, im):
        '''
            Display a given image.

            Inputs:
                im: Image of same size as screen. Can be either RGB or grayscale

            Outputs: None
        '''
        if np.ndim(im) == 2:
            im = np.tile(im[:, :, np.newaxis], [1, 1, 3])

        # Make sure the size of image matches size of screen
        [H, W, _] = im.shape

        if (H != self.bounds.h) or (W != self.bounds.w):
            raise ValueError('Image must be same size as screen')

        # Copy image data to our own buffer
        self._data[:, :, :] = im[:, :, :]

        # And then render
        self.render()

        # Wait for it to update for
        sdl2.SDL_Delay(self.delay)

    def close(self):
        '''
            Close the display
        '''
        # Python has a problem when closing the window -- the window
        # disappears only when python is closed. So just display all black
        # and then close
        self._data[:, :, :] = 0
        self.render()

        # Now it is okay to close
        sdl2.SDL_DestroyWindow(self._window)
        sdl2.SDL_FreeSurface(self._surface)
        sdl2.SDL_Quit()

class SPCamera(object):
    '''
        Create a class to handle Spinnaker (ptgrey) cameras.

        Inputs:
            cam_id: Camera id (default is 0)
            exposure_time: Exposure time in ms
            gain: Gain in dB
            trig_pin: Trigger pin. Set to -1 (default) to disable trigger

        Outputs:
            None
    '''
    def __init__(self, cam_id=0, exposure_time=10, binning=1,
                 trig_pin=-1, roi_pos=[-1, -1], roi_size=[0, 0],
                 fast=True, gain=0, debug=False, isrgb=False, israw=False,
                 isthermal=False):
        # The option fast=True is just to make the SPCamera compatible with
        # HCamera.
        
        # Create a PySpin system
        self.debug = debug
        self.isrgb = isrgb
        self.israw = israw
        self.isthermal = isthermal
        
        if debug:
            print('\tInitializing PySpin system')   
        self._system = PySpin.System.GetInstance()

        if debug:
            print('\tGetting list of cameras')
        self._cam_list = self._system.GetCameras()
        ncameras = self._cam_list.GetSize()

        if type(cam_id) == int:
            if ncameras < cam_id+1:
                raise IOError('Could not find required number of cameras')

            self._cam = self._cam_list[cam_id]
        elif type(cam_id) == str:
            self._cam = self._cam_list.GetBySerial(cam_id)

        # Initialize camera
        if debug:
            print('\tInitializing the camera')

        try:
            self._cam.Init()
        except PySpin.SpinnakerException:
            raise AttributeError('Could not initialize camera')

        if debug:
            print('\tSetting continuous acqusition mode')
        if self._cam.AcquisitionMode.GetAccessMode() != PySpin.RW:
            raise ValueError('Unable to set continuous acquisition mode.')

        self._cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        # Set Gain now
        if debug:
            print('\tSetting gain')
        if not self.isthermal:
            if self._cam.GainAuto.GetAccessMode() != PySpin.RW:
                raise ValueError('Cannot configure gain. Bailing out.')

            self._cam.GainAuto.SetValue(PySpin.GainAuto_Off)

            if self._cam.Gain.GetAccessMode() != PySpin.RW:
                raise ValueError('Cannot set gain. Bailing out.')

            self._cam.Gain.SetValue(gain)

        # Get Nodemap
        nodemap = self._cam.GetNodeMap()
        
        # Need nodemap for this
        if not self.isthermal:
            if debug:
                print('\tSetting trigger')

            # Rely on QuickSpin settings
            if self._cam.TriggerSource.GetAccessMode() != PySpin.RW:
                raise AttributeError('Cannot control trigger.')

            self.software_trigger = False
            
            if trig_pin == -1:
                self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                self.software_trigger = True
            elif trig_pin == 0:
                self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
            elif trig_pin == 1:
                self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Line1)
            elif trig_pin == 2:
                self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Line2)
            else:
                raise AttributeError('Trigger pin is not supported')

            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_On)

            # Before setting exposure, disable framerate control
            if debug:
                print('\tDisabling auto frame rate')

            af_str = 'AcquisitionFrameRateEnabled'
            autoframe_mode = PySpin.CBooleanPtr(nodemap.GetNode(af_str))

            if not PySpin.IsAvailable(autoframe_mode):
                print('autoframe feature does not exist')
            else:
                autoframe_mode.SetValue(False)

            # Set Gamma
            if self._cam.GammaEnable.GetAccessMode() == PySpin.RW:
                self._cam.GammaEnable.SetValue(False)
            else:
                print('Cannot set Gamma level')

            # Disable black level
            if debug:
                print('\tDisabling black level')

            bl_str = 'BlackLevelEnabled'
            blacklevel_mode = PySpin.CBooleanPtr(nodemap.GetNode(bl_str))

            if not PySpin.IsAvailable(blacklevel_mode) or not (
                PySpin.IsReadable(blacklevel_mode)):
                print('Cannot configure black level.')
            else:
                blacklevel_mode.SetValue(False)
            
            # Exposure can be configured only after trigger.
            if debug:
                print('\tSetting exposure')
            if self._cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                raise AttributeError('Cannot configure exposure. Bailing out.')

            self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

            if self._cam.ExposureTime.GetAccessMode() != PySpin.RW:
                raise AttributeError('Unable to set exposure time. Bailing out.')

            #self._cam.ExposureTime.SetValue(exposure_time*1000)
            self._cam.ExposureTime.SetValue(exposure_time)
            
        # Thermal settings
        if self.isthermal:
            if debug:
                print('\tSetting thermal camera properties')
                
            sNodemap = self._cam.GetTLStreamNodeMap()

            # Change bufferhandling mode to NewestOnly
            if debug:
                print('\tChanging buffer handing')
            node_bufferhandling_mode = PySpin.CEnumerationPtr(
                sNodemap.GetNode('StreamBufferHandlingMode'))

            if debug:
                print('\tChanging pixel format')
            node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
            node_pixel_format_mono16 = PySpin.CEnumEntryPtr(
                node_pixel_format.GetEntryByName('Mono16'))
            pixel_format_mono16 = node_pixel_format_mono16.GetValue()
            node_pixel_format.SetIntValue(pixel_format_mono16)
            
            if debug:
                print('\tSetting CMOS bit depth')
            node_bit_depth = PySpin.CEnumerationPtr(nodemap.GetNode('CMOSBitDepth'))
            #node_bit_depth_16bit = PySpin.CEnumEntryPtr(
            #    node_bit_depth.GetEntryByName('bit16bit'))
            #bit_depth = node_bit_depth_16bit.GetValue()
            #node_bit_depth.SetIntValue(bit_depth)

            if debug:
                print('\tEnabling signal linear mode')
            #node_temp_linear = PySpin.CEnumerationPtr(nodemap.GetNode('TemperatureLinearMode'))
            #node_temp_linear_on = PySpin.CEnumEntryPtr(node_temp_linear.GetEntryByName('Off'))
            #node_on = node_temp_linear_on.GetValue()
            #node_temp_linear.SetIntValue(node_on)
            
            if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
                raise AttributeError('Unable to set stream buffer handling mode.. Aborting...')
                
            # Retrieve entry node from enumeration node
            node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
            if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
                raise AttributeError('Unable to set stream buffer handling mode.. Aborting...')

            # Retrieve integer value from entry node
            node_newestonly_mode = node_newestonly.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
            

        # Start continuous acquisition
        if debug:
            print('\tStarting acquisition')
        self._cam.BeginAcquisition()

        # Set pixel format to Mono 16
        self._cam.EndAcquisition()
        
        if not self.isthermal:
            if self.isrgb:
                if self.israw:
                    pixformat = PySpin.PixelFormat_BayerRG16
                else:
                    pixformat = PySpin.PixelFormat_RGB8Packed
            else:
                pixformat = PySpin.PixelFormat_Mono8
                
            if self._cam.PixelFormat.GetAccessMode() == PySpin.RW:
                self._cam.PixelFormat.SetValue(pixformat)
            else:
                print('\tCannot change pixel format')

            # Disable white balancing if pixelformat is RGB
            if self.isrgb:
                if self._cam.BalanceWhiteAuto.GetAccessMode() == PySpin.RW:
                    self._cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
                else:
                    print('\tCannot disable auto white balance')

                if self._cam.BalanceRatioSelector.GetAccessMode() == PySpin.RW:
                    selector = PySpin.BalanceRatioSelector_Blue
                    self._cam.BalanceRatioSelector.SetValue(selector)
                else:
                    print('\tCannot set blue balance ratio selector')

                if self._cam.BalanceRatio.GetAccessMode() == PySpin.RW:
                    self._cam.BalanceRatio.SetValue(2.0)
                else:
                    print('\tCannot set white balance ratio for blue channel')

                if self._cam.BalanceRatioSelector.GetAccessMode() == PySpin.RW:
                    selector = PySpin.BalanceRatioSelector_Red
                    self._cam.BalanceRatioSelector.SetValue(selector)
                else:
                    print('\tCannot set red balance ratio selector')

                if self._cam.BalanceRatio.GetAccessMode() == PySpin.RW:
                    self._cam.BalanceRatio.SetValue(1.0)
                else:
                    print('\tCannot set white balance ratio for red channel')

        self._cam.BeginAcquisition()

        if not self.isthermal:
            # Set binning
            self.set_binning(binning)

            # Set region of interest
            self.set_roi(roi_size, roi_pos)
            
    def set_gain(self, gain):
        if not self.isthermal:
            self._cam.Gain.SetValue(gain)
            
    def set_A655_frame_rate(self, framerate):
        '''
            Set frame rate for A655sc
            
            Frame rates should be one of 3, 6, 12, 25, 50
        '''
        if framerate not in [3, 6, 12, 25, 50]:
            raise ValueError('Frame rate should be 3, 6, 12, 25, 50')
        nodemap = self._cam.GetNodeMap()
        irf_node = PySpin.CEnumerationPtr(nodemap.GetNode('IRFrameRate'))
        
        rate_val = PySpin.CEnumEntryPtr(irf_node.GetEntryByName('Rate%dHz'%framerate))
        irf_node.SetIntValue(rate_val.GetValue())
        
    def set_binning(self, binning=1):
        '''
            Set binning
        '''
        # Stop acquisition before changing binning
        self._cam.EndAcquisition()
        if self._cam.BinningVertical.GetAccessMode() == PySpin.RW:
            self._cam.BinningVertical.SetValue(binning)
        else:
            print('\tCannot change vertical binning')

        if self._cam.BinningHorizontal.GetAccessMode() == PySpin.RW:
            self._cam.BinningHorizontal.SetValue(binning)
        else:
            print('\tCannot change horizontal binning')

        # Then restart acquisition
        self._cam.BeginAcquisition()

    def set_exposure(self, exposure_time=16.5):
        '''
            Set exposure time on the fly.

            Inputs:
                exposure_time: Exposure time in ms
        '''
        if self._cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            raise AttributeError('Cannot configure exposure.')

        self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

        if self._cam.ExposureTime.GetAccessMode() != PySpin.RW:
            raise AttributeError('Unable to set exposure time.')

        self._cam.ExposureTime.SetValue(exposure_time*1000)

    def set_roi(self, roi_size, roi_offset=[-1, -1]):
        '''
            Change region of interest, and binning.

            Inputs:
                roi_size: 2-tuple with Height and width of region of interest
                roi_offset: 2-tuple of top left corner of ROI. If any entry
                    is -1, roi is centered.

            WARNING: ROI size and offset have to respect binning
        '''
        # Stop acquisition before changing ROI configuration
        self._cam.EndAcquisition()
        
        Hmax = self._cam.Height.GetMax()
        Wmax = self._cam.Width.GetMax()

        H, W = roi_size
        Ho, Wo = roi_offset

        # If any entry of roi_size is 0, set roi size to maximum
        if (H == 0) or (W == 0):
            H = Hmax
            W = Wmax

        if ((H + Ho) > Hmax) or ((W + Wo) > Wmax):
            raise ValueError('ROI configuration is wrong')

        # Set ROI dimensions first
        if self._cam.Width.GetAccessMode() == PySpin.RW:
            self._cam.Width.SetValue(W)
        else:
            print('Cannot set width of ROI')

        if self._cam.Height.GetAccessMode() == PySpin.RW:
            self._cam.Height.SetValue(H)
        else:
            print('Cannot set height of ROI')

        # Then set offset
        if (Ho < 0) or (Wo < 0):
            Ho = 8 * ((Hmax - H)//16)
            Wo = 8 * ((Wmax - W)//16)

        if self._cam.OffsetX.GetAccessMode() == PySpin.RW:
            self._cam.OffsetX.SetValue(Wo)
        else:
            print('Cannot set width offset')

        if self._cam.OffsetY.GetAccessMode() == PySpin.RW:
            self._cam.OffsetY.SetValue(Ho)
        else:
            print('Cannot set height offset')

        # Now we can restart acquisition
        self._cam.BeginAcquisition()
            
        
    def capture(self, navg=1):
        '''
            Capture and return an image
        '''
        imgcap = 0
        for idx in range(navg):
            try:
                if not self.isthermal:
                    if self.software_trigger:
                        self._cam.TriggerSoftware.Execute()
                    
                img = self._cam.GetNextImage()
                H = img.GetHeight()
                W = img.GetWidth()

                # Convert to RGB if required
                if self.isrgb:
                    if self.israw:                       
                        imgdata = img.GetData().reshape(H, W)
                    else:
                        imrgb = img.Convert(PySpin.PixelFormat_RGB8,
                                            PySpin.HQ_LINEAR)
                        imgdata = imrgb.GetData().reshape(H, W, -1)
                else:
                    imgdata = img.GetData().reshape(H, W)

                if navg == 1:
                    img.Release()
                    return imgdata
                else:
                    imgcap += imgdata.astype(np.float32)

                img.Release()
            except Exception as err:
                print(type(err))
                print(err.args)
                print(err)
                
                pdb.set_trace()
                print('Not sure what happened, but shutting down system')
                self.close()

                raise SystemError('Camera malfunctioned')

        return imgcap/navg

    def close(self):
        '''
            Close the camera and release the system
        '''
        if not self.isthermal:
            # Reset trigger
            if self._cam.TriggerMode.GetAccessMode() != PySpin.RW:
                raise AttributeError('Unable to disable trigger')

            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        self._cam.EndAcquisition()
        
        if self.debug:
            print('\tDeleting camera')
        del self._cam

        if self.debug:
            print('\tClearing camera list')
        self._cam_list.Clear()

        if self.debug:
            print('\tReleasting PySpin system')
        self._system.ReleaseInstance()
        
class TCam(object):
    '''
        Class to control camera from a Touptek camera.
        
        The camera relies on a callback method to get image as and when the camera sends it.
        
        This class is a thin wrapper to mimic the behavior of spinnaker sdk.
        
        Inputs:
            id: Camera id (0, 1, .. etc.)
            exposure: Exposure in ms
            gain: Gain in dB
            trigger: 0 -- None
                     1 -- Hardware
                     2 -- Software
        
    '''
    def __init__(self, id=0, exposure=10, gain=0, binning=1, roi=None, trigger=0):
        self.id = id
        self.exposure = int(exposure*1000)
        self.gain = int(100*pow(10, gain/10))
        
        self.buffer = None
        self.image = None
        self.im_available = False
        self.binning = binning
        self.roi = roi
        self.trigger = trigger
        
        # An async queue enables better data reading
        self.queue = queue.LifoQueue()
        
        self.connect()

    def connect(self):
        self.camlist = toupcam.Toupcam.EnumV2()
        
        if len(self.camlist) < self.id+1:
            raise AttributeError('Could not find the required number of cameras')
        
        self.cam = toupcam.Toupcam.Open(self.camlist[self.id].id)
        
        self.W, self.H = self.cam.get_FinalSize()
        
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

        self.cam.put_Option(toupcam.TOUPCAM_OPTION_BINNING, self.binning)
        
        # Resolution may change after binning
        self.W, self.H = self.cam.get_FinalSize()
        
        self.cam.put_Speed(9)
        
        # Although measurements are 12 bit, they are aligned 16 bits, and hence
        # this buffer configuration
        bufsize = toupcam.TDIBWIDTHBYTES(self.W * 16) * self.H
        
        self.buffer = bytes(bufsize)
        
        # This function starts the callback mechanism, where the camera informs
        # the software that an image is available.
        self.cam.StartPullModeWithCallback(self.callback, self)
        
        self.init_time = time.time()
        
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
            
            # Put image into the buffer
            self.queue.put_nowait(image)
            
    def _capture(self):
        image = self.queue.get()
        self.queue.task_done()
        return image
    
    def capture(self, nimg=1):
        image = 0

        for _ in range(nimg):
            image += self._capture().astype(np.float32)

        return image/nimg
    
    def flush(self):
        while not self.queue.empty():
            _ = self.queue.get_nowait()
    
    def close(self):
        # Windows causes forrtl error when closing with ctrl+c.
        # You will need a SIGINT handler to ensure the camera is closed
        # correctly.
        #
        # See this link for further info: https://github.com/ContinuumIO/anaconda-issues/issues/905#issuecomment-232498034
        self.cam.Close()
        self.cam = None
        self.buffer = None

def get_camera(camname, camera_id, exposure, gain=0, binning=1,
               trig_pin=-1, roi_pos=[0, 0],
               roi_size=[2048, 2048]):
    '''
        Wrapper function to get a camera controller object
        
        Inputs:
            camname: One of 'pco', 'oryx', 'tcam', 'hcam'
            camera_id: ID of the camera
            exposure: Exposure time in ms
            gain: Gain in dB
            binning: Binning, typically 1, 2, or 4
            trig_pin: Trigger pin. Currently not implemented
            roi_pos: Starting (top left) of ROI
            roi_size: Size of the ROI
    
        Outputs:
            camera: Camera control object
    '''
    if camname == 'oryx':
        camera = SPCamera(camera_id, exposure,
                               gain=gain, trig_pin=trig_pin,
                               isrgb=False, binning=binning)
    elif camname == 'tcam':
        camera = TCam(camera_id, exposure, gain, binning, roi_size)
        
    return camera
