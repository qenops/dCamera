#!/usr/bin/python
 
__author__ = ('David Dunn')
__version__ = '0.3'

import numpy as np
import cv2
import time

BACKEND = {'cv2':0, 'flyCap':1, 'picamera':2, 'spinnaker':3} # different camera APIs that are supported
try:
    from shared_modules.pyfly2 import pyfly2
except ImportError:
    print("Warning: FlyCapture backend is not available.")
    BACKEND['flyCap'] = None
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
except ImportError:
    print("Warning: picamera backend is not available.")
    BACKEND['picamera'] = None
try: 
    import PySpin
except ImportError:
    print("Warning: Spinnaker backend is not available.")
    BACKEND['spinnaker'] = None

__TEST__MODE__ = False

def cv2CloseWindow(window):
    cv2.destroyWindow(window)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

class Camera(object):
    ''' camera objects should store resolution and calibration data as well 
        as provide methods for capturing and viewing content regardless of source of camera data
        By default uses cv2 backend - use subclasses for other backends
    '''
    def __init__(self,id=None,**kwargs):
        self.id = 0 if id is None else id
        self._cap = kwargs.get('_cap', None)
        self.fps = kwargs.get('fps', 30)
        self.shutter_speed = kwargs.get('shutter_speed',0)
        self.iso = kwargs.get('iso',0)
        self.exposure_mode = kwargs.get('exposure_mode','auto')
        self.awb_mode = kwargs.get('awb_mode','auto')
        self._continuousCapture = None
        self.roi = np.array(((0,0),(0,0)), dtype=np.uint16)
        self.roi[1] = (0,0) if self.resolution is None else self.resolution
        self.distortion = CameraDistortion()
        self.nonDefaults = {}
    def open(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.id)
            if not self._cap.isOpened():          # check if we succeeded
                self._cap = None
                return 0
            return 1
        return 2
    def close(self):
        if self._cap is not None:           
            self._cap.release()
            self._cap = None
    release = close
    def read(self, video=False):
        if self.open():
            ret, img = self._cap.read()
            if ret != 1:
                raise IOError('Camera image not captured')
            return img     
    def readUndistort(self, video=False):
        image = self.read(video)
        return self.distortion.undistort(image)
    def view(self):
        streamVideo(self)
    def viewUndistort(self):
        streamVideo(self, True)
    def captureFrames(self):
        return captureFrames(self)
    def captureUndistort(self):
        return captureFrames(self, True)
    @property
    def resolution(self):
        if self.open():
            try:
                return (int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
            except:
                return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    def getNPResolution(self, kwargs=None):
        return (self.resolution[1], self.resolution[0])
    def regionOfInterest(self, img):
        return img[self.roi[0][1]:self.roi[1][1],self.roi[0][0]:self.roi[1][0],:]
    def calibrate(self, gridCorners, gridScale, **kwargs):
        images = captureFrames(self)
        if __TEST__MODE__ and images is None:
            images = [cv2.imread('../data/calibration/%02d.png' % i) for i in range(8)]
        return self.distortion.calibrate(images, gridCorners, gridScale, **kwargs)
    def save(self, file):
        return np.savez(file,distortion=self.distortion,nonDefaults=self.nonDefaults)
    def load(self, file):
        npzfile = np.load(file)
        self.distortion = npzfile['distortion']
        self.nonDefaults = npzfile['nonDefaults']

class FlyCamera(Camera): #TODO: rewrite to be similar to SpinCamera
    def __init__(self,id=None,**kwargs):
        if BACKEND['flyCap'] is None:
            raise ImportError('dCamera: [pyFly2] interface not loaded!')
        super().__init__(id,**kwargs)
        self.fps = kwargs.get('fps', 15)
        self._context = None
    def open(self):
        if self._cap is None:
            self._context = pyfly2.Context()
            if self._context.num_cameras < 1:
                #raise ValueError('No PointGrey cameras found')
                self._cap = None
                return 0
            self._cap = self._context.get_camera(self.id)
            self._cap.Connect()
            self._cap.StartCapture()
            print("Connected PointGrey camera %s" % self.id)
            return 1
        return 2
    def close(self):
        if self._cap is not None:
            self._cap.StopCapture()
        self._cap = None
    release = close
    def read(self, video=False):
        if self.open():
            return self._cap.GrabNumPyImage('bgr')
    @property
    def resolution(self):
        if self.open():
            return self._cap.GetSize()

class PiCamera(Camera):
    def __init__(self,id=None,**kwargs):
        if BACKEND['picamera'] is None:
            raise ImportError('dCamera: [picamera] interface not loaded!')
        super().__init__(id,**kwargs)
        self._cam = None
        # No way to query resolution on a picamera
        self._resolution = kwargs.get('resolution',kwargs.get('res',(640,480)))
        ## TODO get the serial number and put it in self.id
    def open(self):
        if self._cap is None:
            self._cam = PiCamera()
            self._cam.resolution = self.resolution
            self._cam.framerate = self.fps
            self._cap = PiRGBArray(self._cam, size=tuple(self._cam.resolution))
            time.sleep(2)   # pause while the camera inits and settles
            if self.shutter_speed is None:
                self.shutter_speed = self._cam.exposure_speed
            self._cam.shutter_speed = self.shutter_speed
            self._cam.iso = self.iso
            self._cam.exposure_mode = self.exposure_mode
            if self.awb_mode == 'off':
                g = self._cam.awb_gains
                self._cam.awb_mode = self.awb_mode
                self._cam.awb_gains = g
            self._cam.awb_mode = self.awb_mode
            return 1
        return 2
    def close(self):
        if self._cap is not None:
            self._continuousCapture = None
            self._cap.close()
            self._cam.close()
            self._cam = None
        self._cap = None
    def read(self, video=False):
        if self.open():
            if not video:
                self._continuousCapture = None
                self._cam.capture(self._cap, format='bgr')
                image = self._cap.array
                self._cap.truncate(0)
                return image
            else:
                if self._continuousCapture is None:
                    self._continuousCapture = self._cam.capture_continuous(self._cap, format='bgr', use_video_port=True)
                frame = next(self._continuousCapture)
                image = frame.array
                self._cap.truncate(0)
                return image
    @property
    def resolution(self): # No way to query on a picamera (we could get a frame and measure)
        return self._resolution

class SpinCamera(Camera):
    nTypePtr = [PySpin.CValuePtr,
                PySpin.CBasePtr,
                PySpin.CIntegerPtr,
                PySpin.CBooleanPtr,
                PySpin.CCommandPtr,
                PySpin.CFloatPtr,
                PySpin.CStringPtr,
                PySpin.CRegisterPtr,
                PySpin.CCategoryPtr,
                PySpin.CEnumerationPtr,
                PySpin.CEnumEntryPtr,
                None, #PySpin.CPortPtr, #This one doesn't exist?
                ]
    def __init__(self,id=None,**kwargs):
        if BACKEND['spinnaker'] is None:
            raise ImportError('dCamera: [PySpin] interface not loaded!')
        id = 0 if id is None else id
        self._system = PySpin.System.GetInstance()
        cameras = self._system.GetCameras()
        self._cap = None
        self._TLnodemap = None
        self._nodemap = None
        if id < len(cameras):   # id is < len(cameras); it is an index
            self._cap = cameras[id]
            TLnodemap = self._cap.GetTLDeviceNodeMap()
            node = TLnodemap.GetNode('DeviceSerialNumber')
            id = int(PySpin.CStringPtr(node).ToString())
        else:   # id is a serial number 
            for cam in cameras:
                TLnodemap = cam.GetTLDeviceNodeMap()
                node = TLnodemap.GetNode('DeviceSerialNumber')
                if id == int(PySpin.CStringPtr(node).ToString()):
                    self._cap = cam
                    break
            if self._cap is None:
                raise ValueError(f'dCamera: Camera serial number {id} not found')
        self._TLnodemap = self._cap.GetTLDeviceNodeMap()
        self._cap.Init()
        self._nodemap = self._cap.GetNodeMap()
        super().__init__(id, _cap=self._cap, **kwargs)
        self.colorConverter = cv2.COLOR_BAYER_RG2RGB_EA
    def isOpen(self):
        '''None of the following works, so just going to try and set something
        self._cap.AcquisitionStatus
        self._cap.AcquisitionStatusSelector
        node = self._nodemap.GetNode('PixelFormat')
        PySpin.IsWritable(node)
        node.GetAccessMode()
        a = dc.PySpin.gcstring()
        b = dc.PySpin.gcstring()
        PySpin.CEnumerationPtr(node).GetProperty('pIsLocked',a,b)
        '''
        current = self.getNodeValue('PixelFormat')
        try:
            self.setNodeValue('PixelFormat', current)
        except PySpin.SpinnakerException:
            return True
        return False
    def open(self):
        if self._cap is not None:
            if not self.isOpen():
                self._cap.BeginAcquisition()
                return 1
            return 2
        return 0
    def close(self):
        if self._cap is not None and self.isOpen():
            self._cap.EndAcquisition()
    def release(self):
        self.close()
        self._cap.DeInit()
        self._nodemap = None
        self._cap = None
        self._TLnodemap = None
        self._system.ReleaseInstance()
    def read(self, video=False):
        if self.open():
            #try:
            frame = self._cap.GetNextImage()
            if frame.IsIncomplete():
                print('frame incomplete')
                return  # should I do something else here?
            else:
                image = np.array(frame.GetData(), dtype='uint8').reshape((frame.GetHeight(),frame.GetWidth(),-1))
                frame.Release()
                color = cv2.cvtColor(image, self.colorConverter) if self.colorConverter is not None else image
                return color
            #except PySpin.SpinnakerException:
            #    logging.exception()
    def getNodeValue(self, nodeName):
        '''we could cache the pointers for the nodes in a dict, but 
        I don't think that would be advantageous '''
        if self._nodemap is not None:
            node = self._nodemap.GetNode(nodeName)
            if not PySpin.IsReadable(node) and not PySpin.IsAvailable(node):
                print(f"{nodeName} is not readable or available")
                return None
            interface = node.GetPrincipalInterfaceType()
            pType = self.nTypePtr[interface]
            if interface in [2,3,5,6,10]: # Not sure if Enum belongs here or not
                return pType(node).GetValue()
            elif interface == 9: # EnumerationPtrs don't have Get Value()
                return pType(node).GetCurrentEntry().GetSymbolic()
            elif interface == 4: # Command Ptr
                pass
            elif interface == 7: # Register Ptr
                pass
            elif interface == 8: # Category Ptr
                pass
            elif interface == 11: # Ports don't have values?
                pass
    def setNodeValue(self, nodeName, value):
        if self._nodemap is not None:
            node = self._nodemap.GetNode(nodeName)
            if not PySpin.IsWritable(node) and not PySpin.IsAvailable(node):
                print(f"{nodeName} is not writeable or available")
                return 0
            interface = node.GetPrincipalInterfaceType()
            pType = self.nTypePtr[interface]
            if interface in [2,3,5,6]: 
                pType(node).SetValue(value)
                self.nonDefaults[nodeName] = value
                return 1
            elif interface == 9: # Enumeration will pass name of enum
                intValue = pType(node).GetEntryByName(value).GetValue()
                pType(node).SetIntValue(intValue)
                self.nonDefaults[nodeName] = value
                return 1
            elif interface == 4: # Command Ptr
                pass
            elif interface == 7: # Register Ptr
                pass
            elif interface == 8: # Category Ptr
                pass
            elif interface == 11: # Ports don't have values?
                pass
    @property
    def maxResolution(self):
        return self.getNodeValue('WidthMax'), self.getNodeValue('HeightMax')
    @property
    def resolution(self):
        return self.getNodeValue('Width'), self.getNodeValue('Height')
    @resolution.setter
    def resolution(self, value):
        self.setNodeValue('Width', value[0])
        self.setNodeValue('Height', value[1])
    def setFormatFast(self):
        if not self.isOpen():
            self.setNodeValue('PixelFormat', 'BayerRG8')
            self.setNodeValue('IspEnable', False)
            self.colorConverter = cv2.COLOR_BAYER_RG2RGB_EA
    def setFormatHigh(self):
        if not self.isOpen():
            self.setNodeValue('PixelFormat', 'YCbCr8')
            self.colorConverter  = cv2.COLOR_YCrCb2RGB
    def load(self, file):
        super().load(file)
        for k,v in self.nonDefaults:
            self.setNodeValue(k, v)


class CameraDistortion(object):
    ''' these store the parameters and functions for calculating and fixing
        camera distortion
    '''
    def __init__(self, **kwargs):
        self.matrix = kwargs.get('matrix', np.identity(3))
        self.distortion = kwargs.get('distortion', [0,0,0,0,0])
        self.error = kwargs.get('error', 1)
        self.mapX = None
        self.mapY = None
        self.resolution = None
        self.interpolation = cv2.INTER_LINEAR
    def undistort(self, image): #TODO fix image resolution -> numpy flipping
        if self.mapX is None or self.mapY is None or image.shape != self.resolution:
            self.resolution = image.shape
            self.mapX, self.mapY = self.getRemaps()
        return cv2.remap(image, self.mapX, self.mapY, self.interpolation)
    def getRemaps(self):
        return cv2.initUndistortRectifyMap(self.matrix, self.distortion, np.eye(3), self.matrix, self.resolution, cv2.CV_16SC2)
    def calibrate(self, images, gridCorners, gridScale, **kwargs):     # flags=cv2.CALIB_FIX_K3
        ret, matrix, dist = calibrateChess(images, gridCorners, gridScale, **kwargs)
        if ret < self.error:
            self.error = ret
            self.matrix = matrix
            self.distortion = dist
            self.mapX = None
            self.mapY = None
        return ret
    def undistort(self, images, alpha=0.):
        toReturn = []
        frame = cv2.undistort(frame,matrix,dist)
        size = images[0].shape
        newMtx,roi = cv2.getOptimalNewCameraMatrix(self.matrix,self.distortion,(size[1],size[0]),alpha)
        map1, map2 = cv2.initUndistortRectifyMap(self.matrix,self.distortion,np.eye(3),newMtx,(size[1],size[0]),cv2.CV_16SC2)
        for img in images:
            toReturn.append(cv2.remap(img,map1,map2,cv2.INTER_LINEAR))
        return toReturn

def toGray(image):
    iy, ix, channels = image.shape if len(image.shape)>2 else [image.shape[0], image.shape[1], 1]
    if channels == 4:
        gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
    elif channels == 3:  
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray

def calibrateChess(images, gridCorners, gridScale, **kwargs):
    ''' get the calibration of a camera from the images of a chessboard with number of gridCorners given'''
    cameraMatrix = kwargs.get('matrix',np.eye(3))
    distCoeffs = kwargs.get('distortion',np.zeros([14]))
    # termination criteria
    criteria = kwargs.get('criteria',(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    # prepare object points, define top left gridCorner as origin: like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((gridCorners[0]*gridCorners[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:gridCorners[0],0:gridCorners[1]].T.reshape(-1,2)
    objp *= gridScale
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    #markedImages = [] # store the updated images
    for image in images:
        gray = toGray(image)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, gridCorners,None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners) 
        # Draw and display the corners
        #temp = img
        #cv2.drawChessboardCorners(temp, gridCorners, corners,ret)
        #markedImages.append(temp)
    print('Using %s of %s images.'%(len(objpoints), len(images)))
    #slideShow(markedImages)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, **kwargs)
    return ret, mtx, dist

def slideShow(images,scale=False):
    current = 0
    while(True):
        display = images[current]
        display = display.astype(np.float32)/display.max() if scale else display
        cv2.imshow('frame',display)
        ch = cv2.waitKey() 
        if ch == -1 or ch & 0xFF == 27:         # escape
            break
        #else:
        #    print(ch)
        elif ch == 81: #1113937:         # left
            current -= 1
        elif ch == 83: #1113939:         # right
            current += 1
        current = current % len(images)
    cv2CloseWindow('frame')

def captureFrames(cam=None, undistort=False, disparity=False):
    cam = cv2.VideoCapture(0) if cam is None else cam
    if not cam.open():
        return None
    frames = []
    readFunc = cam.read if not undistort else cam.readUndistort if not disparity else cam.readDisparity
    while(True):
        frame = readFunc()          # Capture the frame
        separate = getattr(cam, 'mode', 1)
        img = frame if separate != 0 else np.hstack(frame)
        cv2.imshow('frame',img)   # Display the frame
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:                # escape
            break
        elif ch == 32:              # space bar
            frames.append(frame)
    #cam.release()                   # release the capture
    cv2CloseWindow('frame')
    return frames

def streamVideo(cam=None, undistort=False, disparity=False):
    if cam is None:
        cam = cv2.VideoCapture(0)
    if not cam.open():
        return None
    readFunc = cam.read if not undistort else cam.readUndistort if not disparity else cam.readDisparity
    while(True):
        frame = readFunc(video=True)     # Capture the frame
        separate = getattr(cam, 'mode', 1)
        frame = frame if separate != 0 else np.hstack(frame)
        cv2.imshow('frame',frame)   # Display the frame
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:                # escape
            break
    #cam.release()                   # release the capture
    cv2CloseWindow('frame')
    
def tuneUndistort(cam):
    if not cam.open():
        return None
    readFunc = cam.readUndistort
    scale = .01
    while(True):
        frame = readFunc(video=True)     # Capture the frame
        separate = getattr(cam, 'mode', 1)
        frame = frame if separate != 0 else np.hstack(frame)
        cv2.imshow('frame',frame)   # Display the frame
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:                # escape
            break
        elif ch > 80 and ch < 87:   # arrow key
            if ch == 83: #right
                cam.T += scale * np.array(((1,),(0,),(0,)))
            elif ch == 81: #left
                cam.T += scale * np.array(((-1,),(0,),(0,)))
            elif ch == 82: #up
                cam.T += scale * np.array(((0,),(1,),(0,)))
            elif ch == 84: #down
                cam.T += scale * np.array(((0,),(-1,),(0,)))
            elif ch == 85: #page up
                cam.T += scale * np.array(((0,),(0,),(1,)))
            elif ch == 86: #page down
                cam.T += scale * np.array(((0,),(0,),(-1,)))
            cam.rectify()
        elif ch == 112: #p
            print(cam.T)
        elif ch == 46: #>
            scale *= .1
            print('Scale is %s'%scale)
        elif ch == 44: #<
            scale *= 10
            print('Scale is %s'%scale)
    #cam.release()                   # release the capture
    cv2CloseWindow('frame')

def captureVideo(fname, cam=None, undistort=False, disparity=False):
    # TODO update to use picamera's native capture method
    cam = Camera(0) if cam is None else cam
    if not cam.open():
        return None
    readFunc = cam.read if not undistort else cam.readUndistort if not disparity else cam.readDisparity
    video = cv2.VideoWriter(fname,-1,cam.fps,cam.resolution)
    #frames = []
    while(True):
        frame = readFunc(video=True)     # Capture the frame
        #if view:  # how can I do keyboard controls to escape without streaming the video?
        video.write(frame)
        #frames.append(frame)
        # TODO: add red recording dot in one of the corners of the displayed frame
        cv2.imshow('frame',frame)   # Display the frame
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:                # escape
            break
    #print len(frames)
    #cam.release()                   # release the capture
    video.release()
    cv2CloseWindow('frame')
    

if __name__ == '__main__':
    __TEST__MODE__ = True
    cam = Camera(0)
    ret = cam.calibrate((4,3), flags=cv2.CALIB_FIX_K3)
    temp = cv2.undistort(cv2.imread('../data/calibration/00.png'),cam.matrix,cam.distortion)
    cv2.imwrite('temp.png', temp)
    print(cam.matrix)
    np.savez('camMatrix',mtx=cam.matrix,dist=cam.distortion)
