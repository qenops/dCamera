#!/usr/bin/python
 
__author__ = ('David Dunn')
__version__ = '0.1'

import numpy as np
import cv2
try:
    from shared_modules.pyfly2 import pyfly2
except ImportError:
    print("Warning: PyFly2 is not available.")

modes = {'cv2':0, 'ptGrey':1} # different camera APIs that are supported

__TEST__MODE__ = False

def cv2CloseWindow(window):
    cv2.destroyWindow(window)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

class Camera(object):
    ''' camera objects should store resolution and calibration data as well 
        as provide methods for capturing and viewing content regardless of source of camera data (cv2 or ptGrey)
    '''
    def __init__(self,id,mode=modes['cv2']):
        self.id = id
        self.mode = mode
        self._cap = None
        self._context = None
        self.resolution = [0,0]
        self.resolution = self.getResolution()
        self.fps = 30 if self.mode == modes['cv2'] else 15
        self.close()
        self.matrix = np.identity(3)
        self.distortion = [0,0,0,0,0]
        self.error = 1
    def open(self):
        if self.id is None:
            return 0
        if self._cap is None:
            if self.mode == modes['cv2']:
                self._cap = cv2.VideoCapture(self.id)
                if not self._cap.isOpened():          # check if we succeeded
                    self._cap = None
                    return 0
                return 1
            elif self.mode == modes['ptGrey']:
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
            if self.mode == modes['cv2']:
                self._cap.release()
            elif self.mode == modes['ptGrey']:
                self._cap.StopCapture()
            self._cap = None
    release = close
    def read(self):
        if self.open():
            if self.mode == modes['cv2']:
                return self._cap.read()
            elif self.mode == modes['ptGrey']:
                return (1,self._cap.GrabNumPyImage('bgr'))
    def view(self):
        if self.open():
            streamVideo(self)
    def viewUndistort(self):
        if self.open():
            streamVideo(self, self.matrix, self.distortion)
    def captureFrames(self):
        if self.open():
            return captureFrames(self)
        
    def getResolution(self):
        if self.open():
            if self.mode == modes['cv2']:
                try:
                    return (int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
                except:
                    return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            elif self.mode == modes['ptGrey']:
                return self._cap.GetSize()
    def calibrate(self, gridCorners, **kwargs):     # flags=cv2.CALIB_FIX_K3
        images = captureFrames(self)
        if __TEST__MODE__ and images is None:
            images = [cv2.imread('../data/calibration/%02d.png' % i) for i in range(8)]
        ret, matrix, dist = calibrate(images, gridCorners, **kwargs)
        if ret < self.error:
            self.error = ret
            self.matrix = matrix
            self.distortion = dist
        return ret
    def undistort(self, images, alpha=0.):
        toReturn = []
        size = images[0].shape
        newMtx,roi = cv2.getOptimalNewCameraMatrix(self.matrix,self.distortion,(size[1],size[0]),alpha)
        map1, map2 = cv2.initUndistortRectifyMap(self.matrix,self.distortion,np.eye(3),newMtx,(size[1],size[0]),cv2.CV_16SC2)
        for img in images:
            toReturn.append(cv2.remap(img,map1,map2,cv2.INTER_LINEAR))
        return toReturn
    def save(self, file):
        return np.savez(file,matrix=self.matrix,distortion=self.distortion,error=self.error)
    def load(self, file):
        npzfile = np.load(file)
        self.matrix = npzfile['matrix']
        self.distortion = npzfile['distortion']
        self.error = npzfile['error']

def toGray(image):
    iy, ix, channels = image.shape if len(image.shape)>2 else [image.shape[0], image.shape[1], 1]
    if channels == 4:
        gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
    elif channels == 3:  
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        channels = 4
    else:
        gray = image
    return gray

def calibrate(images, gridCorners, **kwargs):
    ''' get the calibration of a camera from the images of a chessboard with number of gridCorners given'''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((gridCorners[0]*gridCorners[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:gridCorners[0],0:gridCorners[1]].T.reshape(-1,2)
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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], cameraMatrix = np.eye(3), distCoeffs = np.zeros([14]), **kwargs)
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
        elif ch == 1113937:         # left
            current -= 1
        elif ch == 1113939:         # right
            current += 1
        current = current % len(images)
    cv2CloseWindow('frame')

def captureFrames(cam=None):
    cam = cv2.VideoCapture(0) if cam is None else cam
    if not cam.open():
        return None
    frames = []
    while(True):
        ret, frame = cam.read()     # Capture the frame
        cv2.imshow('frame',frame)   # Display the frame
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:                # escape
            break
        elif ch == 32:              # space bar
            frames.append(frame)
    #cam.release()                   # release the capture
    cv2CloseWindow('frame')
    return frames

def streamVideo(cam=None, matrix=None, dist=None):
    cam = cv2.VideoCapture(0) if cam is None else cam
    if not cam.open():
        return None
    while(True):
        ret, frame = cam.read()     # Capture the frame
        if matrix is not None and dist is not None:
            frame = cv2.undistort(frame,matrix,dist)
        cv2.imshow('frame',frame)   # Display the frame
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:                # escape
            break
    #cam.release()                   # release the capture
    cv2CloseWindow('frame')

def captureVideo(fname, cam=None, matrix=None, dist=None, view=True):
    cam = Camera(0) if cam is None else cam
    if not cam.open():
        return None
    video = cv2.VideoWriter(fname,-1,cam.fps,cam.resolution)
    #frames = []
    while(True):
        ret, frame = cam.read()     # Capture the frame
        if matrix is not None and dist is not None:
            frame = cv2.undistort(frame,matrix,dist)
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
