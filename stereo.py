#!/usr/bin/python
 
__author__ = ('David Dunn')
__version__ = '0.1'

import numpy as np
import cv2
import dCamera as dc
try:
    import RPi.GPIO as gp
except ImportError:
    print("Error import RPi.GPIO! This is probably due to lack of superuser privildges. Try running with sudo.")

HARDWARE = {'USB':0,'multiBoard':1,'multiPi':2,'computePi':3}
MODE = {'separate':0,'sideBySide':1,'topBottom':2,'crosseyed':3}

class StereoCamera(Camera):
    ''' A stereo camera which stores the intrinsic matrix and distiortion for right and left cameras
        as well as the transformation between the two cameras including essential nd fundamental matricies 
    '''
    def __init__(self, leftid=0, rightid=1, backend=dc.BACKEND['picamera'], hardware=HARDWARE['multiBoard']):
        self.backend = backend
        self.hardware = hardware
        self.gp_init = False
        self.lfCam = None
        self.lfMatrix = np.identity(3)
        self.lfDistortion = [0,0,0,0,0]
        self.lfError = 1
        self.rtCam = None
        self.rtMatrix = np.identity(3)
        self.rtDistortion = [0,0,0,0,0]
        self.rtError = 1
        self.R = np.identity(3)
        self.T = np.zeros(3)
        self.E = np.identity(3)
        self.F = np.identity(3)
        self.mode = MODE['sideBySide']
        if self.backend == dc.BACKEND['picamera']:
            if self.hardware == HARDWARE['multiBoard']:
                '''using multi camera adaptor module on the raspberry pi for stereo camera capture
                cameras should be connected on the A (left) and B (right) ports '''
                self.pin_sel = 7
                self.pin_en1 = 11
                self.pin_en2 = 12
                self.GPinit()
                self.lfCam = dc.Camera(backend=dc.BACKEND['picamera'],fps=30,shutter_speed=30000,iso=200,awb_mode='off')
            else:
                raise ValueError('dCamera.stereo:   Hardware setup not supported yet.')
        else:
            raise ValueError('dCamera.stereo:   Backend not supported yet.')
    def open(self):
        if self.backend == dc.BACKEND['picamera']:
            if self.hardware == HARDWARE['multiBoard']:
                self.lfCam.open()
    def close(self):
        if self.gp_init:
            gp.cleanup((self.pin_sel, self.pin_en1, self.pin_en2))
            self.gp_init = False
        if self.lfCam is not None:
            self.lfCam.close()
        if self.rtCam is not None:
            self.rtCam.close()
    def GPinit(self):
        gp.setwarnings(False)
        gp.setmode(gp.BOARD)
        self.gp_init = True
        gp.setup(self.pin_sel, gp.OUT, initial=gp.LOW)
        gp.setup(self.pin_en1, gp.OUT, initial=gp.LOW)
        gp.setup(self.pin_en2, gp.OUT, initial=gp.HIGH)
    def GPselectLeft(self):
        gp.output(self.pin_sel, gp.LOW)
        self.lfCam.matrix = self.lfMatrix
        self.lfCam.distortion = self.lfDistortion
        self.lfCam.error = self.lfError
    def GPselectRight(self):
        gp.output(self.pin_sel, gp.HIGH)
        self.lfCam.matrix = self.rtMatrix
        self.lfCam.distortion = self.rtDistortion
        self.lfCam.error = self.rtError
    def read(self, video=True, undistort=False):
        if self.backend == dc.BACKEND['picamera']:
            if self.hardware == HARDWARE['multiBoard']:
                self.GPselectLeft()
                readFunc = lfCam.read if not undistort else lfCam.readUndistort
                leftImg = readFunc(video=video)
                self.GPselectRight()
                readFunc = lfCam.read if not undistort else lfCam.readUndistort
                rightImg = readFunc(video=video)
                if leftImg[0] == 1:
                    leftImg = leftImg[1]
                if rightImg[0] == 1:
                    rightImg = rightImg[1]
        if self.mode == MODE['separate']:
            return leftImg,rightImg
        if self.mode == MODE['sideBySide']:
            return np.hstack((leftImg,rightImg))
        if self.mode == MODE['topBottom']:
            return np.vstack((leftImg,rightImg))
        if self.mode == MODE['crosseyed']:
            return np.hstack((rightImg,leftImg))
    def readUndistort(self, video=True):
        return self.read(video, True)
    def view(self):
        dc.streamVideo(self)
    def viewUndistort(self):
        dc.streamVideo(self, True)
    def captureFrames(self):
        return captureFrames(self)
    def calibrateLeft(self):
        if self.backend == dc.BACKEND['picamera']:
            if self.hardware == HARDWARE['multiBoard']:
                self.GPselectLeft()
                ret = self.lfCam.calibrate()
                if ret < self.lfError:
                    self.lfMatrix = self.lfCam.matrix
                    self.lfDistortion = self.lfCam.distortion
                    self.lfError = self.lfCam.error

    def calibrateLeft(self):
        if self.backend == dc.BACKEND['picamera']:
            if self.hardware == HARDWARE['multiBoard']:
                self.GPselectRight()
                ret = self.lfCam.calibrate()
                if ret < self.rtError:
                    self.rtMatrix = self.lfCam.matrix
                    self.rtDistortion = self.lfCam.distortion
                    self.rtError = self.lfCam.error

    def calibrateStereo(self):
        self.calibrateLeft
        self.calibrateRight

        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = calibrateStereo()


def calibrateStereo(imagesA, imagesB, gridCorners, gridScale, **kwargs)
''' get the calibration of a camera from the images of a chessboard with number of gridCorners given'''
    cameraMatrix1 = kwargs.get('matrixA',np.eye(3))
    distCoeffs1 = kwargs.get('distortionA',np.zeros([14]))
    cameraMatrix2 = kwargs.get('matrixB',np.eye(3))
    distCoeffs2 = kwargs.get('distortionB',np.zeros([14]))
    # termination criteria
    criteria = kwargs.get('criteria',(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    # prepare object points, define top left gridCorner as origin: like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((gridCorners[0]*gridCorners[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:gridCorners[0],0:gridCorners[1]].T.reshape(-1,2)
    objp *= gridScale
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpointsA = [] # 2d points in image plane.
    imgpointsB = [] # 2d points in image plane.
    #markedImages = [] # store the updated images
    for imageA,imageB in zip(imagesA,imagesB):
        grayA = dc.toGray(imageA)
        grayB = dc.toGray(imageB)
        # Find the chess board corners
        retA, cornersA = cv2.findChessboardCorners(grayA, gridCorners,None)
        retB, cornersB = cv2.findChessboardCorners(grayB, gridCorners,None)
        # If found, add object points, image points (after refining them)
        if retA and retB:
            objpoints.append(objp)
            cv2.cornerSubPix(grayA,cornersA,(11,11),(-1,-1),criteria)
            imgpointsA.append(cornersA) 
            cv2.cornerSubPix(grayB,cornersB,(11,11),(-1,-1),criteria)
            imgpointsB.append(cornersB) 
        # Draw and display the corners
        #temp = img
        #cv2.drawChessboardCorners(temp, gridCorners, corners,ret)
        #markedImages.append(temp)
    print('Using %s of %s images.'%(len(objpoints), len(imagesA)))
    #dc.slideShow(markedImages)
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsA, imgpointsB, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, grayA.shape[::-1],**kwargs)
    return ret, mtx1, dist1, mtx2, dist2, R, T, E, F