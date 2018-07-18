#!/usr/bin/python
 
__author__ = ('David Dunn')
__version__ = '0.1'

import numpy as np
import cv2, time
import dCamera as dc
try:
    import RPi.GPIO as gp
except ImportError:
    print("Error import RPi.GPIO! This is probably due to lack of superuser privildges. Try running with sudo.")

HARDWARE = {'USB':0,'multiBoard':1,'multiPi':2,'computePi':3}
MODE = {'separate':0,'sideBySide':1,'topBottom':2,'crosseyed':3}

class StereoCamera(dc.Camera):
    ''' A stereo camera which stores the intrinsic matrix and distiortion for right and left cameras
        as well as the transformation between the two cameras including essential nd fundamental matricies 
    '''
    def __init__(self, leftid=0, rightid=1, backend=dc.BACKEND['picamera'], hardware=HARDWARE['multiBoard']):
        self.backend = backend
        self.hardware = hardware
        self.gp_init = False
        self.resolution = None  # resolution of each of the left and right cameras
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
        self.error = 100
        self.lfR = None
        self.lfP = None
        self.rtR = None
        self.rtP = None
        self.Q = None
        self.lfMapX = None
        self.lfMapY = None
        self.rtMapX = None
        self.rtMapY = None
        self.maxDisparity = 128
        self.blockSize = 9
        self.disparityProcessor = None
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
                self.resolution = self.lfCam.resolution
            else:
                raise ValueError('dCamera.stereo:   Hardware setup not supported yet.')
        else:
            raise ValueError('dCamera.stereo:   Backend not supported yet.')
    def open(self):
        if self.backend == dc.BACKEND['picamera']:
            if self.hardware == HARDWARE['multiBoard']:
                return self.lfCam.open()
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
        self.lfCam.mapX = self.lfMapX
        self.lfCam.mapY = self.lfMapY
    def GPselectRight(self):
        gp.output(self.pin_sel, gp.HIGH)
        self.lfCam.matrix = self.rtMatrix
        self.lfCam.distortion = self.rtDistortion
        self.lfCam.error = self.rtError
        self.lfCam.mapX = self.rtMapX
        self.lfCam.mapY = self.rtMapY
    def read(self, video=True, undistort=False):
        '''note for cv2 backend, we should use grab(), retrieve() instead of read()'''
        if self.backend == dc.BACKEND['picamera']:
            if self.hardware == HARDWARE['multiBoard']:
                readFunc = self.lfCam.read if not undistort else self.lfCam.readUndistort
                self.GPselectLeft()
                time.sleep(.015)
                leftImg = readFunc(video=video)
                self.GPselectRight()
                time.sleep(.015)
                rightImg = readFunc(video=video)
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
    def readDisparity(self, maxDisp=128, blockSize=11):
        disparityProcessor = cv2.StereoSGBM_create(0,maxDisp,blockSize)
        oldMode = self.mode
        self.mode = MODE['separate']
        imageA, imageB = self.read()
        grayA = dc.toGray(imageA)
        grayB = dc.toGray(imageB)
        disparity = disparityProcessor.compute(grayA,grayB)
        #maybe do filtering?
        #cv2.filterSpeckles(disparity, 0, 4000,128)
        #something about disparity being scaled by 16?
        dispScaled = (disparity / 16.).astype(np.uint8) + abs(disparity.min())
        return dispScaled
    def calibrateLeft(self, gridCorners, gridScale):
        if self.backend == dc.BACKEND['picamera']:
            if self.hardware == HARDWARE['multiBoard']:
                self.GPselectLeft()
                ret = self.lfCam.calibrate(gridCorners, gridScale)
                matrix = self.lfCam.matrix
                dist = self.lfCam.distortion
        if ret < self.lfError:
            self.lfMatrix = matrix
            self.lfDistortion = dist
            self.lfError = ret
            self.lfMapX, self.lfMapY = cv2.initUndistortRectifyMap(self.lfMatrix,self.lfDistortion,np.eye(3),self.lfMatrix,self.resolution,cv2.CV_16SC2)
        return ret
    def calibrateRight(self, gridCorners, gridScale):
        if self.backend == dc.BACKEND['picamera']:
            if self.hardware == HARDWARE['multiBoard']:
                self.GPselectRight()
                ret = self.lfCam.calibrate(gridCorners, gridScale)
                matrix = self.lfCam.matrix
                dist = self.lfCam.distortion
        if ret < self.rtError:
            self.rtMatrix = matrix
            self.rtDistortion = dist
            self.rtError = ret
            self.rtMapX, self.rtMapY = cv2.initUndistortRectifyMap(self.rtMatrix,self.rtDistortion,np.eye(3),self.rtMatrix,self.resolution,cv2.CV_16SC2)
        return ret
    def calibrateStereo(self, gridCorners, gridScale):
        if self.lfError >= 1:
            ret = self.calibrateLeft(gridCorners, gridScale)
            print('Left Camera Calibration Error: %f'%ret)
        if self.rtError >= 1:
            ret = self.calibrateRight(gridCorners, gridScale)
            print('Right Camera Calibration Error: %f'%ret)
        oldMode = self.mode
        self.mode = MODE['separate']
        images = dc.captureFrames(self)
        imagesA, imagesB = zip(*images)
        ret, *_, R, T, E, F = calibrateStereo(imagesA,imagesB,gridCorners,gridScale,flags=cv2.CALIB_FIX_INTRINSIC)
        if ret < self.error:
            self.R = R
            self.T = T
            self.E = E
            self.F = F
        self.mode = oldMode
        return ret, R, T, E, F
    def rectify(self):
        self.lfR, self.rtR, self.lfP, self.rtP, self.Q, *_ = cv2.stereoRectify(self.lfMatrix,self.lfDistortion,self.rtMatrix,self.rtDistortion,self.resolution,self.R,self.T)
        self.lfMapX, self.lfMapY = cv2.initUndistortRectifyMap(self.lfMatrix, self.lfDistortion, self.lfR, self.lfP, self.resolution, cv2.CV_16SC2)
        self.rtMapX, self.rtMapY = cv2.initUndistortRectifyMap(self.rtMatrix, self.rtDistortion, self.rtR, self.rtP, self.resolution, cv2.CV_16SC2)

def calibrateStereo(imagesA, imagesB, gridCorners, gridScale, **kwargs):
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

