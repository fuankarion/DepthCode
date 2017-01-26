import cv2
import numpy as np
import os

sourceDirI = '/home/fuanka/Dropbox/OwnVids/LeftFrames/'
sourceDirD = '/home/fuanka/Dropbox/OwnVids/RightFrames/'

targetDirI = '/home/fuanka/Dropbox/OwnVids/rectified/left/'
targetDirD = '/home/fuanka/Dropbox/OwnVids/rectified/right/'

imgsI = []
imgsD = []

filesI = os.listdir(sourceDirI)
filesI.sort()
print('Read I ', len(filesI), ' files')
for aFile in filesI: 
    fileSourcePath = os.path.join(sourceDirI, aFile)
    img = cv2.imread(fileSourcePath, 0)
    imgsI.append(img)
    
filesD = os.listdir(sourceDirD)
filesD.sort()
print('Read D ', len(filesD), ' files')
for aFile in filesD: 
    fileSourcePath = os.path.join(sourceDirD, aFile)
    img = cv2.imread(fileSourcePath, 0)
    imgsD.append(img)
    
img_left_points = []
img_right_points = []
obj_points = []


patternRows = 4
patternCols = 5
pattern_size = (patternRows, patternCols)

objPoints = np.zeros((patternRows * patternCols, 3), np.float32)
objPoints[:,:2] = np.mgrid[0:patternRows, 0:patternCols].T.reshape(-1, 2)
#print('objPoints ', objPoints)

h, w = img.shape[:2]

totalPatternsFound=0
for i in range(0, len(filesD)):
    print('i ', i)
    imgL = imgsI[i]
    imgR = imgsD[i]

    find_chessboard_flags = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, patternRows * patternCols, 0.1)
    
    left_found, left_corners = cv2.findChessboardCorners(imgL, pattern_size, find_chessboard_flags)
    print('left_found ', left_found)
    
    right_found, right_corners = cv2.findChessboardCorners(imgR, pattern_size, find_chessboard_flags)
    print('right_found ', right_found)

    if left_found:
        cv2.cornerSubPix(imgL, left_corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    if right_found:
        cv2.cornerSubPix(imgR, right_corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    if left_found and right_found:
        print('Pattern Found')
        totalPatternsFound=totalPatternsFound+1
        img_left_points.append(left_corners)
        img_right_points.append(right_corners)
        obj_points.append(objPoints)
       
    else:
        print('No Chessborad pattern found')
        continue

print('len(img_left_points)', len(img_left_points))
print('len(img_right_points)', len(img_right_points))
print('totalPatternsFound',totalPatternsFound)

print('Individual camera matrix estimation')
cameraMatrixLeft = cv2.initCameraMatrix2D(obj_points, img_left_points, (w, h))
cameraMatrixRight = cv2.initCameraMatrix2D(obj_points, img_right_points, (w, h))

print('cameraMatrixLeft ', cameraMatrixLeft)
print('cameraMatrixRight ', cameraMatrixRight)

dist_coeffs_l = None
dist_coeffs_r = None
R = None
T = None
E = None
F = None

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

print('Stereo Calibration')
res = cv2.stereoCalibrate(obj_points, img_left_points, img_right_points, 
                          cameraMatrixLeft, dist_coeffs_l, cameraMatrixRight, 
                          dist_coeffs_r, (w, h), R, T, E, F, criteria=stereocalib_criteria,
                          flags=stereocalib_flags)
(rms_stereo, stereoCameraMatrixLeft, dist_coeffs_l, stereoCameraMatrixRight, dist_coeffs_r, R, T, E, F) = res    

print('rms_stereo ', rms_stereo)
print('stereoCameraMatrixLeft ', stereoCameraMatrixLeft)
print('dist_coeffs_l ', dist_coeffs_l)
print('stereoCameraMatrixRight ', stereoCameraMatrixLeft)
print('dist_coeffs_r ', dist_coeffs_r)
print('R ', R)
print('T ', T)
print('E ', E)
print('F ', F)


print('Calculate mapping parametes')
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(stereoCameraMatrixLeft, dist_coeffs_l,
                                                  stereoCameraMatrixRight, dist_coeffs_r, 
                                                  (w, h), R, T)
print('R1 ', R1)
print('R2 ', R2)
print('P1 ', P1)
print('P2 ', P2)
print('Q ', Q)
print('roi1 ', roi1)
print('roi2 ', roi2)

                                                 
mapxLeft, mapyLeft = cv2.initUndistortRectifyMap(cameraMatrixLeft, dist_coeffs_l, R1, stereoCameraMatrixLeft, (w, h), cv2.CV_16SC2)
mapxRight, mapyRight = cv2.initUndistortRectifyMap(cameraMatrixRight, dist_coeffs_r, R2, stereoCameraMatrixRight, (w, h), cv2.CV_16SC2)


for i in range(0, len(filesD)):
    print('Rectify  ', i)
    imgL = imgsI[i]
    imgR = imgsD[i]

    left_img_remap = cv2.remap(imgL, mapxLeft, mapyLeft, cv2.INTER_LINEAR)
    right_img_remap = cv2.remap(imgR, mapxRight, mapyRight, cv2.INTER_LINEAR)
    
    cv2.imwrite(targetDirI + '/' + str(i) + '.png', left_img_remap)
    cv2.imwrite(targetDirD + '/' + str(i) + '.png', right_img_remap)
                                                 

  
