import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import re
 

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

def calculateAndWriteStereoMap(sourceDirI, sourceDirD, targetDir):
    imgsI = []
    imgsD = []

    filesI = os.listdir(sourceDirI)
    filesI = sorted(filesI, key=natural_key)
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

    for i in range(0, len(filesD)):
    #for i in range(0, 100):
        print('Write ', i)
        imgL = imgsI[i]
        imgR = imgsD[i]
        stereo = cv2.StereoBM_create(numDisparities=96, blockSize=17)
        #stereo = cv2.StereoBM_create(numDisparities=80, blockSize=17)
        disparity = stereo.compute(imgL, imgR)

        disparityNorm = np.zeros(disparity.shape)
        disparity = cv2.normalize(disparity, disparityNorm, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        lowerT = 0.0
        upperT = 1.0

        low_values_indices = disparityNorm < lowerT  
        disparityNorm[low_values_indices] = 0.0 

        high_values_indices = disparityNorm > upperT  
        disparityNorm[high_values_indices] = 0.0
        """
        hist, bins = np.histogram(disparityNorm.flatten(), 255, [0.1, 1.0])
        plt.hist(disparityNorm.ravel(), 255, [0.1, 1.0])
        plt.title('Histogram for Filtered')
        plt.show()
        """
        rescaled = np.multiply(disparityNorm, 255)
        typeChange = rescaled.astype('uint8')
        jetMap = cv2.applyColorMap(typeChange, cv2.COLORMAP_JET)

        imgL = np.expand_dims(imgL, axis=2)
        imgL2 = np.concatenate((imgL, imgL), 2)
        imgL = np.concatenate((imgL, imgL2), 2)

        finalImage = np.concatenate((imgL, jetMap), 1)
        cv2.imwrite(targetDir + '/img' + str(i).zfill(5) + '.png', finalImage)
    
sourceDirI = '/home/fuanka/Dropbox/OwnVids/rectified/left/'
sourceDirD = '/home/fuanka/Dropbox/OwnVids/rectified/right/'

simpleStereoTarget = '/home/fuanka/Dropbox/OwnVids/depth/'
calculateAndWriteStereoMap(sourceDirI, sourceDirD, simpleStereoTarget)



