import cv2
import os

def rotateFrames(sourceDir, targetDir):
    files = os.listdir(sourceDir)
    files.sort()
    print('Read  ', len(files), ' files')
    for aFile in files: 
        fileSourcePath = os.path.join(sourceDir, aFile)
        commandRotate = 'convert ' + sourceDir + '/' + aFile + ' -rotate 90 ' + targetDir + '/' + aFile
        print('issue: ' + commandRotate)
        os.system(commandRotate)
        
def getFramesFromVideo(videoPath, frameSkip):
    print ('reading vid', videoPath)
    cap = cv2.VideoCapture(videoPath)
    print ('reading done ')
    frames = []

    print ('Loading frames to memory')
    success, frame = cap.read()#get frame 0
    frames.append(frame)

    idx = 0
    while True:
        for i in range(frameSkip):# read N=frameSkip frames and do nothing
            success, frame = cap.read()
            idx = idx + 1

        if success:
            frames.append(frame)

        if not success:
            break
            
        if idx%100:
            print('Loaded ',idx,)

    print ('Loaded ', len(frames), 'frames')
    return frames

def writeFramesToDisk(targetPath, frames):
    print('Writing ', len(frames), ' Frames To ', targetPath)
    
    for i in range(len(frames)):
        cv2.imwrite(targetPath + '/' + str(i) + '.png', frames[i])
        
videoPath = '/home/fuanka/Dropbox/OwnVids/Left.webm'
framesPath = '/home/fuanka/Dropbox/OwnVids/LeftFrames/'
#rotatedFramesPath = '/home/jcleon/mesurex/mesurex/2016_11_02/videos_horizontal/transcoded/2i70Small/framesRotated/'

frames = getFramesFromVideo(videoPath, 3)
writeFramesToDisk(framesPath, frames)
#rotateFrames(framesPath, rotatedFramesPath)
