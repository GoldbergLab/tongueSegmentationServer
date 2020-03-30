# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import load_model
#from keras.layers import *
from keras.backend import clear_session
from scipy.io import loadmat
from scipy.io import savemat
import glob
import cv2
import numpy as np
from pathlib import Path
#import matplotlib.pyplot as plt

class SegmentationSpecification:
    # A class to hold the info about the parts of the image that should be separately segmented.
    # Provide a list of part names, and correspondingly indexed lists of part widths, heights, xOffsets, and yOffsets
    # Width or height entries may be None, which means the part extends the maximum distance to the edge of the frame
    def __init__(self, partNames=[], widths=[], heights=[], xOffsets=[], yOffsets=[]):
        self.partNames = partNames
        self._specs = dict(zip(partNames, zip(widths, heights)))

    def getPartNames(self):
        return self.partNames

    def getSize(self, partName):
        return self._specs[partName][0:2]

    def getWidth(self, partName):
        return self._specs[partName][0]

    def getXLim(self, partName):
        w, h, x, y = self._specs[partName]
        if w is None:
            return (x, None)
        else:
            return (x, x+w)

    def getYLim(self, partName):
        w, h, x, y = self._specs[partName]
        if h is None:
            return (y, None)
        else:
            return (y, y+h)

    def getXSlice(self, partName):
        return slice(*self.getXLim())

    def getYSlice(self, partName):
        return slice(*self.getYLim())

    def getHeight(self, partName):
        return self._specs[partName][1]

    def getXOffset(self, partName):
        return self._specs[partName][2]

    def getYOffset(self, partName):
        return self._specs[partName][3]

def initializeNeuralNetwork(neuralNetworkPath):
    clear_session()
    return load_model(neuralNetworkPath)

def getVideoList(videoDirs, videoFilter='*'):
    # Generate a list of video Path objects from the given directories using the given path filters
    if len(videoDirs) != len(maskSzveDirectories):
        raise IndexError('Must provide the same # of video and mask save directories')
    videoList = []
    for videoDir in videoDirs:
        p = Path(videoDir)
        for videoPath in p.iterdir():
            if videoPath.match(videoFilter):
                videoList.append(videoPath)
    return videoList

def segmentVideo(neuralNetwork, videoPath, segSpec, maskSaveDirectory, videoIndex):
    # Save one or more predicted mask files for a given video and segmenting neural network
    #   neuralNetwork: A loaded neural network object
    #   videoPath: The path to the video file in question
    #   segSpec: a SegmentationSpecification object, which defines how to split the image up into parts to do separate segmentations
    #   maskSaveDirectory: The directory in which to save the completed binary mask predictions
    #   videoIndex: An integer indicating which video this is in the series of videos. This will be used to number the output masks

    # Open video for reading
    cap = cv2.VideoCapture(videoPath)
    cap.open()

    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wFrame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hFrame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video buffer arrays to receive data
    imageBuffers = {}
    for partName in segSpec.getPartNames():
        imageBuffer[partName] = np.zeros((nFrames,segSpec.getHeight(),segSpect.getWidth(),1))

    k=0
    while(True):
        # Read frame from video
        ret,frame = cap.read()
        if ret == True:
            for partName in segSpec.getPartNames():
                # Get info on how to separate the frame into parts
                xS = segSpec.getXSlice(partName)
                yS = segSpec.getYSlice(partName)
                # Write the frame part into the video buffer array
                h, w, _ = frame[yS, xS, 1].shape
                imageBuffers[partName][k, :, :, :] = frame[yS, xS, 1].reshape(1, h, w, 1)
            k = k+1
        # Break the loop
        else:
            break

    cap.release()

    # Make predictions and save to disk
    maskPredictions = {}
    for partName in segSpec.getPartNames():
        # Convert image to uint8
        imageBuffer[partName] = imageBuffer[partName].astype(np.uint8)
        # Create predicted mask
        maskPredictions[partName] = neuralNetwork.predict(imageBuffer[partName])
        # Threshold mask to make it binary
        maskPredictionsBinary[partName] = maskPredictions[partName] > 0.3
        # Generate save name for mask
        maskSaveName = "{partName}_{index:03d}.mat".format(partName=partName, index=videoIndex)
        savePath = Path(maskSaveDirectory) / maskSaveName
        # Save mask to disk
        savemat(savePath,{'maskPredictions':maskPredictionsBinary[partName]},do_compression=True)
