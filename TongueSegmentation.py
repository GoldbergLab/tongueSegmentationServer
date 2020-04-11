# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import load_model
#from keras.layers import *
from keras.backend import clear_session
from scipy.io import savemat
import glob
import cv2
import numpy as np
from pathlib import Path
from array2gif import write_gif

#import matplotlib.pyplot as plt

class SegmentationSpecification:
    # A class to hold all info about how to segment the parts of the image.
    # Provide a list of part names, and correspondingly indexed lists of part widths, heights, xOffsets, and yOffsets
    # Width or height entries may be None, which means the part extends the maximum distance to the edge of the frame
    # neuralNetworkPaths are a list of file paths leading to .h5 or .hd5 tensorflow trained neural network files
    def __init__(self, partNames=[], widths=[], heights=[], xOffsets=[], yOffsets=[], neuralNetworkPaths=[]):
        N = len(partNames)
        # Fill xOffets with zeros
        xOffsets = xOffsets + [0 for k in range(N - len(xOffsets))]
        yOffsets = yOffsets + [0 for k in range(N - len(yOffsets))]
        widths = widths + [None for k in range(N - len(widths))]
        heights = heights + [None for k in range(N - len(heights))]

        self._partNames = partNames
        self._maskDims = dict(zip(partNames, [list(dims) for dims in zip(widths, heights, xOffsets, yOffsets)]))
        self._networkPaths = dict(zip(partNames, neuralNetworkPaths))
        self._networks = dict(zip(partNames, [None for k in partNames]))

    def getPartNames(self):
        return self._partNames

    def getNetworkPath(self, partName):
        return self._networkPaths[partName]

    def getNetwork(self, partName):
        return self._networks[partName]

    def getSize(self, partName):
        return self._maskDims[partName][0:2]

    def getWidth(self, partName):
        return self._maskDims[partName][0]

    def getXLim(self, partName):
        w, h, x, y = self._maskDims[partName]
        if w is None:
            return (x, None)
        else:
            return (x, x+w)

    def getYLim(self, partName):
        w, h, x, y = self._maskDims[partName]
        if h is None:
            return (y, None)
        else:
            return (y, y+h)

    def getXSlice(self, partName):
        return slice(*self.getXLim(partName))

    def getYSlice(self, partName):
        return slice(*self.getYLim(partName))

    def getHeight(self, partName):
        return self._maskDims[partName][1]

    def getXOffset(self, partName):
        return self._maskDims[partName][2]

    def getYOffset(self, partName):
        return self._maskDims[partName][3]

    def initialize(self, vcap):
        # Initialize segspec with video information, so we can give more informed output
        wFrame = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hFrame = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for partName in self._maskDims:
            [w, h, x, y] = self._maskDims[partName]
            if w is None:
                w = wFrame - x
            if h is None:
                h = hFrame - y
            self._maskDims[partName] = [w, h, x, y]

    def initializeNetworks(self, partNames=None, loadShape=True):
        clear_session()
        if partNames is None:
            partNames = self._networkPaths.keys()
        for partName in partNames:
            self._networks[partName] = load_model(self._networkPaths[partName])
            if loadShape:
                _, h, w, _ = self._networks[partName].input_shape
                self._maskDims[partName][0] = w
                self._maskDims[partName][1] = h

# def initializeNeuralNetwork(neuralNetworkPath):
#     clear_session()
#     return load_model(neuralNetworkPath)

def segmentVideo(videoPath=None, segSpec=None, maskSaveDirectory=None, videoIndex=None, binaryThreshold=0.3):
    # Save one or more predicted mask files for a given video and segmenting neural network
    #   videoPath: The path to the video file in question
    #   segSpec: a SegmentationSpecification object, which defines how to split the image up into parts to do separate segmentations
    #   maskSaveDirectory: The directory in which to save the completed binary mask predictions
    #   videoIndex: An integer indicating which video this is in the series of videos. This will be used to number the output masks
    if None in [videoPath, segSpec, maskSaveDirectory, videoIndex]:
        raise ValueError('Missing argument')

    if type(videoPath) != type(str()):
        videoPath = str(videoPath)

    # Open video for reading
    cap = cv2.VideoCapture(videoPath)
#   This appears to not be necessary?
#    cap.open()

    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize segSpec with video file:
    segSpec.initialize(cap)

    # Prepare video buffer arrays to receive data
    imageBuffers = {}
    for partName in segSpec.getPartNames():
        imageBuffers[partName] = np.zeros((nFrames,segSpec.getHeight(partName),segSpec.getWidth(partName),1))

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
#                h, w = frame[yS, xS, 1].shape
                imageBuffers[partName][k, :, :, :] = frame[yS, xS, 1].reshape(1, segSpec.getHeight(partName), segSpec.getWidth(partName), 1)
            k = k+1
        # Break the loop
        else:
            break

    cap.release()

    gifSaveTemplate = "{partName}.gif"

    # Make predictions and save to disk
    maskPredictions = {}
    for partName in segSpec.getPartNames():
        print('Making prediction for {partName}'.format(partName=partName))
        # Convert image to uint8
        imageBuffers[partName] = imageBuffers[partName].astype(np.uint8)
        # Create predicted mask and threshold to make it binary
        maskPredictions[partName] = segSpec.getNetwork(partName).predict(imageBuffers[partName]) > binaryThreshold
        # Generate save name for mask
        maskSaveName = "{partName}_{index:03d}.mat".format(partName=partName, index=videoIndex)
        savePath = Path(maskSaveDirectory) / maskSaveName
        # Generate gif of the latest mask for monitoring purposes
        try:
            gifSaveName = gifSaveTemplate.format(partName=partName)
            gifSavePath = Path(maskSaveDirectory) / gifSaveName
            print(maskPredictions[partName].shape, maskPredictions[partName].dtype)
            spaceSkip = 3; timeSkip = 15
            gifData = maskPredictions[partName][::timeSkip, ::spaceSkip, ::spaceSkip, 0].astype('uint8')*255
            gifData = np.stack([gifData, gifData, gifData])
            gifData = [gifData[:, k, :, :] for k in range(gifData.shape[1])]
            write_gif(gifData, gifSavePath)
        except:
            print("Mask preview creation failed.")

        # Save mask to disk
        savemat(savePath,{'mask_pred':maskPredictions[partName]},do_compression=True)
