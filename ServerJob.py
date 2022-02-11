import multiprocessing as mp
import logging
from TongueSegmentation import segmentVideo
import copy
import itertools
import os
import queue
import traceback
import time

def clearQueue(q):
    if q is not None:
        while True:
            try:
                stuff = q.get(block=True, timeout=0.1)
            except queue.Empty:
                break

class StateMachineProcess(mp.Process):
    def __init__(self, *args, logger=None, daemon=True, **kwargs):
        mp.Process.__init__(self, *args, daemon=daemon, **kwargs)
        self.ID = "X"
        self.msgQueue = mp.Queue()
        self.logger = logger
        self.publishedStateVar = mp.Value('i', -1)
        self.PID = mp.Value('i', -1)
        self.exitFlag = False
        self.logBuffer = []

    def run(self):
        self.PID.value = os.getpid()

    def updatePublishedState(self, state):
        if self.publishedStateVar is not None:
            L = self.publishedStateVar.get_lock()
            locked = L.acquire(block=False)
            if locked:
                self.publishedStateVar.value = state
                L.release()

    def log(self, msg, lvl=logging.INFO):
        print('SERVER_JOB: ', msg)
#        self.logger.log(logging.INFO, msg)
        self.logBuffer.append(msg) #(msg, lvl))

    # def flushLogBuffer(self):
    #     if len(self.logBuffer) > 0:
    #         lines = []
    #         for msg, lvl in self.logBuffer:
    #             lines.append(msg)
    #         msgs = "\n".join(lines)
    #         self.logger.log(logging.INFO, msgs)
    #     self.logBuffer = []

class TrainJob(StateMachineProcess):
    # Class that the server can use to spawn a separate process state machine
    #   to train a network

    # States:
    STOPPED = 0
    INITIALIZING = 1
    WAITING = 2
    WORKING = 3
    STOPPING = 4
    ERROR = 5
    EXITING = 6
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
        WAITING:'WAITING',
        WORKING :'WORKING',
        STOPPING :'STOPPING',
        ERROR :'ERROR',
        EXITING :'EXITING',
        DEAD :'DEAD'
    }

    # Exit codes:
    INCOMPLETE = -1
    SUCCEEDED = 0
    FAILED = 1
    exitCodeList = {
        INCOMPLETE: 'Incomplete',
        SUCCEEDED: 'Success',
        FAILED: 'Failed'
    }

    #messages:
    START = 'msg_start'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'
    PROCESS = 'msg_process'

    settableParams = [
        'verbose',
        'skipExisting'
    ]

    def __init__(self,
                verbose = False,
                videoList = None,
                maskSaveDirectory = None,
                segSpec = None,
                waitingTimeout = 600,
                binaryThreshold = 0.3,
                jobNum = None,
                generatePreview = True,
                skipExisting = False,
                **kwargs):
        StateMachineProcess.__init__(self, logger=kwargs['logger']) #, **kwargs)
        # Store inputs in instance variables for later access
        self.jobNum = jobNum
        self.errorMessages = []
        self.verbose = verbose
        self.videoList = videoList
        self.maskSaveDirectory = maskSaveDirectory
        self.segSpec = segSpec
        self.progressQueue = mp.Queue()
        self.waitingTimeout = waitingTimeout
        self.binaryThreshold = binaryThreshold
        self.generatePreview = generatePreview
        self.skipExisting = skipExisting
        self.exitCode = SegmentationJob.INCOMPLETE
        self.exitFlag = False

    def setParams(self, **params):
        for key in params:
            if key in SegmentationJob.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def sendProgress(self, currentVideo, processingStartTime): #, finishedVideoList, videoList, currentVideo, processingStartTime):
        # Send progress to server:
        progress = dict(
            # videosCompleted=len(finishedVideoList),
            # videosRemaining=len(videoList),
            log=self.logBuffer,
            exitCode=self.exitCode,
            lastCompletedVideoPath=currentVideo,
            lastProcessingStartTime=processingStartTime
        )
        self.logBuffer = []
        self.progressQueue.put(progress)

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        state = SegmentationJob.STOPPED
        nextState = SegmentationJob.STOPPED
        lastState = SegmentationJob.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPED *********************************
                if state == SegmentationJob.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=self.waitingTimeout)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty:
                        self.exitFlag = True
                        if self.verbose >= 0: self.log('Waiting timeout expired while stopped - exiting')
                        msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = SegmentationJob.EXITING
                    elif msg == '':
                        nextState = state
                    elif msg == SegmentationJob.START:
                        nextState = SegmentationJob.INITIALIZING
                    elif msg == SegmentationJob.EXIT:
                        self.exitFlag = True
                        nextState = SegmentationJob.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *********************************
                elif state == SegmentationJob.INITIALIZING:
                    # DO STUFF
                    if self.verbose >= 1: self.log('Initializing neural networks...')
                    self.segSpec.initializeNetworks()
                    if self.verbose >= 1: self.log('...neural network initialized.')
                    unfinishedVideoList = copy.deepcopy(self.videoList)
                    finishedVideoList = []
                    videoIndex = 0
                    processingStartTime = None
                    currentVideo = None
                    if self.verbose >= 3: self.log('Server job initialized!')
#                    self.sendProgress(finishedVideoList, self.videoList, None, processingStartTime)
                    self.sendProgress(currentVideo, processingStartTime)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = SegmentationJob.STOPPING
                    elif msg in ['', SegmentationJob.START]:
                        nextState = SegmentationJob.WAITING
                    elif msg == SegmentationJob.PROCESS:
                        # Skip straight to working
                        nextState = SegmentationJob.WORKING
                    elif msg == SegmentationJob.EXIT:
                        self.exitFlag = True
                        nextState = SegmentationJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WAITING *********************************
                elif state == SegmentationJob.WAITING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=self.waitingTimeout)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty:
                        self.exitFlag = True
                        if self.verbose >= 0: self.log('Waiting timeout expired - exiting')
                        msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = SegmentationJob.STOPPING
                    elif msg == '':
                        nextState = state
                    elif msg == SegmentationJob.PROCESS:
                        nextState = SegmentationJob.WORKING
                    elif msg == SegmentationJob.EXIT:
                        self.exitFlag = True
                        nextState = SegmentationJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WORKING *********************************
                elif state == SegmentationJob.WORKING:
                    # DO STUFF
                    # Record processing time
                    processingStartTime = time.time_ns()
                    # Segment video
                    currentVideo = self.videoList.pop(0)
                    segmentVideo(
                        videoPath=currentVideo,
                        segSpec=self.segSpec,
                        maskSaveDirectory=self.maskSaveDirectory,
                        videoIndex=videoIndex,
                        binaryThreshold=self.binaryThreshold,
                        generatePreview=self.generatePreview,
                        skipExisting=self.skipExisting
                    )
                    videoIndex += 1
                    finishedVideoList.append(currentVideo)
#                    self.sendProgress(finishedVideoList, self.videoList, currentVideo, processingStartTime)
                    self.sendProgress(currentVideo, processingStartTime)
                    if self.verbose >= 3: self.log('Server job progress: {prog}'.format(prog=progress))
                    # Are we done?
                    if len(self.videoList) == 0:
                        if self.verbose >= 2: self.log('Server job complete, setting exit flag to true.')
                        self.exitFlag = True

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.exitCode = SegmentationJob.SUCCEEDED
                        nextState = SegmentationJob.STOPPING
                    elif msg in ['', SegmentationJob.START]:
                        nextState = SegmentationJob.WORKING
                    elif msg == SegmentationJob.EXIT:
                        self.exitCode = SegmentationJob.SUCCEEDED
                        self.exitFlag = True
                        nextState = SegmentationJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* STOPPING *********************************
                elif state == SegmentationJob.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag or msg in ['', SegmentationJob.EXIT]:
                        nextState = SegmentationJob.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == SegmentationJob.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []
                    self.exitCode = SegmentationJob.FAILED

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == SegmentationJob.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = SegmentationJob.EXITING
                    elif msg == '':
                        if lastState in [SegmentationJob.STOPPING, SegmentationJob.STOPPED]:
                            # We got an error in the stopped or stopping state? Better just exit.
                            nextState = SegmentationJob.EXITING
                        else:
                            self.exitFlag = True
                            nextState = SegmentationJob.STOPPING
                    elif msg == SegmentationJob.EXIT:
                        self.exitFlag = True
                        if lastState == SegmentationJob.STOPPING:
                            nextState = SegmentationJob.EXITING
                        else:
                            nextState = SegmentationJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == SegmentationJob.EXITING:
                    if self.verbose >= 1: self.log('Exiting!')
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 1: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = SegmentationJob.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = SegmentationJob.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.logBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

#            self.flushLogBuffer()

            # Prepare to advance to next state
            lastState = state
            state = nextState

        self.sendProgress(None, None)

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("SegmentationJob process STOPPED")

#        self.flushLogBuffer()
        self.updatePublishedState(self.DEAD)


class SegmentationJob(StateMachineProcess):
    # Class that the server can use to spawn a separate process state machine
    #   to segment a set of videos

    # States:
    STOPPED = 0
    INITIALIZING = 1
    WAITING = 2
    WORKING = 3
    STOPPING = 4
    ERROR = 5
    EXITING = 6
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
        WAITING:'WAITING',
        WORKING :'WORKING',
        STOPPING :'STOPPING',
        ERROR :'ERROR',
        EXITING :'EXITING',
        DEAD :'DEAD'
    }

    # Exit codes:
    INCOMPLETE = -1
    SUCCEEDED = 0
    FAILED = 1
    exitCodeList = {
        INCOMPLETE: 'Incomplete',
        SUCCEEDED: 'Success',
        FAILED: 'Failed'
    }

    #messages:
    START = 'msg_start'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'
    PROCESS = 'msg_process'

    settableParams = [
        'verbose',
        'skipExisting'
    ]

    def __init__(self,
                verbose = False,
                videoList = None,
                maskSaveDirectory = None,
                segSpec = None,
                waitingTimeout = 600,
                binaryThreshold = 0.3,
                jobNum = None,
                generatePreview = True,
                skipExisting = False,
                **kwargs):
        StateMachineProcess.__init__(self, logger=kwargs['logger']) #, **kwargs)
        # Store inputs in instance variables for later access
        self.jobNum = jobNum
        self.errorMessages = []
        self.verbose = verbose
        self.videoList = videoList
        self.maskSaveDirectory = maskSaveDirectory
        self.segSpec = segSpec
        self.progressQueue = mp.Queue()
        self.waitingTimeout = waitingTimeout
        self.binaryThreshold = binaryThreshold
        self.generatePreview = generatePreview
        self.skipExisting = skipExisting
        self.exitCode = SegmentationJob.INCOMPLETE
        self.exitFlag = False

    def setParams(self, **params):
        for key in params:
            if key in SegmentationJob.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def sendProgress(self, currentVideo, processingStartTime): #, finishedVideoList, videoList, currentVideo, processingStartTime):
        # Send progress to server:
        progress = dict(
            # videosCompleted=len(finishedVideoList),
            # videosRemaining=len(videoList),
            log=self.logBuffer,
            exitCode=self.exitCode,
            lastCompletedVideoPath=currentVideo,
            lastProcessingStartTime=processingStartTime
        )
        self.logBuffer = []
        self.progressQueue.put(progress)

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        state = SegmentationJob.STOPPED
        nextState = SegmentationJob.STOPPED
        lastState = SegmentationJob.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPED *********************************
                if state == SegmentationJob.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=self.waitingTimeout)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty:
                        self.exitFlag = True
                        if self.verbose >= 0: self.log('Waiting timeout expired while stopped - exiting')
                        msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = SegmentationJob.EXITING
                    elif msg == '':
                        nextState = state
                    elif msg == SegmentationJob.START:
                        nextState = SegmentationJob.INITIALIZING
                    elif msg == SegmentationJob.EXIT:
                        self.exitFlag = True
                        nextState = SegmentationJob.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *********************************
                elif state == SegmentationJob.INITIALIZING:
                    # DO STUFF
                    if self.verbose >= 1: self.log('Initializing neural networks...')
                    self.segSpec.initializeNetworks()
                    if self.verbose >= 1: self.log('...neural network initialized.')
                    unfinishedVideoList = copy.deepcopy(self.videoList)
                    finishedVideoList = []
                    videoIndex = 0
                    processingStartTime = None
                    currentVideo = None
                    if self.verbose >= 3: self.log('Server job initialized!')
#                    self.sendProgress(finishedVideoList, self.videoList, None, processingStartTime)
                    self.sendProgress(currentVideo, processingStartTime)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = SegmentationJob.STOPPING
                    elif msg in ['', SegmentationJob.START]:
                        nextState = SegmentationJob.WAITING
                    elif msg == SegmentationJob.PROCESS:
                        # Skip straight to working
                        nextState = SegmentationJob.WORKING
                    elif msg == SegmentationJob.EXIT:
                        self.exitFlag = True
                        nextState = SegmentationJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WAITING *********************************
                elif state == SegmentationJob.WAITING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=self.waitingTimeout)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty:
                        self.exitFlag = True
                        if self.verbose >= 0: self.log('Waiting timeout expired - exiting')
                        msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = SegmentationJob.STOPPING
                    elif msg == '':
                        nextState = state
                    elif msg == SegmentationJob.PROCESS:
                        nextState = SegmentationJob.WORKING
                    elif msg == SegmentationJob.EXIT:
                        self.exitFlag = True
                        nextState = SegmentationJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WORKING *********************************
                elif state == SegmentationJob.WORKING:
                    # DO STUFF
                    # Record processing time
                    processingStartTime = time.time_ns()
                    # Segment video
                    currentVideo = self.videoList.pop(0)
                    segmentVideo(
                        videoPath=currentVideo,
                        segSpec=self.segSpec,
                        maskSaveDirectory=self.maskSaveDirectory,
                        videoIndex=videoIndex,
                        binaryThreshold=self.binaryThreshold,
                        generatePreview=self.generatePreview,
                        skipExisting=self.skipExisting
                    )
                    videoIndex += 1
                    finishedVideoList.append(currentVideo)
#                    self.sendProgress(finishedVideoList, self.videoList, currentVideo, processingStartTime)
                    self.sendProgress(currentVideo, processingStartTime)
                    if self.verbose >= 3: self.log('Server job progress: {prog}'.format(prog=progress))
                    # Are we done?
                    if len(self.videoList) == 0:
                        if self.verbose >= 2: self.log('Server job complete, setting exit flag to true.')
                        self.exitFlag = True

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        self.exitCode = SegmentationJob.SUCCEEDED
                        nextState = SegmentationJob.STOPPING
                    elif msg in ['', SegmentationJob.START]:
                        nextState = SegmentationJob.WORKING
                    elif msg == SegmentationJob.EXIT:
                        self.exitCode = SegmentationJob.SUCCEEDED
                        self.exitFlag = True
                        nextState = SegmentationJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* STOPPING *********************************
                elif state == SegmentationJob.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag or msg in ['', SegmentationJob.EXIT]:
                        nextState = SegmentationJob.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == SegmentationJob.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []
                    self.exitCode = SegmentationJob.FAILED

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == SegmentationJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == SegmentationJob.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = SegmentationJob.EXITING
                    elif msg == '':
                        if lastState in [SegmentationJob.STOPPING, SegmentationJob.STOPPED]:
                            # We got an error in the stopped or stopping state? Better just exit.
                            nextState = SegmentationJob.EXITING
                        else:
                            self.exitFlag = True
                            nextState = SegmentationJob.STOPPING
                    elif msg == SegmentationJob.EXIT:
                        self.exitFlag = True
                        if lastState == SegmentationJob.STOPPING:
                            nextState = SegmentationJob.EXITING
                        else:
                            nextState = SegmentationJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == SegmentationJob.EXITING:
                    if self.verbose >= 1: self.log('Exiting!')
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 1: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = SegmentationJob.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = SegmentationJob.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.logBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

#            self.flushLogBuffer()

            # Prepare to advance to next state
            lastState = state
            state = nextState

        self.sendProgress(None, None)

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("SegmentationJob process STOPPED")

#        self.flushLogBuffer()
        self.updatePublishedState(self.DEAD)
