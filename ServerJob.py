import multiprocessing as mp
import logging
from TongueSegmentation import initializeNeuralNetwork, segmentVideo
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
        self.logBuffer.append((msg, lvl))

    def flushLogBuffer(self):
        if len(self.logBuffer) > 0:
            lines = []
            for msg, lvl in self.logBuffer:
                lines.append(msg)
            msgs = "\n".join(lines)
            self.logger.log(logging.INFO, msgs)
        self.logBuffer = []

class ServerJob(StateMachineProcess):
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

    #messages:
    START = 'msg_start'
    STOP = 'msg_stop'
    EXIT = 'msg_exit'
    SETPARAMS = 'msg_setParams'
    PROCESS = 'msg_process'

    settableParams = [
        'verbose'
    ]

    newJobNum = itertools.count().__next__   # Source of this clever little idea: https://stackoverflow.com/a/1045724/1460057

    def __init__(self,
                verbose = False,
                videoList = None,
                maskSaveDirectory = None,
                segmentationSpecification = None,
                waitingTimeout = 600,
                neuralNetworkPath = None,
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        # Store inputs in instance variables for later access
        self.jobNum = ServerJob.newJobNum()
        self.errorMessages = []
        self.verbose = verbose
        self.videoList = videoList
        self.maskSaveDirectory = maskSaveDirectory
        self.segSpec = segmentationSpecification
        self.progressQueue = mp.Queue()
        self.waitingTimeout = waitingTimeout
        self.neuralNetworkPath = neuralNetworkPath
        self.exitFlag = False

    def setParams(self, **params):
        for key in params:
            if key in ServerJob.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def sendProgress(self, finishedVideoList, videoList, currentVideo, processingStartTime):
        # Send progress to server:
        progress = dict(
            videosCompleted=len(finishedVideoList),
            videosRemaining=len(videoList),
            lastCompletedVideoPath=currentVideo,
            lastProcessingStartTime=processingStartTime
        )
        self.progressQueue.put(('PROGRESS', progress))

    def run(self):
        self.PID.value = os.getpid()
        if self.verbose >= 1: self.log("PID={pid}".format(pid=os.getpid()))
        state = ServerJob.STOPPED
        nextState = ServerJob.STOPPED
        lastState = ServerJob.STOPPED
        msg = ''; arg = None

        while True:
            # Publish updated state
            if state != lastState:
                self.updatePublishedState(state)

            try:
# ********************************* STOPPPED *********************************
                if state == ServerJob.STOPPED:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=self.waitingTimeout)
                        if msg == ServerJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty:
                        self.exitFlag = True
                        if self.verbose >= 0: self.log('Waiting timeout expired while stopped - exiting')
                        msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = ServerJob.EXITING
                    elif msg == '':
                        nextState = state
                    elif msg == ServerJob.STOP:
                        nextState = ServerJob.STOPPED
                    elif msg == ServerJob.START:
                        nextState = ServerJob.INITIALIZING
                    elif msg == ServerJob.EXIT:
                        self.exitFlag = True
                        nextState = ServerJob.EXITING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* INITIALIZING *********************************
                elif state == ServerJob.INITIALIZING:
                    # DO STUFF
                    if self.verbose >= 1: self.log('Initializing neural network...')
                    neuralNetwork = initializeNeuralNetwork(self.neuralNetworkPath)
                    if self.verbose >= 1: self.log('...neural network initialized.')
                    unfinishedVideoList = copy.deepcopy(self.videoList)
                    finishedVideoList = []
                    videoIndex = 0
                    processingStartTime = None
                    if self.verbose >= 3: self.log('Server job initialized!')
                    self.sendProgress(finishedVideoList, self.videoList, None, processingStartTime)

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == ServerJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = ServerJob.STOPPING
                    elif msg in ['', ServerJob.START]:
                        nextState = ServerJob.WAITING
                    elif msg == ServerJob.STOP:
                        nextState = ServerJob.STOPPING
                    elif msg == ServerJob.EXIT:
                        self.exitFlag = True
                        nextState = ServerJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WAITING *********************************
                elif state == ServerJob.WAITING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=True, timeout=self.waitingTimeout)
                        if msg == ServerJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty:
                        self.exitFlag = True
                        if self.verbose >= 0: self.log('Waiting timeout expired - exiting')
                        msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = ServerJob.STOPPING
                    elif msg == '':
                        nextState = state
                    elif msg == ServerJob.PROCESS:
                        nextState = ServerJob.WORKING
                    elif msg == ServerJob.STOP:
                        nextState = ServerJob.STOPPING
                    elif msg == ServerJob.EXIT:
                        self.exitFlag = True
                        nextState = ServerJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* WORKING *********************************
                elif state == ServerJob.WORKING:
                    # DO STUFF
                    # Record processing time
                    processingStartTime = time.time_ns()
                    # Segment video
                    currentVideo = self.videoList.pop(0)
                    segmentVideo(neuralNetwork, currentVideo, segSpec, self.maskSaveDirectory, videoIndex)
                    videoIndex += 1
                    finishedVideoList.append(currentVideo)
                    self.sendProgress(finishedVideoList, self.videoList, currentVideo, processingStartTime)
                    if self.verbose >= 3: self.log('Server job progress: {prog}'.format(prog=progress))
                    # Are we done?
                    if len(self.videoList) == 0:
                        if self.verbose >= 2: self.log('Server job complete, setting exit flag to true.')
                        self.exitFlag = True

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == ServerJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = ServerJob.STOPPING
                    elif msg in ['', ServerJob.START]:
                        nextState = ServerJob.WORKING
                    elif msg == ServerJob.STOP:
                        nextState = ServerJob.STOPPING
                    elif msg == ServerJob.EXIT:
                        self.exitFlag = True
                        nextState = ServerJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* STOPPING *********************************
                elif state == ServerJob.STOPPING:
                    # DO STUFF

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == ServerJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = ServerJob.STOPPED
                    elif msg == '':
                        nextState = ServerJob.STOPPED
                    elif msg == ServerJob.STOP:
                        nextState = ServerJob.STOPPED
                    elif msg == ServerJob.EXIT:
                        self.exitFlag = True
                        nextState = ServerJob.STOPPED
                    elif msg == ServerJob.START:
                        nextState = ServerJob.INITIALIZING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ERROR *********************************
                elif state == ServerJob.ERROR:
                    # DO STUFF
                    if self.verbose >= 0:
                        self.log("ERROR STATE. Error messages:\n\n")
                        self.log("\n\n".join(self.errorMessages))
                    self.errorMessages = []

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == ServerJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if lastState == ServerJob.ERROR:
                        # Error ==> Error, let's just exit
                        nextState = ServerJob.EXITING
                    elif msg == '':
                        if lastState == ServerJob.STOPPING:
                            # We got an error in stopping state? Better just stop.
                            nextState = ServerJob.STOPPED
                        elif lastState ==ServerJob.STOPPED:
                            # We got an error in the stopped state? Better just exit.
                            nextState = ServerJob.EXITING
                        else:
                            nextState = ServerJob.STOPPING
                    elif msg == ServerJob.STOP:
                        nextState = ServerJob.STOPPING
                    elif msg == ServerJob.EXIT:
                        self.exitFlag = True
                        if lastState == ServerJob.STOPPING:
                            nextState = ServerJob.EXITING
                        else:
                            nextState = ServerJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* EXIT *********************************
                elif state == ServerJob.EXITING:
                    if self.verbose >= 1: self.log('Exiting!')
                    break
                else:
                    raise KeyError("Unknown state: "+self.stateList[state])
            except KeyboardInterrupt:
                # Handle user using keyboard interrupt
                if self.verbose >= 1: self.log("Keyboard interrupt received - exiting")
                self.exitFlag = True
                nextState = ServerJob.STOPPING
            except:
                # HANDLE UNKNOWN ERROR
                self.errorMessages.append("Error in "+self.stateList[state]+" state\n\n"+traceback.format_exc())
                nextState = ServerJob.ERROR

            if (self.verbose >= 1 and (len(msg) > 0 or self.exitFlag)) or len(self.logBuffer) > 0 or self.verbose >= 3:
                self.log("msg={msg}, exitFlag={exitFlag}".format(msg=msg, exitFlag=self.exitFlag))
                self.log(r'*********************************** /\ {ID} {state} /\ ********************************************'.format(ID=self.ID, state=self.stateList[state]))

            self.flushLogBuffer()

            # Prepare to advance to next state
            lastState = state
            state = nextState

        clearQueue(self.msgQueue)
        if self.verbose >= 1: self.log("ServerJob process STOPPED")

        self.flushLogBuffer()
        self.updatePublishedState(self.DEAD)
