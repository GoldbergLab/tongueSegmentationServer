import multiprocessing as mp
import logging
from TongueSegmentation import initializeNeuralNetwork, getVideoList, segmentVideo

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
    # Class for acquiring an audio signal (or any analog signal) at a rate that
    #   is synchronized to the rising edges on the specified synchronization
    #   channel.

    # States:
    STOPPED = 0
    INITIALIZING = 1
    WORKING = 2
    STOPPING = 3
    ERROR = 4
    EXITING = 5
    DEAD = 100

    stateList = {
        -1:'UNKNOWN',
        STOPPED :'STOPPED',
        INITIALIZING :'INITIALIZING',
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

    settableParams = [
        'verbose'
    ]

    def __init__(self,
                verbose = False,
                videoDirs = None,
                videoFilter = '*',
                **kwargs):
        StateMachineProcess.__init__(self, **kwargs)
        # Store inputs in instance variables for later access
        self.errorMessages = []
        self.verbose = verbose
        self.videoDirs = videoDirs
        self.videoFilter = videoFilter
        self.exitFlag = False

    def setParams(self, **params):
        for key in params:
            if key in ServerJob.settableParams:
                setattr(self, key, params[key])
                if self.verbose >= 1: self.log("Param set: {key}={val}".format(key=key, val=params[key]))
            else:
                if self.verbose >= 0: self.log("Param not settable: {key}={val}".format(key=key, val=params[key]))

    def rescaleAudio(data, maxV=10, minV=-10, maxD=32767, minD=-32767):
        return (data * ((maxD-minD)/(maxV-minV))).astype('int16')

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
                        msg, arg = self.msgQueue.get(block=True)
                        if msg == ServerJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

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
                    neuralNetwork = initializeNeuralNetwork()
                    videoList = getVideoList(self.videoDirs, videoFilter=self.videoFilter)
                    segmentVideo

                    # CHECK FOR MESSAGES
                    try:
                        msg, arg = self.msgQueue.get(block=False)
                        if msg == ServerJob.SETPARAMS: self.setParams(**arg); msg = ''; arg=None
                    except queue.Empty: msg = ''; arg = None

                    # CHOOSE NEXT STATE
                    if self.exitFlag:
                        nextState = ServerJob.STOPPING
                    elif msg in ['', ServerJob.START]:
                        nextState = ServerJob.ACQUIRE_READY
                    elif msg == ServerJob.STOP:
                        nextState = ServerJob.STOPPING
                    elif msg == ServerJob.EXIT:
                        self.exitFlag = True
                        nextState = ServerJob.STOPPING
                    else:
                        raise SyntaxError("Message \"" + msg + "\" not relevant to " + self.stateList[state] + " state")
# ********************************* ACQUIRING *********************************
                elif state == ServerJob.WORKING:
                    # DO STUFF

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
