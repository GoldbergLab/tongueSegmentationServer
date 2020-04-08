from waitress import serve
from wsgi_basic_auth import BasicAuth
import os
if os.name == 'nt':
    import win32net
import traceback
import logging
import datetime as dt
import time
import sys
from subprocess import Popen, PIPE
import urllib
import requests
from pathlib import Path, PureWindowsPath, PurePosixPath
import fnmatch
from ServerJob import ServerJob
from TongueSegmentation import SegmentationSpecification
import queue
import numpy as np
from scipy.io import loadmat
import json
from collections import OrderedDict as odict
import multiprocessing as mp
import itertools

# Tensorflow barfs a ton of debug output - restrict this to only warnings/errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NEURAL_NETWORK_EXTENSIONS = ['.h5', '.hd5']
NETWORKS_SUBFOLDER = 'networks'
LOGS_SUBFOLDER = 'logs'
STATIC_SUBFOLDER = 'static'
ROOT = '.'

logger = logging.getLogger(__name__)

# create logger with 'spam_application'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
datetimeString = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
fh = logging.FileHandler('./{logs}/{n}_{d}.log'.format(d=datetimeString, n=__name__, logs=LOGS_SUBFOLDER))
fh.setLevel(logging.INFO)
# create console handler with a higher log level
# ch = logging.StreamHandler(stream=sys.stdout)
# ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
# logger.addHandler(ch)

rootPath = Path(ROOT)
networksFolder = rootPath / NETWORKS_SUBFOLDER
logsFolder = rootPath / LOGS_SUBFOLDER
staticFolder = rootPath / STATIC_SUBFOLDER
requiredSubfolders = [networksFolder, logsFolder, staticFolder]
for reqFolder in requiredSubfolders:
    if not reqFolder.exists():
        logger.log(logging.INFO, 'Creating required directory: {reqDir}'.format(reqDir=reqFolder))
        reqFolder.mkdir()

# Set environment variables for authentication
envVars = dict(os.environ)  # or os.environ.copy()
USER='glab'
PASSWORD='password'
try:
    envVars['WSGI_AUTH_CREDENTIALS']='{UN}:{PW}'.format(UN=USER, PW=PASSWORD)
finally:
    os.environ.clear()
    os.environ.update(envVars)

def reRootDirectories(videoRootMountPoint, pathStyle, *directories):
    #   videoRootMountPoint - the root of the videoDirs. If videoDirs contains a drive root, replace it.
    #   videoDirs - a list of strings representing video directory paths to look in
    #   pathStyle - the style of the videoDirs paths - either 'windowsStyle' or 'posixStyle'

    reRootedDirectories = []
    if pathStyle == 'windowsStylePaths':
        OSPurePath = PureWindowsPath
    elif pathStyle == 'posixStylePaths':
        OSPurePath = PurePosixPath
    else:
        raise ValueError('Invalid path style: {pathStyle}'.format(pathStyle=pathStyle))

    for directory in directories:
        directoryPath = OSPurePath(directory)
        if directoryPath.parts[0] == directoryPath.anchor:
            # This path includes the root - remove it.
            rootlessDirectoryPathParts = directoryPath.parts[1:]
        else:
            rootlessDirectoryPathParts = directoryPath.parts
        finalizedDirectoryPath = Path(videoRootMountPoint) / Path(*rootlessDirectoryPathParts)
        reRootedDirectories.append(finalizedDirectoryPath)
    return reRootedDirectories

def getVideoList(videoRootMountPoint, videoDirs, pathStyle, videoFilter='*'):
    # Generate a list of video Path objects from the given directories using the given path filters
    #   videoRootMountPoint - the root of the videoDirs. If videoDirs contains a drive root, replace it.
    #   videoDirs - a list of strings representing video directory paths to look in
    #   pathStyle - the style of the videoDirs paths - either 'windowsStyle' or 'posixStyle'
    finalizedVideoDirs = reRootDirectories(videoRootMountPoint, pathStyle, *videoDirs)

    videoList = []
    for p in finalizedVideoDirs:
        for videoPath in p.iterdir():
            if videoPath.match(videoFilter):
                videoList.append(videoPath)
    return videoList

class UpdaterDaemon(mp.Process):
    def __init__(self,
                *args,
                interval=5,             # Time in seconds to wait between update requests
                port=80,                # Port to send request to
                host="localhost",       # Host to send request to
                url="/updateQueue",     # Relative URL for triggering queue update
                **kwargs):
        mp.Process.__init__(self, *args, daemon=True, **kwargs)
        self.fullURL = "http://{host}:{port}{url}".format(host=host, port=port, url=url)
        logger.log(logging.INFO, "UpdaterDaemon ready with update url {url}".format(url=self.fullURL))
        self.interval = interval

        # # create a password manager
        # password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        # # Add the username and password.
        # # If we knew the realm, we could use it instead of None.
        # top_level_url = "http://{host}/".format(host=host)
        # password_mgr.add_password(None, top_level_url, USER, PASSWORD)
        # handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        #
        # # create "opener" (OpenerDirector instance)
        # self.opener = urllib.request.build_opener(handler)

    def run(self):
        while True:
            r = requests.get(self.fullURL, auth=(USER, PASSWORD))
            # # use the opener to fetch a URL
            # self.opener.open(self.fullURL)
            # urllib.request.urlopen(self.fullURL)
            time.sleep(self.interval)

class SegmentationServer:
    newJobNum = itertools.count().__next__   # Source of this clever little idea: https://stackoverflow.com/a/1045724/1460057
    def __init__(self, port=80, webRoot='.'):
        self.port = port
        self.routes = [
            ('/static/*',           self.staticHandler),
            ('/finalizeJob',        self.finalizeJobHandler),
            ('/confirmJob/*',       self.confirmJobHandler),
            ('/checkProgress/*',    self.checkProgressHandler),
            ('/updateQueue',        self.updateJobQueueHandler),
            ('/',                   self.rootHandler)
        ]
        self.webRootPath = Path(webRoot).resolve()
        self.maxActiveJobs = 1          # Maximum # of jobs allowed to be running at once
        self.jobQueue = odict()         # List of job parameters for waiting jobs

        self.cleanupTime = 86400        # Number of seconds to wait before deleting finished/dead jobs

        # Start daemon that periodically makes http request that prompts server to update its job queue
        self.updaterDaemon = UpdaterDaemon(interval=3, port=self.port)
        self.updaterDaemon.start()

    def __call__(self, environ, start_fn):
        for path, handler in self.routes:
            if fnmatch.fnmatch(environ['PATH_INFO'], path):
                logger.log(logging.INFO, 'Matched url {path} to route {route} with handler {handler}'.format(path=environ['PATH_INFO'], route=path, handler=handler))
                return handler(environ, start_fn)
        return self.invalidHandler(environ, start_fn)

    def getMountList(self, includePosixLocal=False):
        mounts = {}
        if os.name == 'nt':
            # Get a list of drives
            resume = 0
            while 1:
                (_drives, total, resume) = win32net.NetShareEnum (None, 2, resume)
                for drive in _drives:
                    mounts[drive['netname']] = drive['path']
                if not resume: break

            # Add to that list the list of network shares
            resume = 0
            while 1:
                (_drives, total, resume) = win32net.NetUseEnum (None, 0, resume)
                for drive in _drives:
                    if drive['local']:
                        mounts[drive['remote']] = drive['local']
                if not resume: break
        elif os.name == 'posix':
            if includeLocal:
                mounts['Local'] = 'LOCAL'
    #        p = Popen('mount', stdout=PIPE, stderr=PIPE, shell=True)
            p = Popen("mount | awk '$5 ~ /cifs|drvfs/ {print $0}'", stdout=PIPE, stderr=PIPE, shell=True)
            stdout, stderr = p.communicate()
            mountLines = stdout.decode('utf-8').strip().split('\n')
            for mountLine in mountLines:
                elements = mountLine.split(' ')
                mounts[elements[0]] = elements[2]
            logger.log(logging.DEBUG, 'Got mount list: ' + str(mounts))
        else:
            # Uh oh...
            raise OSError('This software is only compatible with POSIX or Windows')
        return mounts

    def getNeuralNetworkList(self):
        # Generate a list of available neural networks
        p = Path('.') / NETWORKS_SUBFOLDER
        networks = []
        for item in p.iterdir():
            if item.suffix in NEURAL_NETWORK_EXTENSIONS:
                # This is a neural network file
                networks.append(item.name)
        return networks

    def createOptionList(self, optionValues, optionNames=None):
        if optionNames is None:
            optionNames = optionValues
        options = []
        for optionValue, optionName in zip(optionValues, optionNames):
            options.append('<option value="{v}">{n}</option>'.format(v=optionValue, n=optionName))
        optionText = "\n".join(options)
        return optionText

    def staticHandler(self, environ, start_fn):
        URLparts = environ['PATH_INFO'].split('/')
        requestedStaticFileRelativePath = environ['PATH_INFO'].strip('/')

        if len(URLparts) < 2:
            logger.log(logging.INFO, 'Could not find that static file: {p}'.format(p=requestedStaticFilePath))
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            with open('Error.html', 'r') as f: htmlTemplate = f.read()
            yield [htmlTemplate.format(
                errorTitle='Static file not found',
                errorMsg='Static file {name} not found'.format(name=requestedStaticFileRelativePath),
                linkURL='/',
                linkAction='return to job creation page'
                ).encode('utf-8')]
        else:
            subfolder = int(environ['PATH_INFO'].split('/')[-2])
            if subfolder == "css":
                mimeType = "text/css"
            elif subfolder == "favicon":
                mimeType = "image/x-icon"
            else:
                logger.log(logging.INFO, 'Invalid static file location: {s}'.format(s=subfolder))
                start_fn('404 Not Found', [('Content-Type', 'text/html')])
                with open('Error.html', 'r') as f: htmlTemplate = f.read()
                yield [htmlTemplate.format(
                    errorTitle='Static file not found',
                    errorMsg='Static file {name} not found'.format(name=requestedStaticFileRelativePath),
                    linkURL='/',
                    linkAction='return to job creation page'
                    ).encode('utf-8')]

        logger.log(logging.INFO, 'Serving static file: {path}'.format(path=requestedStaticFileRelativePath))
        requestedStaticFilePath = self.webRootPath / requestedStaticFileRelativePath
        if requestedStaticFilePath.exists():
            logger.log(logging.INFO, 'Found that static file')
            start_fn('200 OK', [('Content-Type', mimeType)])
            for line in requestedStaticFilePath.open('r'):
                yield line.encode('utf-8')
        else:
            logger.log(logging.INFO, 'Could not find that static file: {p}'.format(p=requestedStaticFilePath))
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            with open('Error.html', 'r') as f: htmlTemplate = f.read()
            yield [htmlTemplate.format(
                errorTitle='Static file not found',
                errorMsg='Static file {name} not found'.format(name=requestedStaticFileRelativePath),
                linkURL='/',
                linkAction='return to job creation page'
                ).encode('utf-8')]

    def countJobsRemaining(self, beforeJobNum=None):
        activeJobsAhead = 0
        queuedJobsAhead = 0
        for jobNum in self.jobQueue:
            if beforeJobNum is not None and jobNum == beforeJobNum:
                # This is the specified job num - stop, don't count any more
                break
            if self.jobQueue[jobNum]['completionTime'] is None:
                if self.jobQueue[jobNum]['startTime'] is None:
                    queuedJobsAhead += 1
                else:
                    activeJobsAhead += 1
        jobsAhead = queuedJobsAhead + activeJobsAhead
        return jobsAhead

    def countVideosRemaining(self, beforeJobNum=None):
        completedVideosAhead = 0
        queuedVideosAhead = 0
        for jobNum in self.jobQueue:
            if beforeJobNum is not None and jobNum == beforeJobNum:
                # This is the specified job num - stop, don't count any more
                break
            if self.jobQueue[jobNum]['completionTime'] is None:
                completedVideosAhead += len(self.jobQueue[jobNum]['completedVideoList'])
                queuedVideosAhead += len(self.jobQueue[jobNum]['videoList'])
        videosAhead = queuedVideosAhead - completedVideosAhead
        return videosAhead

    def finalizeJobHandler(self, environ, start_fn):
        # Display page showing what job will be, and offering opportunity to go ahead or cancel
        postDataRaw = environ['wsgi.input'].read().decode('utf-8')
        postData = urllib.parse.parse_qs(postDataRaw, keep_blank_values=False)

        keys = ['videoRootMountPoint', 'videoRoot', 'videoFilter', 'maskSaveDirectory', 'pathStyle', 'neuralNetwork', 'topOffset', 'topHeight', 'botHeight', 'binaryThreshold', 'jobName']
        if not all([key in postData for key in keys]):
            # Not all form parameters got POSTed
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            with open('Error.html', 'r') as f: htmlTemplate = f.read()
            return [htmlTemplate.format(
                errorTitle='Bad job creation parameters',
                errorMsg='One or more job creation parameters missing!',
                linkURL='/',
                linkAction='retry job creation'
                ).encode('utf-8')]
        videoRootMountPoint = postData['videoRootMountPoint'][0]
        videoDirs = postData['videoRoot'][0].strip().splitlines()
        videoFilter = postData['videoFilter'][0]
        maskSaveDirectory = postData['maskSaveDirectory'][0]
        pathStyle = postData['pathStyle'][0]
        networkName = networksFolder / postData['neuralNetwork'][0]
        binaryThreshold = postData['binaryThreshold'][0]
        topOffset = postData['topOffset'][0]
        topHeight = postData['topHeight'][0]
        botHeight = postData['botHeight'][0]
        jobName = postData['jobName'][0]
        segSpec = SegmentationSpecification(
            partNames=['Bot', 'Top'], widths=[None, None], heights=[int(botHeight), int(topHeight)], xOffsets=[0, 0], yOffsets=[0, 0]
        )
        videoList = getVideoList(videoRootMountPoint, videoDirs, pathStyle, videoFilter=videoFilter)

        jobsAhead = len(self.jobQueue)
        videosAhead = self.countVideosRemaining()

        # Add job parameters to queue
        jobNum = SegmentationServer.newJobNum()
        self.jobQueue[jobNum] = dict(
            job=None,                               # Job process object
            jobName=jobName,                        # Name/description of job
            jobNum=jobNum,                          # Job ID
            confirmed=False,                        # Has user confirmed params yet
            videoList=videoList,                    # List of video paths to process
            maskSaveDirectory=maskSaveDirectory,    # Path to save masks
            segmentationSpecification=segSpec,      # SegSpec
            neuralNetworkPath=networkName,          # Path to chosen neural network
            completedVideoList=[],                  # List of processed videos
            times=[],                               # List of video processing start times
            creationTime=time.time_ns(),            # Time job was created
            startTime=None,                         # Time job was started
            completionTime=None,                    # Time job was completed
            log=[],                                 # List of log output from job
            exitCode=ServerJob.INCOMPLETE           # Job exit code
        )

        start_fn('200 OK', [('Content-Type', 'text/html')])
        with open('FinalizeJob.html', 'r') as f: htmlTemplate = f.read()
        return [htmlTemplate.format(
videoList="\n".join(["<li>{v}</li>".format(v=v) for v in videoList]),
networkName=networkName,
binaryThreshold=binaryThreshold,
topOffset=topOffset,
topHeight=topHeight,
botHeight=botHeight,
jobID=jobNum,
jobName=jobName,
jobsAhead=jobsAhead,
videosAhead=videosAhead
        ).encode('utf-8')]

    def startJob(self, jobNum):
        self.jobQueue[jobNum]['job'] = ServerJob(
            verbose = 1,
            logger=logger,
            **self.jobQueue[jobNum]
            )

        self.jobQueue[jobNum]['job'].start()
        self.jobQueue[jobNum]['job'].msgQueue.put((ServerJob.START, None))
        self.jobQueue[jobNum]['job'].msgQueue.put((ServerJob.PROCESS, None))
        self.jobQueue[jobNum]['startTime'] = time.time_ns()

    def getQueuedJobNums(self):
        # Get a list of job nums for queued jobs, in the queue order
        return [jobNum for jobNum in self.jobQueue if self.jobQueue[jobNum]['job'] is None]
    def getActiveJobNums(self):
        # Get a list of active job nums
        return [jobNum for jobNum in self.jobQueue if self.jobQueue[jobNum]['job'] is not None and self.jobQueue[jobNum]['completionTime'] is None]
    def getCompletedJobNums(self):
        # Get a list of completed job nums
        return [jobNum for jobNum in self.jobQueue if self.jobQueue[jobNum]['job'] is not None and self.jobQueue[jobNum]['completionTime'] is not None]
    def getAllJobNums(self):
        # Get a list of all job nums (both queued and active) in the queue order with active jobs at the start
        return [jobNum for jobNum in self.jobQueue]

    def confirmJobHandler(self, environ, start_fn):
        # Get jobNum from URL
        jobNum = int(environ['PATH_INFO'].split('/')[-1])
        if jobNum not in self.getQueuedJobNums():
            # Invalid jobNum
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            with open('Error.html', 'r') as f: htmlTemplate = f.read()
            errorMsg = 'Invalid job ID {jobID}'.format(jobID=jobNum)
            return [htmlTemplate.format(
                errorTitle='Invalid job ID',
                errorMsg=errorMsg,
                linkURL='/',
                linkAction='recreate job'
                ).encode('utf-8')]
        # elif self.jobQueue[jobNum].exitcode is not None:
        #     # Error, process has terminated
        #     start_fn('404 Not Found', [('Content-Type', 'text/html')])
        #     return ['<html><body><h1>Error: Requested job has already terminated. </h1><h2><a href="/">Please click here to re-create job.</a></h2></body></html>'.encode('utf-8')]
        # elif self.pendingJobs[jobNum].publishedStateVar.value != ServerJob.WAITING:
        #     # Error, job is not in the waiting state
        #     jobState = ServerJob.stateList[self.pendingJobs[jobNum].publishedStateVar.value]
        #     start_fn('404 Not Found', [('Content-Type', 'text/html')])
        #     return ['<html><body><h1>Error: Job is not waiting to begin, but instead is in state {state}. </h1><h2><a href="/">Please click here to re-enter.</a></h2></body></html>'.format(state=jobState).encode('utf-8')]
        else:
            # Valid enqueued job - set confirmed flag to True, so it can be started when at the front
            self.jobQueue[jobNum]['confirmed'] = True
        start_fn('303 See Other', [('Location','/checkProgress/{jobID}'.format(jobID=jobNum))])
        return []

    def removeFinishedJob(self, jobNum, waitingPeriod=0):
        # waitingPeriod = amount of time in seconds to wait after job completionTime before removing from queue
        completionTime = self.jobQueue[jobNum]['completionTime']
        if completionTime is None or (time.time_ns() - completionTime) / 1000000 > waitingPeriod:
            # Either completion time has not been set, or the requisite waiting period has elapsed.
            #   Delete job.
            del self.jobQueue[jobNum]

    def updateJobQueueHandler(self, environ, start_fn):
        # Handler for automated calls to update the queue
        self.updateJobQueue()

        # logger.log(logging.INFO, 'Got automated queue update reminder')
        #
        start_fn('200 OK', [('Content-Type', 'text/html')])
        return []

    def updateJobQueue(self):
        # Check if the current job is done. If it is, remove it and start the next job
        for jobNum in self.getActiveJobNums():
            # Loop over active jobs, see if they're done, and pop them off if so
            job = self.jobQueue[jobNum]['job']
            jobState = job.publishedStateVar.value
            # Update progress
            self.updateJobProgress(jobNum)
#            jobStateName = ServerJob.stateList[jobState]
            if jobState == ServerJob.STOPPED:
                pass
            elif jobState == ServerJob.INITIALIZING:
                pass
            elif jobState == ServerJob.WAITING:
                pass
            elif jobState == ServerJob.WORKING:
                pass
            elif jobState == ServerJob.STOPPING:
                pass
            elif jobState == ServerJob.ERROR:
                pass
                # job.terminate()
                # self.removeFinishedJob(jobNum)
                # logger.log(logging.INFO, "Removing job {jobNum} in error state".format(jobNum=jobNum))
            elif jobState == ServerJob.EXITING:
                pass
            elif jobState == ServerJob.DEAD:
                self.jobQueue[jobNum]['completionTime'] = time.time_ns()
                self.removeFinishedJob(jobNum, waitingPeriod=self.cleanupTime)
                logger.log(logging.INFO, "Removing job {jobNum} in dead state".format(jobNum=jobNum))
            elif jobState == -1:
                pass

        if len(self.getActiveJobNums()) < self.maxActiveJobs:
            # Start the next job, if any
            for jobNum in self.getQueuedJobNums():
                if self.jobQueue[jobNum]['confirmed']:
                    # This is the next confirmed job - start it
                    self.startJob(jobNum)
                    break;

    def updateJobProgress(self, jobNum):
        if jobNum in self.jobQueue and self.jobQueue[jobNum]['job'] is not None:
            while True:
                try:
                    progress = self.jobQueue[jobNum]['job'].progressQueue.get(block=False)
                    # Get any new log output from job
                    self.jobQueue[jobNum]['log'].extend(progress['log'])
                    # Get updated exit code from job
                    self.jobQueue[jobNum]['exitCode'] = progress['exitCode']
                    # Get the path to the last video the job has completed
                    if progress['lastCompletedVideoPath'] is not None:
                        self.jobQueue[jobNum]['completedVideoList'].append(progress['lastCompletedVideoPath'])
                    # Get the time when the last video started processing
                    if progress['lastProcessingStartTime'] is not None:
                        self.jobQueue[jobNum]['times'].append(progress['lastProcessingStartTime'])
                except queue.Empty:
                    # Got all progress
                    break

    def formatLogHTML(self, log):
        logHTMLList = []
        for logEntry in log:
            logHTMLList.append('<p>{logEntry}</p>'.format(logEntry=logEntry))
        logHTML = "\n".join(logHTMLList)
        return logHTML


    def checkProgressHandler(self, environ, start_fn):
        # Get jobNum from URL
        jobNum = int(environ['PATH_INFO'].split('/')[-1])
        allJobNums = self.getAllJobNums()
        logger.log(logging.INFO, 'jobNum={jobNum}, allJobNums={allJobNums}, jobQueue={jobQueue}'.format(jobNum=jobNum, allJobNums=allJobNums, jobQueue=self.jobQueue))
        if jobNum not in allJobNums:
            # Invalid jobNum
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            with open('Error.html', 'r') as f: htmlTemplate = f.read()
            errorMsg = 'Invalid job ID {jobID}'.format(jobID=jobNum)
            return [htmlTemplate.format(
                errorTitle='Invalid job ID',
                errorMsg=errorMsg,
                linkURL='/',
                linkAction='create a new job'
                ).encode('utf-8')]

        if self.jobQueue[jobNum]['job'] is not None:
            jobState = self.jobQueue[jobNum]['job'].publishedStateVar.value
            jobStateName = ServerJob.stateList[jobState]
            self.updateJobProgress(jobNum)
        else:
            jobStateName = "ENQUEUED"

        creationTime = ""
        startTime = "Not started yet"
        completionTime = "Not complete yet"
        if self.jobQueue[jobNum]['creationTime'] is not None:
            creationTime = dt.datetime.fromtimestamp(self.jobQueue[jobNum]['creationTime']/1000000000).isoformat()
        if self.jobQueue[jobNum]['startTime'] is not None:
            startTime = dt.datetime.fromtimestamp(self.jobQueue[jobNum]['startTime']/1000000000).isoformat()
        if self.jobQueue[jobNum]['completionTime'] is not None:
            completionTime = dt.datetime.fromtimestamp(self.jobQueue[jobNum]['completionTime']/1000000000).isoformat()

        numVideos = len(self.jobQueue[jobNum]['videoList'])
        if len(self.jobQueue[jobNum]['times']) > 1:
            deltaT = np.diff(self.jobQueue[jobNum]['times'])/1000000000
            meanTime = np.mean(deltaT)
            timeConfInt = np.std(deltaT)*1.96
            numCompletedVideos = len(self.jobQueue[jobNum]['completedVideoList'])
            if self.jobQueue[jobNum]['completionTime'] is None:
                estimatedTimeRemaining = str(dt.timedelta(seconds=(numVideos - numCompletedVideos) * meanTime))
            else:
                estimatedTimeRemaining = "None"
            percentComplete = "{percentComplete:.1}".format(percentComplete=numCompletedVideos/numVideos)
        else:
            meanTime = "Unknown"
            timeConfInt = "Unknown"
            estimatedTimeRemaining = "Unknown"
            numCompletedVideos = "0"
            percentComplete = "0"

        completedVideoListHTML = "\n".join(["<li>{v}</li>".format(v=v) for v in self.jobQueue[jobNum]['completedVideoList']])
        if len(completedVideoListHTML.strip()) == 0:
            completedVieoListHTML = "None"

        exitCode = self.jobQueue[jobNum]['exitCode']
        stateDescription = ''
        if exitCode == ServerJob.INCOMPLETE:
            if self.jobQueue[jobNum]['startTime'] is None:
                jobsAhead = self.countJobsRemaining(beforeJobNum=jobNum)
                videosAhead = self.countVideosRemaining(beforeJobNum=jobNum)
                exitCodePhrase = 'is enqueued, but not started.'
                stateDescription = '<br/>There are <strong>{jobsAhead} jobs</strong> \
                                    ahead of you with <strong>{videosAhead} videos</strong> \
                                    remaining. Your job will be enqueued to start as soon \
                                    as any/all previous jobs are done.'.format(jobsAhead=jobsAhead, videosAhead=videosAhead)
            else:
                exitCodePhrase = 'is <strong>in progress</strong>!'
        elif exitCode == ServerJob.SUCCESS:
            exitCodePhrase = 'is <strong>complete!</strong>'
        elif exitCode == ServerJob.FAILED:
            exitCodePhrase = 'has exited with errors :(  Please see debug output below.'
        else:
            exitCodePhrase = 'is in an unknown exit code state...'

        logHTML = self.formatLogHTML(self.jobQueue[jobNum]['log'])

        start_fn('200 OK', [('Content-Type', 'text/html')])
        with open('CheckProgress.html', 'r') as f: htmlTemplate = f.read()
        return [htmlTemplate.format(
            meanTime=meanTime,
            confInt=timeConfInt,
            videoList=completedVideoListHTML,
            jobStateName=jobStateName,
            jobID=jobNum,
            estimatedTimeRemaining=estimatedTimeRemaining,
            jobName=self.jobQueue[jobNum]['jobName'],
            creationTime=creationTime,
            startTime=startTime,
            completionTime=completionTime,
            exitCodePhrase=exitCodePhrase,
            logHTML=logHTML,
            percentComplete=percentComplete,
            numComplete=numCompletedVideos,
            numTotal=numVideos,
            stateDescription=stateDescription
        ).encode('utf-8')]

    def rootHandler(self, environ, start_fn):
        logger.log(logging.INFO, 'Serving root file')
        neuralNetworkList = self.getNeuralNetworkList()
        mountList = self.getMountList(includePosixLocal=True)
        mountURIs = mountList.keys()
        mountPaths = [mountList[k] for k in mountURIs]
        mountOptionsText = self.createOptionList(mountPaths, mountURIs)
        logger.log(logging.INFO, 'rootHandler 1')
        if 'QUERY_STRING' in environ:
            queryString = environ['QUERY_STRING']
        else:
            queryString = 'None'
        postDataRaw = environ['wsgi.input'].read().decode('utf-8')
        postData = urllib.parse.parse_qs(postDataRaw, keep_blank_values=False)

        logger.log(logging.INFO, 'Creating return data')

        if len(neuralNetworkList) > 0:
            networkOptionText = self.createOptionList(neuralNetworkList)
            with open('Index.html', 'r') as f: htmlTemplate = f.read()
            start_fn('200 OK', [('Content-Type', 'text/html')])
            return [htmlTemplate.format(
                query=queryString,
                mounts=mountList,
                environ=environ,
                input=postData,
                path=environ['PATH_INFO'],
                nopts=networkOptionText,
                mopts=mountOptionsText).encode('utf-8')]
        else:
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            with open('Error.html', 'r') as f: htmlTemplate = f.read()
            errorMsg = 'No neural networks found! Please upload a .h5 or .hd5 neural network file to the ./{nnsubfolder} folder.'.format(nnsubfolder=NETWORKS_SUBFOLDER)
            return [htmlTemplate.format(
                errorTitle='Neural network error',
                errorMsg=errorMsg,
                linkURL='/',
                linkAction='retry job creation once a neural network has been uploaded'
                ).encode('utf-8')]

    def invalidHandler(self, environ, start_fn):
        logger.log(logging.INFO, 'Serving invalid warning')
        requestedPath = environ['PATH_INFO']
        start_fn('404 Not Found', [('Content-Type', 'text/html')])
        with open('Error.html', 'r') as f: htmlTemplate = f.read()
        errorMsg = 'No neural networks found! Please upload a .h5 or .hd5 neural network file to the ./{nnsubfolder} folder.'.format(nnsubfolder=NETWORKS_SUBFOLDER)
        return [htmlTemplate.format(
            errorTitle='Path not recognized',
            errorMsg='Path {name} not recognized!'.format(name=requestedPath),
            linkURL='/',
            linkAction='return to job creation page'
            ).encode('utf-8')]

if __name__ == '__main__':
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 80

    logger.log(logging.INFO, 'Spinning up server!')
    while True:
        s = SegmentationServer(webRoot=ROOT, port=port)
        application = BasicAuth(s)
        try:
            logger.log(logging.INFO, 'Starting segmentation server...')
            serve(application, host='0.0.0.0', port=port)
            logger.log(logging.INFO, '...segmentation server started!')
        except:
            logger.exception('Server crashed!')
        time.sleep(5)
