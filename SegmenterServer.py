from waitress import serve
from wsgi_basic_auth import BasicAuth
import os
import traceback
import logging
import datetime as dt
import time
import sys
from subprocess import Popen, PIPE
import urllib
from pathlib import Path, PureWindowsPath, PurePosixPath
import fnmatch
from ServerJob import ServerJob
from TongueSegmentation import SegmentationSpecification
import queue
import numpy as np
from scipy.io import loadmat
import json
from collections import OrderedDict as odict

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
datetimeString = dt.datetime.now().isoformat()
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
        reqFolder.mkdir()

# Set environment variables for authentication
envVars = dict(os.environ)  # or os.environ.copy()
try:
    envVars['WSGI_AUTH_CREDENTIALS']='glab:password'
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
                interval=5,             # Time in seconds to wait between update requests
                port=80,                # Port to send request to
                host="localhost",       # Host to send request to
                url="/updateQueue",     # Relative URL for triggering queue update
                **kwargs):
        mp.Process.__init__(self, *args, daemon=True, **kwargs)
        self.fullURL = "{host}:{port}{url}".format(host=host, port=port, url=url)
        self.interval = interval

    def run(self):
        while True:
            urllib.request.urlopen(self.fullURL)
            time.sleep(self.inverval)

class SegmentationServer:
    newJobNum = itertools.count().__next__   # Source of this clever little idea: https://stackoverflow.com/a/1045724/1460057
    def __init__(self, webRoot='.'):
        self.routes = [
            ('/static/*',           self.staticHandler),
            ('/finalizeJob',        self.finalizeJobHandler),
            ('/confirmJob/*',       self.confirmJobHandler),
            ('/checkProgress/*',    self.checkProgressHandler),
            ('/updateQueue',        self.updateJobQueueHandler),
            ('/',                   self.rootHandler)
        ]
        self.webRootPath = Path(webRoot).resolve()
        self.maxActiveJobs = 1        # Maximum # of jobs allowed to be running at once
        self.jobQueue = odict() # List of job parameters for waiting jobs

        # Start daemon that periodically makes http request that prompts server to update its job queue
        self.updaterDaemon = UpdaterDaemon(inverval=3)
        self.updateDaemon.start()

    def __call__(self, environ, start_fn):
        for path, handler in self.routes:
            if fnmatch.fnmatch(environ['PATH_INFO'], path):
                return handler(environ, start_fn)
        return self.invalidHandler(environ, start_fn)

    def getMountList(self):
#        p = Popen('mount', stdout=PIPE, stderr=PIPE, shell=True)
        p = Popen("mount | awk '$5 ~ /cifs|drvfs/ {print $0}'", stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = p.communicate()
        mountLines = stdout.decode('utf-8').strip().split('\n')
        mounts = {}
        for mountLine in mountLines:
            elements = mountLine.split(' ')
            mounts[elements[0]] = elements[2]
        logger.log(logging.DEBUG, 'Got mount list: ' + str(mounts))
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
        requestedStaticFileRelativePath = environ['PATH_INFO'].strip('/')
        logger.log(logging.INFO, 'Serving static file: {path}'.format(path=requestedStaticFileRelativePath))
        requestedStaticFilePath = self.webRootPath / requestedStaticFileRelativePath
        if requestedStaticFilePath.exists():
            logger.log(logging.INFO, 'Found that static file')
            start_fn('200 OK', [('Content-Type', 'text/html')])
            for line in requestedStaticFilePath.open('r'):
                yield line.encode('utf-8')
        else:
            logger.log(logging.INFO, 'Could not find that static file: {p}'.format(p=requestedStaticFilePath))
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            yield ['<html><body><h1>Error: Static file {name} not found!</body></html>'.format(name=requestedStaticFileRelativePath).encode('utf-8')]

    def countVideosRemaining(self):
        completedVideosAhead =  sum([len(self.jobQueue[jobNum]['completedVideoList']) for jobNum in self.jobQueue])
        queuedVideosAhead =     sum([len(self.jobQueue[jobNum]['videoList'])          for jobNum in self.jobQueue])
        vidoesAhead = queuedVideosAhead - completedVideosAhead
        return videosAhead

    def finalizeJobHandler(self, environ, start_fn):
        # Display page showing what job will be, and offering opportunity to go ahead or cancel
        postDataRaw = environ['wsgi.input'].read().decode('utf-8')
        postData = urllib.parse.parse_qs(postDataRaw, keep_blank_values=False)

        keys = ['videoRootMountPoint', 'videoRoot', 'videoFilter', 'maskSaveDirectory', 'pathStyle', 'neuralNetwork', 'topOffset', 'topHeight', 'botHeight', 'binaryThreshold']
        if not all([key in postData for key in keys]):
            # Not all form parameters got POSTed
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return ['<html><body><h1>Error: Bad setup parameters. <a href="/">Please click here to re-enter.</a></body></html>'.encode('utf-8')]
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
        segSpec = SegmentationSpecification(
            partNames=['Bot', 'Top'], widths=[None, None], heights=[botHeight, topHeight], xOffsets=[0, 0], yOffsets=[]
        )
        videoList = getVideoList(videoRootMountPoint, videoDirs, pathStyle, videoFilter=videoFilter)

        jobsAhead = len(self.jobQueue)
        videosAhead = self.countVideosRemaining()

        # Add job parameters to queue
        jobNum = SegmentationServer.newJobNum()
        self.jobQueue[jobNum] = dict(
            job=None,                               # Job process object
            jobNum=jobNum,                          # Job ID
            confirmed=False,                        # Has user confirmed params yet
            videoList=videoList,                    # List of video paths to process
            maskSaveDirectory=maskSaveDirectory,    # Path to save masks
            segmentationSpecification=segSpec,      # SegSpec
            neuralNetworkPath=networkName,          # Path to chosen neural network
            completedVideoList=[],                  # List of processed videos
            times=[]                                # List of video processing start times
        )

        start_fn('200 OK', [('Content-Type', 'text/html')])
        return ['''
<html>
    <body>
        <h1>Finalize job:</h1>
        <div>
            There are {jobsAhead} jobs ahead of you with {videosAhead} remaining. Your job (Job ID {jobID}) will be enqueued to start as soon as any/all previous jobs are done.
        </div>
        <div>
            Please review the following job specifications and confirm if correct:
        </div>
        <details>
            <summary>List of video files:</summary>
            <ul>
                {videoList}
            </ul>
        </details>
        <div>Chosen neural network:    {networkName} </div>
        <div>Binary mask threshold:    {binaryThreshold} </div>
        <div>Top mask vertical offset: {topOffset} </div>
        <div>Top mask height:          {topHeight} </div>
        <div>Bottom mask height:       {botHeight} </div>
        <form action="/confirmJob/{jobID}">
            <input type="submit" value="Confirm and enqueue job" />
        </form>
    </body>
</html>
        '''.format(
videoList="\n".join(["<li>{v}</li>".format(v=v) for v in videoList]),
networkName=networkName,
binaryThreshold=binaryThreshold,
topOffset=topOffset,
topHeight=topHeight,
botHeight=botHeight,
jobID=jobNum,
jobsAhead=jobsAhead,
videosAhead=videosAhead
        ).encode('utf-8')]

    def startJob(self, jobNum):
        self.jobQueue[jobNum]['job'] = ServerJob(
            verbose = 1,
            **self.jobQueue[jobNum]
            )

        newJob.start()
        newJob.msgQueue.put((ServerJob.START, None))
        newJob.msgQueue.put((ServerJob.PROCESS, None))

    def getQueuedJobNums(self):
        # Get a list of job nums for queued jobs, in the queue order
        return [self.jobQueue['jobNum'] for jobNum in self.jobQueue if self.jobQueue[jobNum]['job' is None]]
    def getActiveJobNums(self):
        # Get a list of active job nums
        return [self.jobQueue['jobNum'] for jobNum in self.jobQueue if self.jobQueue[jobNum]['job' is not None]]
    def getAllJobNums(self):
        # Get a list of all job nums (both queued and active) in the queue order with active jobs at the start
        return [self.jobQueue['jobNum'] for jobNum in self.jobQueue]

    def confirmJobHandler(self, environ, start_fn):
        # Get jobNum from URL
        jobNum = int(environ['PATH_INFO'].split('/')[-1])
        if jobNum not in self.getQueuedJobNums():
            # Invalid jobNum
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return ['<html><body><h1>Error: Invalid job ID {jobID}. </h1><h2><a href="/">Please click here to re-create job.</a></h2></body></html>'.format(jobID=jobNum).encode('utf-8')]
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

    def removeFinishedJob(self, jobNum):
        del self.jobQueue[jobNum]

    def updateJobQueueHandler(self, environ, start_fn):
        # Handler for automated calls to update the queue
        self.updateJobQueue()

        start_fn('200 OK', [('Content-Type', 'text/html')])
        return []

    def updateJobQueue(self):
        # Check if the current job is done. If it is, remove it and start the next job
        for jobNum in self.getActiveJobNums():
            # Loop over active jobs, see if they're done, and pop them off if so
            job = self.jobQueue[jobNum]['job']
            jobState = job.publishedStateVar.value
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
                job.terminate()
                self.removeFinishedJob()
            elif jobState == ServerJob.EXITING:
                pass
            elif jobState == ServerJob.DEAD:
                self.removeFinishedJob()
            elif jobState == -1:
                pass

        if len(self.getActiveJobNums()) < self.maxActiveJobs:
            # Start the next job, if any
            for jobNum in self.getQueuedJobNums():
                if self.jobQueue[jobNum]['confirmed']:
                    # This is the next confirmed job - start it
                    self.startJob(jobNum)
                    break;

    def checkProgressHandler(self, environ, start_fn):
        # Get jobNum from URL
        jobNum = int(environ['PATH_INFO'].split('/')[-1])
        if jobNum not in self.getAllJobNums():
            # Invalid jobNum
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return ['<html><body><h1>Error: Invalid job ID {jobID}. <a href="/">Click here to create a new job.</a></body></html>'.format(jobID=jobNum).encode('utf-8')]

        if self.jobQueue[jobNum]['job'] is not None:
            jobState = self.jobs[jobNum].publishedStateVar.value
            jobStateName = ServerJob.stateList[jobState]
            while True:
                try:
                    progress = self.jobQueue[jobNum]['job'].progressQueue.get(block=False)
                    self.jobQueue[jobNum]['completedVideoList'].append(progress['lastCompletedVideoPath'])
                    self.jobQueue[jobNum]['times'].append(progress['lastProcessingStartTime'])
                except queue.Empty:
                    # Got all progress
                    break
        else:
            jobStateName = "ENQUEUED"

        if len(self.jobQueue[jobNum]['times']) > 0:
            deltaT = np.diff(self.jobQueue[jobNum]['times'])
            meanTime = np.mean(deltaT)
            stdTime = np.std(deltaT)
            numVideos = len(self.jobQueue[jobNum]['videoList'])
            numCompletedVideos = len(self.jobQueue[jobNum]['completedVideoList'])
            estimatedTimeRemaining = str(dt.timedelta(seconds=(numVideos - numCompletedVideos) * meanTime))
        else:
            meanTime = "Unknown"
            stdTime = "Unknown"
            estimatedTimeRemaining = "Unknown"

        completedVideoListHTML = "\n".join(["<li>{v}</li>".format(v=v) for v in self.jobProgress[jobNum]['completedVideoList']]),

        start_fn('200 OK', [('Content-Type', 'text/html')])
        return ['''
<html>
    <body>
        <h1>Job {jobID} is in state {jobStateName}.</h1>
        <div>
            Processing time per video: {meanTime} &plusmn; {confInt}
        </div>
        <div>
            Estimated time remaining: {estimatedTimeRemaining}
        </div>
        <div>
            <details>
                <summary>List of completed video files:</summary>
                <ul>
                    {videoList}
                </ul>
            </details>
        </div>
    </body>
</html>
        '''.format(
            meanTime=meanTime,
            confInt=stdTime*1.96,
            videoList=completedVideoListHTML,
            jobStateName=jobStateName
        ).encode('utf-8')]

    def rootHandler(self, environ, start_fn):
        logger.log(logging.INFO, 'Serving root file')
        neuralNetworkList = self.getNeuralNetworkList()
        mountList = self.getMountList()
        mountList['Local'] = 'LOCAL'
        mountURIs = mountList.keys()
        mountPaths = [mountList[k] for k in mountURIs]
        mountOptionsText = self.createOptionList(mountPaths, mountURIs)
        start_fn('200 OK', [('Content-Type', 'text/html')])
        if 'QUERY_STRING' in environ:
            queryString = environ['QUERY_STRING']
        else:
            queryString = 'None'
        postDataRaw = environ['wsgi.input'].read().decode('utf-8')
        postData = urllib.parse.parse_qs(postDataRaw, keep_blank_values=False)

        if len(neuralNetworkList) > 0:
            networkOptionText = self.createOptionList(neuralNetworkList)
            formText = '''
<form action="/finalizeJob" method="POST">
    <div class="field-wrap">
        <label class="field-label" for="videoRootMountPoint">Video root mount point:</label>
        <select class="field" name="videoRootMountPoint" id="videoRootMountPoint">
        {mopts}
        </select>
    </div>
    <div class="field-wrap">
        <label class="field-label" for="videoRoot">Video root directories (one per line):</label>
        <textarea class="field" rows="3" cols="30" id="videoRoot" name="videoRoot" value=""></textarea>
    </div>
    <div class="field-wrap">
        <label class="field-label" for="videoFilter">Video filename filter:</label>
        <input class="field" type="text" id="videoFilter" name="videoFilter" value="*">
    </div>
    <div class="field-wrap">
        <label class="field-label" for="maskSaveDirectory">Mask save directory:</label>
        <input class="field" type="text" id="maskSaveDirectory" name="maskSaveDirectory" value="">
    </div>
    <div class="field-wrap"
        <label class="field-label" for="pathStyleWrapper">Path style:</label>
        <div id="pathStyleWrapper" class="field">
            <div>
              <input type="radio" id="windowsStyle" name="pathStyle" value="windowsStylePaths"
                     checked>
              <label for="windowsStyle">Windows-style paths</label>
            </div>
            <div>
              <input type="radio" id="posixStyle" name="pathStyle" value="posixStylePaths">
              <label for="posixStyle">Mac/Linux-style paths</label>
            </div>
        </div>
    </div>
    <div class="field-wrap">
        <label class="field-label" for="networkName">Neural network name:</label>
        <select class="field" name="neuralNetwork" id="neuralNetwork">
        {nopts}
        </select>
    </div>
    <div class="field-wrap">
        <label class="field-label" for="topOffset">Top mask vertical offset:</label>
        <input class="field" type="number" id="topOffset" name="topOffset" min="0" step="1" value="0">
    </div>
    <div class="field-wrap">
        <label class="field-label" for="topHeight">Top mask height:</label>
        <input class="field" type="number" id="topHeight" name="topHeight" min="0" step="1" value="0">
    </div>
    <div class="field-wrap">
        <label class="field-label" for="botHeight">Top mask height:</label>
        <input class="field" type="number" id="botHeight" name="botHeight" min="0" step="1" value="0">
    </div>
    <div class="field-wrap">
        <label class="field-label" for="binaryThreshold">Binary Threshold:</label>
        <input class="field" type="number" id="binaryThreshold" name="binaryThreshold" min="0" max="1" step="0.001" value="0.3">
    </div>
    <input class="field" type="submit" value="Initialize job">
</form>'''.format(nopts=networkOptionText, mopts=mountOptionsText)
        else:
            formText = '''
<h2>No neural networks found!
Please upload a .h5 or .hd5 neural network file to the ./{nnsubfolder} folder.</h2>'''.format(nnsubfolder=NETWORKS_SUBFOLDER)

        response = [
        '''
<html>
<head>
    <link rel="stylesheet" type="text/css" href="static/main.css">
</head>
<body>
    <h1>
        Segmentation Processor for Images of Tongues (SPIT)
    </h1>
    {form}
    <p>Query = {query}</p>
    <p>Mounts = {mounts}</p>
    <p>Input = {input}</p>
    <p>Path = {path}</p>
    <p>environ = {environ}</p>
</body>
</html>
        '''.format(query=queryString, mounts=mountList, environ=environ, input=postData, form=formText, path=environ['PATH_INFO']),
        ]
        response = [line.encode('utf-8') for line in response]
        return response

    def invalidHandler(self, environ, start_fn):
        logger.log(logging.INFO, 'Serving invalid warning')
        requestedPath = environ['PATH_INFO']
        start_fn('404 Not Found', [('Content-Type', 'text/html')])
        return ['<html><body><h1>Error: Path {name} not recognized!</body></html>'.format(name=requestedPath).encode('utf-8')]

if len(sys.argv) > 1:
    port = int(sys.argv[1])
else:
    port = 80

logger.log(logging.INFO, 'Spinning up server!')
while True:
    s = SegmentationServer(webRoot=ROOT)
    application = BasicAuth(s)
    try:
        logger.log(logging.INFO, 'Starting segmentation server...')
        serve(application, host='0.0.0.0', port=port)
        logger.log(logging.INFO, '...segmentation server started!')
    except:
        logger.exception('Server crashed!')
    time.sleep(5)
