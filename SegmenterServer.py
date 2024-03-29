from waitress import serve
from wsgi_basic_auth import BasicAuth
import os
if os.name == 'nt':
    import win32net
import logging
import datetime as dt
import time
import sys
from subprocess import Popen, PIPE
import urllib
import requests
from pathlib import Path, PureWindowsPath, PurePosixPath
import fnmatch
from ServerJob import ServerJob, SegmentationJob, TrainJob
from TongueSegmentation import SegSpec
from NetworkTraining import createDataAugmentationParameters
import queue
import numpy as np
from scipy.io import loadmat
import json
from collections import OrderedDict as odict
import multiprocessing as mp
import itertools
from base64 import b64decode
import json
from webob import Request
import re

# Tensorflow barfs a ton of debug output - restrict this to only warnings/errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

LOGS_SUBFOLDER = 'logs'

def initializeLogger():
    logger = logging.getLogger(__name__)

    # create logger with 'spam_application'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    datetimeString = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')

    # If logs folder doesn't exist, create it
    if not Path(LOGS_SUBFOLDER).exists():
        Path(LOGS_SUBFOLDER).mkdir()

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
    return logger

logger = initializeLogger()

TRAIN_TYPE = 'train'
SEGMENT_TYPE = 'segment'

NEURAL_NETWORK_EXTENSIONS = ['.h5', '.hd5', '.hdf5']
NETWORKS_SUBFOLDER = 'networks'
STATIC_SUBFOLDER = 'static'
ROOT = '.'
PRIVATE_SUBFOLDER = 'private'
AUTH_NAME = 'Auth.json'
REMOVED_NETWORKS_SUBFOLDER = 'deleted'

DEFAULT_MOUNT_PATH = 'X:'

DEFAULT_TOP_NETWORK_NAME="lickbot_net_9952_loss0_0111_09262018_top.h5"
DEFAULT_BOT_NETWORK_NAME="lickbot_net_9973_loss_0062_10112018_scale3_Bot.h5"
RANDOM_TRAINING_NETWORK_NAME="**RANDOM**"

HTML_DATE_FORMAT='%Y-%m-%d %H:%M:%S'
ROOT_PATH = Path(ROOT).resolve()
NETWORKS_FOLDER = ROOT_PATH / NETWORKS_SUBFOLDER
REMOVED_NETWORKS_FOLDER = NETWORKS_FOLDER / REMOVED_NETWORKS_SUBFOLDER
LOGS_FOLDER = ROOT_PATH / LOGS_SUBFOLDER
STATIC_FOLDER = ROOT_PATH / STATIC_SUBFOLDER
PRIVATE_FOLDER = ROOT_PATH / PRIVATE_SUBFOLDER
REQUIRED_SUBFOLDERS = [NETWORKS_FOLDER, LOGS_FOLDER, STATIC_FOLDER, PRIVATE_FOLDER, REMOVED_NETWORKS_FOLDER]
for reqFolder in REQUIRED_SUBFOLDERS:
    if not reqFolder.exists():
        logger.log(logging.INFO, 'Creating required directory: {reqDir}'.format(reqDir=reqFolder))
        reqFolder.mkdir()

AUTH_FILE = PRIVATE_FOLDER / AUTH_NAME

def validPassword(password):
    valid = re.search('[a-zA-Z]', password) is not None and re.search('[0-9]', password) is not None and len(password) >= 6 and len(password) <= 15
    return valid

def changePassword(user, currentPass, newPass):
    reason = None
    passwordChanged = False
    if USERS[user] == currentPass:
        # Authentication succeeded
        if validPassword(newPass):
            USERS[user] = newPass
            userData = {U:(USERS[U], USER_LVLS[U]) for U in USERS}
            with open(AUTH_FILE, 'w') as f:
                f.write(json.dumps(userData))
                passwordChanged = True
        else:
            reason = "Password must be 6 - 15 characters with at least one number and one letter"
    else:
        reason = "Current password incorrect"
    return passwordChanged, reason

def loadAuth():
    try:
        if not AUTH_FILE.is_file():
            # No auth file found - create default one
            logger.log(logging.WARNING, "No authentication file found - creating a default auth file at {f}. It is recommended to change the default passwords.".format(f=AUTH_FILE))
            with open(AUTH_FILE, 'w') as f:
                f.write(json.dumps(DEFAULT_AUTH))
        with open(AUTH_FILE, 'r') as f:
            userData = json.loads(f.read())
    except:
        logger.log(logging.ERROR, "Error loading authentication file")
        sys.exit()
    users = dict((user, userData[user][0]) for user in userData)
    user_lvls = dict((user, userData[user][1]) for user in userData)
    logger.log(logging.INFO, "Authentication reloaded.")
    return users, user_lvls

BASE_USER='user'
ADMIN_USER='admin'
DEFAULT_PASSWORD = 'password'
# DEFAULT_AUTH is used to create a default auth file if none is found
DEFAULT_AUTH = {BASE_USER:[DEFAULT_PASSWORD, 0], ADMIN_USER:[DEFAULT_PASSWORD, 2]}

USERS, USER_LVLS = loadAuth()

# How often monitoring pages auto-reload, in ms
AUTO_RELOAD_INTERVAL=5000

def isWriteAuthorized(user, owner):
    # Check if user is authorized to modify/terminate owner's job
    userLvl = USER_LVLS[user]
    ownerLvl = USER_LVLS[owner]
    return (user == owner) or (userLvl > ownerLvl)
def isAdmin(user):
    # Check if user has at least lvl 2 privileges
    return USER_LVLS[user] >= 2

# Set environment variables for authentication
# envVars = dict(os.environ)  # or os.environ.copy()
# try:
#     envVars['WSGI_AUTH_CREDENTIALS']='{UN}:{PW}'.format(UN=USER, PW=PASSWORD)
# finally:
#     os.environ.clear()
#     os.environ.update(envVars)

def reRootDirectory(rootMountPoint, pathStyle, directory):
    #   rootMountPoint - the root of the videoDirs. If videoDirs contains a drive root, replace it.
    #   directories - a list of strings representing directory paths to re-root
    #   pathStyle - the style of the videoDirs paths - either 'windowsStyle' or 'posixStyle'

    reRootedDirectory = []
    if pathStyle == 'windowsStylePaths':
        OSPurePath = PureWindowsPath
    elif pathStyle == 'posixStylePaths':
        OSPurePath = PurePosixPath
    else:
        raise ValueError('Invalid path style: {pathStyle}'.format(pathStyle=pathStyle))

    directoryPath = OSPurePath(directory)
    if directoryPath.parts[0] == directoryPath.anchor:
        # This path includes the root - remove it.
        rootlessDirectoryPathParts = directoryPath.parts[1:]
    else:
        rootlessDirectoryPathParts = directoryPath.parts
    reRootedDirectory = Path(rootMountPoint) / Path(*rootlessDirectoryPathParts)
    return reRootedDirectory

def getVideoList(videoDirs, videoFilter='*'):
    # Generate a set of lists of video Path objects from the given directories using the given path filters
    #   videoDirs - a list of strings representing video directory paths to look in
    #   pathStyle - the style of the videoDirs paths - either 'windowsStyle' or 'posixStyle'
    videoLists = []
    for p in videoDirs:
        videoList = []
        for videoPath in p.iterdir():
            if videoPath.suffix.lower() == ".avi":
                if videoPath.match(videoFilter):
                    videoList.append(videoPath)
        videoLists.append(videoList)
    return videoLists

def getUsername(environ):
    request = Request(environ)
    auth = request.authorization
    if auth and auth[0] == 'Basic':
        credentials = b64decode(auth[1]).decode('UTF-8')
        username, password = credentials.split(':', 1)
    return username

def addMessage(environ, message):
    environ["segserver.message"] = message

def makeNaturalLanguageList(strings):
    NLList = ""
    if len(strings) == 1:
        return strings[0]
    for k, string in enumerate(strings):
        if k < len(strings)-1:
            NLList += "{string}, ".format(string=string)
        else:
            NLList += "and {string}".format(string=string)
    return NLList

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
            r = requests.get(self.fullURL, auth=(BASE_USER, USERS[BASE_USER]))
            # # use the opener to fetch a URL
            # self.opener.open(self.fullURL)
            # urllib.request.urlopen(self.fullURL)
            time.sleep(self.interval)

class SegmentationServer:
    newJobNum = itertools.count().__next__   # Source of this clever little idea: https://stackoverflow.com/a/1045724/1460057
    def __init__(self, port=80, webRoot='.'):
        self.port = port
        self.routes = [
            ('/static/*',                   self.staticHandler),
            ('/finalizeSegmentationJob',    self.finalizeSegmentationJobHandler),
            ('/finalizeTrainJob',           self.finalizeTrainJobHandler),
            ('/confirmJob/*',               self.confirmJobHandler),
            ('/checkProgress/*',            self.checkProgressHandler),
            ('/updateQueue',                self.updateJobQueueHandler),
            ('/cancelJob/*',                self.cancelJobHandler),
            ('/serverManagement',           self.serverManagementHandler),
            ('/restartServer',              self.restartServerHandler),
            ('/maskPreview/*',              self.getMaskPreviewHandler),
            ('/myJobs',                     self.myJobsHandler),
            ('/help',                       self.helpHandler),
            ('/changePassword',             self.changePasswordHandler),
            ('/finalizePassword',           self.finalizePasswordHandler),
            ('/reloadAuth',                 self.reloadAuthHandler),
            ('/train',                      self.trainHandler),
            ('/networkManagement',          self.networkManagementHandler),
            ('/networkManagement/rename/*', self.networkRenameHandler),
            ('/networkManagement/remove/*', self.networkRemoveHandler),
            ('/',                           self.rootHandler)
        ]
        self.webRootPath = Path(webRoot).resolve()
        self.maxActiveJobs = 1          # Maximum # of jobs allowed to be running at once
        self.jobQueue = odict()         # List of job parameters for waiting jobs

        self.startTime = dt.datetime.now()

        self.cleanupTime = 86400        # Number of seconds to wait before deleting finished/dead jobs

        self.basic_auth_app = None

        # Start daemon that periodically makes http request that prompts server to update its job queue
        self.updaterDaemon = UpdaterDaemon(interval=3, port=self.port)
        self.updaterDaemon.start()

    def __call__(self, environ, start_fn):
        # Handle routing
        for path, handler in self.routes:
            if fnmatch.fnmatch(environ['PATH_INFO'], path):
                logger.log(logging.DEBUG, 'Matched url {path} to route {route} with handler {handler}'.format(path=environ['PATH_INFO'], route=path, handler=handler))
                return handler(environ, start_fn)
        return self.invalidHandler(environ, start_fn)

    def linkBasicAuth(self, basic_auth_app):
        # Link auth app to allow for dynamic changes in authentication
        self.basic_auth_app = basic_auth_app

    def reloadPasswords(self):
        USERS, USER_LVLS = loadAuth()
        self.basic_auth_app._users = USERS

    def formatHTML(self, environ, templateFilename, **parameters):
        # Check to see if we should be putting up an alert
        if 'segserver.message' in environ:
            message = environ['segserver.message']
        else:
            message = ''

        with open('html/NavBar.html', 'r') as f:
            navBarHTML = f.read()
            jobsRemaining = self.countJobsRemaining()
            videosRemaining = self.countVideosRemaining()
            if jobsRemaining > 0 and videosRemaining > 0:
                serverStatus = "Status: {jobsRemaining} jobs, {videosRemaining} videos".format(jobsRemaining=jobsRemaining, videosRemaining=videosRemaining)
            else:
                serverStatus = "Status: idle"
            user = getUsername(environ)
            navBarHTML = navBarHTML.format(user=user, serverStatus=serverStatus)
        with open('html/HeadLinks.html', 'r') as f:
            headLinksHTML = f.read()
            headLinksHTML = headLinksHTML.format(message=message)

        with open(templateFilename, 'r') as f:
            htmlTemplate = f.read()
            html = htmlTemplate.format(
                navBarHTML=navBarHTML,
                headLinksHTML=headLinksHTML,
                **parameters
            )
        return [html.encode('utf-8')]

    def formatError(self, environ, errorTitle='Error', errorMsg='Unknown error!', linkURL='/', linkAction='retry job creation'):
        return self.formatHTML(environ, 'html/Error.html', errorTitle=errorTitle, errorMsg=errorMsg, linkURL=linkURL, linkAction=linkAction)

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

    def removeNeuralNetwork(self, name):
        # Move a neural network file out of the main networks directory into
        #   the "deleted" subdirectory so they won't be listed any more.
        oldPath = NETWORKS_FOLDER / name
        newPath = REMOVED_NETWORKS_FOLDER / name
        self.checkNetworkPathSafety(oldPath)
        self.checkNetworkPathSafety(newPath)
        if oldPath.is_file():
            logger.log(logging.INFO, 'Removing net by moving it from \"{oldPath}\" to \"{newPath}\"'.format(oldPath=oldPath, newPath=newPath))
            if newPath.is_file():
                # Deleted network already exists in deleted folder
                os.remove(newPath)
            os.rename(oldPath, newPath)
        else:
            raise FileNotFoundError('Could not find file {f}'.format(name))

    def checkNetworkPathSafety(self, path):
        if (not NETWORKS_FOLDER in path.parents) and (not NETWORKS_FOLDER in path.resolve().parents):
            # Ensure both paths are children of NETWORKS_FOLDER, to prevent hijinks
            raise OSError('Disallowed network path accessed: {p} - permission denied.'.format(p=path))

    def renameNeuralNetwork(self, oldName, newName, overwrite=False):
        # Rename a neural network within the neural networks directory.
        oldPath = NETWORKS_FOLDER / oldName
        newPath = NETWORKS_FOLDER / newName
        self.checkNetworkPathSafety(oldPath)
        self.checkNetworkPathSafety(newPath)
        if not oldPath.is_file():
            return False, 'Not a file'
        if newPath.is_file():
            if not overwrite:
                return False, 'New name already exists'
        else:
            logger.log(logging.INFO, 'Renaming net \"{oldPath}\" to \"{newPath}\"'.format(oldPath=oldPath, newPath=newPath))
            os.rename(oldPath, newPath)
            return True, None

    def getNeuralNetworkList(self, namesOnly=False, includeTimestamps=False):
        # Generate a list of available neural networks
        p = Path('.') / NETWORKS_SUBFOLDER
        networks = []
        timestamps = []
        for item in p.iterdir():
            if item.suffix in NEURAL_NETWORK_EXTENSIONS:
                # This is a neural network file
                if namesOnly:
                    networks.append(item.name)
                else:
                    networks.append(item)
                if includeTimestamps:
                    ts = item.stat().st_mtime
                    t = dt.datetime.fromtimestamp(ts)
                    timestamps.append(t)
        if includeTimestamps:
            return networks, timestamps
        else:
            return networks

    def createBulletedList(d):
        # Create an HTML string representing a bulleted list from a dictionary
        html = '<ul>\n{items}\n</ul>'.format(
            items = '\n'.join(['<li>{key}: {val}</li>'.format(key=key, val=d[key]) for key in d])
        )
        return html

    def createOptionList(self, optionValues, defaultValue=None, optionNames=None):
        # Create an HTML string representing a set of drop-down list options
        if optionNames is None:
            optionNames = optionValues
        options = []
        for optionValue, optionName in zip(optionValues, optionNames):
            if optionValue == defaultValue:
                selected = "selected"
            else:
                selected = ""
            options.append('<option value="{v}" {s}>{n}</option>'.format(v=optionValue, n=optionName, s=selected))
        optionText = "\n".join(options)
        return optionText

    def staticHandler(self, environ, start_fn):
        URLparts = environ['PATH_INFO'].split('/')
        requestedStaticFileRelativePath = environ['PATH_INFO'].strip('/')

        if len(URLparts) < 2:
            logger.log(logging.ERROR, 'Could not find that static file: {p}'.format(p=requestedStaticFilePath))
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            with open('html/Error.html', 'r') as f: htmlTemplate = f.read()
            return [htmlTemplate.format(
                errorTitle='Static file not found',
                errorMsg='Static file {name} not found'.format(name=requestedStaticFileRelativePath),
                linkURL='/',
                linkAction='return to job creation page'
                ).encode('utf-8')]
        else:
            subfolder = environ['PATH_INFO'].split('/')[-2]

        logger.log(logging.DEBUG, 'Serving static file: {path}'.format(path=requestedStaticFileRelativePath))
        requestedStaticFilePath = self.webRootPath / requestedStaticFileRelativePath
        if requestedStaticFilePath.exists():
            logger.log(logging.DEBUG, 'Found that static file')
            if subfolder == "css":
                start_fn('200 OK', [('Content-Type', 'text/css')])
                with requestedStaticFilePath.open('r') as f:
                    return [f.read().encode('utf-8')]
            elif subfolder == "favicon":
                start_fn('200 OK', [('Content-Type', "image/x-icon")])
                with requestedStaticFilePath.open('rb') as f:
                    return [f.read()]
            elif subfolder == "images":
                type = requestedStaticFilePath.suffix.strip('.').lower()
                if type not in ['png', 'gif', 'bmp', 'jpg', 'jpeg', 'ico', 'tiff']:
                    start_fn('404 Not Found', [('Content-Type', 'text/html')])
                    return self.formatError(
                        environ,
                        errorTitle='Unknown image type',
                        errorMsg='Unknown image type: {type}'.format(type=type),
                        linkURL='/',
                        linkAction='return to job creation page (or use browser back button)'
                        )
                else:
                    # Convert some extensions to mime types
                    if type == 'jpg': type = 'jpeg'
                    if type in ['ico', 'cur']: type = 'x-icon'
                    if type == 'svg': type = 'svg+xml'
                    if type == 'tif': type = 'tiff'
                    start_fn('200 OK', [('Content-Type', "image/{type}".format(type=type))])
                    with requestedStaticFilePath.open('rb') as f:
                        return [f.read()]
        else:
            logger.log(logging.ERROR, 'Could not find that static file: {p}'.format(p=requestedStaticFilePath))
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            with open('html/Error.html', 'r') as f: htmlTemplate = f.read()
            return [htmlTemplate.format(
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
            if not self.isComplete(jobNum):
                if not self.isStarted(jobNum):
                    queuedJobsAhead += 1
                else:
                    activeJobsAhead += 1
        jobsAhead = queuedJobsAhead + activeJobsAhead
        return jobsAhead

    def countVideosRemaining(self, beforeJobNum=None):
        completedVideosAhead = 0
        queuedVideosAhead = 0
        for jobNum in self.jobQueue:
            if self.jobQueue[jobNum]['jobType'] == SEGMENT_TYPE:
                # Job is a segmentation job, not a training job.
                if beforeJobNum is not None and jobNum == beforeJobNum:
                    # This is the specified job num - stop, don't count any more
                    break
                if self.jobQueue[jobNum]['completionTime'] is None:
                    completedVideosAhead += len(self.jobQueue[jobNum]['completedVideoList'])
                    queuedVideosAhead += len(self.jobQueue[jobNum]['videoList'])
        videosAhead = queuedVideosAhead - completedVideosAhead
        return videosAhead

    def countEpochsRemaining(self, beforeJobNum=None):
        epochsAhead = 0
        for jobNum in self.jobQueue:
            if self.jobQueue[jobNum]['jobType'] == TRAIN_TYPE:
                # Job is a training job, not a segmentation job.
                if beforeJobNum is not None and jobNum == beforeJobNum:
                    # This is the specified job num - stop, don't count any more
                    break
                if self.jobQueue[jobNum]['completionTime'] is None:
                    epochsAhead += self.jobQueue[jobNum]['numEpochs'] - self.jobQueue[jobNum]['lastEpochNum']
        return epochsAhead

    def finalizeSegmentationJobHandler(self, environ, start_fn):
        # Display page showing what job will be, and offering opportunity to go ahead or cancel
        postDataRaw = environ['wsgi.input'].read().decode('utf-8')
        postData = urllib.parse.parse_qs(postDataRaw, keep_blank_values=False)

        try:
            rootMountPoint = postData['rootMountPoint'][0]
            videoDirs = postData['videoSearchDirs'][0].strip().splitlines()
            videoFilter = postData['videoFilter'][0]
            maskSaveDirs = postData['maskSaveDirs'][0].strip().splitlines()
            pathStyle = postData['pathStyle'][0]
            topNetworkName = postData['topNetworkName'][0]
            botNetworkName = postData['botNetworkName'][0]
            topNetworkPath = NETWORKS_FOLDER / topNetworkName
            botNetworkPath = NETWORKS_FOLDER / botNetworkName
            binaryThreshold = float(postData['binaryThreshold'][0])
            topOffset = int(postData['topOffset'][0])
            if 'topHeight' not in postData or len(postData['topHeight'][0]) == 0:
                topHeight = None
            else:
                topHeight = int(postData['topHeight'][0])
            if 'botHeight' not in postData or len(postData['botHeight'][0]) == 0:
                botHeight = None
            else:
                botHeight = int(postData['botHeight'][0])

            if 'topWidth' not in postData or len(postData['topWidth'][0]) == 0:
                topWidth = None
            else:
                topWidth = int(postData['topWidth'][0])
            if 'botWidth' not in postData or len(postData['botWidth'][0]) == 0:
                botWidth = None
            else:
                botWidth = int(postData['botWidth'][0])

            if 'generatePreview' in postData:
                generatePreview = True
            else:
                generatePreview = False
            if 'skipExisting' in postData:
                skipExisting = True
            else:
                skipExisting = False
            jobName = postData['jobName'][0]

        except KeyError:
            # Missing one of the postData arguments
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Missing parameter',
                errorMsg='A required field is missing. Please retry with all required fields filled in.',
                linkURL='/',
                linkAction='return to job creation page (or use browser back button)'
                )

        if len(maskSaveDirs) != len(videoDirs):
            # Different # of mask and video dirs
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Non-corresponding video and mask directories',
                errorMsg='A different number of mask directories than video directories was provided. Please make sure there is a 1:1 correspondence between video and mask directories.',
                linkURL='/',
                linkAction='return to job creation page (or use browser back button)'
                )

        segSpec = SegSpec(
            partNames=['Bot', 'Top'],
            heights=[botHeight, topHeight],
            widths=[botWidth, topWidth],
            yOffsets=[0, topOffset],
            offsetAnchors=[SegSpec.SW, SegSpec.NW],
            neuralNetworkPaths=[botNetworkPath, topNetworkPath]
        )
        # Re-root directories
        reRootedVideoDirs = [reRootDirectory(rootMountPoint, pathStyle, videoDir) for videoDir in videoDirs]
        reRootedMaskSaveDirs = [reRootDirectory(rootMountPoint, pathStyle, maskSaveDir) for maskSaveDir in maskSaveDirs]

        # Check if all parameters are valid. If not, display error and offer to go back
        valid = True
        errorMessages = []
        for maskSaveDir in reRootedMaskSaveDirs:
            if not maskSaveDir.exists():
                valid = False
                errorMessages.append('Mask save directory not found: {maskSaveDirectory}. Hint: Did you pick the right root?'.format(maskSaveDirectory=maskSaveDir))
        for videoDir in reRootedVideoDirs:
            if not videoDir.exists():
                valid = False
                errorMessages.append('Video directory not found: {videoDir}'.format(videoDir=videoDir))
        # keys = ['rootMountPoint', 'videoSearchDirs', 'videoFilter', 'maskSaveDirectory', 'pathStyle', 'topNetworkName', 'botNetworkName', 'topOffset', 'topHeight', 'botHeight', 'binaryThreshold', 'jobName']
        # missingKeys = [key for key in keys if key not in postData]
        # if len(missingKeys) > 0:
        #     # Not all form parameters got POSTed
        #     valid = False
        #     errorMessages.append('Job creation parameters missing: {params}'.format(params=', '.join(missingKeys)))

        if not valid:
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Invalid job parameter',
                errorMsg="<br/>".join(errorMessages),
                linkURL='/',
                linkAction='return to job creation page (or use browser back button)'
                )

        # Generate list of lists of videos (one list per videoDir)
        videoLists = getVideoList(reRootedVideoDirs, videoFilter=videoFilter)

        # Error out if no videos are found
        if len(videoLists) == 0 or all([len(videoList) == 0 for videoList in videoLists]):
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='No videos found',
                errorMsg="No videos found with the given video root and filters. Please adjust the parameters, or upload videos, then try again.",
                linkURL='/',
                linkAction='return to job creation page (or use browser back button)'
                )

        jobNums = []

        # Loop over (usually one, potentially multiple) pairs of videoLists and mask save dirs, create one job for each.
        for videoDir, videoList, maskDir in zip(reRootedVideoDirs, videoLists, reRootedMaskSaveDirs):
            # Add job parameters to queue
            jobNum = SegmentationServer.newJobNum()
            jobNums.append(jobNum)

            # Prepare to record job parameters for posterity
            paramRecord = {
                "rootMountPoint":rootMountPoint,
                "videoDir":str(videoDir),
                "videoFilter":videoFilter,
                "maskDir":str(maskDir),
                "pathStyle":pathStyle,
                "topNetworkName":topNetworkName,
                "botNetworkName":botNetworkName,
                "binaryThreshold":binaryThreshold,
                "topOffset":topOffset,
                "topHeight":topHeight,
                "topHeight":topHeight,
                "botHeight":botHeight,
                "topWidth":topWidth,
                "botWidth":botWidth,
                "botWidth":botWidth,
                "generatePreview":generatePreview,
                "skipExisting":skipExisting,
                "jobName":jobName
            }

            self.jobQueue[jobNum] = dict(
                job=None,                               # Job process object
                jobName=jobName,                        # Name/description of job
                jobType=SEGMENT_TYPE,                   # Type of job (segment or train)
                jobClass=SegmentationJob,               # Reference to job class
                owner=getUsername(environ),             # Owner of job, has special privileges
                jobNum=jobNum,                          # Job ID
                confirmed=False,                        # Has user confirmed params yet
                cancelled=False,                        # Has the user cancelled this job?
                videoList=videoList,                    # List of video paths to process
                maskSaveDirectory=maskDir,              # Path to save masks
                segSpec=segSpec,                        # segSpec
                generatePreview=generatePreview,        # Should we generate gif previews of masks?
                skipExisting=skipExisting,              # Should we skip generating masks that already exist?
                binaryThreshold=binaryThreshold,        # Threshold to use to change grayscale masks to binary
                completedVideoList=[],                  # List of processed videos
                times=[],                               # List of video processing start times
                creationTime=time.time_ns(),            # Time job was created
                startTime=None,                         # Time job was started
                completionTime=None,                    # Time job was completed
                log=[],                                 # List of log output from job
                exitCode=ServerJob.INCOMPLETE,          # Job exit code
                paramRecord=paramRecord                 # Record of relevant parameters for posterity
            )

        jobsAhead = self.countJobsRemaining(beforeJobNum=jobNums[0])
        videosAhead = self.countVideosRemaining(beforeJobNum=jobNums[0])
        epochsAhead = self.countEpochsRemaining(beforeJobNum=jobNums[0])

        if len(videoLists) == 1:
            # Just one video list
            videoListText = "\n".join(["<li>{v}</li>".format(v=v) for v in videoLists[0]])
        else:
            videoListTexts = []
            for k, videoList in enumerate(videoLists):
                subListText = "\n".join(["\t<li>{v}</li>".format(v=v) for v in videoList])
                subListText = "<li>Job {jobID}:\n<ul>\n{subListText}\n</ul></li>".format(subListText=subListText, jobID=jobNums[k])
                videoListTexts.append(subListText)
            videoListText = "\n".join(videoListTexts)

        if topHeight is None:
            topHeightText = "Use network size"
        else:
            topHeightText = str(topHeight)
        if topWidth is None:
            topWidthText = "Use network size"
        else:
            topWidthText = str(topWidth)
        if botHeight is None:
            botHeightText = "Use network size"
        else:
            botHeightText = str(botHeight)
        if botWidth is None:
            botWidthText = "Use network size"
        else:
            botWidthText = str(botWidth)

        if len(jobNums) <= 1:
            jobIDText = "job (job ID {jobID})".format(jobID=jobNums[0])
        else:
            jobIDText = "jobs (job IDs {jobIDs})".format(jobIDs=makeNaturalLanguageList(jobNums))

        jobIDQueryString = '/'.join([str(jobNum) for jobNum in jobNums])

        start_fn('200 OK', [('Content-Type', 'text/html')])
        return self.formatHTML(
            environ,
            'html/FinalizeSegmentationJob.html',
            videoList=videoListText,
            topNetworkName=topNetworkPath.name,
            botNetworkName=botNetworkPath.name,
            binaryThreshold=binaryThreshold,
            topOffset=topOffset,
            topHeight=topHeightText,
            topWidth=topWidthText,
            botHeight=botHeightText,
            botWidth=botWidthText,
            generatePreview=generatePreview,
            skipExisting=skipExisting,
            jobIDText=jobIDText,
            jobIDQueryString=jobIDQueryString,
            jobName=jobName,
            jobsAhead=jobsAhead,
            videosAhead=videosAhead,
            epochsAhead=epochsAhead
        )

    def finalizeTrainJobHandler(self, environ, start_fn):
        # Display page showing what job will be, and offering opportunity to go ahead or cancel
        postDataRaw = environ['wsgi.input'].read().decode('utf-8')
        postData = urllib.parse.parse_qs(postDataRaw, keep_blank_values=False)

        try:
            rootMountPoint = postData['rootMountPoint'][0]
            startNetworkName = postData['startNetworkName'][0]
            if startNetworkName == RANDOM_TRAINING_NETWORK_NAME:
                startNetworkPath = None
            else:
                startNetworkPath = NETWORKS_FOLDER / startNetworkName
            newNetworkName = postData['newNetworkName'][0]
            newNetworkPath = NETWORKS_FOLDER / newNetworkName
            if newNetworkPath.is_file():
                # Chosen network name is already taken
                start_fn('500 Internal Server Error', [('Content-Type', 'text/html')])
                return self.formatError(
                    environ,
                    errorTitle='Network name already in use',
                    errorMsg='A network by the name \"{newName}\" already exists. Please choose a different name, or rename/remove the preexisting network first.'.format(newName=newNetworkName),
                    linkURL='/networkManagement',
                    linkAction='go to network management, or use your browser\'s back button to go back to your new train job page.'
                )
            trainingDataPath = Path(postData['trainingDataPath'][0])
            pathStyle = postData['pathStyle'][0]
            batchSize = int(postData['batchSize'][0])
            numEpochs = int(postData['numEpochs'][0])
            if 'augmentData' in postData:
                augmentData = True
                rotationRange = float(postData['rotationRange'][0])
                widthShiftRange = float(postData['widthShiftRange'][0])
                heightShiftRange = float(postData['heightShiftRange'][0])
                zoomRange = float(postData['zoomRange'][0])
                if 'horizontalFlip' in postData:
                    horizontalFlip = True
                else:
                    horizontalFlip = False
                if 'verticalFlip' in postData:
                    verticalFlip = True
                else:
                    verticalFlip = False
            else:
                augmentData = False
                rotationRange = None
                widthShiftRange = None
                heightShiftRange = None
                zoomRange = None
                horizontalFlip = None
                verticalFlip = None

            if 'generateValidationPreview' in postData:
                generateValidationPreview = True
            else:
                generateValidationPreview = False

            jobName = postData['jobName'][0]

            # Prepare to record job parameters for posterity
            paramRecord = {
                "rootMountPoint":rootMountPoint,
                "startNetworkName":startNetworkName,
                "newNetworkName":newNetworkName,
                "trainingDataPath":str(trainingDataPath),
                "pathStyle":pathStyle,
                "batchSize":batchSize,
                "numEpochs":numEpochs,
                "augmentData":augmentData,
                "rotationRange":rotationRange,
                "widthShiftRange":widthShiftRange,
                "heightShiftRange":heightShiftRange,
                "zoomRange":zoomRange,
                "horizontalFlip":horizontalFlip,
                "verticalFlip":verticalFlip,
                "generateValidationPreview":generateValidationPreview,
                "jobName":jobName
            }
        except KeyError:
            # Missing one of the postData arguments
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Missing parameter',
                errorMsg='A required field is missing. Please retry with all required fields filled in.',
                linkURL='/',
                linkAction='return to job creation page (or use browser back button)'
                )

        augmentationParameters = createDataAugmentationParameters(
            rotation_range=rotationRange,
            width_shift_range=widthShiftRange,
            height_shift_range=heightShiftRange,
            zoom_range=zoomRange,
            horizontal_flip=horizontalFlip,
            vertical_flip=verticalFlip
        )

        # Re-root training data path
        reRootedTrainingDataPath = reRootDirectory(rootMountPoint, pathStyle, trainingDataPath)

        # Check if all parameters are valid. If not, display error and offer to go back
        valid = True
        errorMessages = []
        if not (trainingDataPath.suffix.lower() == ".mat"):
            valid = False
            errorMessages.append('Training data file provided must be a .mat file. Instead, you provided {trainingDataPath}.'.format(trainingDataPath=trainingDataPath))
        if not reRootedTrainingDataPath.is_file():
            valid = False
            errorMessages.append('Training data file not found: {reRootedTrainingDataPath}. Hint: Did you pick the right root?'.format(reRootedTrainingDataPath=reRootedTrainingDataPath))
        if startNetworkPath is not None and not startNetworkPath.is_file():
            valid = False
            errorMessages.append('Starting network file not found: {startNetworkPath}'.format(startNetworkPath=startNetworkPath))
        if newNetworkPath.suffix.lower() not in NEURAL_NETWORK_EXTENSIONS:
            valid = False
            errorMessages.append('Invalid network file extension. Please use one of the following file extensions to name your network: {e}'.format(e=', '.join(NEURAL_NETWORK_EXTENSIONS)))

        if not valid:
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Invalid job parameter',
                errorMsg="<br/>".join(errorMessages),
                linkURL='/',
                linkAction='return to job creation page (or use browser back button)'
                )

        # Add job parameters to queue
        jobNum = SegmentationServer.newJobNum()

        jobsAhead = self.countJobsRemaining(beforeJobNum=jobNum)
        videosAhead = self.countVideosRemaining(beforeJobNum=jobNum)
        epochsAhead = self.countEpochsRemaining(beforeJobNum=jobNum)

        self.jobQueue[jobNum] = dict(
            job=None,                                       # Job process object
            jobName=jobName,                                # Name/description of job
            jobType=TRAIN_TYPE,                             # Type of job (segment or train)
            jobClass=TrainJob,                              # Reference to job class
            owner=getUsername(environ),                     # Owner of job, has special privileges
            jobNum=jobNum,                                  # Job ID
            confirmed=False,                                # Has user confirmed params yet
            cancelled=False,                                # Has the user cancelled this job?
            startNetworkPath=startNetworkPath,              # Either path to an existing network to train, or None
            newNetworkPath=newNetworkPath,                  # Path to save the newly trained network to
            batchSize=batchSize,                            # Size of a training batch
            numEpochs=numEpochs,                            # Number of epochs in the training session
            augmentData=augmentData,                        # Should training dataset be randomly augmented?
            augmentationParameters=augmentationParameters,  # Dictionary of augmentation parameters
            trainingDataPath=reRootedTrainingDataPath,      # Path to the .mat file containing the training data
            generatePreview=generateValidationPreview,      # Should we generate validation results between each epoch?
            lastEpochNum=0,                                 # Last completed epoch number
            loss=[],                                        # List of the calculated loss after each epoch
            accuracy=[],                                    # List of the calculated accuracy after each epoch
            times=[],                                       # List of epoch completion times
            creationTime=time.time_ns(),                    # Time job was created
            startTime=None,                                 # Time job was started
            completionTime=None,                            # Time job was completed
            log=[],                                         # List of log output from job
            exitCode=ServerJob.INCOMPLETE,                  # Job exit code
            paramRecord=paramRecord                         # Record of relevant parameters for posterity
        )

        if startNetworkPath is None:
            startNetworkNameText = "Randomized (naive) network"
        else:
            startNetworkNameText = str(startNetworkName)
        if augmentData:
            augmentDataText = 'Yes'
        else:
            augmentDataText = 'No'

        start_fn('200 OK', [('Content-Type', 'text/html')])
        return self.formatHTML(
            environ,
            'html/FinalizeTrainJob.html',
            startNetworkName=startNetworkNameText,
            newNetworkName=newNetworkName,
            batchSize=batchSize,
            numEpochs=numEpochs,
            augmentData=augmentDataText,
            rotationRange=augmentationParameters['rotation_range'],
            widthShiftRange=augmentationParameters['width_shift_range'],
            heightShiftRange=augmentationParameters['height_shift_range'],
            zoomRange=augmentationParameters['zoom_range'],
            horizontalFlip=augmentationParameters['horizontal_flip'],
            verticalFlip=augmentationParameters['vertical_flip'],
            generatePreview=generateValidationPreview,
            jobID=jobNum,
            jobName=jobName,
            jobsAhead=jobsAhead,
            videosAhead=videosAhead,
            epochsAhead=epochsAhead
        )

    def startJob(self, jobNum):
        # Instantiate new job object of the appropriate class
        self.jobQueue[jobNum]['job'] = self.jobQueue[jobNum]['jobClass'](
            verbose = 1,
            logger=logger,
            **self.jobQueue[jobNum]
            )

        logger.log(logging.INFO, 'Starting job {jobNum}'.format(jobNum=jobNum))
        self.jobQueue[jobNum]['job'].start()
        self.jobQueue[jobNum]['job'].msgQueue.put((ServerJob.START, None))
        self.jobQueue[jobNum]['job'].msgQueue.put((ServerJob.PROCESS, None))
        self.jobQueue[jobNum]['startTime'] = time.time_ns()

    def isConfirmed(self, jobNum):
        return self.jobQueue[jobNum]['confirmed']
    def isCancelled(self, jobNum):
        return self.jobQueue[jobNum]['cancelled']
    def isStarted(self, jobNum):
        return (self.jobQueue[jobNum]['job'] is not None) and (self.jobQueue[jobNum]['startTime'] is not None)
    def isActive(self, jobNum):
        return (self.isStarted(jobNum)) and (not self.isComplete(jobNum))
    def isComplete(self, jobNum):
        return (self.jobQueue[jobNum]['exitCode'] != ServerJob.INCOMPLETE) or (self.jobQueue[jobNum]['completionTime'] is not None)
    def isSucceeded(self, jobNum):
        return (self.jobQueue[jobNum]['exitCode'] == ServerJob.SUCCEEDED)
    def isFailed(self, jobNum):
        return (self.jobQueue[jobNum]['exitCode'] == ServerJob.FAILED)
    def isOwnedBy(self, jobNum, owner):
        return (self.jobQueue[jobNum]['owner'] == owner)
    def isEnqueued(self, jobNum):
        return self.isConfirmed(jobNum) and (not self.isStarted(jobNum)) and (not self.isCancelled(jobNum)) and (not self.isComplete(jobNum))

    def getJobNums(self, confirmed=None, started=None, active=None, completed=None, owner=None, succeeded=None, failed=None, cancelled=None):
        # For each filter argument, "None" means do not filter
        jobNums = []
        for jobNum in self.jobQueue:
            job = self.jobQueue[jobNum]
            # logger.log(logging.INFO, "Job {jobNum} checking for inclusion...".format(jobNum=jobNum))
            if   (owner is not None) and (not self.isOwnedBy(jobNum, owner)):
                # logger.log(logging.INFO, "Job {jobNum} rejected by owned filter".format(jobNum=jobNum))
                continue
            elif (confirmed is not None) and (confirmed != self.isConfirmed(jobNum)):
                # logger.log(logging.INFO, "Job {jobNum} rejected by confirmed filter".format(jobNum=jobNum))
                continue
            elif (active is not None) and (active != self.isActive(jobNum)):
                # logger.log(logging.INFO, "Job {jobNum} rejected by active filter".format(jobNum=jobNum))
                continue
            elif (completed is not None) and (completed != self.isComplete(jobNum)):
                # logger.log(logging.INFO, "Job {jobNum} rejected by completed filter".format(jobNum=jobNum))
                continue
            elif (succeeded is not None) and (succeeded != self.isSucceeded(jobNum)):
                # logger.log(logging.INFO, "Job {jobNum} rejected by succeeded filter".format(jobNum=jobNum))
                continue
            elif (failed is not None) and (failed != self.isFailed(jobNum)):
                # logger.log(logging.INFO, "Job {jobNum} rejected by failed filter".format(jobNum=jobNum))
                continue
            elif (started is not None) and (started != self.isStarted(jobNum)):
                # logger.log(logging.INFO, "Job {jobNum} rejected by started filter".format(jobNum=jobNum))
                continue
            # logger.log(logging.INFO, "Job {jobNum} accepted".format(jobNum=jobNum))
            jobNums.append(jobNum)
        return jobNums

    def confirmJobHandler(self, environ, start_fn):
        # On either finalize job page, if the user clicks "confirm", the request
        #   goes to this handler, which changes the "confirmed" state of the job
        #   to True.

        # Get jobNums from URL
        jobNums = [int(jobNum) for jobNum in environ['PATH_INFO'].split('/')[2:]]

        invalidJobNums = []
        unauthorizedJobNums = []

        # Loop over each job, check if it is valid, then confirm it if not.
        for jobNum in jobNums:
            if jobNum not in self.getJobNums(active=False, completed=False):
                # Invalid jobNum
                invalidJobNums.append(jobNum)
                continue

            # Check if user is authorized to start job
            if not isWriteAuthorized(getUsername(environ), self.jobQueue[jobNum]['owner']):
                # User is not authorized
                unauthorizedJobNums.append(jobNum)
                continue

            # Job is valid and enqueued - set confirmed flag to True, so they can be started when at the front of the queue
            self.jobQueue[jobNum]['confirmed'] = True

        # Check if there were any invalid jobs
        if len(invalidJobNums) > 0:
            invalidJobNumsText = makeNaturalLanguageList(invalidJobNums)
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Invalid job ID',
                errorMsg='Invalid job ID: {jobIDs}'.format(jobIDs=invalidJobNums),
                linkURL='/',
                linkAction='recreate job'
                )

        # Check if there were any unauthorized jobs
        if len(unauthorizedJobNums) > 0:
            # At least one unauthorized job
            return self.unauthorizedHandler(environ, start_fn)

        # All jobs were enqueued hunky dory
        if len(jobNums) > 1:
            # If we're starting more than one job, send user to myJobs page
            start_fn('303 See Other', [('Location','/myJobs')])
        else:
            # IF there's just one job, send user directly to checkProgress page
            start_fn('303 See Other', [('Location','/checkProgress/{jobID}'.format(jobID=jobNums[0]))])
        return []


    def removeJob(self, jobNum, waitingPeriod=0):
        # waitingPeriod = amount of time in seconds to wait after job completionTime before removing from queue
        expired = False
        if self.jobQueue[jobNum]['confirmed']:
            if self.jobQueue[jobNum]['completionTime'] is not None:
                if (time.time_ns() - self.jobQueue[jobNum]['completionTime']) / 1000000000 > waitingPeriod:
                    expired = True
        else:
            if self.jobQueue[jobNum]['creationTime'] is not None:
                if (time.time_ns() - self.jobQueue[jobNum]['creationTime']) / 1000000000 > waitingPeriod:
                    expired = True
            else:
                expire = True
                # Creation time should never be None
                raise ValueError('Job creation time should never be None')

        if expired:
            # Delete expired job
            logger.log(logging.INFO, 'Removing job {jobNum}'.format(jobNum=jobNum))
            del self.jobQueue[jobNum]

    def updateJobQueueHandler(self, environ, start_fn):
        # Handler for automated calls to update the queue
        self.updateJobQueue()

        # logger.log(logging.INFO, 'Got automated queue update reminder')
        #
        start_fn('200 OK', [('Content-Type', 'text/html')])
        return []

    def updateJobQueue(self):
        # Remove stale unconfirmed jobs:
#        logger.log(logging.INFO, "getJobNums(confirmed=False) - removing unconfirmed jobs")
        for jobNum in self.getJobNums(confirmed=False):
            self.removeJob(jobNum, waitingPeriod=self.cleanupTime)
        # Check if the current job is done. If it is, remove it and start the next job
#        logger.log(logging.INFO, "getJobNums(active=True) checking if active job is done")
        for jobNum in self.getJobNums(active=True):
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
                # self.removeJob(jobNum)
                # logger.log(logging.INFO, "Removing job {jobNum} in error state".format(jobNum=jobNum))
            elif jobState == ServerJob.EXITING:
                pass
            elif jobState == ServerJob.DEAD:
                self.jobQueue[jobNum]['completionTime'] = time.time_ns()
                self.removeJob(jobNum, waitingPeriod=self.cleanupTime)
                logger.log(logging.INFO, "Removing job {jobNum} in dead state".format(jobNum=jobNum))
            elif jobState == -1:
                pass

#        logger.log(logging.INFO, "getJobNums(active=True) - checking if room for new job")
        if len(self.getJobNums(active=True)) < self.maxActiveJobs:
            # Start the next job, if any
            # Loop over confirmed, inactive (queued) job nums
#            logger.log(logging.INFO, "getJobNums(active=False, confirmed=True) - looking for job to start")
            for jobNum in self.getJobNums(confirmed=True, started=False, completed=False):
                # This is the next queued confirmed job - start it
                self.startJob(jobNum)
                break;

    def updateJobProgress(self, jobNum):
        # Retrieve and record any progress reports sent by the specified job
        if jobNum in self.jobQueue and self.jobQueue[jobNum]['job'] is not None:
            while True:
                try:
                    progress = self.jobQueue[jobNum]['job'].progressQueue.get(block=False)
                    # Get any new log output from job
                    self.jobQueue[jobNum]['log'].extend(progress['log'])
                    # Get updated exit code from job
                    self.jobQueue[jobNum]['exitCode'] = progress['exitCode']
                    if self.jobQueue[jobNum]['jobType'] == SEGMENT_TYPE:
                        # Get the path to the last video the job has completed
                        if progress['lastCompletedVideoPath'] is not None:
                            self.jobQueue[jobNum]['completedVideoList'].append(progress['lastCompletedVideoPath'])
                        # Get the time when the last video started processing
                        if progress['lastProcessingStartTime'] is not None:
                            self.jobQueue[jobNum]['times'].append(progress['lastProcessingStartTime'])
                    elif self.jobQueue[jobNum]['jobType'] == TRAIN_TYPE:
                        # Get the last completed epoch number
                        if progress['lastEpochNum'] is not None:
                            self.jobQueue[jobNum]['lastEpochNum'] = progress['lastEpochNum']
                        # Get the time when the last epoch completed
                        if progress['lastEpochTime'] is not None:
                            self.jobQueue[jobNum]['times'].append(progress['lastEpochTime'])
                        if progress['loss'] is not None:
                            self.jobQueue[jobNum]['loss'].append(progress['loss'])
                        if progress['accuracy'] is not None:
                            self.jobQueue[jobNum]['accuracy'].append(progress['accuracy'])
                    else:
                        raise ValueError('Unknown type: {t}'.format(t=self.jobQueue[jobNum]['jobType']))
                except queue.Empty:
                    # Got all progress
                    break

    def formatLogHTML(self, log):
        logHTMLList = []
        for logEntry in log:
            logHTMLList.append('<p>{logEntry}</p>'.format(logEntry=logEntry))
        logHTML = "\n".join(logHTMLList)
        return logHTML

    def getMaskPreviewHandler(self, environ, start_fn):
        # Get jobNum from URL
        jobNum = int(environ['PATH_INFO'].split('/')[-2])
        if jobNum not in self.jobQueue:
            # Invalid jobNum
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Invalid job ID',
                errorMsg='Invalid job ID {jobID}'.format(jobID=jobNum),
                linkURL='/',
                linkAction='create a new job'
                )

        maskPart = environ['PATH_INFO'].split('/')[-1].lower()
        if maskPart == "top":
            preview = self.jobQueue[jobNum]['maskSaveDirectory'] / 'Top.gif'
        elif maskPart == "bot":
            preview = self.jobQueue[jobNum]['maskSaveDirectory'] / 'Bot.gif'
        else:
            # Invalid mask part
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Invalid mask part',
                errorMsg='Invalid mask part: {maskPart}'.format(maskPart=maskPart),
                linkURL='/',
                linkAction='create a new job'
                )

        if not preview.exists():
            # Preview mask doesn't exist. Instead, serve a static placeholder gif
            environ['PATH_INFO'] = "/static/images/MaskPreviewPlaceholder.gif"
            return self.staticHandler(environ, start_fn)

        start_fn('200 OK', [('Content-Type', "image/gif")])
        with preview.open('rb') as f:
            return [f.read()]

    def getJobTimeStats(self, jobNum):
        creationTime = ""
        startTime = "Not started yet"
        completionTime = "Not complete yet"
        if self.jobQueue[jobNum]['creationTime'] is not None:
            creationTime = dt.datetime.fromtimestamp(self.jobQueue[jobNum]['creationTime']/1000000000).strftime(HTML_DATE_FORMAT)
        if self.jobQueue[jobNum]['startTime'] is not None:
            startTime = dt.datetime.fromtimestamp(self.jobQueue[jobNum]['startTime']/1000000000).strftime(HTML_DATE_FORMAT)
        if self.jobQueue[jobNum]['completionTime'] is not None:
            completionTime = dt.datetime.fromtimestamp(self.jobQueue[jobNum]['completionTime']/1000000000).strftime(HTML_DATE_FORMAT)

        if len(self.jobQueue[jobNum]['times']) > 1:
            deltaT = np.diff(self.jobQueue[jobNum]['times'])/1000000000
            meanTime = np.mean(deltaT)
            meanTimeStr = "{meanTime:.2f}".format(meanTime=meanTime)
            timeConfInt = np.std(deltaT)*1.96
            timeConfIntStr = "{timeConfInt:.2f}".format(timeConfInt=timeConfInt)
            if self.jobQueue[jobNum]['completionTime'] is None:
                numTasks, numCompletedTasks = self.getJobProgress(jobNum)
                estimatedSecondsRemaining = (numTasks - numCompletedTasks) * meanTime
                days, remainder = divmod(estimatedSecondsRemaining, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                if days > 0:
                    estimatedDaysRemaining = '{days} d, '.format(days=int(days))
                else:
                    estimatedDaysRemaining = ''
                if hours > 0 or days > 0:
                    estimatedHoursRemaining = '{hours} h, '.format(hours=int(hours))
                else:
                    estimatedHoursRemaining = ''
                if minutes > 0 or hours > 0 or days > 0:
                    estimatedMinutesRemaining = '{minutes} m, '.format(minutes=int(minutes))
                else:
                    estimatedMinutesRemaining = ''
                estimatedSecondsRemaining = '{seconds} s'.format(seconds=int(seconds))
                estimatedTimeRemaining = estimatedDaysRemaining + estimatedHoursRemaining + estimatedMinutesRemaining + estimatedSecondsRemaining
            else:
                estimatedTimeRemaining = "None"
        else:
            meanTime = 0
            meanTimeStr = "Unknown"
            timeConfInt = 0
            timeConfIntStr = "Unknown"
            estimatedTimeRemaining = "Unknown"

        return creationTime, startTime, completionTime, meanTime, meanTimeStr, timeConfInt, timeConfIntStr, estimatedTimeRemaining

    def getJobStateText(self, jobNum):
        exitCode = self.jobQueue[jobNum]['exitCode']
        stateDescription = ''
        processDead = "true"
        if exitCode == ServerJob.INCOMPLETE:
            processDead = "false"
            if not self.isStarted(jobNum):
                jobsAhead = self.countJobsRemaining(beforeJobNum=jobNum)
                videosAhead = self.countVideosRemaining(beforeJobNum=jobNum)
                epochsAhead = self.countEpochsRemaining(beforeJobNum=jobNum)
                if self.isCancelled(jobNum):
                    exitCodePhrase = 'has been cancelled.'
                    stateDescription = 'This job has been cancelled.'
                elif self.isConfirmed(jobNum):
                    exitCodePhrase = 'is enqueued, but not started.'
                    stateDescription = '<br/>There are <strong>{jobsAhead} jobs</strong> \
                                        ahead of you with <strong>{videosAhead} total videos</strong> \
                                        and <strong>{epochsAhead} total epochs</strong> \
                                        remaining. Your job will be enqueued to start as soon \
                                        as any/all previous jobs are done.'.format(jobsAhead=jobsAhead, videosAhead=videosAhead, epochsAhead=epochsAhead)
                else:
                    exitCodePhrase = 'has not been confirmed yet. <form action="/confirmJob/{jobID}"><input class="button button-primary" type="submit" value="Confirm and enqueue job" /></form>'.format(jobID=jobNum)
                    stateDescription = '<br/>There are <strong>{jobsAhead} jobs</strong> \
                                        ahead of you with <strong>{videosAhead} total videos</strong> \
                                        and <strong>{epochsAhead} total epochs</strong> \
                                        remaining. Your job will be enqueued after you confirm it.'.format(jobsAhead=jobsAhead, videosAhead=videosAhead, epochsAhead=epochsAhead)
            else:
                if self.isCancelled(jobNum):
                    exitCodePhrase = 'has been cancelled.'
                    stateDescription = 'This job has been cancelled, and will stop after the current task is complete. All existing output will remain in place. Stand by...'
                else:
                    exitCodePhrase = 'is <strong>in progress</strong>!'
        elif self.isSucceeded(jobNum):
            if self.isCancelled(jobNum):
                exitCodePhrase = 'has been <strong>cancelled</strong>.'
            else:
                exitCodePhrase = 'is <strong>complete!</strong>'
        elif self.isFailed(jobNum):
            exitCodePhrase = 'has exited with errors :(  Please see debug output below.'
        else:
            exitCodePhrase = 'is in an unknown exit code state...'

        return processDead, exitCodePhrase, stateDescription

    def checkProgressHandler(self, environ, start_fn):
        # Get jobNum from URL
        jobNum = int(environ['PATH_INFO'].split('/')[-1])
        allJobNums = self.getJobNums()
#        logger.log(logging.INFO, 'jobNum={jobNum}, allJobNums={allJobNums}, jobQueue={jobQueue}'.format(jobNum=jobNum, allJobNums=allJobNums, jobQueue=self.jobQueue))
        if jobNum not in allJobNums:
            # Invalid jobNum
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Invalid job ID',
                errorMsg='Invalid job ID {jobID}'.format(jobID=jobNum),
                linkURL='/',
                linkAction='create a new job'
                )

        jobEntry = self.jobQueue[jobNum]

        if jobEntry['job'] is not None:
            # Job has started. Check its state
            jobState = jobEntry['job'].publishedStateVar.value
            jobStateName = ServerJob.stateList[jobState]
            # Get all pending updates on its progress
            self.updateJobProgress(jobNum)
        else:
            # Job has not started
            jobStateName = "ENQUEUED"

        creationTime, startTime, completionTime, meanTime, meanTimeStr, timeConfInt, timeConfIntStr, estimatedTimeRemaining = self.getJobTimeStats(jobNum)

        processDead, exitCodePhrase, stateDescription = self.getJobStateText(jobNum)

        numTasks, numCompletedTasks = self.getJobProgress(jobNum)
        percentComplete = "{percentComplete:.1f}".format(percentComplete=100*numCompletedTasks/numTasks)

        logHTML = self.formatLogHTML(jobEntry['log'])

        owner = jobEntry['owner']
        if owner == getUsername(environ):
            owner = owner + " (you)"

        generatePreview = jobEntry['generatePreview']
        if not generatePreview:
            hidePreview = "hidden"
        else:
            hidePreview = ""

        if jobEntry['jobType'] == SEGMENT_TYPE:
            # Get some parameters about job ready for display
            completedVideoListHTML = "\n".join(["<li>{v}</li>".format(v=v) for v in jobEntry['completedVideoList']])
            if len(completedVideoListHTML.strip()) == 0:
                completedVieoListHTML = "None"

            binaryThreshold = jobEntry['binaryThreshold']
            maskSaveDirectory = jobEntry['maskSaveDirectory']
            segSpec = jobEntry['segSpec']
            topNetworkName = segSpec.getNetworkPath('Top').name
            botNetworkName = segSpec.getNetworkPath('Bot').name
            topOffset = segSpec.getYOffset('Top')
            topHeight = segSpec.getHeight('Top')
            botHeight = segSpec.getHeight('Bot')
            topWidth = segSpec.getWidth('Top')
            botWidth = segSpec.getWidth('Bot')
            if topHeight is None:
                topHeightText = "Use network size"
            else:
                topHeightText = str(topHeight)
            if botHeight is None:
                botHeightText = "Use network size"
            else:
                botHeightText = str(botHeight)

            if topWidth is None:
                topWidthText = "Use network size"
            else:
                topWidthText = str(topWidth)
            if botWidth is None:
                botWidthText = "Use network size"
            else:
                botWidthText = str(botWidth)

            topMaskPreviewSrc = '/maskPreview/{jobNum}/top'.format(jobNum=jobNum)
            botMaskPreviewSrc = '/maskPreview/{jobNum}/bot'.format(jobNum=jobNum)

            skipExisting = jobEntry['skipExisting']

            start_fn('200 OK', [('Content-Type', 'text/html')])
            # with open('html/CheckSegmentationProgress.html', 'r') as f: htmlTemplate = f.read()
            return self.formatHTML(
                environ,
                'html/CheckSegmentationProgress.html',
                meanTime=meanTimeStr,
                confInt=timeConfIntStr,
                videoList=completedVideoListHTML,
                jobStateName=jobStateName,
                jobNum=jobNum,
                estimatedTimeRemaining=estimatedTimeRemaining,
                jobName=jobEntry['jobName'],
                owner=owner,
                creationTime=creationTime,
                startTime=startTime,
                completionTime=completionTime,
                exitCodePhrase=exitCodePhrase,
                logHTML=logHTML,
                percentComplete=percentComplete,
                numComplete=numCompletedTasks,
                numTotal=numTasks,
                stateDescription=stateDescription,
                processDead=processDead,
                binaryThreshold=binaryThreshold,
                maskSaveDirectory=maskSaveDirectory,
                topNetworkName=topNetworkName,
                botNetworkName=botNetworkName,
                topOffset=topOffset,
                topHeight=topHeightText,
                botHeight=botHeightText,
                topWidth=topWidthText,
                botWidth=botWidthText,
                generatePreview=generatePreview,
                skipExisting=skipExisting,
                topMaskPreviewSrc=topMaskPreviewSrc,
                botMaskPreviewSrc=botMaskPreviewSrc,
                autoReloadInterval=AUTO_RELOAD_INTERVAL,
                hidePreview=hidePreview
            )
        elif jobEntry['jobType'] == TRAIN_TYPE:
            if jobEntry['startNetworkPath'] is None:
                startNetworkNameText = "Randomized (naive) network"
            else:
                startNetworkNameText = jobEntry['startNetworkPath'].name
            if jobEntry['augmentData']:
                augmentDataText = 'Yes'
            else:
                augmentDataText = 'No'

            if len(jobEntry['loss']) > 0:
                bestLoss = '{:0.4f}'.format(min(jobEntry['loss']))
                lastLoss = '{:0.4f}'.format(jobEntry['loss'][-1])
            else:
                bestLoss = '--'
                lastLoss = '--'
            if len(jobEntry['loss']) > 0:
                bestAccuracy = '{:0.4f}'.format(max(jobEntry['accuracy']))
                lastAccuracy = '{:0.4f}'.format(jobEntry['accuracy'][-1])
            else:
                bestAccuracy = '--'
                lastAccuracy = '--'

            augmentationParameters = jobEntry['augmentationParameters']

            start_fn('200 OK', [('Content-Type', 'text/html')])
            return self.formatHTML(
                environ,
                'html/CheckTrainProgress.html',
                numEpochsComplete=numCompletedTasks,
                numEpochsTotal=numTasks,
                bestLoss=bestLoss,
                bestAccuracy=bestAccuracy,
                lastLoss=lastLoss,
                lastAccuracy=lastAccuracy,
                startNetworkName=startNetworkNameText,
                newNetworkName=jobEntry['newNetworkPath'].name,
                batchSize=jobEntry['batchSize'],
                augmentData=augmentDataText,
                rotationRange=augmentationParameters['rotation_range'],
                widthShiftRange=augmentationParameters['width_shift_range'],
                heightShiftRange=augmentationParameters['height_shift_range'],
                zoomRange=augmentationParameters['zoom_range'],
                horizontalFlip=augmentationParameters['horizontal_flip'],
                verticalFlip=augmentationParameters['vertical_flip'],
                meanTime=meanTimeStr,
                confInt=timeConfIntStr,
                jobStateName=jobStateName,
                jobName=jobEntry['jobName'],
                jobNum=jobNum,
                estimatedTimeRemaining=estimatedTimeRemaining,
                owner=owner,
                creationTime=creationTime,
                startTime=startTime,
                completionTime=completionTime,
                exitCodePhrase=exitCodePhrase,
                logHTML=logHTML,
                percentComplete=percentComplete,
                stateDescription=stateDescription,
                processDead=processDead,
                generatePreview=generatePreview,
                autoReloadInterval=AUTO_RELOAD_INTERVAL,
                hidePreview=hidePreview
            )
        else:
            raise ValueError('Unrecognized job type: {t}'.format(t=self.jobQueue[jobNum]['jobType']))

    def rootHandler(self, environ, start_fn):
        logger.log(logging.INFO, 'Serving root file')
        neuralNetworkList = self.getNeuralNetworkList(namesOnly=True)
        mountList = self.getMountList(includePosixLocal=True)
        mountURIs = mountList.keys()
        mountPaths = [mountList[k] for k in mountURIs]
        mountOptionsText = self.createOptionList(mountPaths, optionNames=mountURIs, defaultValue=DEFAULT_MOUNT_PATH)
        if 'QUERY_STRING' in environ:
            queryString = environ['QUERY_STRING']
        else:
            queryString = 'None'
        postDataRaw = environ['wsgi.input'].read().decode('utf-8')
        postData = urllib.parse.parse_qs(postDataRaw, keep_blank_values=False)

        logger.log(logging.INFO, 'Creating return data')

        username = getUsername(environ)

        if len(neuralNetworkList) > 0:
            topNetworkOptionText = self.createOptionList(neuralNetworkList, defaultValue=DEFAULT_TOP_NETWORK_NAME)
            botNetworkOptionText = self.createOptionList(neuralNetworkList, defaultValue=DEFAULT_BOT_NETWORK_NAME)
            start_fn('200 OK', [('Content-Type', 'text/html')])
            return self.formatHTML(
                environ,
                'html/Index.html',
                query=queryString,
                # mounts=mountList,
                # environ=environ,
                input=postData,
                remoteUser=username,
                path=environ['PATH_INFO'],
                nopts_bot=botNetworkOptionText,
                nopts_top=topNetworkOptionText,
                mopts=mountOptionsText
                )
        else:
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Neural network error',
                errorMsg='No neural networks found! Please upload a .h5 or .hd5 neural network file to the ./{nnsubfolder} folder.'.format(nnsubfolder=NETWORKS_SUBFOLDER),
                linkURL='/',
                linkAction='retry job creation once a neural network has been uploaded'
            )

    def trainHandler(self, environ, start_fn):
        logger.log(logging.INFO, 'Serving network training file')
        existingNeuralNetworkList = [RANDOM_TRAINING_NETWORK_NAME] + self.getNeuralNetworkList(namesOnly=True)
        mountList = self.getMountList(includePosixLocal=True)
        mountURIs = mountList.keys()
        mountPaths = [mountList[k] for k in mountURIs]
        mountOptionsText = self.createOptionList(mountPaths, optionNames=mountURIs, defaultValue=DEFAULT_MOUNT_PATH)
        if 'QUERY_STRING' in environ:
            queryString = environ['QUERY_STRING']
        else:
            queryString = 'None'
        postDataRaw = environ['wsgi.input'].read().decode('utf-8')
        postData = urllib.parse.parse_qs(postDataRaw, keep_blank_values=False)

        logger.log(logging.INFO, 'Creating return data')

        username = getUsername(environ)

        existingNetworkOptionText = self.createOptionList(existingNeuralNetworkList, defaultValue=RANDOM_TRAINING_NETWORK_NAME)
        start_fn('200 OK', [('Content-Type', 'text/html')])
        return self.formatHTML(
            environ,
            'html/Train.html',
            query=queryString,
            # mounts=mountList,
            # environ=environ,
            input=postData,
            remoteUser=username,
            path=environ['PATH_INFO'],
            nopts=existingNetworkOptionText,
            mopts=mountOptionsText
            )

    def cancelJobHandler(self, environ, start_fn):
        # Get jobNums from URL
        jobNums = [int(jobNum) for jobNum in environ['PATH_INFO'].split('/')[2:]]

        invalidJobNums = []
        alreadyCancelledJobNums = []
        unauthorizedJobNums = []

        for jobNum in jobNums:
            if jobNum not in self.getJobNums():
                # Invalid jobNum
                invalidJobNums.append(jobNum)
                continue

            if jobNum in self.getJobNums(completed=True):
                # Job is already finished, can't cancel
                alreadyCancelledJobNums.append(jobNum)
                continue

            if not isWriteAuthorized(getUsername(environ), self.jobQueue[jobNum]['owner']):
                # User is not authorized
                unauthorizedJobNums.append(jobNum)
                continue

            # Valid enqueued job - set cancelled flag to True, and
            logger.log(logging.INFO, 'Cancelling job {jobNum}'.format(jobNum=jobNum))
            self.jobQueue[jobNum]['cancelled'] = True
            now = time.time_ns()
            if self.jobQueue[jobNum]['creationTime'] is None:
                self.jobQueue[jobNum]['creationTime'] = now
            # if self.jobQueue[jobNum]['startTime'] is None:
            #     self.jobQueue[jobNum]['startTime'] = now
            self.jobQueue[jobNum]['completionTime'] = now
            if self.jobQueue[jobNum]['job'] is not None:
                self.jobQueue[jobNum]['job'].msgQueue.put((ServerJob.EXIT, None))

        # Check if there were any invalid job nums
        if len(invalidJobNums) > 0:
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Invalid job ID',
                errorMsg='Invalid job ID: {jobID}'.format(jobID=makeNaturalLanguageList(invalidJobNums)),
                linkURL='/',
                linkAction='recreate job'
            )
        # Check if there were any already cancelled job num
        if len(alreadyCancelledJobNums) > 0:
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Cannot terminate completed job',
                errorMsg='Can\'t terminate job {jobID} because it has already finished processing.'.format(jobID=makeNaturalLanguageList(alreadyCancelledJobNums)),
                linkURL='/',
                linkAction='create a new job'
            )
        # Check if there were any unauthorized job nums
        if len(unauthorizedJobNums) > 0:
            return self.unauthorizedHandler(environ, start_fn)

        if len(jobNums) <= 1:
            start_fn('303 See Other', [('Location','/checkProgress/{jobID}'.format(jobID=jobNums[0]))])
        else:
            start_fn('303 See Other', [('Location','/myJobs')])
        return []

    def getHumanReadableJobState(self, jobNum):
        state = 'Unknown'
        if self.isCancelled(jobNum):
            state = 'Cancelled'
        elif self.jobQueue[jobNum]['exitCode'] == ServerJob.INCOMPLETE:
            if not self.isConfirmed(jobNum):
                state = 'Unconfirmed'
            elif self.isEnqueued(jobNum):
                state = 'Enqueued'
            else:
                state = 'Working'
        elif self.isSucceeded(jobNum):
            state = 'Succeeded'
        elif self.isFailed(jobNum):
            state = 'Failed'
        return state

    def myJobsHandler(self, environ, start_fn):
        user = getUsername(environ)

        with open('html/MyJobsTableRowTemplate.html', 'r') as f:
            jobEntryTemplate = f.read()

        jobEntries = []
        for jobNum in self.getJobNums(owner=user):
            state = self.getHumanReadableJobState(jobNum)

            numTasks, numCompletedTasks = self.getJobProgress(jobNum)
            percentComplete = "{percentComplete:.1f}".format(percentComplete=100*numCompletedTasks/numTasks)

            if self.jobQueue[jobNum]['jobType'] == TRAIN_TYPE:
                jobType = 'Train'
            elif self.jobQueue[jobNum]['jobType'] == SEGMENT_TYPE:
                jobType = 'Segment'
            else:
                raise ValueError('Unknown job type {t}'.format(t=self.jobQueue[jobNum]['jobType']))

            jobEntries.append(jobEntryTemplate.format(
                percentComplete=percentComplete,
                jobNum=jobNum,
                jobType=jobType,
                numTasks=numTasks,
                jobDescription=self.jobQueue[jobNum]['jobName'],
                state=state
            ))
        jobEntryTableBody = '\n'.join(jobEntries)
        start_fn('200 OK', [('Content-Type', 'text/html')])
        return self.formatHTML(
            environ,
            'html/MyJobs.html',
            tbody=jobEntryTableBody,
            autoReloadInterval=AUTO_RELOAD_INTERVAL,
            user=user
        )

    def networkRenameHandler(self, environ, start_fn):
        # if not isAdmin(getUsername(environ)):
        #     # User is not authorized
        #     return self.unauthorizedHandler(environ, start_fn)

        # Get old and new names from URL
        oldName, newName = environ['PATH_INFO'].split('/')[-2:]
        if oldName not in self.getNeuralNetworkList(namesOnly=True):
            # Invalid name
            start_fn('500 Internal Server Error', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Cannot rename - network does not exist',
                errorMsg='Something went wrong - could not find network {netName}'.format(netName=oldName),
                linkURL='/networkManagement',
                linkAction='go back to network management'
            )
        elif newName in self.getNeuralNetworkList(namesOnly=True):
            # New name already exists
            start_fn('500 Internal Server Error', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Cannot rename - name already in use',
                errorMsg='A network by the name \"{newName}\" already exists. Please rename or remove that network first.'.format(newName=newName),
                linkURL='/networkManagement',
                linkAction='go back to network management'
            )
        else:
            # Valid network names - execute rename operation
            try:
                success, msg = self.renameNeuralNetwork(oldName, newName)
                if not success:
                    raise OSError(msg)
            except FileNotFoundError:
                start_fn('500 Internal Server Error', [('Content-Type', 'text/html')])
                return self.formatError(
                    environ,
                    errorTitle='Error renaming file',
                    errorMsg='An error occurred when renaming \"{oldName}\" to \"{newName}\". Please check that the new name only contains valid filename characters.'.format(oldName=oldName, newName=newName),
                    linkURL='/networkManagement',
                    linkAction='go back to network management'
                )
            except PermissionError:
                start_fn('500 Internal Server Error', [('Content-Type', 'text/html')])
                return self.formatError(
                    environ,
                    errorTitle='Error renaming file',
                    errorMsg='A permission error occurred when renaming \"{oldName}\" to \"{newName}\". Please check the permissions for the network file.'.format(oldName=oldName, newName=newName),
                    linkURL='/networkManagement',
                    linkAction='go back to network management'
                )
            except OSError:
                start_fn('500 Internal Server Error', [('Content-Type', 'text/html')])
                return self.formatError(
                    environ,
                    errorTitle='Error renaming file',
                    errorMsg='An error occurred when renaming \"{oldName}\" to \"{newName}\": {msg}'.format(oldName=oldName, newName=newName, msg=msg),
                    linkURL='/networkManagement',
                    linkAction='go back to network management'
                )
        start_fn('303 See Other', [('Location','/networkManagement')])
        return []

    def networkRemoveHandler(self, environ, start_fn):
        # if not isAdmin(getUsername(environ)):
        #     # User is not authorized
        #     return self.unauthorizedHandler(environ, start_fn)

        # Get old and new names from URL
        netName = environ['PATH_INFO'].split('/')[-1]
        if netName not in self.getNeuralNetworkList(namesOnly=True):
            # Invalid name
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Cannot remove - network does not exist',
                errorMsg='Something went wrong - could not find network {netName}'.format(netName=netName),
                linkURL='/networkManagement',
                linkAction='go back to network management'
            )
        else:
            # Valid network names - execute remove operation
            logger.log(logging.INFO, 'Removing net \"{netName}\"'.format(netName=netName))
            try:
                self.removeNeuralNetwork(netName)
            except FileNotFoundError:
                return self.formatError(
                    environ,
                    errorTitle='Error renaming file',
                    errorMsg='An error occurred when removing \"{netName}\" - could not find network.'.format(netName=netName),
                    linkURL='/networkManagement',
                    linkAction='go back to network management'
                )
            except PermissionError:
                return self.formatError(
                    environ,
                    errorTitle='Error renaming file',
                    errorMsg='A permission error occurred when removing \"{netName}\". Please check the permissions for the network file.'.format(netName=netName),
                    linkURL='/networkManagement',
                    linkAction='go back to network management'
                )
        start_fn('303 See Other', [('Location','/networkManagement')])
        return []

    def networkManagementHandler(self, environ, start_fn):
        # if not isAdmin(getUsername(environ)):
        #     # User is not authorized
        #     return self.unauthorizedHandler(environ, start_fn)

        networkPaths, networkTimestamps = self.getNeuralNetworkList(includeTimestamps=True)
        networkTimestampStrings = [timestamp.strftime(HTML_DATE_FORMAT) for timestamp in networkTimestamps]

        with open('html/NetworkManagementTableRowTemplate.html', 'r') as f:
            networkEntryTemplate = f.read()

        networkEntries = []
        for k, (networkPath, timestamp) in enumerate(zip(networkPaths, networkTimestampStrings)):
            networkEntries.append(
                networkEntryTemplate.format(
                    netNum = k,
                    netName = networkPath.name,
                    netTime = timestamp
                )
            )
        networkEntryTableBody = '\n'.join(networkEntries)
        start_fn('200 OK', [('Content-Type', 'text/html')])
        return self.formatHTML(
            environ,
            'html/NetworkManagement.html',
            tbody=networkEntryTableBody,
            numNetworks=len(networkPaths),
            autoReloadInterval=AUTO_RELOAD_INTERVAL,
        )

    def getJobProgress(self, jobNum):
        if self.jobQueue[jobNum]['jobType'] == SEGMENT_TYPE:
            numTasks = len(self.jobQueue[jobNum]['videoList'])
            numCompletedTasks = len(self.jobQueue[jobNum]['completedVideoList'])
        elif self.jobQueue[jobNum]['jobType'] == TRAIN_TYPE:
            numTasks = self.jobQueue[jobNum]['numEpochs']
            numCompletedTasks = self.jobQueue[jobNum]['lastEpochNum'] + 1
        else:
            raise ValueError('Unknown job type {t}'.format(t=self.jobQueue[jobNum]['jobType']))
        return numTasks, numCompletedTasks

    def serverManagementHandler(self, environ, start_fn):
        if not isAdmin(getUsername(environ)):
            # User is not authorized
            return self.unauthorizedHandler(environ, start_fn)

        allJobNums = self.getJobNums()

        with open('html/ServerManagementTableRowTemplate.html', 'r') as f:
            jobEntryTemplate = f.read()

        serverStartTime = self.startTime.strftime(HTML_DATE_FORMAT)

        jobEntries = []
        for jobNum in allJobNums:
            state = self.getHumanReadableJobState(jobNum)

            numTasks, numCompletedTasks = self.getJobProgress(jobNum)
            percentComplete = "{percentComplete:.1f}".format(percentComplete=100*numCompletedTasks/numTasks)

            if self.jobQueue[jobNum]['jobType'] == TRAIN_TYPE:
                jobType = 'Train'
            elif self.jobQueue[jobNum]['jobType'] == SEGMENT_TYPE:
                jobType = 'Segment'
            else:
                raise ValueError('Unknown job type {t}'.format(t=self.jobQueue[jobNum]['jobType']))

            jobEntries.append(jobEntryTemplate.format(
                jobNum=jobNum,
                jobDescription = self.jobQueue[jobNum]['jobName'],
                jobType=jobType,
                owner=self.jobQueue[jobNum]['owner'],
                percentComplete = percentComplete,
                numTasks=numTasks,
                confirmed=self.jobQueue[jobNum]['confirmed'],
                cancelled=self.jobQueue[jobNum]['cancelled'],
                state=state
            ))
        jobEntryTableBody = '\n'.join(jobEntries)
        start_fn('200 OK', [('Content-Type', 'text/html')])
        return self.formatHTML(
            environ,
            'html/ServerManagement.html',
            tbody=jobEntryTableBody,
            startTime=serverStartTime,
            autoReloadInterval=AUTO_RELOAD_INTERVAL,
        )

    def restartServerHandler(self, environ, start_fn):
        if not isAdmin(getUsername(environ)):
            # User is not authorized
            return self.unauthorizedHandler(environ, start_fn)
        raise SystemExit("Server restart requested")

    def reloadAuthHandler(self, environ, start_fn):
        if not isAdmin(getUsername(environ)):
            # User is not authorized
            return self.unauthorizedHandler(environ, start_fn)

        message = "Authentication file successfully reloaded!"
        addMessage(environ, message)
        return self.serverManagementHandler(environ, start_fn)

    def helpHandler(self, environ, start_fn):
        user = getUsername(environ)

        start_fn('200 OK', [('Content-Type', 'text/html')])
        return self.formatHTML(
            environ,
            'html/Help.html',
            user=user
        )

    def changePasswordHandler(self, environ, start_fn):
        user = getUsername(environ)

        start_fn('200 OK', [('Content-Type', 'text/html')])
        return self.formatHTML(
            environ,
            'html/ChangePassword.html',
            user=user
        )

    def finalizePasswordHandler(self, environ, start_fn):
        # Display page showing what job will be, and offering opportunity to go ahead or cancel
        postDataRaw = environ['wsgi.input'].read().decode('utf-8')
        postData = urllib.parse.parse_qs(postDataRaw, keep_blank_values=False)

        user = getUsername(environ)
        try:
            oldPassword  = postData['oldPassword' ][0]
            newPassword  = postData['newPassword' ][0]
            newPassword2 = postData['newPassword2'][0]
        except KeyError:
            # Missing one of the postData arguments
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Missing parameter',
                errorMsg='A required field is missing. Please retry with all required fields filled in.',
                linkURL='/changePassword',
                linkAction='return to change password page (or use browser back button)'
                )

        # Check if all parameters are valid. If not, display error and offer to go back
        if newPassword == newPassword2:
            success, reason = changePassword(user, oldPassword, newPassword)
        else:
            success = False
            reason = "New password does not match confirmation. Please retype."

        if not success:
            start_fn('404 Not Found', [('Content-Type', 'text/html')])
            return self.formatError(
                environ,
                errorTitle='Invalid job parameter',
                errorMsg=reason,
                linkURL='/changePassword',
                linkAction='return to change password page (or use browser back button)'
                )

        start_fn('200 OK', [('Content-Type', 'text/html')])
        return self.formatHTML(
            environ,
            'html/PasswordChanged.html',
            user=user,
            message="Password succesfully changed!"
        )

    def invalidHandler(self, environ, start_fn):
        logger.log(logging.INFO, 'Serving invalid warning')
        start_fn('404 Not Found', [('Content-Type', 'text/html')])
        return self.formatError(
            environ,
            errorTitle='Path not recognized',
            errorMsg='Path {name} not recognized!'.format(name=environ['PATH_INFO']),
            linkURL='/',
            linkAction='return to job creation page'
        )

    def unauthorizedHandler(self, environ, start_fn):
        user=getUsername(environ)
        logger.log(logging.INFO, 'Unauthorized access attempt: {user}'.format(user=user))
        start_fn('404 Not Found', [('Content-Type', 'text/html')])
        return self.formatError(
            environ,
            errorTitle='Not authorized',
            errorMsg='User {user} is not authorized to perform that action!'.format(user=user),
            linkURL='/',
            linkAction='return to job creation page'
        )

if __name__ == '__main__':
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 80

    logger.log(logging.INFO, 'Spinning up server!')
    while True:
        s = SegmentationServer(webRoot=ROOT, port=port)
        application = BasicAuth(s, users=USERS)
        s.linkBasicAuth(application)
        try:
            logger.log(logging.INFO, 'Starting segmentation server...')
            serve(application, host='0.0.0.0', port=port, url_scheme='http')
            logger.log(logging.INFO, '...segmentation server started!')
        except KeyboardInterrupt:
            logger.exception('Keyboard interrupt')
            break
        except SystemExit:
            logger.exception('Server restart requested by user')
        except:
            logger.exception('Server crashed!')
        time.sleep(1)
