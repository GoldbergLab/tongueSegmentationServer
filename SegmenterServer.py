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
from pathlib import Path
import fnmatch

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


class SegmentationServer:
    def __init__(self, webRoot='.'):
        self.routes = [
            ('/static/*',   self.staticHandler),
            ('/',           self.rootHandler)
        ]
        self.webRootPath = Path(webRoot).resolve()

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
            return ['<html><body><h1>Static file {name} not found!</body></html>'.format(name=requestedStaticFileRelativePath).encode('utf-8')]

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
<form action="/" method="POST">
    <div class="field-wrap">
        <label class="field-label" for="videoRootMountPoint">Video root mount point:</label>
        <select class="field" name="videoRootMountPoint">
        {mopts}
        </select>
    </div>
    <div class="field-wrap">
        <label class="field-label" for="videoRoot">Video root directory:</label>
        <input class="field" type="text" id="videoRoot" name="videoRoot" value="">
    </div>
    <div class="field-wrap">
        <label class="field-label" for="networkName">Neural network name:</label>
        <select class="field" name="neuralNetwork">
        {nopts}
        </select>
    </div>
    <input class="field" type="submit" value="Submit">
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
        Hello World!
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
        return ['<html><body><h1>Path {name} not recognized!</body></html>'.format(name=requestedPath).encode('utf-8')]

logger.log(logging.INFO, 'Spinning up server!')
while True:
    s = SegmentationServer(webRoot=ROOT)
    application = BasicAuth(s)
    try:
        logger.log(logging.INFO, 'Starting segmentation server...')
        serve(application, host='0.0.0.0', port=5000)
        logger.log(logging.INFO, '...segmentation server started!')
    except e:
        logger.exception('Server crashed!')
    time.sleep(5)
