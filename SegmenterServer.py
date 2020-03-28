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

NEURAL_NETWORK_EXTENSIONS = ['.h5', '.hd5']
NETWORKS_SUBFOLDER = 'networks'

logger = logging.getLogger(__name__)

# create logger with 'spam_application'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
datetimeString = dt.datetime.now().isoformat()
fh = logging.FileHandler('./logs/{n}_{d}.log'.format(d=datetimeString, n=__name__))
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

envVars = dict(os.environ)  # or os.environ.copy()
try:
    envVars['WSGI_AUTH_CREDENTIALS']='glab:password'
finally:
    os.environ.clear()
    os.environ.update(envVars)

class SegmentationServer:
    def __init__(self):
        pass

    def getMountList(self):
#        p = Popen('mount', stdout=PIPE, stderr=PIPE, shell=True)
        p = Popen("mount | awk '$5 ~ /cifs|drvfs/ {print $0}'", stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = p.communicate()
        mountLines = stdout.decode('utf-8').strip().split('\n')
        mounts = {}
        for mountLine in mountLines:
            elements = mountLine.split(' ')
            mounts[elements[0]] = elements[2]
        logger.log(logging.INFO, 'Got mount list: ' + str(mounts))
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

    def __call__(self, environ, start_fn):
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

        logger.log(logging.INFO, 'Serving page with queryString {queryString}'.format(queryString=queryString))

        if len(neuralNetworkList) > 0:
            networkOptionText = self.createOptionList(neuralNetworkList)
            formText = '''
<form action="/" method="POST">
    <div>
        <label for="videoRootMountPoint">Video root mount point:</label><br>
        <select name="videoRootMountPoint">
        {mopts}
        </select>
    </div>
    <div>
        <label for="videoRoot">Video root directory:</label><br>
        <input type="text" id="videoRoot" name="videoRoot" value=""><br>
    </div>
    <div>
        <label for="networkName">Neural network name:</label><br>
        <select name="neuralNetwork">
        {nopts}
        </select>
    </div>
    <input type="submit" value="Submit">
</form>'''.format(nopts=networkOptionText, mopts=mountOptionsText)
        else:
            formText = '''
<h2>No neural networks found!
Please upload a .h5 or .hd5 neural network file to the ./{nnsubfolder} folder.</h2>'''.format(nnsubfolder=NETWORKS_SUBFOLDER)

        response = [
        '''
<html>
<head>
    <link rel="stylesheet" type="text/css" href="main.css">
</head>
<body>
    <h1>
        Hello World!
    </h1>
    {form}
    <p>Query = {query}</p>
    <p>Mounts = {mounts}</p>
    <p>Input = {input}</p>
    <p>environ = {environ}</p>
</body>
</html>
        '''.format(query=queryString, mounts=mountList, environ=environ, input=postData, form=formText),
        ]
        response = [line.encode('utf-8') for line in response]
        return response

logger.log(logging.INFO, 'Spinning up server!')
while True:
    s = SegmentationServer()
    application = BasicAuth(s)
    try:
        logger.log(logging.INFO, 'Starting segmentation server...')
        serve(application, host='0.0.0.0', port=5000)
        logger.log(logging.INFO, '...segmentation server started!')
    except e:
        logger.exception('Server crashed!')
    time.sleep(5)
