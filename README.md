# tongueSegmentationServer
A client/server application for applying neural networks to automatically
segment sets of videos of mouse tongues, as well as training or retraining
neural networks to do that task. It provides a queueing system to allow
multiple users to schedule segmentation jobs on the server, and a HTTP interface
for managing and monitoring the server and the progress of segmentation jobs.

## Requirements

- Computer to run the server that can be accessed by client computers
- Windows 10 recommended
- [Anaconda](https://www.anaconda.com/)
- Libraries installed in Anaconda as per [conda_requirements.yml](conda_requirements.yml)
- Firewall with TCP connections allowed on port 80

## Installation

 - Download and install Anaconda on the server computer
 - Open an Anaconda terminal
 - Navigate your current working directory to the directory where conda_requirements.yml is
 - Install the required libraries in a new Anaconda environment called "segServer":

```>> conda env update -n segServer --file conda_requirements.yaml```

## Starting server

 - Open an Anaconda terminal
 - Navigate to the directory where you cloned this repository, which contains SegmenterServer.py
 - Optionally fetch the latest version of the repository:

```
git fetch
git pull
```
 - Activate the environment you created with the required libraries:

```activate segServer2```

 - Start the server:

```
python SegmenterServer.py
```

## Accessing the server

 - Find the IP address or host name of the server computer (see [ipconfig](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/ipconfig) and [nslookup](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/nslookup))
 - On another computer, enter a browser, and simply type the IP address or host name of the server computer into the address bar.
 - An authentication request should pop up. Enter your username and password. If you haven't added any users yet, the default user/pass is "user" and "password", or "admin" and "password" for an administrator. It is strongly recommended that you add your own usernames and passwords immediately (see below)

## Adding users

 - Find and open the file "private/Auth.json" in a text editor.
 - Add a new entry to the list of users, following the exact same pattern as the existing users:

```
"username": ["password", ZERO_OR_ONE]
```

 - ZERO_OR_ONE is either the number 0 or the number 1. 0 indicates the user has administrative permissions, 1 indicates they are a regular user.
 - Restart the server if it was running (press control-c in the Anaconda terminal to stop the server), and the new users should be available
