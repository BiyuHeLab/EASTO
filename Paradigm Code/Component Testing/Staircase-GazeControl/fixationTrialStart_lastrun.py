#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.1.1),
    on April 05, 2023, at 14:40
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, iohub, hardware
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from elConnect
# DESCRIPTION:
# This is a basic example, which shows how to implement a gaze-based trigger. 
# First a fixation cross is shown at the center of the screen. The trial moves 
# on only when gaze has been directed to the fixation cross.  
# A keyboard press terminates the trial.

# The code components in the eyelinkSetup, eyelinkStartRecording, fixWindow, 
# trial, and eyelinkStopRecording routines handle communication with the 
# Host PC/EyeLink system.  All the code components are set to Code Type Py, and 
# each code component may have code in the various tabs 
# (e.g., Before Experiment, Begin Experiment, etc.)

# Last updated: March 7 2023

# This Before Experiment tab of the elConnect component imports some
# modules we need, manages data filenames, allows for dummy mode configuration
# (for testing), connects to the Host PC, configures some tracker settings,
# and defines some helper function definitions (which are called later)

import pylink
import time
import platform
from PIL import Image  # for preparing the Host backdrop image
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from string import ascii_letters, digits
from math import fabs

# Switch to the script folder
script_path = os.path.dirname(sys.argv[0])
if len(script_path) != 0:
    os.chdir(script_path)


# Set this variable to True if you use the built-in retina screen as your
# primary display device on macOS. If have an external monitor, set this
# variable True if you choose to "Optimize for Built-in Retina Display"
# in the Displays preference settings.
use_retina = False

# Set this variable to True to run the script in "Dummy Mode"
dummy_mode = False

# Prompt user to specify an EDF data filename
# before we open a fullscreen window
dlg_title = 'Enter EDF File Name'
dlg_prompt = 'Please enter a file name with 8 or fewer characters\n' + \
             '[letters, numbers, and underscore].'

# Set up EDF data file name and local data folder
#
# The EDF data filename should not exceed 8 alphanumeric characters
# use ONLY number 0-9, letters, & _ (underscore) in the filename
edf_fname = 'TEST'

# Prompt user to specify an EDF data filename
# before we open a fullscreen window
dlg_title = 'Enter EDF File Name'
dlg_prompt = 'Please enter a file name with 8 or fewer characters\n' + \
             '[letters, numbers, and underscore].'

# loop until we get a valid filename
while True:
    dlg = gui.Dlg(dlg_title)
    dlg.addText(dlg_prompt)
    dlg.addField('File Name:', edf_fname)
    # show dialog and wait for OK or Cancel
    ok_data = dlg.show()
    if dlg.OK:  # if ok_data is not None
        print('EDF data filename: {}'.format(ok_data[0]))
    else:
        print('user cancelled')
        core.quit()
        sys.exit()

    # get the string entered by the experimenter
    tmp_str = dlg.data[0]
    # strip trailing characters, ignore the ".edf" extension
    edf_fname = tmp_str.rstrip().split('.')[0]

    # check if the filename is valid (length <= 8 & no special char)
    allowed_char = ascii_letters + digits + '_'
    if not all([c in allowed_char for c in edf_fname]):
        print('ERROR: Invalid EDF filename')
    elif len(edf_fname) > 8:
        print('ERROR: EDF filename should not exceed 8 characters')
    else:
        break
        
# Set up a folder to store the EDF data files and the associated resources
# e.g., files defining the interest areas used in each trial
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# We download EDF data file from the EyeLink Host PC to the local hard
# drive at the end of each testing session, here we rename the EDF to
# include session start date/time
time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
session_identifier = edf_fname + time_str

# create a folder for the current testing session in the "results" folder
session_folder = os.path.join(results_folder, session_identifier)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

# For macOS users check if they have a retina screen
if 'Darwin' in platform.system():
        dlg = gui.Dlg("Retina Screen?", labelButtonOK='Yes', labelButtonCancel='No')
        dlg.addText("Does your Mac have a Retina screen?")
        # show dialog and wait for OK or Cancel
        ok_data = dlg.show()
        if dlg.OK:  # if ok_data is not None
            use_retina = True
        else:
            use_retina = False
    
# Step 1: Connect to the EyeLink Host PC
#
# The Host IP address, by default, is "100.1.1.1".
# the "el_tracker" objected created here can be accessed through the Pylink
# Set the Host PC address to "None" (without quotes) to run the script
# in "Dummy Mode"
if dummy_mode:
    el_tracker = pylink.EyeLink(None)
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        dlg = gui.Dlg("Dummy Mode?")
        dlg.addText("Couldn't connect to tracker at 100.1.1.1 -- continue in Dummy Mode?")
        # show dialog and wait for OK or Cancel
        ok_data = dlg.show()
        if dlg.OK:  # if ok_data is not None
            #print('EDF data filename: {}'.format(ok_data[0]))
            dummy_mode = True
            el_tracker = pylink.EyeLink(None)
        else:
            print('user cancelled')
            core.quit()
            sys.exit()

# Step 2: Open an EDF data file on the Host PC
edf_file = edf_fname + ".EDF"
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print('ERROR:', err)
    # close the link if we have one open
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# Add a header text to the EDF file to identify the current experiment name
# This is OPTIONAL. If your text starts with "RECORDED BY " it will be
# available in DataViewer's Inspector window by clicking
# the EDF session node in the top panel and looking for the "Recorded By:"
# field in the bottom panel of the Inspector.
preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

# Step 3: Configure the tracker
#
# Put the tracker in offline mode before we change tracking parameters
el_tracker.setOfflineMode()

# Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
# 5-EyeLink 1000 Plus, 6-Portable DUO
eyelink_ver = 0  # set version to 0, in case running in Dummy mode
if not dummy_mode:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split('.')[0])
    # print out some version info in the shell
    print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

# File and Link data control
# what eye events to save in the EDF file, include everything by default
file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
# what eye events to make available over the link, include everything by default
link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
# what sample data to save in the EDF data file and to make available
# over the link, include the 'HTARGET' flag to save head target sticker
# data for supported eye trackers
if eyelink_ver > 3:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
else:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    
el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# Optional tracking parameters
# Sample rate, 250, 500, 1000, or 2000, check your tracker specification
# if eyelink_ver > 2:
#     el_tracker.sendCommand("sample_rate 1000")
# Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
el_tracker.sendCommand("calibration_type = HV9")
# Set a gamepad button to accept calibration/drift check target
# You need a supported gamepad/button box that is connected to the Host PC
el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

def clear_screen(win):
    """ clear up the PsychoPy window"""
    win.fillColor = genv.getBackgroundColor()
    win.flip()


def show_msg(win, text, wait_for_keypress=True):
    """ Show task instructions on screen"""
    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win)


def terminate_task():
    """ Terminate the task gracefully and retrieve the EDF data file
    """
    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, msg, wait_for_keypress=False)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()


def abort_trial():
    """Ends recording """
    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(win)
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR    
    



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2023.1.1'
expName = 'fixationTrialStart'  # from the Builder filename that created this script
expInfo = {
    'participant': '',
    'session': '001',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\rhardstone\\Documents\\Brandon\\Staircase-GazeControl\\fixationTrialStart_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='EEG', color=[0,0,0], colorSpace='rgb',
    backgroundImage='', backgroundFit='none',
    blendMode='avg', useFBO=True, 
    units='deg')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "eyelinkSetup" ---
elInstructions = visual.TextStim(win=win, name='elInstructions',
    text='Press any key to start Camera Setup',
    font='Open Sans',
    units='norm', pos=(0, 0), height=0.7, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_2 = keyboard.Keyboard()
# Run 'Begin Experiment' code from elConnect
# This Begin Experiment tab of the elConnect component gets graphic 
# information from Psychopy, sets the screen_pixel_coords on the Host PC based
# on these values, and logs the screen resolution for Data Viewer via 
# a DISPLAY_COORDS message

# get the native screen resolution used by PsychoPy
scn_width, scn_height = win.size

# resolution fix for Mac retina displays
if 'Darwin' in platform.system():
    if use_retina:
        scn_width = int(scn_width/2.0)
        scn_height = int(scn_height/2.0)

# Pass the display pixel coordinates (left, top, right, bottom) to the tracker
# see the EyeLink Installation Guide, "Customizing Screen Settings"
el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendCommand(el_coords)

# Write a DISPLAY_COORDS message to the EDF file
# Data Viewer needs this piece of info for proper visualization, see Data
# Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendMessage(dv_coords)  
    

# --- Initialize components for Routine "start" ---
text_2 = visual.TextStim(win=win, name='text_2',
    text='Any text\n\nincluding line breaks',
    font='Open Sans',
    units='norm', pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_3 = keyboard.Keyboard()

# --- Initialize components for Routine "eyelinkStartRecording" ---

# --- Initialize components for Routine "forceFixation" ---
fixation = visual.ShapeStim(
    win=win, name='fixation', vertices='cross',
    size=(2, 2),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
# Run 'Begin Experiment' code from elFixWindow
########## Start section from the Begin Experiment part of 
########## the eyelinkTrialCode code in the trial routine

trial_index = 1

########## End section from the Begin Experiment part of 
########## the eyelinkTrialCode code in the trial routine

# --- Initialize components for Routine "stim" ---
fixation_2afc = visual.ShapeStim(
    win=win, name='fixation_2afc', vertices='cross',units='deg', 
    size=(2, 2),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
left_img = visual.ImageStim(
    win=win,
    name='left_img', 
    image='images/img1.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(5, 5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
ISI_load_images = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI_load_images')
right_img = visual.ImageStim(
    win=win,
    name='right_img', 
    image='images/img2.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(5, 5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-4.0)
key_resp = keyboard.Keyboard()

# --- Initialize components for Routine "eyelinkStopRecording" ---

# --- Initialize components for Routine "endTask" ---
text = visual.TextStim(win=win, name='text',
    text='Thanks\n',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "eyelinkSetup" ---
continueRoutine = True
# update component parameters for each repeat
key_resp_2.keys = []
key_resp_2.rt = []
_key_resp_2_allKeys = []
# keep track of which components have finished
eyelinkSetupComponents = [elInstructions, key_resp_2]
for thisComponent in eyelinkSetupComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "eyelinkSetup" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *elInstructions* updates
    
    # if elInstructions is starting this frame...
    if elInstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        elInstructions.frameNStart = frameN  # exact frame index
        elInstructions.tStart = t  # local t and not account for scr refresh
        elInstructions.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(elInstructions, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'elInstructions.started')
        # update status
        elInstructions.status = STARTED
        elInstructions.setAutoDraw(True)
    
    # if elInstructions is active this frame...
    if elInstructions.status == STARTED:
        # update params
        pass
    
    # *key_resp_2* updates
    waitOnFlip = False
    
    # if key_resp_2 is starting this frame...
    if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_2.frameNStart = frameN  # exact frame index
        key_resp_2.tStart = t  # local t and not account for scr refresh
        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'key_resp_2.started')
        # update status
        key_resp_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_2.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_2.getKeys(keyList=None, waitRelease=False)
        _key_resp_2_allKeys.extend(theseKeys)
        if len(_key_resp_2_allKeys):
            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in eyelinkSetupComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "eyelinkSetup" ---
for thisComponent in eyelinkSetupComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if key_resp_2.keys in ['', [], None]:  # No response was made
    key_resp_2.keys = None
thisExp.addData('key_resp_2.keys',key_resp_2.keys)
if key_resp_2.keys != None:  # we had a response
    thisExp.addData('key_resp_2.rt', key_resp_2.rt)
thisExp.nextEntry()
# Run 'End Routine' code from elConnect
# This End Routine tab of the elConnect component configures some
# graphics options for calibration, and then performs a camera setup
# so that you can set up the eye tracker and calibrate/validate the participant

# Configure a graphics environment (genv) for tracker calibration
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
print(genv)  # print out the version number of the CoreGraphics library

# Set background and foreground colors for the calibration target
# in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
foreground_color = (-1, -1, -1)
background_color = tuple(win.color)
genv.setCalibrationColors(foreground_color, background_color)

# Set up the calibration target
#
# The target could be a "circle" (default), a "picture", a "movie" clip,
# or a rotating "spiral". To configure the type of calibration target, set
# genv.setTargetType to "circle", "picture", "movie", or "spiral", e.g.,
# genv.setTargetType('picture')
#
# Use gen.setPictureTarget() to set a "picture" target
# genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))
#
# Use genv.setMovieTarget() to set a "movie" target
# genv.setMovieTarget(os.path.join('videos', 'calibVid.mov'))

# Use a picture as the calibration target
genv.setTargetType('picture')
genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))

# Configure the size of the calibration target (in pixels)
# this option applies only to "circle" and "spiral" targets
# genv.setTargetSize(24)

# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
genv.setCalibrationSounds('', '', '')

# resolution fix for macOS retina display issues
if use_retina:
    genv.fixMacRetinaDisplay()

#clear the screen before we begin Camera Setup mode
clear_screen(win)

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)

# skip this step if running the script in Dummy Mode
if not dummy_mode:
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()

# the Routine "eyelinkSetup" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "start" ---
continueRoutine = True
# update component parameters for each repeat
key_resp_3.keys = []
key_resp_3.rt = []
_key_resp_3_allKeys = []
# keep track of which components have finished
startComponents = [text_2, key_resp_3]
for thisComponent in startComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "start" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_2* updates
    
    # if text_2 is starting this frame...
    if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_2.frameNStart = frameN  # exact frame index
        text_2.tStart = t  # local t and not account for scr refresh
        text_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text_2.started')
        # update status
        text_2.status = STARTED
        text_2.setAutoDraw(True)
    
    # if text_2 is active this frame...
    if text_2.status == STARTED:
        # update params
        pass
    
    # *key_resp_3* updates
    waitOnFlip = False
    
    # if key_resp_3 is starting this frame...
    if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_3.frameNStart = frameN  # exact frame index
        key_resp_3.tStart = t  # local t and not account for scr refresh
        key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'key_resp_3.started')
        # update status
        key_resp_3.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_3.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_3.getKeys(keyList=None, waitRelease=False)
        _key_resp_3_allKeys.extend(theseKeys)
        if len(_key_resp_3_allKeys):
            key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
            key_resp_3.rt = _key_resp_3_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in startComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "start" ---
for thisComponent in startComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if key_resp_3.keys in ['', [], None]:  # No response was made
    key_resp_3.keys = None
thisExp.addData('key_resp_3.keys',key_resp_3.keys)
if key_resp_3.keys != None:  # we had a response
    thisExp.addData('key_resp_3.rt', key_resp_3.rt)
thisExp.nextEntry()
# the Routine "start" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
blocks = data.TrialHandler(nReps=5.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='blocks')
thisExp.addLoop(blocks)  # add the loop to the experiment
thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
if thisBlock != None:
    for paramName in thisBlock:
        exec('{} = thisBlock[paramName]'.format(paramName))

for thisBlock in blocks:
    currentLoop = blocks
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            exec('{} = thisBlock[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "eyelinkStartRecording" ---
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from elStartRecord
    # This Begin Routine tab of the elStartRecord component draws some feedback 
    # graphics (image and a simple shape) on the Host PC, sends a trial start 
    # message to the EDF, performs drift check/drift correct, and starts eye tracker 
    # recording
    
    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()
    
    # put the tracker in the offline mode first
    el_tracker.setOfflineMode()
    
    # send a "TRIALID" message to mark the start of a trial, see Data
    ## Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    #el_tracker.sendMessage('TRIALID %d' % trial_index)
    
    # record_status_message : show some info on the Host PC
    # here we show how many trial has been tested
    status_msg = 'TRIAL number %d' % trial_index
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)
    
    # Start recording
    # arguments: sample_to_file, events_to_file, sample_over_link,
    # event_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1)
    except RuntimeError as error:
        print("ERROR:", error)
        abort_trial()
    
    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)
    
    # determine which eye(s) is/are available
    # 0-left, 1-right, 2-binocular
    eye_used = el_tracker.eyeAvailable()
    if eye_used == 1:
        el_tracker.sendMessage("EYE_USED 1 RIGHT")
    elif eye_used == 0 or eye_used == 2:
        el_tracker.sendMessage("EYE_USED 0 LEFT")
        eye_used = 0
    else:
        print("ERROR: Could not get eye information!")
    
    # keep track of which components have finished
    eyelinkStartRecordingComponents = []
    for thisComponent in eyelinkStartRecordingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyelinkStartRecording" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyelinkStartRecordingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyelinkStartRecording" ---
    for thisComponent in eyelinkStartRecordingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "eyelinkStartRecording" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=5.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    for thisTrial in trials:
        currentLoop = trials
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "forceFixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from elFixWindow
        # This Begin Routine tab of the elFixWindow component sets some 
        # variables that specify the fixation window characteristics (fix window width, 
        # fix window height, ,location, minimum duration) and resets some variables that 
        # help keep track of whether the gaze criteria on the fix window have been met
        
        # the width of the fixation window in pixels
        fix_win_width = 120
        
        #the height of the fixation window in pixels
        fix_win_height = 120
        
        #the X and Y location of the fixation window
        fix_x, fix_y = (scn_width/2.0, scn_height/2.0)
        
        # the minimum consecutive time that gaze must be within the fixation window
        # to move on to the trial stimulus
        minimum_duration = 0.3
        
        # keeps track of the time when the eye most recently entered the fix window
        gaze_start = -1
        
        # keeps track of whether the eye is currently in the fix window region
        in_hit_region = False
        
        # keeps track of whether the gaze criteria have been met (i.e., the trigger fired)
        trigger_fired = False
        
        # keeps track of whether we have marked the onset of the fixation target screen
        sentFixationMessage = 0
        
        el_tracker.sendMessage('TRIALID %d' % trial_index)
        # keep track of which components have finished
        forceFixationComponents = [fixation]
        for thisComponent in forceFixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "forceFixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from elFixWindow
            # This Each Frame tab of the elFixWindow component checks to see whether
            # the fixation window target has onset (and marks it with a message when it
            # does), and grabs the gaze data online and uses it to check whether the
            # fixation window criteria have been satisfied
            
            # Checks whether it is the first frame of the fixation presentation
            if fixation.tStartRefresh is not None and sentFixationMessage == 0:
                # calculate the difference between the current time and the time of the 
                # fixation onset. This offset value will be sent at the beginning of the message
                # and will automatically be subtracted by Data Viewer from the timestamp
                # of the message to position the message at the correct point in time
                # then send a message marking the event
                offsetValue = int(round((core.getTime() - fixation.tStartRefresh)*1000))
                el_tracker.sendMessage('%i FIXATION_ONSET' % offsetValue)
                
                # send some Data Viewer drawing commands so that you can see a representation
                # of the fixation cross in Data Viewer's various visualizations
                # For more information on this, see section "Protocol for EyeLink Data to 
                # Viewer Integration" section of the Data Viewer User Manual (Help -> Contents)
                el_tracker.sendMessage('%i !V CLEAR 128 128 128' % offsetValue)
                el_tracker.sendMessage('%i !V DRAWLINE 255 255 255 %i %i %i %i' % \
                    (offsetValue,scn_width/2 - 25,scn_height/2,scn_width/2 + 25,\
                    scn_height/2))  
                el_tracker.sendMessage('%i !V DRAWLINE 255 255 255 %i %i %i %i' % \
                    (offsetValue,scn_width/2,scn_height/2 - 25,scn_width/2,\
                    scn_height/2 + 25)) 
                    
                #log the fixation onset time (in Display PC time) as a Trial Variable
                fixationTime = fixation.tStartRefresh*1000
                el_tracker.sendMessage('!V TRIAL_VAR fixationTime %i' % fixationTime)
                
                # set this variable to 1 to ensure we don't write the event message/
                # draw command messages again on future frames
                sentFixationMessage = 1
            
            # Gaze checking / Fix Window Section
            # Do we have a sample in the sample buffer?
            # and does it differ from the one we've seen before?
            new_sample = el_tracker.getNewestSample()
            if new_sample is not None and fixation.tStartRefresh is not None:
                
                # check if the new sample has data for the eye
                # currently being tracked; if so, we retrieve the current
                # gaze position and PPD (how many pixels correspond to 1
                # deg of visual angle, at the current gaze position)
                if eye_used == 1 and new_sample.isRightSample():
                    g_x, g_y = new_sample.getRightEye().getGaze()
                if eye_used == 0 and new_sample.isLeftSample():
                    g_x, g_y = new_sample.getLeftEye().getGaze()
            
                # break the while loop if the current gaze position is
                # in a 120 x 120 pixels region around the screen centered
                if fabs(g_x - fix_x) < fix_win_width/2 and fabs(g_y - fix_y) < fix_win_height/2:
                    # record gaze start time
                    if not in_hit_region:
                        if gaze_start == -1:
                            gaze_start = core.getTime()
                            in_hit_region = True
                    # check the gaze duration and fire
                    if in_hit_region:
                        gaze_dur = core.getTime() - gaze_start
                        if gaze_dur > minimum_duration:
                            trigger_fired = True
                else:  # gaze outside the hit region, reset variables
                    in_hit_region = False
                    gaze_start = -1
            
            # if the gaze criteria have been met then send an event marking message 
            # and log the time of occurrence as a trial variable for Data Viewer
            if trigger_fired == True and continueRoutine == True:
                el_tracker.sendMessage('FIX_WINDOW_GAZE_COMPLETED')
                el_tracker.sendMessage('!V TRIAL_VAR gazeOnFixationTime %i' % (core.getTime()*1000))
                continueRoutine = False
            
            # abort the current trial if the tracker is no longer recording
            error = el_tracker.isRecording()
            if error is not pylink.TRIAL_OK:
                el_tracker.sendMessage('tracker_disconnected')
                abort_trial()
            
            # check keyboard events
            for keycode, modifier in event.getKeys(modifiers=True):
            
            # Abort a trial if "ESCAPE" is pressed
                if keycode == 'escape':
                    el_tracker.sendMessage('trial_skipped_by_user')
                    # clear the screen
                    clear_screen(win)
                    # abort trial
                    abort_trial()
                    continueRoutine = False
                    
                # Terminate the task if Ctrl-c
                if keycode == 'c' and (modifier['ctrl'] is True):
                    el_tracker.sendMessage('terminated_by_user')
                    terminate_task()
                    
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in forceFixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "forceFixation" ---
        for thisComponent in forceFixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "forceFixation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "stim" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from elTrial
        # This Begin Routine tab of the elTrial component resets some 
        # variables that are used to keep track of whether certain trial events have
        # happened and sends trial variable messages to the EDF to mark condition
        # information
        
        # these variables keep track of whether the fixation presentation, image 
        # presentation, and trial response have occured yet (0 = no, 1 = yes).
        # They later help us to ensure that each event marking message only gets
        # sent once, at the time of each event
        sentFixationMessage = 0
        sentImageMessage = 0
        
        ## record trial variables to the EDF data file, for details, see Data
        ## Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        #el_tracker.sendMessage('!V TRIAL_VAR condition %s' % condition)
        #el_tracker.sendMessage('!V TRIAL_VAR identifier %s' % identifier)
        #el_tracker.sendMessage('!V TRIAL_VAR image %s' % trialImage)
        ## if sending many messages in a row, add a 1 msec pause between after 
        ## every 5 messages or so
        #time.sleep(0.001)
        #el_tracker.sendMessage('!V TRIAL_VAR corrAns %s' % corrAns)
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        stimComponents = [fixation_2afc, left_img, ISI_load_images, right_img, key_resp]
        for thisComponent in stimComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stim" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from elTrial
            ## This Each Frame tab of the elTrial component handles the marking
            ## of the trial image onset via a message to the EDF, sends additional messages
            ## to allow visualization of trial stimuli in Data Viewer, logs trial variable
            ## information associated with stimulus timing, and checks whether
            ## the eye tracker is still properly recording (and aborts the trial if not)
            #
            ##Check whether it is the first frame of the image presentation
            #if image.tStartRefresh is not None and sentImageMessage == 0:
            #
            #    # calculate the difference between the current time and the time of the 
            #    # image onset. This offset value will be sent at the beginning of the message
            #    # and will automatically be subtracted by Data Viewer from the timestamp
            #    # of the message to position the message at the correct point in time
            #    # then send a message marking the event
            #    offsetValue = int(round((core.getTime()-image.tStartRefresh)*1000))
            #    el_tracker.sendMessage(str(offsetValue) + ' IMAGE_ONSET')  
            #
            #    # send some Data Viewer drawing commands so that you can see the trial image
            #    # in Data Viewer's various visualizations
            #    # For more information on this, see section "Protocol for EyeLink Data to 
            #    # Viewer Integration" section of the Data Viewer User Manual (Help -> Contents)
            #    el_tracker.sendMessage('%i !V CLEAR 128 128 128' % offsetValue)
            #    el_tracker.sendMessage('%i !V IMGLOAD CENTER ../../%s %i %i' % \
            #        (offsetValue,trialImage,scn_width/2,scn_height/2)) 
            #        
            #    #log the fixation onset time (in Display PC time) as a Trial Variable
            #    imageTime = image.tStartRefresh*1000
            #    el_tracker.sendMessage('!V TRIAL_VAR imageTime %i' % imageTime)
            #    
            #    # set this variable to 1 to ensure we don't write the event message/
            #    # draw command messages again on future frames
            #    sentImageMessage = 1
            #
            ## abort the current trial if the tracker is no longer recording
            #error = el_tracker.isRecording()
            #if error is not pylink.TRIAL_OK:
            #    el_tracker.sendMessage('tracker_disconnected')
            #    abort_trial()
            #
            ## check keyboard events
            #for keycode, modifier in event.getKeys(modifiers=True):
            #
            ## Abort a trial if "ESCAPE" is pressed
            #    if keycode == 'escape':
            #        el_tracker.sendMessage('trial_skipped_by_user')
            #        # clear the screen
            #        clear_screen(win)
            #        # abort trial
            #        abort_trial()
            #        continueRoutine = False
            #        
            #    # Terminate the task if Ctrl-c
            #    if keycode == 'c' and (modifier['ctrl'] is True):
            #        el_tracker.sendMessage('terminated_by_user')
            #        terminate_task()
            #        
            
            # *fixation_2afc* updates
            
            # if fixation_2afc is starting this frame...
            if fixation_2afc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_2afc.frameNStart = frameN  # exact frame index
                fixation_2afc.tStart = t  # local t and not account for scr refresh
                fixation_2afc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_2afc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_2afc.started')
                # update status
                fixation_2afc.status = STARTED
                fixation_2afc.setAutoDraw(True)
            
            # if fixation_2afc is active this frame...
            if fixation_2afc.status == STARTED:
                # update params
                pass
            
            # *left_img* updates
            
            # if left_img is starting this frame...
            if left_img.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                left_img.frameNStart = frameN  # exact frame index
                left_img.tStart = t  # local t and not account for scr refresh
                left_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_img.started')
                # update status
                left_img.status = STARTED
                left_img.setAutoDraw(True)
            
            # if left_img is active this frame...
            if left_img.status == STARTED:
                # update params
                pass
            
            # *right_img* updates
            
            # if right_img is starting this frame...
            if right_img.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                right_img.frameNStart = frameN  # exact frame index
                right_img.tStart = t  # local t and not account for scr refresh
                right_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_img.started')
                # update status
                right_img.status = STARTED
                right_img.setAutoDraw(True)
            
            # if right_img is active this frame...
            if right_img.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=None, waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False
            # *ISI_load_images* period
            
            # if ISI_load_images is starting this frame...
            if ISI_load_images.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                ISI_load_images.frameNStart = frameN  # exact frame index
                ISI_load_images.tStart = t  # local t and not account for scr refresh
                ISI_load_images.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ISI_load_images, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('ISI_load_images.started', t)
                # update status
                ISI_load_images.status = STARTED
                ISI_load_images.start(0.5)
            elif ISI_load_images.status == STARTED:  # one frame should pass before updating params and completing
                ISI_load_images.complete()  # finish the static period
                ISI_load_images.tStop = ISI_load_images.tStart + 0.5  # record stop time
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in stimComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stim" ---
        for thisComponent in stimComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from elTrial
        # send a 'TRIAL_RESULT' message to mark the end of trial, see Data
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        el_tracker.sendMessage('TRIAL_RESULT %d' % 0)
        
        # update the trial counter for the next trial
        trial_index = trial_index + 1
        
        
        
        ## This End Routine tab of the elTrial component handles the marking
        ## of response events via messages to the EDF and sends additional messages
        ## to log data about the response as trial variables.  It also marks the end
        ## of the trial
        #
        ## if a key was presssed, calculate the difference between the current time 
        ## and the time of the key press onset. 
        ## This offset value will be sent at the beginning of the message
        ## and will automatically be subtracted by Data Viewer from the timestamp
        ## of the message to position the message at the correct point in time
        ## then send a message marking the event
        #if not isinstance(resp.rt,list):
        #    offsetValue = int(round((core.getTime() - \
        #        (image.tStartRefresh + resp.rt))*1000))
        #    el_tracker.sendMessage('%i KEY_PRESSED' % offsetValue)
        #    el_tracker.sendMessage('!V TRIAL_VAR accuracy %i' % resp.corr)
        #    el_tracker.sendMessage('!V TRIAL_VAR keyPressed %s' % resp.keys)
        #    # if sending many messages in a row, add a 1 msec pause between after 
        #    # every 5 messages or so
        #    time.sleep(0.001)
        #    el_tracker.sendMessage('!V TRIAL_VAR RT %i' % int(round(resp.rt * 1000)))
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
        # the Routine "stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "eyelinkStopRecording" ---
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    eyelinkStopRecordingComponents = []
    for thisComponent in eyelinkStopRecordingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyelinkStopRecording" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyelinkStopRecordingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyelinkStopRecording" ---
    for thisComponent in eyelinkStopRecordingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # Run 'End Routine' code from elStopRecord
    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    el_tracker.stopRecording()
        
    
    # the Routine "eyelinkStopRecording" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 5.0 repeats of 'blocks'


# --- Prepare to start Routine "endTask" ---
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
endTaskComponents = [text]
for thisComponent in endTaskComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "endTask" ---
routineForceEnded = not continueRoutine
while continueRoutine and routineTimer.getTime() < 1.0:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text* updates
    
    # if text is starting this frame...
    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text.started')
        # update status
        text.status = STARTED
        text.setAutoDraw(True)
    
    # if text is active this frame...
    if text.status == STARTED:
        # update params
        pass
    
    # if text is stopping this frame...
    if text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text.tStartRefresh + 1.0-frameTolerance:
            # keep track of stop time/frame for later
            text.tStop = t  # not accounting for scr refresh
            text.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.stopped')
            # update status
            text.status = FINISHED
            text.setAutoDraw(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in endTaskComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "endTask" ---
for thisComponent in endTaskComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
if routineForceEnded:
    routineTimer.reset()
else:
    routineTimer.addTime(-1.000000)
# Run 'End Experiment' code from elConnect
# This End Experiment tab of the elConnect component calls the 
# terminate_task helper function to get the EDF file and close the connection
# to the Host PC

# Disconnect, download the EDF file, then terminate the task
terminate_task()

# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
