#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.10),
    on Tue Mar 15 22:00:54 2022
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.10'
expName = 'block_cue_test'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
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
    originPath='/Users/brandonchen93/Downloads/block_cue_test_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1440, 900], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='deg')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "Block_Cue"
Block_CueClock = core.Clock()
stim_cats = ['AM', 'AN', 'FF', 'FM', 'HB', 'HH', 'OH', 'ON']


#Get path for real and scrambled images and generate lists of images to be used 
real_imgs = ['images/%s%s.npy' %(stim_cats[i], x) for i in range(len(stim_cats)) for x in range(1,3)]

scram_imgs = ['images/%s%sscram.npy' %(stim_cats[i], x) for i in range(len(stim_cats)) for x in range(1,3)]

place_right = visual.Rect(
    win=win, name='place_right',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4,0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
place_left = visual.Rect(
    win=win, name='place_left',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4,0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)
long_back_R = visual.Rect(
    win=win, name='long_back_R',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4,0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_R = visual.Rect(
    win=win, name='hi_back_R',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4,0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
hi_back_L = visual.Rect(
    win=win, name='hi_back_L',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4,0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
long_back_L = visual.Rect(
    win=win, name='long_back_L',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(-4,0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-6.0, interpolate=True)
probability_left = visual.ImageStim(
    win=win,
    name='probability_left', 
    image='sin', mask=None,
    ori=0, pos=(-4,0), size=(4, 4),
    color=[1,1,1], colorSpace='rgb', opacity=.7,
    flipHoriz=False, flipVert=True,
    texRes=512, interpolate=True, depth=-7.0)
probability_right = visual.ImageStim(
    win=win,
    name='probability_right', 
    image='sin', mask=None,
    ori=0, pos=(4,0), size=(4, 4),
    color=[1,1,1], colorSpace='rgb', opacity=.7,
    flipHoriz=False, flipVert=True,
    texRes=512, interpolate=True, depth=-8.0)

# Initialize components for Routine "trial"
trialClock = core.Clock()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "Block_Cue"-------
continueRoutine = True
routineTimer.add(3.000000)
# update component parameters for each repeat
prob_right = [-1,-1,-1]
prob_left = [1,1,1]

left_img = np.load(np.random.choice(real_imgs))

right_img = np.load(np.random.choice(scram_imgs))


place_right.setLineColor(prob_right)
place_left.setLineColor(prob_left)
probability_left.setImage(left_img)
probability_right.setImage(right_img)
# keep track of which components have finished
Block_CueComponents = [place_right, place_left, long_back_R, hi_back_R, hi_back_L, long_back_L, probability_left, probability_right]
for thisComponent in Block_CueComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Block_CueClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Block_Cue"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = Block_CueClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Block_CueClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *place_right* updates
    if place_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        place_right.frameNStart = frameN  # exact frame index
        place_right.tStart = t  # local t and not account for scr refresh
        place_right.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(place_right, 'tStartRefresh')  # time at next scr refresh
        place_right.setAutoDraw(True)
    if place_right.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > place_right.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            place_right.tStop = t  # not accounting for scr refresh
            place_right.frameNStop = frameN  # exact frame index
            win.timeOnFlip(place_right, 'tStopRefresh')  # time at next scr refresh
            place_right.setAutoDraw(False)
    
    # *place_left* updates
    if place_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        place_left.frameNStart = frameN  # exact frame index
        place_left.tStart = t  # local t and not account for scr refresh
        place_left.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(place_left, 'tStartRefresh')  # time at next scr refresh
        place_left.setAutoDraw(True)
    if place_left.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > place_left.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            place_left.tStop = t  # not accounting for scr refresh
            place_left.frameNStop = frameN  # exact frame index
            win.timeOnFlip(place_left, 'tStopRefresh')  # time at next scr refresh
            place_left.setAutoDraw(False)
    
    # *long_back_R* updates
    if long_back_R.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        long_back_R.frameNStart = frameN  # exact frame index
        long_back_R.tStart = t  # local t and not account for scr refresh
        long_back_R.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(long_back_R, 'tStartRefresh')  # time at next scr refresh
        long_back_R.setAutoDraw(True)
    if long_back_R.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > long_back_R.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            long_back_R.tStop = t  # not accounting for scr refresh
            long_back_R.frameNStop = frameN  # exact frame index
            win.timeOnFlip(long_back_R, 'tStopRefresh')  # time at next scr refresh
            long_back_R.setAutoDraw(False)
    
    # *hi_back_R* updates
    if hi_back_R.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        hi_back_R.frameNStart = frameN  # exact frame index
        hi_back_R.tStart = t  # local t and not account for scr refresh
        hi_back_R.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(hi_back_R, 'tStartRefresh')  # time at next scr refresh
        hi_back_R.setAutoDraw(True)
    if hi_back_R.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > hi_back_R.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            hi_back_R.tStop = t  # not accounting for scr refresh
            hi_back_R.frameNStop = frameN  # exact frame index
            win.timeOnFlip(hi_back_R, 'tStopRefresh')  # time at next scr refresh
            hi_back_R.setAutoDraw(False)
    
    # *hi_back_L* updates
    if hi_back_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        hi_back_L.frameNStart = frameN  # exact frame index
        hi_back_L.tStart = t  # local t and not account for scr refresh
        hi_back_L.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(hi_back_L, 'tStartRefresh')  # time at next scr refresh
        hi_back_L.setAutoDraw(True)
    if hi_back_L.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > hi_back_L.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            hi_back_L.tStop = t  # not accounting for scr refresh
            hi_back_L.frameNStop = frameN  # exact frame index
            win.timeOnFlip(hi_back_L, 'tStopRefresh')  # time at next scr refresh
            hi_back_L.setAutoDraw(False)
    
    # *long_back_L* updates
    if long_back_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        long_back_L.frameNStart = frameN  # exact frame index
        long_back_L.tStart = t  # local t and not account for scr refresh
        long_back_L.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(long_back_L, 'tStartRefresh')  # time at next scr refresh
        long_back_L.setAutoDraw(True)
    if long_back_L.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > long_back_L.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            long_back_L.tStop = t  # not accounting for scr refresh
            long_back_L.frameNStop = frameN  # exact frame index
            win.timeOnFlip(long_back_L, 'tStopRefresh')  # time at next scr refresh
            long_back_L.setAutoDraw(False)
    
    # *probability_left* updates
    if probability_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        probability_left.frameNStart = frameN  # exact frame index
        probability_left.tStart = t  # local t and not account for scr refresh
        probability_left.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(probability_left, 'tStartRefresh')  # time at next scr refresh
        probability_left.setAutoDraw(True)
    if probability_left.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > probability_left.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            probability_left.tStop = t  # not accounting for scr refresh
            probability_left.frameNStop = frameN  # exact frame index
            win.timeOnFlip(probability_left, 'tStopRefresh')  # time at next scr refresh
            probability_left.setAutoDraw(False)
    
    # *probability_right* updates
    if probability_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        probability_right.frameNStart = frameN  # exact frame index
        probability_right.tStart = t  # local t and not account for scr refresh
        probability_right.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(probability_right, 'tStartRefresh')  # time at next scr refresh
        probability_right.setAutoDraw(True)
    if probability_right.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > probability_right.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            probability_right.tStop = t  # not accounting for scr refresh
            probability_right.frameNStop = frameN  # exact frame index
            win.timeOnFlip(probability_right, 'tStopRefresh')  # time at next scr refresh
            probability_right.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Block_CueComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Block_Cue"-------
for thisComponent in Block_CueComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('place_right.started', place_right.tStartRefresh)
thisExp.addData('place_right.stopped', place_right.tStopRefresh)
thisExp.addData('place_left.started', place_left.tStartRefresh)
thisExp.addData('place_left.stopped', place_left.tStopRefresh)
thisExp.addData('long_back_R.started', long_back_R.tStartRefresh)
thisExp.addData('long_back_R.stopped', long_back_R.tStopRefresh)
thisExp.addData('hi_back_R.started', hi_back_R.tStartRefresh)
thisExp.addData('hi_back_R.stopped', hi_back_R.tStopRefresh)
thisExp.addData('hi_back_L.started', hi_back_L.tStartRefresh)
thisExp.addData('hi_back_L.stopped', hi_back_L.tStopRefresh)
thisExp.addData('long_back_L.started', long_back_L.tStartRefresh)
thisExp.addData('long_back_L.stopped', long_back_L.tStopRefresh)
thisExp.addData('probability_left.started', probability_left.tStartRefresh)
thisExp.addData('probability_left.stopped', probability_left.tStopRefresh)
thisExp.addData('probability_right.started', probability_right.tStartRefresh)
thisExp.addData('probability_right.stopped', probability_right.tStopRefresh)

# ------Prepare to start Routine "trial"-------
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
trialComponents = []
for thisComponent in trialComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "trial"-------
while continueRoutine:
    # get current time
    t = trialClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=trialClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "trial"-------
for thisComponent in trialComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "trial" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
