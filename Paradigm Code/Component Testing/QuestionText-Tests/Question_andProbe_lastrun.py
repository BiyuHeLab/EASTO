#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.10),
    on April 27, 2022, at 15:12
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
expName = 'Question_andProbe'  # from the Builder filename that created this script
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
    originPath='\\\\homedir-cifs.nyumc.org\\bc1693\\Personal\\EASTO\\Paradigm\\Paradigms\\Component Testing\\Question_Testing\\Question_andProbe_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1680, 1050], fullscr=True, screen=0, 
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

# Initialize components for Routine "q_cat"
q_catClock = core.Clock()
q_category = visual.TextStim(win=win, name='q_category',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.1, wrapWidth=500, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
cat_resp = keyboard.Keyboard()
probe_cue_cat = visual.ImageStim(
    win=win,
    name='probe_cue_cat', 
    image='sin', mask=None,
    ori=0, pos=(0,0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# Initialize components for Routine "q_rec"
q_recClock = core.Clock()
rec_question  = (u'Meaningful visual experience: \n\n\n\n\n '
            u'{Y:\u00a0^15} {N:\u00a0^15}''\n\n\n '
            u'{b2:\u00a0^15} {b3:\u00a0^15}'.format(Y = u'Yes',
                                               N = u'No',
                                               b2 = u'2',
                                               b3 = u'3 '))
q_recognition = visual.TextStim(win=win, name='q_recognition',
    text=rec_question,
    font='Courier New',
    units='norm', pos=(0, 0), height=0.1, wrapWidth=500, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
rec_resp = keyboard.Keyboard()
probe_cue_rec = visual.ImageStim(
    win=win,
    name='probe_cue_rec', 
    image='sin', mask=None,
    ori=0, pos=(0,0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "q_cat"-------
continueRoutine = True
routineTimer.add(3.000000)
# update component parameters for each repeat
## Write Code to set cue to be valid or not valid 
#based on conditions file 

probe_loc = 'L'
if probe_loc == 'L':
    probe_cue_img = 'cues/cue_left.png'
elif probe_loc == 'R':
    probe_cue_img = 'cues/cue_right.png'

# Randomize Order of FAOH 
#Initialize cat components


cats = ['Face', 'Animal', 'Object', 'House'] 
shuffle(cats)
cat_question = (
                u'Image Category:\n\n\n\n\n'
                u'{cats1:\u00a0^9} {cats2:\u00a0^9} {cats3:\u00a0^9} {cats4:\u00a0^9}\n\n\n'
                u'{b1:\u00a0^9} {b2:\u00a0^9} {b3:\u00a0^9} {b4:\u00a0^9}'.format(cats1 = cats[0].encode().decode('utf-8'), 
                                                           cats2 = cats[1].encode().decode('utf-8'), 
                                                           cats3 = cats[2].encode().decode('utf-8'), 
                                                           cats4 = cats[3].encode().decode('utf-8'),
                                                           b1 = u'1',b2 = u'2',b3 = u'3',b4 = u'4'
                                                            ))

q_category.setText(cat_question)
cat_resp.keys = []
cat_resp.rt = []
_cat_resp_allKeys = []
probe_cue_cat.setImage(probe_cue_img)
# keep track of which components have finished
q_catComponents = [q_category, cat_resp, probe_cue_cat]
for thisComponent in q_catComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
q_catClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "q_cat"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = q_catClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=q_catClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    if frameN == 2: 
        win.getMovieFrame()
    
    # *q_category* updates
    if q_category.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        q_category.frameNStart = frameN  # exact frame index
        q_category.tStart = t  # local t and not account for scr refresh
        q_category.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(q_category, 'tStartRefresh')  # time at next scr refresh
        q_category.setAutoDraw(True)
    if q_category.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > q_category.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            q_category.tStop = t  # not accounting for scr refresh
            q_category.frameNStop = frameN  # exact frame index
            win.timeOnFlip(q_category, 'tStopRefresh')  # time at next scr refresh
            q_category.setAutoDraw(False)
    
    # *cat_resp* updates
    waitOnFlip = False
    if cat_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        cat_resp.frameNStart = frameN  # exact frame index
        cat_resp.tStart = t  # local t and not account for scr refresh
        cat_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(cat_resp, 'tStartRefresh')  # time at next scr refresh
        cat_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(cat_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(cat_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if cat_resp.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > cat_resp.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            cat_resp.tStop = t  # not accounting for scr refresh
            cat_resp.frameNStop = frameN  # exact frame index
            win.timeOnFlip(cat_resp, 'tStopRefresh')  # time at next scr refresh
            cat_resp.status = FINISHED
    if cat_resp.status == STARTED and not waitOnFlip:
        theseKeys = cat_resp.getKeys(keyList=['6', '7', '8', '9'], waitRelease=False)
        _cat_resp_allKeys.extend(theseKeys)
        if len(_cat_resp_allKeys):
            cat_resp.keys = _cat_resp_allKeys[-1].name  # just the last key pressed
            cat_resp.rt = _cat_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *probe_cue_cat* updates
    if probe_cue_cat.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        probe_cue_cat.frameNStart = frameN  # exact frame index
        probe_cue_cat.tStart = t  # local t and not account for scr refresh
        probe_cue_cat.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(probe_cue_cat, 'tStartRefresh')  # time at next scr refresh
        probe_cue_cat.setAutoDraw(True)
    if probe_cue_cat.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > probe_cue_cat.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            probe_cue_cat.tStop = t  # not accounting for scr refresh
            probe_cue_cat.frameNStop = frameN  # exact frame index
            win.timeOnFlip(probe_cue_cat, 'tStopRefresh')  # time at next scr refresh
            probe_cue_cat.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in q_catComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "q_cat"-------
for thisComponent in q_catComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
win.saveMovieFrames('categoryQ.png')
## Map buttons to shuffled cats, assumed read L-R
#
#buttons = ['6','7','8','9']
#categories = [c[0] for c in cats]
#
#button_map = dict(zip(categories,buttons))
#cat_map  = dict(zip(buttons,categories))
##
#if probe_loc == 'L':
#    corAns = button_map[target_left[10]] # Index first letter after " images/" in target
#    
#elif probe_loc == 'R':
#    corAns = button_map[target_right[10]] # Index first letter after " images/" in target
#
#correct_category = cat_map[corAns]
#trials.addData('objective_category', cat_map[corAns])
#
#if cat_resp.keys:
#    category_response = cat_map[cat_resp.keys[-1]]
#    trials.addData('category_response', category_response)
#
#else: #If no response
#    category_response = None
#    trials.addData('category_response', category_response)
#
#
##Calculate categorization accuracy
#if correct_category == category_response:
#    trials.addData('category_correct', 1)
#    category_correct_count += 1 
#else:
#    trials.addData('category_correct', 0)
#
#
#
#
#    
#
thisExp.addData('q_category.started', q_category.tStartRefresh)
thisExp.addData('q_category.stopped', q_category.tStopRefresh)
# check responses
if cat_resp.keys in ['', [], None]:  # No response was made
    cat_resp.keys = None
thisExp.addData('cat_resp.keys',cat_resp.keys)
if cat_resp.keys != None:  # we had a response
    thisExp.addData('cat_resp.rt', cat_resp.rt)
thisExp.addData('cat_resp.started', cat_resp.tStartRefresh)
thisExp.addData('cat_resp.stopped', cat_resp.tStopRefresh)
thisExp.nextEntry()
thisExp.addData('probe_cue_cat.started', probe_cue_cat.tStartRefresh)
thisExp.addData('probe_cue_cat.stopped', probe_cue_cat.tStopRefresh)

# ------Prepare to start Routine "q_rec"-------
continueRoutine = True
routineTimer.add(3.000000)
# update component parameters for each repeat
## Write Code to set cue to be valid or not valid 
#based on conditions file 
probe_loc = 'L'
if probe_loc == 'L':
    probe_cue_img_2 = 'cues/cue_left.png'
    
elif probe_loc == 'R':
    probe_cue_img_2 = 'cues/cue_right.png'

rec_resp.keys = []
rec_resp.rt = []
_rec_resp_allKeys = []
probe_cue_rec.setImage(probe_cue_img_2)
# keep track of which components have finished
q_recComponents = [q_recognition, rec_resp, probe_cue_rec]
for thisComponent in q_recComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
q_recClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "q_rec"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = q_recClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=q_recClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    if frameN == 2: 
        win.getMovieFrame()
    
    # *q_recognition* updates
    if q_recognition.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        q_recognition.frameNStart = frameN  # exact frame index
        q_recognition.tStart = t  # local t and not account for scr refresh
        q_recognition.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(q_recognition, 'tStartRefresh')  # time at next scr refresh
        q_recognition.setAutoDraw(True)
    if q_recognition.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > q_recognition.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            q_recognition.tStop = t  # not accounting for scr refresh
            q_recognition.frameNStop = frameN  # exact frame index
            win.timeOnFlip(q_recognition, 'tStopRefresh')  # time at next scr refresh
            q_recognition.setAutoDraw(False)
    
    # *rec_resp* updates
    waitOnFlip = False
    if rec_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        rec_resp.frameNStart = frameN  # exact frame index
        rec_resp.tStart = t  # local t and not account for scr refresh
        rec_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rec_resp, 'tStartRefresh')  # time at next scr refresh
        rec_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(rec_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(rec_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if rec_resp.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > rec_resp.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            rec_resp.tStop = t  # not accounting for scr refresh
            rec_resp.frameNStop = frameN  # exact frame index
            win.timeOnFlip(rec_resp, 'tStopRefresh')  # time at next scr refresh
            rec_resp.status = FINISHED
    if rec_resp.status == STARTED and not waitOnFlip:
        theseKeys = rec_resp.getKeys(keyList=['7', '8'], waitRelease=False)
        _rec_resp_allKeys.extend(theseKeys)
        if len(_rec_resp_allKeys):
            rec_resp.keys = _rec_resp_allKeys[-1].name  # just the last key pressed
            rec_resp.rt = _rec_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *probe_cue_rec* updates
    if probe_cue_rec.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        probe_cue_rec.frameNStart = frameN  # exact frame index
        probe_cue_rec.tStart = t  # local t and not account for scr refresh
        probe_cue_rec.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(probe_cue_rec, 'tStartRefresh')  # time at next scr refresh
        probe_cue_rec.setAutoDraw(True)
    if probe_cue_rec.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > probe_cue_rec.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            probe_cue_rec.tStop = t  # not accounting for scr refresh
            probe_cue_rec.frameNStop = frameN  # exact frame index
            win.timeOnFlip(probe_cue_rec, 'tStopRefresh')  # time at next scr refresh
            probe_cue_rec.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in q_recComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "q_rec"-------
for thisComponent in q_recComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
win.saveMovieFrames('recQ.png')
thisExp.addData('q_recognition.started', q_recognition.tStartRefresh)
thisExp.addData('q_recognition.stopped', q_recognition.tStopRefresh)
# check responses
if rec_resp.keys in ['', [], None]:  # No response was made
    rec_resp.keys = None
thisExp.addData('rec_resp.keys',rec_resp.keys)
if rec_resp.keys != None:  # we had a response
    thisExp.addData('rec_resp.rt', rec_resp.rt)
thisExp.addData('rec_resp.started', rec_resp.tStartRefresh)
thisExp.addData('rec_resp.stopped', rec_resp.tStopRefresh)
thisExp.nextEntry()
thisExp.addData('probe_cue_rec.started', probe_cue_rec.tStartRefresh)
thisExp.addData('probe_cue_rec.stopped', probe_cue_rec.tStopRefresh)

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
