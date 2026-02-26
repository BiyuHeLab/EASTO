#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.10),
    on Wed Jun  1 19:07:29 2022
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

#Import necessary packages 
import pandas as pd
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
    
import EASTO_funcs as ea 


import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


import math


#Make Filter store in params dict 
# Make param dict 
params = {'imsize' : [300,300],
      'imbgrd' : 127,
      'categories' : ['AM', 'AN', 'FF', 'FM', 'HB', 'HH', 'OH','ON']}
      
params['filter'] = ea.make_filter(params) 


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.10'
expName = 'SEA_Quest'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001', 'QuestAlt': False, 'EyeTrack': False, 'HPColor': 'black', 'numTrials': '40'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s_%s' % (expInfo['participant'],expInfo['session'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/brandonchen93/EASTO/Paradigm/Paradigms/Master/Spatial/Quest/SEA_Quest_lastrun.py',
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
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='EEG', color=[0,0,0], colorSpace='rgb',
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

# Initialize components for Routine "Quest_Type"
Quest_TypeClock = core.Clock()
if expInfo['QuestAlt'] == True:
    #Minmize FullScreen so GUI appears 
    win.winHandle.minimize()
    win.winHandle.set_fullscreen(False)
    win.flip()
    
    # Read Q_Tracks Excel (all images) and convert to dictionary 
    quest_tracks_xls = os.getcwd() + os.sep + 'Q_tracks.xlsx'
    q_df = pd.read_excel(quest_tracks_xls)
    q_dict = q_df.to_dict('record')
    
    #Read in qMini excel as dataframe
    qMini_tracks_xls = os.getcwd() + os.sep + 'QMini_tracks.xlsx'
    qMini_df = pd.read_excel(qMini_tracks_xls)

    stim_cats = ['AM', 'AN', 'FF', 'FM', 'HB', 'HH', 'OH', 'ON']

    #Get list of Images into dict for UI
    imgs_list = ['%s%s' %(stim_cats[i], x) for i in range(len(stim_cats)) for x in range(1,3)]
    imgs_dict = {'%s%s' %(stim_cats[i], x): False  for i in range(len(stim_cats)) for x in range(1,3)}
    imgs_dict['qMini'] = False
    dlg = gui.DlgFromDict(dictionary=imgs_dict, sortKeys=False, title='Choose Images/Thresholding')
    
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    
    # If qMini is selected, load quest_estimate. 
    if imgs_dict['qMini']:
        #Import QUEST estimates for subject 
        quest_file = gui.fileOpenDlg('.')
        if not quest_file:
            #    core.quit()
            print('Using Arbitrary Contrasts for Tests')
            #Gets initiated with target stim loading
            quest_estimates = []
        else:
            with open(quest_file[0], 'rb') as handle:
                quest_estimates = pickle.load(handle)
    
    # Get Which Images were selected from UI, will ignore qMini since it is not in imgs_list
    q_select = [cond for cond, img in zip(q_dict, imgs_list) if imgs_dict[img]]
    # Rewrite as dataframe (Probably do not need to turn to a dict in the first place)
    q_alt_df = pd.DataFrame.from_records(q_select)

    if imgs_dict['qMini']:
        q_alt_df = pd.concat([q_alt_df, qMini_df])
    #Save as CSV
    q_alt_df.to_csv('Q_AltTracks.csv', index = False)
        
    q_conditions = 'Q_AltTracks.csv'
    
    #After Making Selection put the screen back on
    win.winHandle.maximize()
    win.winHandle.set_fullscreen(True)
    win.winHandle.activate()
    win.flip()  
else:
    q_conditions = 'Q_tracks.xlsx'
    


# Initialize components for Routine "gen_instr"
gen_instrClock = core.Clock()
instrText_3 = visual.TextStim(win=win, name='instrText_3',
    text='In this part of the experiment you will be asked to make judgements about simple images\n\nYou will see images of faces, animals, objects, and houses presented on either the left or the right side of the screen.\n\nYou will be asked to indicate the image category and to report your visual experience of the image presented.\n\nPress any button to continue\n',
    font='Courier New',
    units='norm', pos=[0, 0], height=.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_3 = keyboard.Keyboard()

# Initialize components for Routine "q_instr"
q_instrClock = core.Clock()
Instr_Img = visual.TextStim(win=win, name='Instr_Img',
    text='Every trial an image will appear at the left and right the screen. \n\nA given image may become harder or easier to see over subsequent trials. \n\nIf you do not see anything, make a genuine guess on the category of the image. \n\nNo matter your visual experience, please answer every question.\n\n\n\nPress any button to continue',
    font='Courier New',
    units='norm', pos=[0, 0], height=.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_4 = keyboard.Keyboard()

# Initialize components for Routine "fixation_2"
fixation_2Clock = core.Clock()
fix_instr = visual.TextStim(win=win, name='fix_instr',
    text='As a reminder: remember to always keep your eyes on this cross when you see it\n\n\n\n\n\n\n\n\n\n\n\nPress any button to continue',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
Fix_Example = visual.ShapeStim(
    win=win, name='Fix_Example', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=.7, depth=-1.0, interpolate=True)
endInstructions_5 = keyboard.Keyboard()

# Initialize components for Routine "begin_exp"
begin_expClock = core.Clock()
beginexp = visual.TextStim(win=win, name='beginexp',
    text='If you do not have any questions or concerns:\n\n\n\n\n\npress any button to begin the experiment',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_3 = keyboard.Keyboard()

# Initialize components for Routine "eyetrack_calib"
eyetrack_calibClock = core.Clock()
#%% Eyelink Init Code Comparison 
# SETP 2: open a connection to the tracker
# replace the IP address with None will open a simulated connection


if expInfo['EyeTrack']: 
    tk = pylink.EyeLink('100.1.1.1')
    
    scnWidth, scnHeight = win.size

    # STEP 3: Open an EDF data file on the Host and write a file header
    # The file name should not exceeds 8 characters
    dataFileName = expInfo['subject'] + '_' + expInfo['session']+ 'Q.EDF'
    tk.openDataFile(dataFileName)
    # add personalized data file header (preamble text)
    tk.sendCommand("add_file_preamble_text 'SEA QUESTExp'") 

else: 
    pylink.EyeLink.dummy_open
    tk = pylink.EyeLink(None)




# Initialize components for Routine "block_break"
block_breakClock = core.Clock()
blocks_count = 0
num_exemps = 16
num_trials = int(expInfo['numTrials'])
#Initialize correct category counter
category_correct_count = 0
trial_counter = 0
end_break = keyboard.Keyboard()
break_text = visual.TextStim(win=win, name='break_text',
    text='Great job! Take a break to rest your eyes if you need to.\n\n\n\n\n\n\n\n\nPress any button to resume when you are ready',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
block_count = visual.TextStim(win=win, name='block_count',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);
cat_accuracy = visual.TextStim(win=win, name='cat_accuracy',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-4.0);
countdown_timer = visual.TextStim(win=win, name='countdown_timer',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-5.0);

# Initialize components for Routine "eyetrack_recalib"
eyetrack_recalibClock = core.Clock()

# Initialize components for Routine "init_fix"
init_fixClock = core.Clock()
if expInfo['HPColor'] == 'black':
    place_real = [-1.0,-1.0,-1.0]
    place_scram = [1.0,1.0,1.0]
elif expInfo['HPColor'] == 'white':
    place_real = [1.0,1.0,1.0]
    place_scram = [-1.0,-1.0,-1.0]
    
### Make Stim list for qMini 
if imgs_dict['qMini']:
    stim_cats = ['AM', 'AN', 'FF', 'FM', 'HB', 'HH', 'OH', 'ON']

    real_imgs = ['../images/%s%s.npy' %(stim_cats[i], x) for i in range(len(stim_cats)) for x in range(1,3)]

#For qMini, run 80 trials (2 staircases of 40 trials), thus 5 repeats per image
    qMiniStimulusList = list(np.repeat(real_imgs, 5))
    shuffle(qMiniStimulusList)

#Define possible ITI Vals & number of trials in a block
numTotalTrials = num_exemps * num_trials #Number of exemplars (staircases) * num trials per staircase
ITI_vals = np.array([.75, 1])  #Reduced from 1, 1.5 to account for feedback time
#Sample from exponential distribution 
lam = 1.33 # This is set s.t. 1/lambda = value of first interval. i.e exp dist mean is T1 
xdist = lam * np.exp(-lam * ITI_vals)

#Normalize and generate counts 
xdist_norm = numTotalTrials * (xdist/ sum(xdist));
ITI_counts = np.round(xdist_norm).astype(int)

#Set ITI for each trial for one block
#And randomly shuffle 
ITI_list = np.random.permutation(np.repeat(ITI_vals, ITI_counts))

ITI_list = list(ITI_list)

place_left_3 = visual.Rect(
    win=win, name='place_left_3',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
place_right_3 = visual.Rect(
    win=win, name='place_right_3',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)
long_back_R_3 = visual.Rect(
    win=win, name='long_back_R_3',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_R_3 = visual.Rect(
    win=win, name='hi_back_R_3',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
hi_back_L_3 = visual.Rect(
    win=win, name='hi_back_L_3',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
long_back_L_3 = visual.Rect(
    win=win, name='long_back_L_3',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-6.0, interpolate=True)
pre_cue_fix_2 = visual.ShapeStim(
    win=win, name='pre_cue_fix_2', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=0.7, depth=-7.0, interpolate=True)

# Initialize components for Routine "stim"
stimClock = core.Clock()
place_left_4 = visual.Rect(
    win=win, name='place_left_4',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
place_right_4 = visual.Rect(
    win=win, name='place_right_4',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)
long_back_R_4 = visual.Rect(
    win=win, name='long_back_R_4',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_R_4 = visual.Rect(
    win=win, name='hi_back_R_4',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
hi_back_L_4 = visual.Rect(
    win=win, name='hi_back_L_4',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
long_back_L_4 = visual.Rect(
    win=win, name='long_back_L_4',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-6.0, interpolate=True)
stim_fix_2 = visual.ShapeStim(
    win=win, name='stim_fix_2', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=0.7, depth=-7.0, interpolate=True)
stim_right_2 = visual.ImageStim(
    win=win,
    name='stim_right_2', 
    image='sin', mask=None,
    ori=0, pos=(4, 0), size=(4, 4),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=True,
    texRes=512, interpolate=True, depth=-8.0)
stim_left_2 = visual.ImageStim(
    win=win,
    name='stim_left_2', 
    image='sin', mask=None,
    ori=0, pos=(-4, 0), size=(4, 4),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=True,
    texRes=512, interpolate=True, depth=-9.0)

# Initialize components for Routine "blank_post"
blank_postClock = core.Clock()
post_stim_fix = visual.ShapeStim(
    win=win, name='post_stim_fix', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=0.7, depth=0.0, interpolate=True)

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

# Initialize components for Routine "thanks"
thanksClock = core.Clock()
thanksMsg = visual.TextStim(win=win, name='thanksMsg',
    text='Thank you! \nYou can relax now\n\n\n\n\n\nThe next part of the experiment will begin shortly. It may take a few minutes.',
    font='Courier New',
    units='norm', pos=[0, 0], height=0.1, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
cat_accuracy_2 = visual.TextStim(win=win, name='cat_accuracy_2',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
EndExp = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "Quest_Type"-------
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
Quest_TypeComponents = []
for thisComponent in Quest_TypeComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Quest_TypeClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Quest_Type"-------
while continueRoutine:
    # get current time
    t = Quest_TypeClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Quest_TypeClock)
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
    for thisComponent in Quest_TypeComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Quest_Type"-------
for thisComponent in Quest_TypeComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "Quest_Type" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "gen_instr"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_3.keys = []
endInstructions_3.rt = []
_endInstructions_3_allKeys = []
# keep track of which components have finished
gen_instrComponents = [instrText_3, endInstructions_3]
for thisComponent in gen_instrComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
gen_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "gen_instr"-------
while continueRoutine:
    # get current time
    t = gen_instrClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=gen_instrClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instrText_3* updates
    if instrText_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instrText_3.frameNStart = frameN  # exact frame index
        instrText_3.tStart = t  # local t and not account for scr refresh
        instrText_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instrText_3, 'tStartRefresh')  # time at next scr refresh
        instrText_3.setAutoDraw(True)
    
    # *endInstructions_3* updates
    waitOnFlip = False
    if endInstructions_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endInstructions_3.frameNStart = frameN  # exact frame index
        endInstructions_3.tStart = t  # local t and not account for scr refresh
        endInstructions_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endInstructions_3, 'tStartRefresh')  # time at next scr refresh
        endInstructions_3.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endInstructions_3.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endInstructions_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endInstructions_3.status == STARTED and not waitOnFlip:
        theseKeys = endInstructions_3.getKeys(keyList=None, waitRelease=False)
        _endInstructions_3_allKeys.extend(theseKeys)
        if len(_endInstructions_3_allKeys):
            endInstructions_3.keys = _endInstructions_3_allKeys[-1].name  # just the last key pressed
            endInstructions_3.rt = _endInstructions_3_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in gen_instrComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "gen_instr"-------
for thisComponent in gen_instrComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('instrText_3.started', instrText_3.tStartRefresh)
thisExp.addData('instrText_3.stopped', instrText_3.tStopRefresh)
# check responses
if endInstructions_3.keys in ['', [], None]:  # No response was made
    endInstructions_3.keys = None
thisExp.addData('endInstructions_3.keys',endInstructions_3.keys)
if endInstructions_3.keys != None:  # we had a response
    thisExp.addData('endInstructions_3.rt', endInstructions_3.rt)
thisExp.addData('endInstructions_3.started', endInstructions_3.tStartRefresh)
thisExp.addData('endInstructions_3.stopped', endInstructions_3.tStopRefresh)
thisExp.nextEntry()
# the Routine "gen_instr" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "q_instr"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_4.keys = []
endInstructions_4.rt = []
_endInstructions_4_allKeys = []
# keep track of which components have finished
q_instrComponents = [Instr_Img, endInstructions_4]
for thisComponent in q_instrComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
q_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "q_instr"-------
while continueRoutine:
    # get current time
    t = q_instrClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=q_instrClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Instr_Img* updates
    if Instr_Img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Instr_Img.frameNStart = frameN  # exact frame index
        Instr_Img.tStart = t  # local t and not account for scr refresh
        Instr_Img.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Instr_Img, 'tStartRefresh')  # time at next scr refresh
        Instr_Img.setAutoDraw(True)
    
    # *endInstructions_4* updates
    waitOnFlip = False
    if endInstructions_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endInstructions_4.frameNStart = frameN  # exact frame index
        endInstructions_4.tStart = t  # local t and not account for scr refresh
        endInstructions_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endInstructions_4, 'tStartRefresh')  # time at next scr refresh
        endInstructions_4.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endInstructions_4.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endInstructions_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endInstructions_4.status == STARTED and not waitOnFlip:
        theseKeys = endInstructions_4.getKeys(keyList=None, waitRelease=False)
        _endInstructions_4_allKeys.extend(theseKeys)
        if len(_endInstructions_4_allKeys):
            endInstructions_4.keys = _endInstructions_4_allKeys[-1].name  # just the last key pressed
            endInstructions_4.rt = _endInstructions_4_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in q_instrComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "q_instr"-------
for thisComponent in q_instrComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Instr_Img.started', Instr_Img.tStartRefresh)
thisExp.addData('Instr_Img.stopped', Instr_Img.tStopRefresh)
# check responses
if endInstructions_4.keys in ['', [], None]:  # No response was made
    endInstructions_4.keys = None
thisExp.addData('endInstructions_4.keys',endInstructions_4.keys)
if endInstructions_4.keys != None:  # we had a response
    thisExp.addData('endInstructions_4.rt', endInstructions_4.rt)
thisExp.addData('endInstructions_4.started', endInstructions_4.tStartRefresh)
thisExp.addData('endInstructions_4.stopped', endInstructions_4.tStopRefresh)
thisExp.nextEntry()
# the Routine "q_instr" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "fixation_2"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_5.keys = []
endInstructions_5.rt = []
_endInstructions_5_allKeys = []
# keep track of which components have finished
fixation_2Components = [fix_instr, Fix_Example, endInstructions_5]
for thisComponent in fixation_2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
fixation_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "fixation_2"-------
while continueRoutine:
    # get current time
    t = fixation_2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=fixation_2Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *fix_instr* updates
    if fix_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        fix_instr.frameNStart = frameN  # exact frame index
        fix_instr.tStart = t  # local t and not account for scr refresh
        fix_instr.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(fix_instr, 'tStartRefresh')  # time at next scr refresh
        fix_instr.setAutoDraw(True)
    
    # *Fix_Example* updates
    if Fix_Example.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Fix_Example.frameNStart = frameN  # exact frame index
        Fix_Example.tStart = t  # local t and not account for scr refresh
        Fix_Example.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Fix_Example, 'tStartRefresh')  # time at next scr refresh
        Fix_Example.setAutoDraw(True)
    
    # *endInstructions_5* updates
    waitOnFlip = False
    if endInstructions_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endInstructions_5.frameNStart = frameN  # exact frame index
        endInstructions_5.tStart = t  # local t and not account for scr refresh
        endInstructions_5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endInstructions_5, 'tStartRefresh')  # time at next scr refresh
        endInstructions_5.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endInstructions_5.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endInstructions_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endInstructions_5.status == STARTED and not waitOnFlip:
        theseKeys = endInstructions_5.getKeys(keyList=None, waitRelease=False)
        _endInstructions_5_allKeys.extend(theseKeys)
        if len(_endInstructions_5_allKeys):
            endInstructions_5.keys = _endInstructions_5_allKeys[-1].name  # just the last key pressed
            endInstructions_5.rt = _endInstructions_5_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in fixation_2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "fixation_2"-------
for thisComponent in fixation_2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('fix_instr.started', fix_instr.tStartRefresh)
thisExp.addData('fix_instr.stopped', fix_instr.tStopRefresh)
thisExp.addData('Fix_Example.started', Fix_Example.tStartRefresh)
thisExp.addData('Fix_Example.stopped', Fix_Example.tStopRefresh)
# check responses
if endInstructions_5.keys in ['', [], None]:  # No response was made
    endInstructions_5.keys = None
thisExp.addData('endInstructions_5.keys',endInstructions_5.keys)
if endInstructions_5.keys != None:  # we had a response
    thisExp.addData('endInstructions_5.rt', endInstructions_5.rt)
thisExp.addData('endInstructions_5.started', endInstructions_5.tStartRefresh)
thisExp.addData('endInstructions_5.stopped', endInstructions_5.tStopRefresh)
thisExp.nextEntry()
# the Routine "fixation_2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "begin_exp"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_3.keys = []
key_resp_3.rt = []
_key_resp_3_allKeys = []
# keep track of which components have finished
begin_expComponents = [beginexp, key_resp_3]
for thisComponent in begin_expComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
begin_expClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "begin_exp"-------
while continueRoutine:
    # get current time
    t = begin_expClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=begin_expClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *beginexp* updates
    if beginexp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        beginexp.frameNStart = frameN  # exact frame index
        beginexp.tStart = t  # local t and not account for scr refresh
        beginexp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(beginexp, 'tStartRefresh')  # time at next scr refresh
        beginexp.setAutoDraw(True)
    
    # *key_resp_3* updates
    waitOnFlip = False
    if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_3.frameNStart = frameN  # exact frame index
        key_resp_3.tStart = t  # local t and not account for scr refresh
        key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
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
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in begin_expComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "begin_exp"-------
for thisComponent in begin_expComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('beginexp.started', beginexp.tStartRefresh)
thisExp.addData('beginexp.stopped', beginexp.tStopRefresh)
# check responses
if key_resp_3.keys in ['', [], None]:  # No response was made
    key_resp_3.keys = None
thisExp.addData('key_resp_3.keys',key_resp_3.keys)
if key_resp_3.keys != None:  # we had a response
    thisExp.addData('key_resp_3.rt', key_resp_3.rt)
thisExp.addData('key_resp_3.started', key_resp_3.tStartRefresh)
thisExp.addData('key_resp_3.stopped', key_resp_3.tStopRefresh)
thisExp.nextEntry()
# the Routine "begin_exp" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "eyetrack_calib"-------
continueRoutine = True
# update component parameters for each repeat

if expInfo['EyeTrack']: 


    # set up a custom graphics envrionment (EyeLinkCoreGraphicsPsychopy) for calibration
    genv = EyeLinkCoreGraphicsPsychoPy(tk, win)

    # Configure the calibration target, could be a 'circle', 
    # a movie clip ('movie'), a 'picture', or a 'spiral', the default is a circle
    genv.calTarget = 'circle'


    pylink.openGraphicsEx(genv)

    # STEP 5: Set up the tracker
    # put the tracker in idle mode before we change its parameters
    tk.setOfflineMode()
    pylink.pumpDelay(100)

    # IMPORTANT: send screen resolution to the tracker
    # see Eyelink Installation Guide, Section 8.4: Customizing Your PHYSICAL.INI Settings
    tk.sendCommand("screen_pixel_coords = 0 0 %d %d" % (scnWidth-1, scnHeight-1))

    # save screen resolution in EDF data, so Data Viewer can correctly load experimental graphics
    # see Data Viewer User Manual, Section 7: Protocol for EyeLink Data to Viewer Integration
    tk.sendMessage("DISPLAY_COORDS = 0 0 %d %d" % (scnWidth-1, scnHeight-1))

    # sampling rate, 250, 500, 1000, or 2000; this command is not supported for EyeLInk II/I trackers
    # tk.sendCommand("sample_rate 1000")

    # detect eye events based on "GAZE" (or "HREF") data
    tk.sendCommand("recording_parse_type = GAZE")

    # Saccade detection thresholds: 0-> standard/coginitve, 1-> sensitive/psychophysiological
    # see Eyelink User Manual, Section 4.3: EyeLink Parser Configuration
    tk.sendCommand("select_parser_configuration 0") 

    # choose a calibration type, H3, HV3, HV5, HV13 (HV = horiztonal/vertical), 
    # tk.setCalibrationType('HV9') also works, see the Pylink manual
    tk.sendCommand("calibration_type = HV9") 

    # tracker hardware, 1-EyeLink I, 2-EyeLink II, 3-Newer models (1000/1000Plus/Portable DUO)
    hardware_ver = tk.getTrackerVersion()

    # tracking software version
    software_ver = 0
    if hardware_ver == 3:
        tvstr = tk.getTrackerVersionString()
        vindex = tvstr.find("EYELINK CL")
        software_ver = float(tvstr.split()[-1])

    # sample and event data saved in EDF data file
    # see sectin 4.6 of the EyeLink user manual, software version > 4 adds remote tracking (and thus HTARGET)
    tk.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
    if software_ver >= 4:
        tk.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,HTARGET,INPUT")
    else:
        tk.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,INPUT")

    # sample and event data available over the link    
    tk.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,BUTTON,INPUT")
    if software_ver >= 4:
        tk.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,HTARGET,INPUT")
    else:
        tk.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,INPUT")

    # STEP 6:  Show task instructions and calibrate the tracker
    msg = visual.TextStim(win, text='Calibration will begin shortly\n' +
                                    'In the task, quickly move your eyes to look at a dot (target) when it appears')
    msg.draw()
    win.flip()
    event.waitKeys()

    # set up the camera and calibrate the tracker
    tk.doTrackerSetup()
    
else: 
    continueRoutine = False

# keep track of which components have finished
eyetrack_calibComponents = []
for thisComponent in eyetrack_calibComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
eyetrack_calibClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "eyetrack_calib"-------
while continueRoutine:
    # get current time
    t = eyetrack_calibClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=eyetrack_calibClock)
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
    for thisComponent in eyetrack_calibComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "eyetrack_calib"-------
for thisComponent in eyetrack_calibComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "eyetrack_calib" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of trials etc
conditions = data.importConditions(q_conditions)
trials = data.MultiStairHandler(stairType='quest', name='trials',
    nTrials=num_trials,
    conditions=conditions,
    method='random',
    originPath=-1)
thisExp.addLoop(trials)  # add the loop to the experiment
# initialise values for first condition
level = trials._nextIntensity  # initialise some vals
condition = trials.currentStaircase.condition

for level, condition in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb=condition.rgb)
    for paramName in condition:
        exec(paramName + '= condition[paramName]')
    
    # ------Prepare to start Routine "block_break"-------
    continueRoutine = True
    routineTimer.add(120.000000)
    # update component parameters for each repeat
    #Taking a break every 80 trials
    break_freq = 80
    blocks_total = math.ceil((num_exemps*num_trials)/break_freq)
    block_complete = ''
    categorization_accuracy = ''
    if trials.totalTrials == 0 or trials.totalTrials %break_freq != 0:
        continueRoutine = False
    else:
        blocks_count += 1
        block_complete = '\n\n\n\n\n\nBlocks Completed: {num_done:n}/{num_total:n}'.format(num_done = blocks_count, num_total = blocks_total)
        categorization_accuracy = '\n\n You correctly categorized {percent_correct:.0%} of the images!'.format(percent_correct =  category_correct_count/trial_counter)
        win.callOnFlip(tk.sendMessage, 'break')
        #Reset category correct and trial counters 
        category_correct_count = 0 
        trial_counter = 0 
    
    
    
    end_break.keys = []
    end_break.rt = []
    _end_break_allKeys = []
    block_count.setText(block_complete)
    cat_accuracy.setText(categorization_accuracy )
    # keep track of which components have finished
    block_breakComponents = [end_break, break_text, block_count, cat_accuracy, countdown_timer]
    for thisComponent in block_breakComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    block_breakClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "block_break"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = block_breakClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=block_breakClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_break* updates
        waitOnFlip = False
        if end_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_break.frameNStart = frameN  # exact frame index
            end_break.tStart = t  # local t and not account for scr refresh
            end_break.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_break, 'tStartRefresh')  # time at next scr refresh
            end_break.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_break.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_break.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_break.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_break.tStartRefresh + 120-frameTolerance:
                # keep track of stop time/frame for later
                end_break.tStop = t  # not accounting for scr refresh
                end_break.frameNStop = frameN  # exact frame index
                win.timeOnFlip(end_break, 'tStopRefresh')  # time at next scr refresh
                end_break.status = FINISHED
        if end_break.status == STARTED and not waitOnFlip:
            theseKeys = end_break.getKeys(keyList=None, waitRelease=False)
            _end_break_allKeys.extend(theseKeys)
            if len(_end_break_allKeys):
                end_break.keys = _end_break_allKeys[-1].name  # just the last key pressed
                end_break.rt = _end_break_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *break_text* updates
        if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            break_text.frameNStart = frameN  # exact frame index
            break_text.tStart = t  # local t and not account for scr refresh
            break_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
            break_text.setAutoDraw(True)
        if break_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > break_text.tStartRefresh + 120-frameTolerance:
                # keep track of stop time/frame for later
                break_text.tStop = t  # not accounting for scr refresh
                break_text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(break_text, 'tStopRefresh')  # time at next scr refresh
                break_text.setAutoDraw(False)
        
        # *block_count* updates
        if block_count.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            block_count.frameNStart = frameN  # exact frame index
            block_count.tStart = t  # local t and not account for scr refresh
            block_count.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(block_count, 'tStartRefresh')  # time at next scr refresh
            block_count.setAutoDraw(True)
        if block_count.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > block_count.tStartRefresh + 120-frameTolerance:
                # keep track of stop time/frame for later
                block_count.tStop = t  # not accounting for scr refresh
                block_count.frameNStop = frameN  # exact frame index
                win.timeOnFlip(block_count, 'tStopRefresh')  # time at next scr refresh
                block_count.setAutoDraw(False)
        
        # *cat_accuracy* updates
        if cat_accuracy.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cat_accuracy.frameNStart = frameN  # exact frame index
            cat_accuracy.tStart = t  # local t and not account for scr refresh
            cat_accuracy.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cat_accuracy, 'tStartRefresh')  # time at next scr refresh
            cat_accuracy.setAutoDraw(True)
        if cat_accuracy.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cat_accuracy.tStartRefresh + 120-frameTolerance:
                # keep track of stop time/frame for later
                cat_accuracy.tStop = t  # not accounting for scr refresh
                cat_accuracy.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cat_accuracy, 'tStopRefresh')  # time at next scr refresh
                cat_accuracy.setAutoDraw(False)
        
        # *countdown_timer* updates
        if countdown_timer.status == NOT_STARTED and tThisFlip >= 110-frameTolerance:
            # keep track of start time/frame for later
            countdown_timer.frameNStart = frameN  # exact frame index
            countdown_timer.tStart = t  # local t and not account for scr refresh
            countdown_timer.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(countdown_timer, 'tStartRefresh')  # time at next scr refresh
            countdown_timer.setAutoDraw(True)
        if countdown_timer.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > countdown_timer.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                countdown_timer.tStop = t  # not accounting for scr refresh
                countdown_timer.frameNStop = frameN  # exact frame index
                win.timeOnFlip(countdown_timer, 'tStopRefresh')  # time at next scr refresh
                countdown_timer.setAutoDraw(False)
        if countdown_timer.status == STARTED:  # only update if drawing
            countdown_timer.setText(






round(120 - t, ndigits = 1))
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in block_breakComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "block_break"-------
    for thisComponent in block_breakComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if end_break.keys in ['', [], None]:  # No response was made
        end_break.keys = None
    trials.addOtherData('end_break.started', end_break.tStartRefresh)
    trials.addOtherData('end_break.stopped', end_break.tStopRefresh)
    trials.addOtherData('break_text.started', break_text.tStartRefresh)
    trials.addOtherData('break_text.stopped', break_text.tStopRefresh)
    trials.addOtherData('block_count.started', block_count.tStartRefresh)
    trials.addOtherData('block_count.stopped', block_count.tStopRefresh)
    trials.addOtherData('cat_accuracy.started', cat_accuracy.tStartRefresh)
    trials.addOtherData('cat_accuracy.stopped', cat_accuracy.tStopRefresh)
    trials.addOtherData('countdown_timer.started', countdown_timer.tStartRefresh)
    trials.addOtherData('countdown_timer.stopped', countdown_timer.tStopRefresh)
    
    # ------Prepare to start Routine "eyetrack_recalib"-------
    continueRoutine = True
    # update component parameters for each repeat
    #Taking a break every N trials
    #Recalibrate eyetracker after break
    if trials.totalTrials == 0 or trials.totalTrials %break_freq != 0:
        continueRoutine = False
    else:
        if expInfo['EyeTrack']: 
            # set up a custom graphics envrionment (EyeLinkCoreGraphicsPsychopy) for calibration
            genv = EyeLinkCoreGraphicsPsychoPy(tk, win)
    
            # Configure the calibration target, could be a 'circle', 
            # a movie clip ('movie'), a 'picture', or a 'spiral', the default is a circle
            genv.calTarget = 'circle'
    
            pylink.openGraphicsEx(genv)
    
        #    # STEP 5: Set up the tracker
        #    # put the tracker in idle mode before we change its parameters
        #    tk.setOfflineMode()
        #    pylink.pumpDelay(100)
    
            # IMPORTANT: send screen resolution to the tracker
            # see Eyelink Installation Guide, Section 8.4: Customizing Your PHYSICAL.INI Settings
            tk.sendCommand("screen_pixel_coords = 0 0 %d %d" % (scnWidth-1, scnHeight-1))
    
            # save screen resolution in EDF data, so Data Viewer can correctly load experimental graphics
            # see Data Viewer User Manual, Section 7: Protocol for EyeLink Data to Viewer Integration
            tk.sendMessage("DISPLAY_COORDS = 0 0 %d %d" % (scnWidth-1, scnHeight-1))
    
            # sampling rate, 250, 500, 1000, or 2000; this command is not supported for EyeLInk II/I trackers
            # tk.sendCommand("sample_rate 1000")
    
            # detect eye events based on "GAZE" (or "HREF") data
            tk.sendCommand("recording_parse_type = GAZE")
    
            # Saccade detection thresholds: 0-> standard/coginitve, 1-> sensitive/psychophysiological
            # see Eyelink User Manual, Section 4.3: EyeLink Parser Configuration
            tk.sendCommand("select_parser_configuration 0") 
    
            # choose a calibration type, H3, HV3, HV5, HV13 (HV = horiztonal/vertical), 
            # tk.setCalibrationType('HV9') also works, see the Pylink manual
            tk.sendCommand("calibration_type = HV9") 
    
            # tracker hardware, 1-EyeLink I, 2-EyeLink II, 3-Newer models (1000/1000Plus/Portable DUO)
            hardware_ver = tk.getTrackerVersion()
    
            # tracking software version
            software_ver = 0
            if hardware_ver == 3:
                tvstr = tk.getTrackerVersionString()
                vindex = tvstr.find("EYELINK CL")
                software_ver = float(tvstr.split()[-1])
    
            # sample and event data saved in EDF data file
            # see sectin 4.6 of the EyeLink user manual, software version > 4 adds remote tracking (and thus HTARGET)
            tk.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
            if software_ver >= 4:
                tk.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,HTARGET,INPUT")
            else:
                tk.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,INPUT")
    
            # sample and event data available over the link    
            tk.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,BUTTON,INPUT")
            if software_ver >= 4:
                tk.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,HTARGET,INPUT")
            else:
                tk.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,INPUT")
    
            # STEP 6:  Show task instructions and calibrate the tracker
            msg = visual.TextStim(win, text='Calibration will begin shortly\n' +
                                            'In the task, quickly move your eyes to look at a dot (target) when it appears')
            msg.draw()
            win.flip()
            event.waitKeys()
    
    
            # set up the camera and calibrate the tracker
            tk.doTrackerSetup()
        else:
            continueRoutine = False
    # keep track of which components have finished
    eyetrack_recalibComponents = []
    for thisComponent in eyetrack_recalibComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    eyetrack_recalibClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "eyetrack_recalib"-------
    while continueRoutine:
        # get current time
        t = eyetrack_recalibClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=eyetrack_recalibClock)
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
        for thisComponent in eyetrack_recalibComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "eyetrack_recalib"-------
    for thisComponent in eyetrack_recalibComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "eyetrack_recalib" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "init_fix"-------
    continueRoutine = True
    # update component parameters for each repeat
    #Manual trial counter for categorization accuracy
    trial_counter += 1
    #Get ITI for trial 
    ITI_duration = ITI_list.pop()
    
    thisExp.addData('ITI', ITI_duration)
    win.callOnFlip(tk.sendMessage, 'ITI')
    
    # Loading .npy files
    #Select which stimuli to present 
    
    # For mini Quest, each exemplar is presented same number of times 
    # Decide which stimulus image to present randomly. 
    if label == 'qMini': 
        stim_img = qMiniStimulusList.pop()
    
    # Randomly choose between stim presented left or right
    # Scrambled image is always the scrambled version of the real counterpart 
    
    left_or_right = random()
    
    if left_or_right>0.5:
        #Display stim on left and scrambled on right
        target_left = stim_img
        target_right = stim_img[:13] + 'scram.npy'
        probe_loc = 'L'
        prob_cue_L = place_real
        prob_cue_R = place_scram
        thisExp.addData('target_stimulus', target_left)
    
    else:
        #Display image on right not on left
        target_left = stim_img[:13] + 'scram.npy'
        target_right = stim_img
        probe_loc = 'R'
        prob_cue_L= place_scram
        prob_cue_R = place_real
        thisExp.addData('target_stimulus', target_right)
        
    thisExp.addData('stim_left', target_left)
    thisExp.addData('stim_right',target_right) 
    
    
    
    # Load Stimulus images
    
    img_left = np.load(target_left)  
    img_right = np.load(target_right) 
    
    
    #Ramp up and down stim contrast with peak 
    #contrast at 66.7 mS (4 frames assuming 60Hz)
    
    minC = 0.001 
    
    #For qMini staircases, the "level" is being defined as a deviation from the initial QUEST estimate done in a previous session 
    if label == 'qMini':
        maxC = quest_estimates[stim_img[:13]] * 10**level
    else:
        maxC = 10**level
    
    #Presenting Stimuli for 300 ms for practice instead of 66 ms
    if not expInfo['frameRate']: #Can't measure FR, e.g. dropping frames
        nC = 4
    elif np.isclose(expInfo['frameRate'], 120, rtol = .1):
        nC = 8 #Num frames of presentation
    elif np.isclose(expInfo['frameRate'], 60, rtol = .1):
        nC = 4
    else: 
        nC = 8
    
    
    grad_contr = np.linspace(minC,maxC,nC)
    
    #New stim duration in frames
    stim_duration = len(grad_contr)
    place_left_3.setLineColor(prob_cue_L)
    place_right_3.setLineColor(prob_cue_R)
    # keep track of which components have finished
    init_fixComponents = [place_left_3, place_right_3, long_back_R_3, hi_back_R_3, hi_back_L_3, long_back_L_3, pre_cue_fix_2]
    for thisComponent in init_fixComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    init_fixClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "init_fix"-------
    while continueRoutine:
        # get current time
        t = init_fixClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=init_fixClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *place_left_3* updates
        if place_left_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            place_left_3.frameNStart = frameN  # exact frame index
            place_left_3.tStart = t  # local t and not account for scr refresh
            place_left_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(place_left_3, 'tStartRefresh')  # time at next scr refresh
            place_left_3.setAutoDraw(True)
        if place_left_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > place_left_3.tStartRefresh + ITI_duration-frameTolerance:
                # keep track of stop time/frame for later
                place_left_3.tStop = t  # not accounting for scr refresh
                place_left_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(place_left_3, 'tStopRefresh')  # time at next scr refresh
                place_left_3.setAutoDraw(False)
        
        # *place_right_3* updates
        if place_right_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            place_right_3.frameNStart = frameN  # exact frame index
            place_right_3.tStart = t  # local t and not account for scr refresh
            place_right_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(place_right_3, 'tStartRefresh')  # time at next scr refresh
            place_right_3.setAutoDraw(True)
        if place_right_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > place_right_3.tStartRefresh + ITI_duration-frameTolerance:
                # keep track of stop time/frame for later
                place_right_3.tStop = t  # not accounting for scr refresh
                place_right_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(place_right_3, 'tStopRefresh')  # time at next scr refresh
                place_right_3.setAutoDraw(False)
        
        # *long_back_R_3* updates
        if long_back_R_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            long_back_R_3.frameNStart = frameN  # exact frame index
            long_back_R_3.tStart = t  # local t and not account for scr refresh
            long_back_R_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(long_back_R_3, 'tStartRefresh')  # time at next scr refresh
            long_back_R_3.setAutoDraw(True)
        if long_back_R_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > long_back_R_3.tStartRefresh + ITI_duration-frameTolerance:
                # keep track of stop time/frame for later
                long_back_R_3.tStop = t  # not accounting for scr refresh
                long_back_R_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(long_back_R_3, 'tStopRefresh')  # time at next scr refresh
                long_back_R_3.setAutoDraw(False)
        
        # *hi_back_R_3* updates
        if hi_back_R_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hi_back_R_3.frameNStart = frameN  # exact frame index
            hi_back_R_3.tStart = t  # local t and not account for scr refresh
            hi_back_R_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hi_back_R_3, 'tStartRefresh')  # time at next scr refresh
            hi_back_R_3.setAutoDraw(True)
        if hi_back_R_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > hi_back_R_3.tStartRefresh + ITI_duration-frameTolerance:
                # keep track of stop time/frame for later
                hi_back_R_3.tStop = t  # not accounting for scr refresh
                hi_back_R_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(hi_back_R_3, 'tStopRefresh')  # time at next scr refresh
                hi_back_R_3.setAutoDraw(False)
        
        # *hi_back_L_3* updates
        if hi_back_L_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hi_back_L_3.frameNStart = frameN  # exact frame index
            hi_back_L_3.tStart = t  # local t and not account for scr refresh
            hi_back_L_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hi_back_L_3, 'tStartRefresh')  # time at next scr refresh
            hi_back_L_3.setAutoDraw(True)
        if hi_back_L_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > hi_back_L_3.tStartRefresh + ITI_duration-frameTolerance:
                # keep track of stop time/frame for later
                hi_back_L_3.tStop = t  # not accounting for scr refresh
                hi_back_L_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(hi_back_L_3, 'tStopRefresh')  # time at next scr refresh
                hi_back_L_3.setAutoDraw(False)
        
        # *long_back_L_3* updates
        if long_back_L_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            long_back_L_3.frameNStart = frameN  # exact frame index
            long_back_L_3.tStart = t  # local t and not account for scr refresh
            long_back_L_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(long_back_L_3, 'tStartRefresh')  # time at next scr refresh
            long_back_L_3.setAutoDraw(True)
        if long_back_L_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > long_back_L_3.tStartRefresh + ITI_duration-frameTolerance:
                # keep track of stop time/frame for later
                long_back_L_3.tStop = t  # not accounting for scr refresh
                long_back_L_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(long_back_L_3, 'tStopRefresh')  # time at next scr refresh
                long_back_L_3.setAutoDraw(False)
        
        # *pre_cue_fix_2* updates
        if pre_cue_fix_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pre_cue_fix_2.frameNStart = frameN  # exact frame index
            pre_cue_fix_2.tStart = t  # local t and not account for scr refresh
            pre_cue_fix_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pre_cue_fix_2, 'tStartRefresh')  # time at next scr refresh
            pre_cue_fix_2.setAutoDraw(True)
        if pre_cue_fix_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pre_cue_fix_2.tStartRefresh + ITI_duration-frameTolerance:
                # keep track of stop time/frame for later
                pre_cue_fix_2.tStop = t  # not accounting for scr refresh
                pre_cue_fix_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(pre_cue_fix_2, 'tStopRefresh')  # time at next scr refresh
                pre_cue_fix_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in init_fixComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "init_fix"-------
    for thisComponent in init_fixComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addOtherData('place_left_3.started', place_left_3.tStartRefresh)
    trials.addOtherData('place_left_3.stopped', place_left_3.tStopRefresh)
    trials.addOtherData('place_right_3.started', place_right_3.tStartRefresh)
    trials.addOtherData('place_right_3.stopped', place_right_3.tStopRefresh)
    trials.addOtherData('long_back_R_3.started', long_back_R_3.tStartRefresh)
    trials.addOtherData('long_back_R_3.stopped', long_back_R_3.tStopRefresh)
    trials.addOtherData('hi_back_R_3.started', hi_back_R_3.tStartRefresh)
    trials.addOtherData('hi_back_R_3.stopped', hi_back_R_3.tStopRefresh)
    trials.addOtherData('hi_back_L_3.started', hi_back_L_3.tStartRefresh)
    trials.addOtherData('hi_back_L_3.stopped', hi_back_L_3.tStopRefresh)
    trials.addOtherData('long_back_L_3.started', long_back_L_3.tStartRefresh)
    trials.addOtherData('long_back_L_3.stopped', long_back_L_3.tStopRefresh)
    trials.addOtherData('pre_cue_fix_2.started', pre_cue_fix_2.tStartRefresh)
    trials.addOtherData('pre_cue_fix_2.stopped', pre_cue_fix_2.tStopRefresh)
    # the Routine "init_fix" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "stim"-------
    continueRoutine = True
    # update component parameters for each repeat
    win.callOnFlip(tk.sendMessage, 'stimOnset')
    place_left_4.setLineColor(prob_cue_L)
    place_right_4.setLineColor(prob_cue_R)
    # keep track of which components have finished
    stimComponents = [place_left_4, place_right_4, long_back_R_4, hi_back_R_4, hi_back_L_4, long_back_L_4, stim_fix_2, stim_right_2, stim_left_2]
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
    stimClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "stim"-------
    while continueRoutine:
        # get current time
        t = stimClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=stimClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        #Set graded conntrast of images for entire presentation
        
        if frameN <stim_duration: 
            F = grad_contr[frameN] * params['filter']  #Keep real and scrambled at same contrast regardless 
            left_stim = np.array(img_left * F)
            right_stim = np.array(img_right * F)
            
        
        # *place_left_4* updates
        if place_left_4.status == NOT_STARTED and frameN >= 0.0:
            # keep track of start time/frame for later
            place_left_4.frameNStart = frameN  # exact frame index
            place_left_4.tStart = t  # local t and not account for scr refresh
            place_left_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(place_left_4, 'tStartRefresh')  # time at next scr refresh
            place_left_4.setAutoDraw(True)
        if place_left_4.status == STARTED:
            if frameN >= (place_left_4.frameNStart + stim_duration):
                # keep track of stop time/frame for later
                place_left_4.tStop = t  # not accounting for scr refresh
                place_left_4.frameNStop = frameN  # exact frame index
                win.timeOnFlip(place_left_4, 'tStopRefresh')  # time at next scr refresh
                place_left_4.setAutoDraw(False)
        
        # *place_right_4* updates
        if place_right_4.status == NOT_STARTED and frameN >= 0.0:
            # keep track of start time/frame for later
            place_right_4.frameNStart = frameN  # exact frame index
            place_right_4.tStart = t  # local t and not account for scr refresh
            place_right_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(place_right_4, 'tStartRefresh')  # time at next scr refresh
            place_right_4.setAutoDraw(True)
        if place_right_4.status == STARTED:
            if frameN >= (place_right_4.frameNStart + stim_duration):
                # keep track of stop time/frame for later
                place_right_4.tStop = t  # not accounting for scr refresh
                place_right_4.frameNStop = frameN  # exact frame index
                win.timeOnFlip(place_right_4, 'tStopRefresh')  # time at next scr refresh
                place_right_4.setAutoDraw(False)
        
        # *long_back_R_4* updates
        if long_back_R_4.status == NOT_STARTED and frameN >= 0.0:
            # keep track of start time/frame for later
            long_back_R_4.frameNStart = frameN  # exact frame index
            long_back_R_4.tStart = t  # local t and not account for scr refresh
            long_back_R_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(long_back_R_4, 'tStartRefresh')  # time at next scr refresh
            long_back_R_4.setAutoDraw(True)
        if long_back_R_4.status == STARTED:
            if frameN >= (long_back_R_4.frameNStart + stim_duration):
                # keep track of stop time/frame for later
                long_back_R_4.tStop = t  # not accounting for scr refresh
                long_back_R_4.frameNStop = frameN  # exact frame index
                win.timeOnFlip(long_back_R_4, 'tStopRefresh')  # time at next scr refresh
                long_back_R_4.setAutoDraw(False)
        
        # *hi_back_R_4* updates
        if hi_back_R_4.status == NOT_STARTED and frameN >= 0.0:
            # keep track of start time/frame for later
            hi_back_R_4.frameNStart = frameN  # exact frame index
            hi_back_R_4.tStart = t  # local t and not account for scr refresh
            hi_back_R_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hi_back_R_4, 'tStartRefresh')  # time at next scr refresh
            hi_back_R_4.setAutoDraw(True)
        if hi_back_R_4.status == STARTED:
            if frameN >= (hi_back_R_4.frameNStart + stim_duration):
                # keep track of stop time/frame for later
                hi_back_R_4.tStop = t  # not accounting for scr refresh
                hi_back_R_4.frameNStop = frameN  # exact frame index
                win.timeOnFlip(hi_back_R_4, 'tStopRefresh')  # time at next scr refresh
                hi_back_R_4.setAutoDraw(False)
        
        # *hi_back_L_4* updates
        if hi_back_L_4.status == NOT_STARTED and frameN >= 0.0:
            # keep track of start time/frame for later
            hi_back_L_4.frameNStart = frameN  # exact frame index
            hi_back_L_4.tStart = t  # local t and not account for scr refresh
            hi_back_L_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hi_back_L_4, 'tStartRefresh')  # time at next scr refresh
            hi_back_L_4.setAutoDraw(True)
        if hi_back_L_4.status == STARTED:
            if frameN >= (hi_back_L_4.frameNStart + stim_duration):
                # keep track of stop time/frame for later
                hi_back_L_4.tStop = t  # not accounting for scr refresh
                hi_back_L_4.frameNStop = frameN  # exact frame index
                win.timeOnFlip(hi_back_L_4, 'tStopRefresh')  # time at next scr refresh
                hi_back_L_4.setAutoDraw(False)
        
        # *long_back_L_4* updates
        if long_back_L_4.status == NOT_STARTED and frameN >= 0.0:
            # keep track of start time/frame for later
            long_back_L_4.frameNStart = frameN  # exact frame index
            long_back_L_4.tStart = t  # local t and not account for scr refresh
            long_back_L_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(long_back_L_4, 'tStartRefresh')  # time at next scr refresh
            long_back_L_4.setAutoDraw(True)
        if long_back_L_4.status == STARTED:
            if frameN >= (long_back_L_4.frameNStart + stim_duration):
                # keep track of stop time/frame for later
                long_back_L_4.tStop = t  # not accounting for scr refresh
                long_back_L_4.frameNStop = frameN  # exact frame index
                win.timeOnFlip(long_back_L_4, 'tStopRefresh')  # time at next scr refresh
                long_back_L_4.setAutoDraw(False)
        
        # *stim_fix_2* updates
        if stim_fix_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            stim_fix_2.frameNStart = frameN  # exact frame index
            stim_fix_2.tStart = t  # local t and not account for scr refresh
            stim_fix_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stim_fix_2, 'tStartRefresh')  # time at next scr refresh
            stim_fix_2.setAutoDraw(True)
        if stim_fix_2.status == STARTED:
            if frameN >= (stim_fix_2.frameNStart + stim_duration):
                # keep track of stop time/frame for later
                stim_fix_2.tStop = t  # not accounting for scr refresh
                stim_fix_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stim_fix_2, 'tStopRefresh')  # time at next scr refresh
                stim_fix_2.setAutoDraw(False)
        
        # *stim_right_2* updates
        if stim_right_2.status == NOT_STARTED and frameN >= 0:
            # keep track of start time/frame for later
            stim_right_2.frameNStart = frameN  # exact frame index
            stim_right_2.tStart = t  # local t and not account for scr refresh
            stim_right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stim_right_2, 'tStartRefresh')  # time at next scr refresh
            stim_right_2.setAutoDraw(True)
        if stim_right_2.status == STARTED:
            if frameN >= (stim_right_2.frameNStart + stim_duration):
                # keep track of stop time/frame for later
                stim_right_2.tStop = t  # not accounting for scr refresh
                stim_right_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stim_right_2, 'tStopRefresh')  # time at next scr refresh
                stim_right_2.setAutoDraw(False)
        if stim_right_2.status == STARTED:  # only update if drawing
            stim_right_2.setImage(right_stim)
        
        # *stim_left_2* updates
        if stim_left_2.status == NOT_STARTED and frameN >= 0:
            # keep track of start time/frame for later
            stim_left_2.frameNStart = frameN  # exact frame index
            stim_left_2.tStart = t  # local t and not account for scr refresh
            stim_left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stim_left_2, 'tStartRefresh')  # time at next scr refresh
            stim_left_2.setAutoDraw(True)
        if stim_left_2.status == STARTED:
            if frameN >= (stim_left_2.frameNStart + stim_duration):
                # keep track of stop time/frame for later
                stim_left_2.tStop = t  # not accounting for scr refresh
                stim_left_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stim_left_2, 'tStopRefresh')  # time at next scr refresh
                stim_left_2.setAutoDraw(False)
        if stim_left_2.status == STARTED:  # only update if drawing
            stim_left_2.setImage(left_stim)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in stimComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "stim"-------
    for thisComponent in stimComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    ## End Routine
    #win.saveMovieFrames('stim_img'+str(trials.totalTrials)+'.png')
    trials.addOtherData('place_left_4.started', place_left_4.tStartRefresh)
    trials.addOtherData('place_left_4.stopped', place_left_4.tStopRefresh)
    trials.addOtherData('place_right_4.started', place_right_4.tStartRefresh)
    trials.addOtherData('place_right_4.stopped', place_right_4.tStopRefresh)
    trials.addOtherData('long_back_R_4.started', long_back_R_4.tStartRefresh)
    trials.addOtherData('long_back_R_4.stopped', long_back_R_4.tStopRefresh)
    trials.addOtherData('hi_back_R_4.started', hi_back_R_4.tStartRefresh)
    trials.addOtherData('hi_back_R_4.stopped', hi_back_R_4.tStopRefresh)
    trials.addOtherData('hi_back_L_4.started', hi_back_L_4.tStartRefresh)
    trials.addOtherData('hi_back_L_4.stopped', hi_back_L_4.tStopRefresh)
    trials.addOtherData('long_back_L_4.started', long_back_L_4.tStartRefresh)
    trials.addOtherData('long_back_L_4.stopped', long_back_L_4.tStopRefresh)
    trials.addOtherData('stim_fix_2.started', stim_fix_2.tStartRefresh)
    trials.addOtherData('stim_fix_2.stopped', stim_fix_2.tStopRefresh)
    trials.addOtherData('stim_right_2.started', stim_right_2.tStartRefresh)
    trials.addOtherData('stim_right_2.stopped', stim_right_2.tStopRefresh)
    trials.addOtherData('stim_left_2.started', stim_left_2.tStartRefresh)
    trials.addOtherData('stim_left_2.stopped', stim_left_2.tStopRefresh)
    # the Routine "stim" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "blank_post"-------
    continueRoutine = True
    routineTimer.add(0.400000)
    # update component parameters for each repeat
    # keep track of which components have finished
    blank_postComponents = [post_stim_fix]
    for thisComponent in blank_postComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    blank_postClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "blank_post"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = blank_postClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=blank_postClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *post_stim_fix* updates
        if post_stim_fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            post_stim_fix.frameNStart = frameN  # exact frame index
            post_stim_fix.tStart = t  # local t and not account for scr refresh
            post_stim_fix.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(post_stim_fix, 'tStartRefresh')  # time at next scr refresh
            post_stim_fix.setAutoDraw(True)
        if post_stim_fix.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > post_stim_fix.tStartRefresh + .4-frameTolerance:
                # keep track of stop time/frame for later
                post_stim_fix.tStop = t  # not accounting for scr refresh
                post_stim_fix.frameNStop = frameN  # exact frame index
                win.timeOnFlip(post_stim_fix, 'tStopRefresh')  # time at next scr refresh
                post_stim_fix.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blank_postComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "blank_post"-------
    for thisComponent in blank_postComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addOtherData('post_stim_fix.started', post_stim_fix.tStartRefresh)
    trials.addOtherData('post_stim_fix.stopped', post_stim_fix.tStopRefresh)
    
    # ------Prepare to start Routine "q_cat"-------
    continueRoutine = True
    routineTimer.add(3.000000)
    # update component parameters for each repeat
    # Randomize Order of FAOH 
    #Initialize cat components
    win.callOnFlip(tk.sendMessage, 'qCatOnset')
    if probe_loc == 'L':
        probe_cue_img_2 = '../cues/cue_left.png'
        
    elif probe_loc == 'R':
        probe_cue_img_2 = '../cues/cue_right.png'
    
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
    q_category.setText(cat_question
)
    cat_resp.keys = []
    cat_resp.rt = []
    _cat_resp_allKeys = []
    probe_cue_cat.setImage(probe_cue_img_2)
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
    # Map buttons to shuffled cats, assumed read L-R
    
    buttons = ['6','7','8','9']
    categories = [c[0] for c in cats]
    
    button_map = dict(zip(categories,buttons))
    cat_map  = dict(zip(buttons,categories))
    #
    if probe_loc == 'L':
        corAns = button_map[target_left[10]] # Index first letter after " images/" in target
        
    elif probe_loc == 'R':
        corAns = button_map[target_right[10]] # Index first letter after " images/" in target
    
    correct_category = cat_map[corAns]
    thisExp.addData('objective_category', correct_category)
    
    if cat_resp.keys:
        category_response = cat_map[cat_resp.keys[-1]]
        thisExp.addData('category_response', category_response)
    
    else: #If no response
        category_response = None
        thisExp.addData('category_response', category_response)
    
    
    #Calculate categorization accuracy
    if correct_category == category_response:
        thisExp.addData('category_correct', 1)
        category_correct_count += 1 
    else:
        thisExp.addData('category_correct', 0)
    
    
        
    
    trials.addOtherData('q_category.started', q_category.tStartRefresh)
    trials.addOtherData('q_category.stopped', q_category.tStopRefresh)
    # check responses
    if cat_resp.keys in ['', [], None]:  # No response was made
        cat_resp.keys = None
    trials.addOtherData('cat_resp.started', cat_resp.tStartRefresh)
    trials.addOtherData('cat_resp.stopped', cat_resp.tStopRefresh)
    trials.addOtherData('probe_cue_cat.started', probe_cue_cat.tStartRefresh)
    trials.addOtherData('probe_cue_cat.stopped', probe_cue_cat.tStopRefresh)
    
    # ------Prepare to start Routine "q_rec"-------
    continueRoutine = True
    routineTimer.add(3.000000)
    # update component parameters for each repeat
    win.callOnFlip(tk.sendMessage, 'qRecOnset')
    
    if probe_loc == 'L':
        probe_cue_img_2 = '../cues/cue_left.png'
        
    elif probe_loc == 'R':
        probe_cue_img_2 = '../cues/cue_right.png'
    
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
                # was this correct?
                if (rec_resp.keys == str('7')) or (rec_resp.keys == '7'):
                    rec_resp.corr = 1
                else:
                    rec_resp.corr = 0
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
    if rec_resp.keys == '7':
        thisExp.addData('recognition', 'R')
    else: #For "No" or No Response
        thisExp.addData('recognition', 'U')
    trials.addOtherData('q_recognition.started', q_recognition.tStartRefresh)
    trials.addOtherData('q_recognition.stopped', q_recognition.tStopRefresh)
    # check responses
    if rec_resp.keys in ['', [], None]:  # No response was made
        rec_resp.keys = None
        # was no response the correct answer?!
        if str('7').lower() == 'none':
           rec_resp.corr = 1;  # correct non-response
        else:
           rec_resp.corr = 0;  # failed to respond (incorrectly)
    # store data for trials (MultiStairHandler)
    trials.addResponse(rec_resp.corr)
    trials.addOtherData('rec_resp.rt', rec_resp.rt)
    trials.addOtherData('rec_resp.started', rec_resp.tStartRefresh)
    trials.addOtherData('rec_resp.stopped', rec_resp.tStopRefresh)
    trials.addOtherData('probe_cue_rec.started', probe_cue_rec.tStartRefresh)
    trials.addOtherData('probe_cue_rec.stopped', probe_cue_rec.tStopRefresh)
    thisExp.nextEntry()
    
# all staircases completed


# ------Prepare to start Routine "thanks"-------
continueRoutine = True
# update component parameters for each repeat
categorization_accuracy = '\n\n\n You correctly categorized {percent_correct:.0%} images!'.format(percent_correct =  category_correct_count/trial_counter)
cat_accuracy_2.setText(categorization_accuracy)
EndExp.keys = []
EndExp.rt = []
_EndExp_allKeys = []
# keep track of which components have finished
thanksComponents = [thanksMsg, cat_accuracy_2, EndExp]
for thisComponent in thanksComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
thanksClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "thanks"-------
while continueRoutine:
    # get current time
    t = thanksClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=thanksClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *thanksMsg* updates
    if thanksMsg.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        thanksMsg.frameNStart = frameN  # exact frame index
        thanksMsg.tStart = t  # local t and not account for scr refresh
        thanksMsg.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(thanksMsg, 'tStartRefresh')  # time at next scr refresh
        thanksMsg.setAutoDraw(True)
    
    # *cat_accuracy_2* updates
    if cat_accuracy_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        cat_accuracy_2.frameNStart = frameN  # exact frame index
        cat_accuracy_2.tStart = t  # local t and not account for scr refresh
        cat_accuracy_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(cat_accuracy_2, 'tStartRefresh')  # time at next scr refresh
        cat_accuracy_2.setAutoDraw(True)
    
    # *EndExp* updates
    waitOnFlip = False
    if EndExp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        EndExp.frameNStart = frameN  # exact frame index
        EndExp.tStart = t  # local t and not account for scr refresh
        EndExp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(EndExp, 'tStartRefresh')  # time at next scr refresh
        EndExp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(EndExp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(EndExp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if EndExp.status == STARTED and not waitOnFlip:
        theseKeys = EndExp.getKeys(keyList=['e'], waitRelease=False)
        _EndExp_allKeys.extend(theseKeys)
        if len(_EndExp_allKeys):
            EndExp.keys = _EndExp_allKeys[-1].name  # just the last key pressed
            EndExp.rt = _EndExp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in thanksComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "thanks"-------
for thisComponent in thanksComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('thanksMsg.started', thanksMsg.tStartRefresh)
thisExp.addData('thanksMsg.stopped', thanksMsg.tStopRefresh)
thisExp.addData('cat_accuracy_2.started', cat_accuracy_2.tStartRefresh)
thisExp.addData('cat_accuracy_2.stopped', cat_accuracy_2.tStopRefresh)
# check responses
if EndExp.keys in ['', [], None]:  # No response was made
    EndExp.keys = None
thisExp.addData('EndExp.keys',EndExp.keys)
if EndExp.keys != None:  # we had a response
    thisExp.addData('EndExp.rt', EndExp.rt)
thisExp.addData('EndExp.started', EndExp.tStartRefresh)
thisExp.addData('EndExp.stopped', EndExp.tStopRefresh)
thisExp.nextEntry()
# the Routine "thanks" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='comma')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
