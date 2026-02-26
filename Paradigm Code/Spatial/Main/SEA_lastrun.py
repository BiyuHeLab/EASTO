#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.10),
    on Fri Sep 16 10:26:20 2022
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
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
    
import EASTO_funcs as ea 
import pandas as pd

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
import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
import math
#Make Filter store in params dict 
# Make param dict 
params = {'imsize' : [300,300],
      'imbgrd' : 127,
      'categories' : ['AM', 'AN', 'FF', 'FM', 'HB', 'HH', 'OH','ON']}
      
params['filter'] = ea.make_filter(params) 
from psychopy.tools.monitorunittools import deg2pix
from psychopy import monitors


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.10'
expName = 'SEA'  # from the Builder filename that created this script
expInfo = {'subject': '', 'session': '001', 'EyeTrack': False, 'ExpMon': 'EEG', 'HPColor': 'black', 'Control': False, 'feedback': False}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s_%s' % (expInfo['subject'],expInfo['session'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/brandonchen93/EASTO/Paradigm/Master/Spatial/Main/SEA_lastrun.py',
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
    monitor=expInfo['ExpMon'], color=[0,0,0], colorSpace='rgb',
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

# Initialize components for Routine "instructions"
instructionsClock = core.Clock()
if expInfo['Control']: 
    instructions_type = 'task_instructions_control.xlsx'
    trial_conditions = 'conditions_control.xlsx'
    if expInfo['HPColor'] == 'black':
        block_conditions = 'blocks_black-HP-control.xlsx'
    elif expInfo['HPColor'] == 'white':
        block_conditions = 'blocks_white-HP-control.xlsx'
else: 
    instructions_type = 'task_instructions.xlsx'
    trial_conditions = 'conditions.xlsx'
    if expInfo['HPColor'] == 'black':
        block_conditions = 'blocks_black-HP.xlsx'
    elif expInfo['HPColor'] == 'white':
        block_conditions = 'blocks_white-HP.xlsx'

instr_txt = visual.TextStim(win=win, name='instr_txt',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
Fix_Example_2 = visual.ShapeStim(
    win=win, name='Fix_Example_2', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1.0, depth=-2.0, interpolate=True)
instr_resp = keyboard.Keyboard()

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
    dataFileName = expInfo['subject'] + '_' + expInfo['session']+ '.EDF'
    tk.openDataFile(dataFileName)
    # add personalized data file header (preamble text)
    tk.sendCommand("add_file_preamble_text 'SEA Exp'") 
else: 
    pylink.EyeLink.dummy_open
    tk = pylink.EyeLink(None)





# Initialize components for Routine "break_2"
break_2Clock = core.Clock()
blocks_count = 0
#Counter for eye tracker
trial_count = 0
#Counter for categorization accuracy
category_correct_count = 0
trial_counter = 0 
end_break = keyboard.Keyboard()
break_text = visual.TextStim(win=win, name='break_text',
    text='Great job! Take a  break to rest your eyes if you need to.\n\n\n\n\n\n\n\n\nPress any button to resume when you are ready',
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

# Initialize components for Routine "Block_Cue"
Block_CueClock = core.Clock()
#Set up image lists & experiment parameters
n_imgs = 16
n_trial_types = 18 
n_block_types = 3

stim_cats = ['AM', 'AN', 'FF', 'FM', 'HB', 'HH', 'OH', 'ON']

if not quest_estimates:
    quest_estimates = {'../images/%s%s' %(stim_cats[i], x) : 1 for i in range(len(stim_cats)) for x in range(1,3)} 

#Get path for real and scrambled images 
real_imgs = ['../images/%s%s.npy' %(stim_cats[i], x) for i in range(len(stim_cats)) for x in range(1,3)]

scram_imgs = ['../images/%s%sscram.npy' %(stim_cats[i], x) for i in range(len(stim_cats)) for x in range(1,3)]


#Generate Stimulus Lists for the whole experiment to ensure that each exemplar is presented in each condition equally 

expect_left_trials, expect_right_trials, expect_neutral_trials = ea.generate_stim(real_imgs, scram_imgs, trial_conditions, [2/3, 1/3], 'Spatial')

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

# Initialize components for Routine "init_fix"
init_fixClock = core.Clock()
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

# Initialize components for Routine "att_cue"
att_cueClock = core.Clock()
place_right_5 = visual.Rect(
    win=win, name='place_right_5',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
place_left_5 = visual.Rect(
    win=win, name='place_left_5',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)
long_back_R_5 = visual.Rect(
    win=win, name='long_back_R_5',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_R_5 = visual.Rect(
    win=win, name='hi_back_R_5',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
hi_back_L_5 = visual.Rect(
    win=win, name='hi_back_L_5',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
long_back_L_5 = visual.Rect(
    win=win, name='long_back_L_5',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-6.0, interpolate=True)
att_left_2 = visual.ImageStim(
    win=win,
    name='att_left_2', 
    image='../cues/cue_left.png', mask=None,
    ori=0, pos=(-0.4, 0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-7.0)
att_right_2 = visual.ImageStim(
    win=win,
    name='att_right_2', 
    image='sin', mask=None,
    ori=0, pos=(0.4, 0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-8.0)
att_cue_fix_3 = visual.ShapeStim(
    win=win, name='att_cue_fix_3', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=0.7, depth=-9.0, interpolate=True)

# Initialize components for Routine "blank_pre"
blank_preClock = core.Clock()
place_right_4 = visual.Rect(
    win=win, name='place_right_4',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=0.0, interpolate=True)
place_left_4 = visual.Rect(
    win=win, name='place_left_4',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
long_back_R_4 = visual.Rect(
    win=win, name='long_back_R_4',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)
hi_back_R_4 = visual.Rect(
    win=win, name='hi_back_R_4',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_L_4 = visual.Rect(
    win=win, name='hi_back_L_4',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
long_back_L_4 = visual.Rect(
    win=win, name='long_back_L_4',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
post_cue_fix2 = visual.ShapeStim(
    win=win, name='post_cue_fix2', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=0.7, depth=-6.0, interpolate=True)

# Initialize components for Routine "stim"
stimClock = core.Clock()
place_right_2 = visual.Rect(
    win=win, name='place_right_2',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
place_left_2 = visual.Rect(
    win=win, name='place_left_2',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)
long_back_R_2 = visual.Rect(
    win=win, name='long_back_R_2',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_R_2 = visual.Rect(
    win=win, name='hi_back_R_2',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
hi_back_L_2 = visual.Rect(
    win=win, name='hi_back_L_2',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
long_back_L_2 = visual.Rect(
    win=win, name='long_back_L_2',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-6.0, interpolate=True)
stim_left_2 = visual.ImageStim(
    win=win,
    name='stim_left_2', 
    image='sin', mask=None,
    ori=0, pos=(-4, 0), size=(4, 4),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=True,
    texRes=512, interpolate=True, depth=-7.0)
stim_right_2 = visual.ImageStim(
    win=win,
    name='stim_right_2', 
    image='sin', mask=None,
    ori=0, pos=(4, 0), size=(4, 4),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=True,
    texRes=512, interpolate=True, depth=-8.0)
stim_fix_2 = visual.ShapeStim(
    win=win, name='stim_fix_2', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=0.7, depth=-9.0, interpolate=True)

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
rec_question  = (u'Real Image or Noise?: \n\n\n\n\n '
            u'{Y:\u00a0^15} {N:\u00a0^15}''\n\n\n '
            u'{b2:\u00a0^15} {b3:\u00a0^15}'.format(Y = u'Real',
                                               N = u'Noise',
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

# Initialize components for Routine "q_conf"
q_confClock = core.Clock()
q_confidence = visual.TextStim(win=win, name='q_confidence',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.1, wrapWidth=500, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
conf_resp = keyboard.Keyboard()
probe_cue_cat_2 = visual.ImageStim(
    win=win,
    name='probe_cue_cat_2', 
    image='sin', mask=None,
    ori=0, pos=(0,0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# Initialize components for Routine "feedback"
feedbackClock = core.Clock()
mon = monitors.Monitor('EEG') 

incor_lw = deg2pix(0.4, mon)
fbck_cor = visual.ShapeStim(
    win=win, name='fbck_cor', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-.75,.75,-.75], lineColorSpace='rgb',
    fillColor=[-.75,.75,-.75], fillColorSpace='rgb',
    opacity=.7, depth=-1.0, interpolate=True)
fbck_incor = visual.Line(
    win=win, name='fbck_incor',
    start=(-(0.4, 0.4)[0]/2.0, 0), end=(+(0.4, 0.4)[0]/2.0, 0),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[.75,-.75,-.75], lineColorSpace='rgb',
    fillColor=[.75,-.75,-.75], fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)

# Initialize components for Routine "Thanks_msg"
Thanks_msgClock = core.Clock()
thanks_txt = visual.TextStim(win=win, name='thanks_txt',
    text='Thank you for being a participant! ',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
EndExp = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# set up handler to look after randomisation of conditions etc
instructions_loop = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions(instructions_type),
    seed=None, name='instructions_loop')
thisExp.addLoop(instructions_loop)  # add the loop to the experiment
thisInstructions_loop = instructions_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisInstructions_loop.rgb)
if thisInstructions_loop != None:
    for paramName in thisInstructions_loop:
        exec('{} = thisInstructions_loop[paramName]'.format(paramName))

for thisInstructions_loop in instructions_loop:
    currentLoop = instructions_loop
    # abbreviate parameter names if possible (e.g. rgb = thisInstructions_loop.rgb)
    if thisInstructions_loop != None:
        for paramName in thisInstructions_loop:
            exec('{} = thisInstructions_loop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "instructions"-------
    continueRoutine = True
    # update component parameters for each repeat
    if instruction_purpose == 'fixation':
        fix_on = 0.7 
    else:
        fix_on = 0
     
    instruction = instruction_text
    instr_txt.setText(instruction)
    Fix_Example_2.setOpacity(fix_on)
    instr_resp.keys = []
    instr_resp.rt = []
    _instr_resp_allKeys = []
    # keep track of which components have finished
    instructionsComponents = [instr_txt, Fix_Example_2, instr_resp]
    for thisComponent in instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "instructions"-------
    while continueRoutine:
        # get current time
        t = instructionsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=instructionsClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr_txt* updates
        if instr_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_txt.frameNStart = frameN  # exact frame index
            instr_txt.tStart = t  # local t and not account for scr refresh
            instr_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_txt, 'tStartRefresh')  # time at next scr refresh
            instr_txt.setAutoDraw(True)
        
        # *Fix_Example_2* updates
        if Fix_Example_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Fix_Example_2.frameNStart = frameN  # exact frame index
            Fix_Example_2.tStart = t  # local t and not account for scr refresh
            Fix_Example_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Fix_Example_2, 'tStartRefresh')  # time at next scr refresh
            Fix_Example_2.setAutoDraw(True)
        
        # *instr_resp* updates
        waitOnFlip = False
        if instr_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_resp.frameNStart = frameN  # exact frame index
            instr_resp.tStart = t  # local t and not account for scr refresh
            instr_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_resp, 'tStartRefresh')  # time at next scr refresh
            instr_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr_resp.getKeys(keyList=None, waitRelease=False)
            _instr_resp_allKeys.extend(theseKeys)
            if len(_instr_resp_allKeys):
                instr_resp.keys = _instr_resp_allKeys[-1].name  # just the last key pressed
                instr_resp.rt = _instr_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "instructions"-------
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    instructions_loop.addData('instr_txt.started', instr_txt.tStartRefresh)
    instructions_loop.addData('instr_txt.stopped', instr_txt.tStopRefresh)
    instructions_loop.addData('Fix_Example_2.started', Fix_Example_2.tStartRefresh)
    instructions_loop.addData('Fix_Example_2.stopped', Fix_Example_2.tStopRefresh)
    # check responses
    if instr_resp.keys in ['', [], None]:  # No response was made
        instr_resp.keys = None
    instructions_loop.addData('instr_resp.keys',instr_resp.keys)
    if instr_resp.keys != None:  # we had a response
        instructions_loop.addData('instr_resp.rt', instr_resp.rt)
    instructions_loop.addData('instr_resp.started', instr_resp.tStartRefresh)
    instructions_loop.addData('instr_resp.stopped', instr_resp.tStopRefresh)
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1 repeats of 'instructions_loop'


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
    tk.sendCommand("calibration_type = HV5") 

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
                                        'In the task, quickly move your eyes to look at a dot (target) when it appears',
                                        font='Courier New',
                                        units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
                                        color='white', colorSpace='rgb', opacity=1, 
                                        languageStyle='LTR',
                                        depth=0.0)    
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

# set up handler to look after randomisation of conditions etc
blocks = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions(block_conditions),
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
    
    # ------Prepare to start Routine "break_2"-------
    continueRoutine = True
    routineTimer.add(120.000000)
    # update component parameters for each repeat
    #Taking a break every Block
    blocks_total = blocks.nTotal #Total # of blocks to be run 
    blocks_count = blocks.thisN
    block_complete = ''
    categorization_accuracy = ''
    
    if blocks_count == 0:
        continueRoutine = False
    else:
    #    blocks_count +=1
        block_complete = '\n\n\n\n\n\nBlocks Completed: {num_done}/{num_total}'.format(num_done = blocks_count, num_total = blocks_total)
        categorization_accuracy = '\n\n You correctly categorized {percent_correct:.0%} of the images!'.format(percent_correct =  category_correct_count/trial_counter)
        # STEP 5: Set up the tracker
        # put the tracker in idle mode before we change its parameters
        tk.setOfflineMode()
        tk.sendMessage("Break")
        pylink.pumpDelay(100)
        #Reset category count every block and trial count
        category_correct_count = 0
        trial_counter = 0
    end_break.keys = []
    end_break.rt = []
    _end_break_allKeys = []
    block_count.setText(block_complete)
    cat_accuracy.setText(categorization_accuracy )
    # keep track of which components have finished
    break_2Components = [end_break, break_text, block_count, cat_accuracy, countdown_timer]
    for thisComponent in break_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    break_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "break_2"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = break_2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=break_2Clock)
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
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "break_2"-------
    for thisComponent in break_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if end_break.keys in ['', [], None]:  # No response was made
        end_break.keys = None
    blocks.addData('end_break.keys',end_break.keys)
    if end_break.keys != None:  # we had a response
        blocks.addData('end_break.rt', end_break.rt)
    blocks.addData('end_break.started', end_break.tStartRefresh)
    blocks.addData('end_break.stopped', end_break.tStopRefresh)
    blocks.addData('break_text.started', break_text.tStartRefresh)
    blocks.addData('break_text.stopped', break_text.tStopRefresh)
    blocks.addData('block_count.started', block_count.tStartRefresh)
    blocks.addData('block_count.stopped', block_count.tStopRefresh)
    blocks.addData('cat_accuracy.started', cat_accuracy.tStartRefresh)
    blocks.addData('cat_accuracy.stopped', cat_accuracy.tStopRefresh)
    blocks.addData('countdown_timer.started', countdown_timer.tStartRefresh)
    blocks.addData('countdown_timer.stopped', countdown_timer.tStopRefresh)
    
    # ------Prepare to start Routine "eyetrack_recalib"-------
    continueRoutine = True
    # update component parameters for each repeat
    #Taking a break every N trials
    #Recalibrate eyetracker after break
    if blocks_count == 0:
        continueRoutine = False
    else:
        if expInfo['EyeTrack']:
            # STEP 6:  Show task instructions and calibrate the tracker
            msg = visual.TextStim(win, text='Calibration will begin shortly\n' +
                                            'In the task, quickly move your eyes to look at a dot (target) when it appears',
                                            font='Courier New',
                                            units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
                                            color='white', colorSpace='rgb', opacity=1, 
                                            languageStyle='LTR',
                                            depth=0.0)
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
    
    # ------Prepare to start Routine "Block_Cue"-------
    continueRoutine = True
    routineTimer.add(3.000000)
    # update component parameters for each repeat
    ##% Eye Track Code 
    # put the tracker in idle mode before we start recording
    tk.setOfflineMode()
    pylink.pumpDelay(100)
    
    # Set which trials get implemented in the condition 
    
    #Rows that refer to condition file set in loop
    conditions = pd.read_excel(trial_conditions)
    # Create dict to store indices of conditions file by block
    trials = {cond : '' for cond in conditions.exp_loc.unique() }
    for cond in conditions.exp_loc.unique():
        start_ind = conditions.index[conditions['exp_loc'] == cond][0]
        end_ind = conditions.index[conditions['exp_loc'] == cond][-1] + 1 #Adding +1 to include last index
        trials[cond] = '{start}:{end}'.format(start = start_ind, end = end_ind)
    
    
    
    #Refer to the previously copied dicts that contain all possible trials 
    #(this enables having arbitrary block size while maintaing each exemplar is presented with the same frequency
    if exp_block == 'left':
        bloc_type = trials['L']
        targets_left = expect_left_trials['left_stims'] # Higher prob real left
        targets_right = expect_left_trials['right_stims']
        
        trial_repeats = 4
        
    elif exp_block == 'right':
        bloc_type = trials['R']
        targets_left = expect_right_trials['left_stims'] # Higher prob real left
        targets_right = expect_right_trials['right_stims']
        
        trial_repeats = 4
        
    elif exp_block == 'neutral':
        bloc_type = trials['N']
        
        targets_left = expect_neutral_trials['left_stims'] # Higher prob real left
        targets_right = expect_neutral_trials['right_stims']
        
        if expInfo['Control']: 
            trial_repeats = 2 
        else: 
            trial_repeats = 4
    
    #randomize order of stimuli & freq of Real/Scram
    for exp_types in targets_left.keys(): #dict keys should be identical regardless of left vs right or 1st vs 2nd
        #Shuffle images
        shuffle(targets_left[exp_types])
        shuffle(targets_right[exp_types])
       
    # Define ITI Distribution per block 
    #Define possible ITI Vals & number of trials in a block
    
    numTrials = n_imgs * n_trial_types * n_block_types # num exemplars * num trials * blocks (Total trials in a block) (Extra to prevent empty lists)
    if expInfo['feedback']:
        ITI_vals = np.array([0.5, 1])  
    else:
        ITI_vals = np.array([1, 1.5])  
    #Sample from exponential distribution 
    lam = 1 # This is set s.t. 1/lambda = value of first interval. i.e exp dist mean is T1 
    xdist = lam * np.exp(-lam * ITI_vals)
    
    #Normalize and generate counts 
    xdist_norm = numTrials * (xdist/ sum(xdist));
    ITI_counts = np.round(xdist_norm).astype(int)
    
    #Set ITI for each trial for one block
    #And randomly shuffle 
    ITI_list = np.random.permutation(np.repeat(ITI_vals, ITI_counts))
    
    ITI_list = list(ITI_list)
    
    
    place_right.setLineColor(prob_right)
    place_left.setLineColor(prob_left)
    # keep track of which components have finished
    Block_CueComponents = [place_right, place_left, long_back_R, hi_back_R, hi_back_L, long_back_L]
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
    blocks.addData('place_right.started', place_right.tStartRefresh)
    blocks.addData('place_right.stopped', place_right.tStopRefresh)
    blocks.addData('place_left.started', place_left.tStartRefresh)
    blocks.addData('place_left.stopped', place_left.tStopRefresh)
    blocks.addData('long_back_R.started', long_back_R.tStartRefresh)
    blocks.addData('long_back_R.stopped', long_back_R.tStopRefresh)
    blocks.addData('hi_back_R.started', hi_back_R.tStartRefresh)
    blocks.addData('hi_back_R.stopped', hi_back_R.tStopRefresh)
    blocks.addData('hi_back_L.started', hi_back_L.tStartRefresh)
    blocks.addData('hi_back_L.stopped', hi_back_L.tStopRefresh)
    blocks.addData('long_back_L.started', long_back_L.tStartRefresh)
    blocks.addData('long_back_L.stopped', long_back_L.tStopRefresh)
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=trial_repeats, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(trial_conditions, selection=bloc_type),
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
        
        # ------Prepare to start Routine "init_fix"-------
        continueRoutine = True
        # update component parameters for each repeat
        ##% Eyelink Code 
        # send the standard "TRIALID" message to mark the start of a trial
        # see Data Viewer User Manual, Section 7: Protocol for EyeLink Data to Viewer Integration
        trial_count += 1 # Manual trial counter for eyelink
        trial_counter += 1 # Manual counter for categorization accuracy (gets reset every block)
        tk.sendMessage('TRIALID %d' % trial_count)
        
        # record_status_message : show some info on the Host PC - OPTIONAL
        # here we show how many trial has been tested
        tk.sendCommand("record_status_message 'Trial number %s'"% trial_count)
        # start recording    
        # arguments: sample_to_file, events_to_file, sample_over_link, event_over_link (1-yes, 0-no)
        err = tk.startRecording(1, 1, 1, 1)
        pylink.pumpDelay(100)  # wait for 100 ms to cache some samples
        
        #Get ITI for trial 
        #Initiation of ITI_list is in "Block_Cue" Routine
        ITI_duration = ITI_list.pop()
        
        thisExp.addData('ITI', ITI_duration)
        
        win.callOnFlip(tk.sendMessage,'init_fix')
        place_left_3.setLineColor(prob_left)
        place_right_3.setLineColor(prob_right)
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
        trials.addData('place_left_3.started', place_left_3.tStartRefresh)
        trials.addData('place_left_3.stopped', place_left_3.tStopRefresh)
        trials.addData('place_right_3.started', place_right_3.tStartRefresh)
        trials.addData('place_right_3.stopped', place_right_3.tStopRefresh)
        trials.addData('long_back_R_3.started', long_back_R_3.tStartRefresh)
        trials.addData('long_back_R_3.stopped', long_back_R_3.tStopRefresh)
        trials.addData('hi_back_R_3.started', hi_back_R_3.tStartRefresh)
        trials.addData('hi_back_R_3.stopped', hi_back_R_3.tStopRefresh)
        trials.addData('hi_back_L_3.started', hi_back_L_3.tStartRefresh)
        trials.addData('hi_back_L_3.stopped', hi_back_L_3.tStopRefresh)
        trials.addData('long_back_L_3.started', long_back_L_3.tStartRefresh)
        trials.addData('long_back_L_3.stopped', long_back_L_3.tStopRefresh)
        trials.addData('pre_cue_fix_2.started', pre_cue_fix_2.tStartRefresh)
        trials.addData('pre_cue_fix_2.stopped', pre_cue_fix_2.tStopRefresh)
        # the Routine "init_fix" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "att_cue"-------
        continueRoutine = True
        routineTimer.add(0.050000)
        # update component parameters for each repeat
        if att_loc == 'L':
            att_left_opac = 1
            att_right_opac = 0
            
        elif att_loc == 'R':
            att_left_opac = 0
            att_right_opac = 1
            
        elif att_loc == 'N':
            #Maybe change this later to be the same in both control and main task
            if expInfo['Control']:
                att_left_opac = 0
                att_right_opac = 0
            else:
                att_left_opac = 1
                att_right_opac = 1
        
        #EyeLink Code 
        win.callOnFlip(tk.sendMessage,'attention_cue_onset')
        place_right_5.setLineColor(prob_right)
        place_left_5.setLineColor(prob_left)
        att_left_2.setOpacity(att_left_opac)
        att_right_2.setOpacity(att_right_opac)
        att_right_2.setImage('../cues/cue_right.png')
        # keep track of which components have finished
        att_cueComponents = [place_right_5, place_left_5, long_back_R_5, hi_back_R_5, hi_back_L_5, long_back_L_5, att_left_2, att_right_2, att_cue_fix_3]
        for thisComponent in att_cueComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        att_cueClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "att_cue"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = att_cueClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=att_cueClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *place_right_5* updates
            if place_right_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                place_right_5.frameNStart = frameN  # exact frame index
                place_right_5.tStart = t  # local t and not account for scr refresh
                place_right_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(place_right_5, 'tStartRefresh')  # time at next scr refresh
                place_right_5.setAutoDraw(True)
            if place_right_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > place_right_5.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    place_right_5.tStop = t  # not accounting for scr refresh
                    place_right_5.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(place_right_5, 'tStopRefresh')  # time at next scr refresh
                    place_right_5.setAutoDraw(False)
            
            # *place_left_5* updates
            if place_left_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                place_left_5.frameNStart = frameN  # exact frame index
                place_left_5.tStart = t  # local t and not account for scr refresh
                place_left_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(place_left_5, 'tStartRefresh')  # time at next scr refresh
                place_left_5.setAutoDraw(True)
            if place_left_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > place_left_5.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    place_left_5.tStop = t  # not accounting for scr refresh
                    place_left_5.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(place_left_5, 'tStopRefresh')  # time at next scr refresh
                    place_left_5.setAutoDraw(False)
            
            # *long_back_R_5* updates
            if long_back_R_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                long_back_R_5.frameNStart = frameN  # exact frame index
                long_back_R_5.tStart = t  # local t and not account for scr refresh
                long_back_R_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(long_back_R_5, 'tStartRefresh')  # time at next scr refresh
                long_back_R_5.setAutoDraw(True)
            if long_back_R_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > long_back_R_5.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    long_back_R_5.tStop = t  # not accounting for scr refresh
                    long_back_R_5.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(long_back_R_5, 'tStopRefresh')  # time at next scr refresh
                    long_back_R_5.setAutoDraw(False)
            
            # *hi_back_R_5* updates
            if hi_back_R_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hi_back_R_5.frameNStart = frameN  # exact frame index
                hi_back_R_5.tStart = t  # local t and not account for scr refresh
                hi_back_R_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hi_back_R_5, 'tStartRefresh')  # time at next scr refresh
                hi_back_R_5.setAutoDraw(True)
            if hi_back_R_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > hi_back_R_5.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    hi_back_R_5.tStop = t  # not accounting for scr refresh
                    hi_back_R_5.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(hi_back_R_5, 'tStopRefresh')  # time at next scr refresh
                    hi_back_R_5.setAutoDraw(False)
            
            # *hi_back_L_5* updates
            if hi_back_L_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hi_back_L_5.frameNStart = frameN  # exact frame index
                hi_back_L_5.tStart = t  # local t and not account for scr refresh
                hi_back_L_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hi_back_L_5, 'tStartRefresh')  # time at next scr refresh
                hi_back_L_5.setAutoDraw(True)
            if hi_back_L_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > hi_back_L_5.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    hi_back_L_5.tStop = t  # not accounting for scr refresh
                    hi_back_L_5.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(hi_back_L_5, 'tStopRefresh')  # time at next scr refresh
                    hi_back_L_5.setAutoDraw(False)
            
            # *long_back_L_5* updates
            if long_back_L_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                long_back_L_5.frameNStart = frameN  # exact frame index
                long_back_L_5.tStart = t  # local t and not account for scr refresh
                long_back_L_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(long_back_L_5, 'tStartRefresh')  # time at next scr refresh
                long_back_L_5.setAutoDraw(True)
            if long_back_L_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > long_back_L_5.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    long_back_L_5.tStop = t  # not accounting for scr refresh
                    long_back_L_5.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(long_back_L_5, 'tStopRefresh')  # time at next scr refresh
                    long_back_L_5.setAutoDraw(False)
            
            # *att_left_2* updates
            if att_left_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                att_left_2.frameNStart = frameN  # exact frame index
                att_left_2.tStart = t  # local t and not account for scr refresh
                att_left_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(att_left_2, 'tStartRefresh')  # time at next scr refresh
                att_left_2.setAutoDraw(True)
            if att_left_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > att_left_2.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    att_left_2.tStop = t  # not accounting for scr refresh
                    att_left_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(att_left_2, 'tStopRefresh')  # time at next scr refresh
                    att_left_2.setAutoDraw(False)
            
            # *att_right_2* updates
            if att_right_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                att_right_2.frameNStart = frameN  # exact frame index
                att_right_2.tStart = t  # local t and not account for scr refresh
                att_right_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(att_right_2, 'tStartRefresh')  # time at next scr refresh
                att_right_2.setAutoDraw(True)
            if att_right_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > att_right_2.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    att_right_2.tStop = t  # not accounting for scr refresh
                    att_right_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(att_right_2, 'tStopRefresh')  # time at next scr refresh
                    att_right_2.setAutoDraw(False)
            
            # *att_cue_fix_3* updates
            if att_cue_fix_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                att_cue_fix_3.frameNStart = frameN  # exact frame index
                att_cue_fix_3.tStart = t  # local t and not account for scr refresh
                att_cue_fix_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(att_cue_fix_3, 'tStartRefresh')  # time at next scr refresh
                att_cue_fix_3.setAutoDraw(True)
            if att_cue_fix_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > att_cue_fix_3.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    att_cue_fix_3.tStop = t  # not accounting for scr refresh
                    att_cue_fix_3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(att_cue_fix_3, 'tStopRefresh')  # time at next scr refresh
                    att_cue_fix_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in att_cueComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "att_cue"-------
        for thisComponent in att_cueComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        trials.addData('place_right_5.started', place_right_5.tStartRefresh)
        trials.addData('place_right_5.stopped', place_right_5.tStopRefresh)
        trials.addData('place_left_5.started', place_left_5.tStartRefresh)
        trials.addData('place_left_5.stopped', place_left_5.tStopRefresh)
        trials.addData('long_back_R_5.started', long_back_R_5.tStartRefresh)
        trials.addData('long_back_R_5.stopped', long_back_R_5.tStopRefresh)
        trials.addData('hi_back_R_5.started', hi_back_R_5.tStartRefresh)
        trials.addData('hi_back_R_5.stopped', hi_back_R_5.tStopRefresh)
        trials.addData('hi_back_L_5.started', hi_back_L_5.tStartRefresh)
        trials.addData('hi_back_L_5.stopped', hi_back_L_5.tStopRefresh)
        trials.addData('long_back_L_5.started', long_back_L_5.tStartRefresh)
        trials.addData('long_back_L_5.stopped', long_back_L_5.tStopRefresh)
        trials.addData('att_left_2.started', att_left_2.tStartRefresh)
        trials.addData('att_left_2.stopped', att_left_2.tStopRefresh)
        trials.addData('att_right_2.started', att_right_2.tStartRefresh)
        trials.addData('att_right_2.stopped', att_right_2.tStopRefresh)
        trials.addData('att_cue_fix_3.started', att_cue_fix_3.tStartRefresh)
        trials.addData('att_cue_fix_3.stopped', att_cue_fix_3.tStopRefresh)
        
        # ------Prepare to start Routine "blank_pre"-------
        continueRoutine = True
        routineTimer.add(0.900000)
        # update component parameters for each repeat
        place_right_4.setLineColor(prob_right)
        place_left_4.setLineColor(prob_left)
        win.callOnFlip(tk.sendMessage, 'prestim_fix')
        
        #Select which stimuli to present 
        #Pick for left image 
        target_left = targets_left[trial_type].pop()
        
        #Pick right image 
        target_right = targets_right[trial_type].pop()
        
        thisExp.addData('stim_left', target_left)
        thisExp.addData('stim_right', target_right) 
        
        #Load .npy of images 
        img_left = np.load(target_left)
        img_right = np.load(target_right)
        
        #Ramp up and down stim contrast with peak 
        #contrast at 66.7 mS (4 frames assuming 60Hz)
        stim_left_contrast = quest_estimates[target_left[:13]] 
        stim_right_contrast = quest_estimates[target_right[:13]] 
        
        if not expInfo['frameRate']: #Can't measure FR, e.g. dropping frames
            nC = 4
        elif np.isclose(expInfo['frameRate'], 120, rtol = .1):
            nC = 8 #Num frames of presentation
        elif np.isclose(expInfo['frameRate'], 60, rtol = .1):
            nC =4
        else: 
            nC = 8
            
        minC = 0.001 
        maxC = stim_left_contrast
        grad_contr_left = np.linspace(minC,maxC,nC) 
        
        minC = 0.001 
        maxC = stim_right_contrast
        grad_contr_right = np.linspace(minC,maxC,nC)
        
        #New stim duration in frames
        stim_duration = len(grad_contr_left) #left and right should have same duration 
        
        
        # keep track of which components have finished
        blank_preComponents = [place_right_4, place_left_4, long_back_R_4, hi_back_R_4, hi_back_L_4, long_back_L_4, post_cue_fix2]
        for thisComponent in blank_preComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        blank_preClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "blank_pre"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = blank_preClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=blank_preClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *place_right_4* updates
            if place_right_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                place_right_4.frameNStart = frameN  # exact frame index
                place_right_4.tStart = t  # local t and not account for scr refresh
                place_right_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(place_right_4, 'tStartRefresh')  # time at next scr refresh
                place_right_4.setAutoDraw(True)
            if place_right_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > place_right_4.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    place_right_4.tStop = t  # not accounting for scr refresh
                    place_right_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(place_right_4, 'tStopRefresh')  # time at next scr refresh
                    place_right_4.setAutoDraw(False)
            
            # *place_left_4* updates
            if place_left_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                place_left_4.frameNStart = frameN  # exact frame index
                place_left_4.tStart = t  # local t and not account for scr refresh
                place_left_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(place_left_4, 'tStartRefresh')  # time at next scr refresh
                place_left_4.setAutoDraw(True)
            if place_left_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > place_left_4.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    place_left_4.tStop = t  # not accounting for scr refresh
                    place_left_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(place_left_4, 'tStopRefresh')  # time at next scr refresh
                    place_left_4.setAutoDraw(False)
            
            # *long_back_R_4* updates
            if long_back_R_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                long_back_R_4.frameNStart = frameN  # exact frame index
                long_back_R_4.tStart = t  # local t and not account for scr refresh
                long_back_R_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(long_back_R_4, 'tStartRefresh')  # time at next scr refresh
                long_back_R_4.setAutoDraw(True)
            if long_back_R_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > long_back_R_4.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    long_back_R_4.tStop = t  # not accounting for scr refresh
                    long_back_R_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(long_back_R_4, 'tStopRefresh')  # time at next scr refresh
                    long_back_R_4.setAutoDraw(False)
            
            # *hi_back_R_4* updates
            if hi_back_R_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hi_back_R_4.frameNStart = frameN  # exact frame index
                hi_back_R_4.tStart = t  # local t and not account for scr refresh
                hi_back_R_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hi_back_R_4, 'tStartRefresh')  # time at next scr refresh
                hi_back_R_4.setAutoDraw(True)
            if hi_back_R_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > hi_back_R_4.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    hi_back_R_4.tStop = t  # not accounting for scr refresh
                    hi_back_R_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(hi_back_R_4, 'tStopRefresh')  # time at next scr refresh
                    hi_back_R_4.setAutoDraw(False)
            
            # *hi_back_L_4* updates
            if hi_back_L_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hi_back_L_4.frameNStart = frameN  # exact frame index
                hi_back_L_4.tStart = t  # local t and not account for scr refresh
                hi_back_L_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hi_back_L_4, 'tStartRefresh')  # time at next scr refresh
                hi_back_L_4.setAutoDraw(True)
            if hi_back_L_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > hi_back_L_4.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    hi_back_L_4.tStop = t  # not accounting for scr refresh
                    hi_back_L_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(hi_back_L_4, 'tStopRefresh')  # time at next scr refresh
                    hi_back_L_4.setAutoDraw(False)
            
            # *long_back_L_4* updates
            if long_back_L_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                long_back_L_4.frameNStart = frameN  # exact frame index
                long_back_L_4.tStart = t  # local t and not account for scr refresh
                long_back_L_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(long_back_L_4, 'tStartRefresh')  # time at next scr refresh
                long_back_L_4.setAutoDraw(True)
            if long_back_L_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > long_back_L_4.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    long_back_L_4.tStop = t  # not accounting for scr refresh
                    long_back_L_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(long_back_L_4, 'tStopRefresh')  # time at next scr refresh
                    long_back_L_4.setAutoDraw(False)
            
            # *post_cue_fix2* updates
            if post_cue_fix2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                post_cue_fix2.frameNStart = frameN  # exact frame index
                post_cue_fix2.tStart = t  # local t and not account for scr refresh
                post_cue_fix2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(post_cue_fix2, 'tStartRefresh')  # time at next scr refresh
                post_cue_fix2.setAutoDraw(True)
            if post_cue_fix2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > post_cue_fix2.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    post_cue_fix2.tStop = t  # not accounting for scr refresh
                    post_cue_fix2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(post_cue_fix2, 'tStopRefresh')  # time at next scr refresh
                    post_cue_fix2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank_preComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "blank_pre"-------
        for thisComponent in blank_preComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        trials.addData('place_right_4.started', place_right_4.tStartRefresh)
        trials.addData('place_right_4.stopped', place_right_4.tStopRefresh)
        trials.addData('place_left_4.started', place_left_4.tStartRefresh)
        trials.addData('place_left_4.stopped', place_left_4.tStopRefresh)
        trials.addData('long_back_R_4.started', long_back_R_4.tStartRefresh)
        trials.addData('long_back_R_4.stopped', long_back_R_4.tStopRefresh)
        trials.addData('hi_back_R_4.started', hi_back_R_4.tStartRefresh)
        trials.addData('hi_back_R_4.stopped', hi_back_R_4.tStopRefresh)
        trials.addData('hi_back_L_4.started', hi_back_L_4.tStartRefresh)
        trials.addData('hi_back_L_4.stopped', hi_back_L_4.tStopRefresh)
        trials.addData('long_back_L_4.started', long_back_L_4.tStartRefresh)
        trials.addData('long_back_L_4.stopped', long_back_L_4.tStopRefresh)
        trials.addData('post_cue_fix2.started', post_cue_fix2.tStartRefresh)
        trials.addData('post_cue_fix2.stopped', post_cue_fix2.tStopRefresh)
        
        # ------Prepare to start Routine "stim"-------
        continueRoutine = True
        # update component parameters for each repeat
        win.callOnFlip(tk.sendMessage,'stim_onset')
        place_right_2.setLineColor(prob_right)
        place_left_2.setLineColor(prob_left)
        # keep track of which components have finished
        stimComponents = [place_right_2, place_left_2, long_back_R_2, hi_back_R_2, hi_back_L_2, long_back_L_2, stim_left_2, stim_right_2, stim_fix_2]
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
                F1 = grad_contr_left[frameN] * params['filter'] 
                F2 = grad_contr_right[frameN] * params['filter'] 
                left_stim = np.array(img_left * F1)
                right_stim = np.array(img_right * F2)
            
            
            
            
            # *place_right_2* updates
            if place_right_2.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                place_right_2.frameNStart = frameN  # exact frame index
                place_right_2.tStart = t  # local t and not account for scr refresh
                place_right_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(place_right_2, 'tStartRefresh')  # time at next scr refresh
                place_right_2.setAutoDraw(True)
            if place_right_2.status == STARTED:
                if frameN >= (place_right_2.frameNStart + stim_duration):
                    # keep track of stop time/frame for later
                    place_right_2.tStop = t  # not accounting for scr refresh
                    place_right_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(place_right_2, 'tStopRefresh')  # time at next scr refresh
                    place_right_2.setAutoDraw(False)
            
            # *place_left_2* updates
            if place_left_2.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                place_left_2.frameNStart = frameN  # exact frame index
                place_left_2.tStart = t  # local t and not account for scr refresh
                place_left_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(place_left_2, 'tStartRefresh')  # time at next scr refresh
                place_left_2.setAutoDraw(True)
            if place_left_2.status == STARTED:
                if frameN >= (place_left_2.frameNStart + stim_duration):
                    # keep track of stop time/frame for later
                    place_left_2.tStop = t  # not accounting for scr refresh
                    place_left_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(place_left_2, 'tStopRefresh')  # time at next scr refresh
                    place_left_2.setAutoDraw(False)
            
            # *long_back_R_2* updates
            if long_back_R_2.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                long_back_R_2.frameNStart = frameN  # exact frame index
                long_back_R_2.tStart = t  # local t and not account for scr refresh
                long_back_R_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(long_back_R_2, 'tStartRefresh')  # time at next scr refresh
                long_back_R_2.setAutoDraw(True)
            if long_back_R_2.status == STARTED:
                if frameN >= (long_back_R_2.frameNStart + stim_duration):
                    # keep track of stop time/frame for later
                    long_back_R_2.tStop = t  # not accounting for scr refresh
                    long_back_R_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(long_back_R_2, 'tStopRefresh')  # time at next scr refresh
                    long_back_R_2.setAutoDraw(False)
            
            # *hi_back_R_2* updates
            if hi_back_R_2.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                hi_back_R_2.frameNStart = frameN  # exact frame index
                hi_back_R_2.tStart = t  # local t and not account for scr refresh
                hi_back_R_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hi_back_R_2, 'tStartRefresh')  # time at next scr refresh
                hi_back_R_2.setAutoDraw(True)
            if hi_back_R_2.status == STARTED:
                if frameN >= (hi_back_R_2.frameNStart + stim_duration):
                    # keep track of stop time/frame for later
                    hi_back_R_2.tStop = t  # not accounting for scr refresh
                    hi_back_R_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(hi_back_R_2, 'tStopRefresh')  # time at next scr refresh
                    hi_back_R_2.setAutoDraw(False)
            
            # *hi_back_L_2* updates
            if hi_back_L_2.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                hi_back_L_2.frameNStart = frameN  # exact frame index
                hi_back_L_2.tStart = t  # local t and not account for scr refresh
                hi_back_L_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hi_back_L_2, 'tStartRefresh')  # time at next scr refresh
                hi_back_L_2.setAutoDraw(True)
            if hi_back_L_2.status == STARTED:
                if frameN >= (hi_back_L_2.frameNStart + stim_duration):
                    # keep track of stop time/frame for later
                    hi_back_L_2.tStop = t  # not accounting for scr refresh
                    hi_back_L_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(hi_back_L_2, 'tStopRefresh')  # time at next scr refresh
                    hi_back_L_2.setAutoDraw(False)
            
            # *long_back_L_2* updates
            if long_back_L_2.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                long_back_L_2.frameNStart = frameN  # exact frame index
                long_back_L_2.tStart = t  # local t and not account for scr refresh
                long_back_L_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(long_back_L_2, 'tStartRefresh')  # time at next scr refresh
                long_back_L_2.setAutoDraw(True)
            if long_back_L_2.status == STARTED:
                if frameN >= (long_back_L_2.frameNStart + stim_duration):
                    # keep track of stop time/frame for later
                    long_back_L_2.tStop = t  # not accounting for scr refresh
                    long_back_L_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(long_back_L_2, 'tStopRefresh')  # time at next scr refresh
                    long_back_L_2.setAutoDraw(False)
            
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
        trials.addData('place_right_2.started', place_right_2.tStartRefresh)
        trials.addData('place_right_2.stopped', place_right_2.tStopRefresh)
        trials.addData('place_left_2.started', place_left_2.tStartRefresh)
        trials.addData('place_left_2.stopped', place_left_2.tStopRefresh)
        trials.addData('long_back_R_2.started', long_back_R_2.tStartRefresh)
        trials.addData('long_back_R_2.stopped', long_back_R_2.tStopRefresh)
        trials.addData('hi_back_R_2.started', hi_back_R_2.tStartRefresh)
        trials.addData('hi_back_R_2.stopped', hi_back_R_2.tStopRefresh)
        trials.addData('hi_back_L_2.started', hi_back_L_2.tStartRefresh)
        trials.addData('hi_back_L_2.stopped', hi_back_L_2.tStopRefresh)
        trials.addData('long_back_L_2.started', long_back_L_2.tStartRefresh)
        trials.addData('long_back_L_2.stopped', long_back_L_2.tStopRefresh)
        trials.addData('stim_left_2.started', stim_left_2.tStartRefresh)
        trials.addData('stim_left_2.stopped', stim_left_2.tStopRefresh)
        trials.addData('stim_right_2.started', stim_right_2.tStartRefresh)
        trials.addData('stim_right_2.stopped', stim_right_2.tStopRefresh)
        trials.addData('stim_fix_2.started', stim_fix_2.tStartRefresh)
        trials.addData('stim_fix_2.stopped', stim_fix_2.tStopRefresh)
        # the Routine "stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "blank_post"-------
        continueRoutine = True
        routineTimer.add(0.400000)
        # update component parameters for each repeat
        win.callOnFlip(tk.sendMessage,'poststim_fix')
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
        trials.addData('post_stim_fix.started', post_stim_fix.tStartRefresh)
        trials.addData('post_stim_fix.stopped', post_stim_fix.tStopRefresh)
        
        # ------Prepare to start Routine "q_cat"-------
        continueRoutine = True
        routineTimer.add(3.000000)
        # update component parameters for each repeat
        ## Write Code to set cue to be valid or not valid 
        #based on conditions file 
        
        if probe_loc == 'L':
            probe_cue_img = '../cues/cue_left.png'
        elif probe_loc == 'R':
            probe_cue_img = '../cues/cue_right.png'
        
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
        #Eyetrack
        win.callOnFlip(tk.sendMessage,'q_cat_onset')
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
        
        if probe_loc == 'L':
            corAns = button_map[target_left[10]] # Index first letter after " images/" in target
            
        elif probe_loc == 'R':
            corAns = button_map[target_right[10]] # Index first letter after " images/" in target
        
        correct_category = cat_map[corAns]
        trials.addData('objective_category', cat_map[corAns])
        
        if cat_resp.keys:
            category_response = cat_map[cat_resp.keys[-1]]
            trials.addData('category_response', category_response)
        
        else: #If no response
            category_response = None
            trials.addData('category_response', category_response)
        
        #Calculate categorization accuracy
        if correct_category == category_response:
            trials.addData('category_correct', 1)
            category_correct_count += 1 
        else:
            trials.addData('category_correct', 0)
        
        
        
        
            
        
        trials.addData('q_category.started', q_category.tStartRefresh)
        trials.addData('q_category.stopped', q_category.tStopRefresh)
        # check responses
        if cat_resp.keys in ['', [], None]:  # No response was made
            cat_resp.keys = None
        trials.addData('cat_resp.keys',cat_resp.keys)
        if cat_resp.keys != None:  # we had a response
            trials.addData('cat_resp.rt', cat_resp.rt)
        trials.addData('cat_resp.started', cat_resp.tStartRefresh)
        trials.addData('cat_resp.stopped', cat_resp.tStopRefresh)
        trials.addData('probe_cue_cat.started', probe_cue_cat.tStartRefresh)
        trials.addData('probe_cue_cat.stopped', probe_cue_cat.tStopRefresh)
        
        # ------Prepare to start Routine "q_rec"-------
        continueRoutine = True
        routineTimer.add(3.000000)
        # update component parameters for each repeat
        ## Write Code to set cue to be valid or not valid 
        #based on conditions file 
        if probe_loc == 'L':
            probe_cue_img_2 = '../cues/cue_left.png'
            
        elif probe_loc == 'R':
            probe_cue_img_2 = '../cues/cue_right.png'
        
        #Eyetracker msg
        win.callOnFlip(tk.sendMessage,'q_rec_onset')
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
        # Real/Scram is always mapped to 7/8, assumed read L-R
        buttons = ['7','8']
        realScram = ['Real', 'Scram']
        
        if probe_loc == 'L':
            if 'scram' in target_left:
                corAns = '8'
            else:
                corAns = '7'
        
        elif probe_loc == 'R':
            if 'scram' in target_right:
                corAns = '8'
            else:
                corAns = '7'
        
        if rec_resp.keys:
            detect_response = rec_resp.keys[-1]
        else: #If no response
            detect_response = None
        
        
        if detect_response == corAns:
            corDetect = 1 
        else:
            corDetect = 0 
        
        # Add whether trial was valid or not 
        trials.addData('validity', is_valid)
        
        trials.addData('q_recognition.started', q_recognition.tStartRefresh)
        trials.addData('q_recognition.stopped', q_recognition.tStopRefresh)
        # check responses
        if rec_resp.keys in ['', [], None]:  # No response was made
            rec_resp.keys = None
        trials.addData('rec_resp.keys',rec_resp.keys)
        if rec_resp.keys != None:  # we had a response
            trials.addData('rec_resp.rt', rec_resp.rt)
        trials.addData('rec_resp.started', rec_resp.tStartRefresh)
        trials.addData('rec_resp.stopped', rec_resp.tStopRefresh)
        trials.addData('probe_cue_rec.started', probe_cue_rec.tStartRefresh)
        trials.addData('probe_cue_rec.stopped', probe_cue_rec.tStopRefresh)
        
        # ------Prepare to start Routine "q_conf"-------
        continueRoutine = True
        routineTimer.add(3.000000)
        # update component parameters for each repeat
        ## Write Code to set cue to be valid or not valid 
        #based on conditions file 
        
        if probe_loc == 'L':
            probe_cue_img = '../cues/cue_left.png'
        elif probe_loc == 'R':
            probe_cue_img = '../cues/cue_right.png'
        
        # Randomize Order of FAOH 
        #Initialize cat components
        
        
        conf_lvls = ['1', '2', '3', '4'] 
        conf_question = (
                        u'Confidence:\n\n\n\n\n'
                        u'{cats1:\u00a0^9} {cats2:\u00a0^9} {cats3:\u00a0^9} {cats4:\u00a0^9}\n\n\n'
                        u'{b1:\u00a0^9} {b2:\u00a0^9} {b3:\u00a0^9} {b4:\u00a0^9}'.format(cats1 = conf_lvls[0].encode().decode('utf-8'), 
                                                                   cats2 = conf_lvls[1].encode().decode('utf-8'), 
                                                                   cats3 = conf_lvls[2].encode().decode('utf-8'), 
                                                                   cats4 = conf_lvls[3].encode().decode('utf-8'),
                                                                   b1 = u'1',b2 = u'2',b3 = u'3',b4 = u'4'
                                                                    ))
        #Eyetrack
        win.callOnFlip(tk.sendMessage,'q_conf_onset')
        q_confidence.setText(conf_question)
        conf_resp.keys = []
        conf_resp.rt = []
        _conf_resp_allKeys = []
        probe_cue_cat_2.setImage(probe_cue_img)
        # keep track of which components have finished
        q_confComponents = [q_confidence, conf_resp, probe_cue_cat_2]
        for thisComponent in q_confComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        q_confClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "q_conf"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = q_confClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=q_confClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *q_confidence* updates
            if q_confidence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                q_confidence.frameNStart = frameN  # exact frame index
                q_confidence.tStart = t  # local t and not account for scr refresh
                q_confidence.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(q_confidence, 'tStartRefresh')  # time at next scr refresh
                q_confidence.setAutoDraw(True)
            if q_confidence.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > q_confidence.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    q_confidence.tStop = t  # not accounting for scr refresh
                    q_confidence.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(q_confidence, 'tStopRefresh')  # time at next scr refresh
                    q_confidence.setAutoDraw(False)
            
            # *conf_resp* updates
            waitOnFlip = False
            if conf_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                conf_resp.frameNStart = frameN  # exact frame index
                conf_resp.tStart = t  # local t and not account for scr refresh
                conf_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(conf_resp, 'tStartRefresh')  # time at next scr refresh
                conf_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(conf_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(conf_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if conf_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > conf_resp.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    conf_resp.tStop = t  # not accounting for scr refresh
                    conf_resp.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(conf_resp, 'tStopRefresh')  # time at next scr refresh
                    conf_resp.status = FINISHED
            if conf_resp.status == STARTED and not waitOnFlip:
                theseKeys = conf_resp.getKeys(keyList=['6', '7', '8', '9'], waitRelease=False)
                _conf_resp_allKeys.extend(theseKeys)
                if len(_conf_resp_allKeys):
                    conf_resp.keys = _conf_resp_allKeys[-1].name  # just the last key pressed
                    conf_resp.rt = _conf_resp_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False
            
            # *probe_cue_cat_2* updates
            if probe_cue_cat_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                probe_cue_cat_2.frameNStart = frameN  # exact frame index
                probe_cue_cat_2.tStart = t  # local t and not account for scr refresh
                probe_cue_cat_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe_cue_cat_2, 'tStartRefresh')  # time at next scr refresh
                probe_cue_cat_2.setAutoDraw(True)
            if probe_cue_cat_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > probe_cue_cat_2.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    probe_cue_cat_2.tStop = t  # not accounting for scr refresh
                    probe_cue_cat_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(probe_cue_cat_2, 'tStopRefresh')  # time at next scr refresh
                    probe_cue_cat_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in q_confComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "q_conf"-------
        for thisComponent in q_confComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        trials.addData('q_confidence.started', q_confidence.tStartRefresh)
        trials.addData('q_confidence.stopped', q_confidence.tStopRefresh)
        # check responses
        if conf_resp.keys in ['', [], None]:  # No response was made
            conf_resp.keys = None
        trials.addData('conf_resp.keys',conf_resp.keys)
        if conf_resp.keys != None:  # we had a response
            trials.addData('conf_resp.rt', conf_resp.rt)
        trials.addData('conf_resp.started', conf_resp.tStartRefresh)
        trials.addData('conf_resp.stopped', conf_resp.tStopRefresh)
        trials.addData('probe_cue_cat_2.started', probe_cue_cat_2.tStartRefresh)
        trials.addData('probe_cue_cat_2.stopped', probe_cue_cat_2.tStopRefresh)
        
        # ------Prepare to start Routine "feedback"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        if expInfo['feedback']:
            if corDetect:
                fbck_cor.opacity = 0.7
                fbck_incor.opacity = 0
            else: 
                fbck_cor.opacity = 0
                fbck_incor.opacity = 0.7
        else: 
            continueRoutine = False
            
            
            
        # keep track of which components have finished
        feedbackComponents = [fbck_cor, fbck_incor]
        for thisComponent in feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "feedback"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = feedbackClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=feedbackClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fbck_cor* updates
            if fbck_cor.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fbck_cor.frameNStart = frameN  # exact frame index
                fbck_cor.tStart = t  # local t and not account for scr refresh
                fbck_cor.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fbck_cor, 'tStartRefresh')  # time at next scr refresh
                fbck_cor.setAutoDraw(True)
            if fbck_cor.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fbck_cor.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    fbck_cor.tStop = t  # not accounting for scr refresh
                    fbck_cor.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fbck_cor, 'tStopRefresh')  # time at next scr refresh
                    fbck_cor.setAutoDraw(False)
            
            # *fbck_incor* updates
            if fbck_incor.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fbck_incor.frameNStart = frameN  # exact frame index
                fbck_incor.tStart = t  # local t and not account for scr refresh
                fbck_incor.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fbck_incor, 'tStartRefresh')  # time at next scr refresh
                fbck_incor.setAutoDraw(True)
            if fbck_incor.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fbck_incor.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    fbck_incor.tStop = t  # not accounting for scr refresh
                    fbck_incor.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fbck_incor, 'tStopRefresh')  # time at next scr refresh
                    fbck_incor.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "feedback"-------
        for thisComponent in feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        #Send Trial Information to Eyetracker for later condition sorting
        (tk.sendMessage, trial_type)
        ##%  Eye Link Code 
        #Stop recording
        tk.stopRecording()
        # send trial variables to record in the EDF data file
        # see Data Viewer User Manual, Section 7: Protocol for EyeLink Data to Viewer Integration
        
        # send over the standard 'TRIAL_RESULT' message to mark the end of trial
        # see Data Viewer User Manual, Section 7: Protocol for EyeLink Data to Viewer Integration
        tk.sendMessage('TRIAL_RESULT %d' % trial_count)
        trials.addData('fbck_cor.started', fbck_cor.tStartRefresh)
        trials.addData('fbck_cor.stopped', fbck_cor.tStopRefresh)
        trials.addData('fbck_incor.started', fbck_incor.tStartRefresh)
        trials.addData('fbck_incor.stopped', fbck_incor.tStopRefresh)
        thisExp.nextEntry()
        
    # completed trial_repeats repeats of 'trials'
    
# completed 1 repeats of 'blocks'


# ------Prepare to start Routine "Thanks_msg"-------
continueRoutine = True
# update component parameters for each repeat
if expInfo['EyeTrack']:
    # STEP 8: close the EDF data file and put the tracker in idle mode
    tk.setOfflineMode()
    pylink.pumpDelay(100)
    tk.closeDataFile()

    # STEP 9: download EDF file to Display PC and put it in local folder ('edfData')
    msg = 'Thank you for being a participant! \n\n\n\n\n\nEDF data is transfering from EyeLink Host PC...'
    edfTransfer = visual.TextStim(win, text=msg, color='white')
    edfTransfer.draw()
    win.flip()
    pylink.pumpDelay(500)

    # make sure the 'edfData' folder is there, create one if not
    dataFolder = os.getcwd() + '/edfData/'
    if not os.path.exists(dataFolder): 
        os.makedirs(dataFolder)
    tk.receiveDataFile(dataFileName, 'edfData' + os.sep + dataFileName)

    # STEP 10: close the connection to tracker
    tk.close()

#    # STEP 11: make sure everything is closed down
#    core.quit()
else:
    pass
EndExp.keys = []
EndExp.rt = []
_EndExp_allKeys = []
# keep track of which components have finished
Thanks_msgComponents = [thanks_txt, EndExp]
for thisComponent in Thanks_msgComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Thanks_msgClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Thanks_msg"-------
while continueRoutine:
    # get current time
    t = Thanks_msgClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Thanks_msgClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *thanks_txt* updates
    if thanks_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        thanks_txt.frameNStart = frameN  # exact frame index
        thanks_txt.tStart = t  # local t and not account for scr refresh
        thanks_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(thanks_txt, 'tStartRefresh')  # time at next scr refresh
        thanks_txt.setAutoDraw(True)
    
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
    for thisComponent in Thanks_msgComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Thanks_msg"-------
for thisComponent in Thanks_msgComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('thanks_txt.started', thanks_txt.tStartRefresh)
thisExp.addData('thanks_txt.stopped', thanks_txt.tStopRefresh)
# check responses
if EndExp.keys in ['', [], None]:  # No response was made
    EndExp.keys = None
thisExp.addData('EndExp.keys',EndExp.keys)
if EndExp.keys != None:  # we had a response
    thisExp.addData('EndExp.rt', EndExp.rt)
thisExp.addData('EndExp.started', EndExp.tStartRefresh)
thisExp.addData('EndExp.stopped', EndExp.tStopRefresh)
thisExp.nextEntry()
# the Routine "Thanks_msg" was not non-slip safe, so reset the non-slip timer
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
