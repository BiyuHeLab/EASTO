#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.10),
    on February 02, 2022, at 17:28
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

import numpy as np 
import math
import EASTO_funcs as ea 

#Define possible ITI Vals & number of trials in a block
numTrials = 24*35 * 2 # num trials * num quest tracks
ITI_vals = np.array([.75, 1])  #Reduced from 1, 1.5 to account for feedback time
#Sample from exponential distribution 
lam = 1.33 # This is set s.t. 1/lambda = value of first interval. i.e exp dist mean is T1 
xdist = lam * np.exp(-lam * ITI_vals)

#Normalize and generate counts 
xdist_norm = numTrials * (xdist/ sum(xdist));
ITI_counts = np.round(xdist_norm).astype(int)

#Set ITI for each trial for one block
#And randomly shuffle 
ITI_list = np.random.permutation(np.repeat(ITI_vals, ITI_counts))

ITI_list = list(ITI_list)

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
expName = 'SEA_PracticeQ'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
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
    originPath='\\\\homedir-cifs.nyumc.org\\bc1693\\Personal\\EASTO\\Paradigm\\Paradigms\\Master\\Spatial\\Practice\\SEA_PracticeQ_lastrun.py',
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
    size=[1050, 1680], fullscr=True, screen=0, 
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

# Initialize components for Routine "gen_instr"
gen_instrClock = core.Clock()
instrText_4 = visual.TextStim(win=win, name='instrText_4',
    text='In this part of the experiment you will be asked to make judgements about simple images\n\nYou will see images of faces, animals, objects, and houses\n\nYou will be asked to indicate the image category and to report your visual experience of the image presented\n\nPress any button to continue\n',
    font='Courier New',
    units='norm', pos=[0, 0], height=.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_5 = keyboard.Keyboard()

# Initialize components for Routine "overview"
overviewClock = core.Clock()
exp_overview = visual.TextStim(win=win, name='exp_overview',
    text='This experiment will include the following parts: \n\n(1) Perception Task (~35 minutes)\n(2) Guided Perception Task (~1.5 hours)\n\nYou will be able to take mental breaks every 5-10 minutes\n\nPress any button to continue',
    font='Courier New',
    units='norm', pos=[0, 0], height=.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_3 = keyboard.Keyboard()

# Initialize components for Routine "fixation_2"
fixation_2Clock = core.Clock()
fix_instr = visual.TextStim(win=win, name='fix_instr',
    text='In the middle of the screen you can see a fixation cross,\n\n\n\n\n\n\n\n\n\nRemember to always keep your eyes on this cross when you see it\n\nPress any button to continue',
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
endInstructions_4 = keyboard.Keyboard()

# Initialize components for Routine "Exemplars"
ExemplarsClock = core.Clock()
stim_fix_3 = visual.ShapeStim(
    win=win, name='stim_fix_3', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=.7, depth=-1.0, interpolate=True)
stim_pres = visual.ImageStim(
    win=win,
    name='stim_pres', 
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=(4, 4),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=True,
    texRes=512, interpolate=True, depth=-2.0)
endInstructions_6 = keyboard.Keyboard()
img_instr = visual.TextStim(win=win, name='img_instr',
    text='Here are examples of images you will see in both tasks. Please make sure you are able to define the category of each image\n\n\n\n\n\n\n\n\n\n\n\nPress any button to continue to the next image',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-4.0);
img_cats = visual.TextStim(win=win, name='img_cats',
    text='\n\n\n\n\n\n\n\n\n\nFace | Animal | House | Object',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=500, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-5.0);

# Initialize components for Routine "instr_cues"
instr_cuesClock = core.Clock()
instr_txt_cue = visual.TextStim(win=win, name='instr_txt_cue',
    text='Every trial you will see two images on the left and right side of the fixation cross. \n\nOne image will have meaningful content and the other will not. \n\nThe questions will be presented alongside an arrow cue that indicates which image you should answer the question about. \n\nFor this part of the experiment this cue will ALWAYS correspond to the image that had meaningful content. \n\n\npress any button to continue\n',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
instr_resp_cue = keyboard.Keyboard()

# Initialize components for Routine "instr_exp_cue"
instr_exp_cueClock = core.Clock()
place_right = visual.Rect(
    win=win, name='place_right',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)
place_left = visual.Rect(
    win=win, name='place_left',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)
long_back_R = visual.Rect(
    win=win, name='long_back_R',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)
hi_back_R = visual.Rect(
    win=win, name='hi_back_R',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_L = visual.Rect(
    win=win, name='hi_back_L',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
long_back_L = visual.Rect(
    win=win, name='long_back_L',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
place_explan = visual.TextStim(win=win, name='place_explan',
    text='To help you identify which image will have meaningful content and which will not, both stimuli will be contained in a placeholder.\n\n\n\n\n\n\n\n\n\n\n\nThe BLACK placeholder indicates that a meaningful image will appear within the placeholder. The WHITE placeholder indicates that a noise image will appear within the placeholder\n\npress any button to continue',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-6.0);
endInstructions_9 = keyboard.Keyboard()

# Initialize components for Routine "Q_cat"
Q_catClock = core.Clock()
text_5 = visual.TextStim(win=win, name='text_5',
    text='The images will be presented very briefly. \nThe questions will always be presented with a cue (left or right pointing arrow) that indicates which image you should answer the question about\n\nIf you do not see anything - this is normal, just try your best to guess the correct image category. \n\nIf you need to make a guess, please make a genuine guess. That is, do not use a systematic strategy for your guesses, \nlike always pressing "1" or always pressing the opposite of what you pressed last.\n\nNo matter your visual experience, please answer every question. \n\nPress any button to continue\n',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_7 = keyboard.Keyboard()

# Initialize components for Routine "q_cat_exemp"
q_cat_exempClock = core.Clock()
q_category_2 = visual.TextStim(win=win, name='q_category_2',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.1, wrapWidth=500, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
cat_resp_2 = keyboard.Keyboard()
probe_cue_cat_3 = visual.ImageStim(
    win=win,
    name='probe_cue_cat_3', 
    image='sin', mask=None,
    ori=0, pos=(0,0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# Initialize components for Routine "Q_Rec"
Q_RecClock = core.Clock()
text_6 = visual.TextStim(win=win, name='text_6',
    text='You will also be asked if you had any meaningful visual experience.\n\nFor example: if you saw an animal, or only part of an animal, as long as you have a sense of the image\'s identity - please answer "yes".\n\nThere will be cases where you will see only a noisy glimpse of light or simply nothing. If this is the case, the experience is not meaningful - please answer "no".\n\nPress any button to continue',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_8 = keyboard.Keyboard()

# Initialize components for Routine "q_rec_exemp"
q_rec_exempClock = core.Clock()
rec_question  = (u'Meaningful visual experience: \n\n\n\n\n '
            u'{Y:\u00a0^15} {N:\u00a0^15}''\n\n\n '
            u'{b2:\u00a0^15} {b3:\u00a0^15}'.format(Y = u'Yes',
                                               N = u'No',
                                               b2 = u'2',
                                               b3 = u'3 '))
q_recognition_3 = visual.TextStim(win=win, name='q_recognition_3',
    text=rec_question,
    font='Courier New',
    units='norm', pos=(0, 0), height=0.1, wrapWidth=500, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
rec_resp = keyboard.Keyboard()
probe_cue_rec_2 = visual.ImageStim(
    win=win,
    name='probe_cue_rec_2', 
    image='sin', mask=None,
    ori=0, pos=(0,0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# Initialize components for Routine "Summary"
SummaryClock = core.Clock()
text_8 = visual.TextStim(win=win, name='text_8',
    text='Summary of task: \n\n1. Fixation \n2. Image Presentation (Left and Right)\n3. Category Question: Guess if unsure \n4. Visual Experience: Yes or No \n5. Always respond \n\n\nPress any button to continue',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_10 = keyboard.Keyboard()

# Initialize components for Routine "begin_practice"
begin_practiceClock = core.Clock()
begin = visual.TextStim(win=win, name='begin',
    text='Time to practice some trials \n\n\n\n\nPress any button to begin',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_4 = keyboard.Keyboard()

# Initialize components for Routine "block_break"
block_breakClock = core.Clock()
blocks_count = 0
num_exemps = 16
num_trials = 4
#Initialize correct category counter
category_correct_count = 0
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

# Initialize components for Routine "init_fix"
init_fixClock = core.Clock()
place_left_3 = visual.Rect(
    win=win, name='place_left_3',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)
place_right_3 = visual.Rect(
    win=win, name='place_right_3',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)
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
    opacity=1, depth=-1.0, interpolate=True)
place_right_4 = visual.Rect(
    win=win, name='place_right_4',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)
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
rec_resp_2 = keyboard.Keyboard()
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
    text='Thank you! \n\nYou can relax now\n\n\n\n\n\nThe next part of the experiment will begin shortly. It may take a few minutes.',
    font='Courier New',
    units='norm', pos=[0, 0], height=0.07, wrapWidth=None, ori=0, 
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

# ------Prepare to start Routine "gen_instr"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_5.keys = []
endInstructions_5.rt = []
_endInstructions_5_allKeys = []
# keep track of which components have finished
gen_instrComponents = [instrText_4, endInstructions_5]
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
    
    # *instrText_4* updates
    if instrText_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instrText_4.frameNStart = frameN  # exact frame index
        instrText_4.tStart = t  # local t and not account for scr refresh
        instrText_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instrText_4, 'tStartRefresh')  # time at next scr refresh
        instrText_4.setAutoDraw(True)
    
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
thisExp.addData('instrText_4.started', instrText_4.tStartRefresh)
thisExp.addData('instrText_4.stopped', instrText_4.tStopRefresh)
# check responses
if endInstructions_5.keys in ['', [], None]:  # No response was made
    endInstructions_5.keys = None
thisExp.addData('endInstructions_5.keys',endInstructions_5.keys)
if endInstructions_5.keys != None:  # we had a response
    thisExp.addData('endInstructions_5.rt', endInstructions_5.rt)
thisExp.addData('endInstructions_5.started', endInstructions_5.tStartRefresh)
thisExp.addData('endInstructions_5.stopped', endInstructions_5.tStopRefresh)
thisExp.nextEntry()
# the Routine "gen_instr" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "overview"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_3.keys = []
endInstructions_3.rt = []
_endInstructions_3_allKeys = []
# keep track of which components have finished
overviewComponents = [exp_overview, endInstructions_3]
for thisComponent in overviewComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
overviewClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "overview"-------
while continueRoutine:
    # get current time
    t = overviewClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=overviewClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *exp_overview* updates
    if exp_overview.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        exp_overview.frameNStart = frameN  # exact frame index
        exp_overview.tStart = t  # local t and not account for scr refresh
        exp_overview.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(exp_overview, 'tStartRefresh')  # time at next scr refresh
        exp_overview.setAutoDraw(True)
    
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
    for thisComponent in overviewComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "overview"-------
for thisComponent in overviewComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('exp_overview.started', exp_overview.tStartRefresh)
thisExp.addData('exp_overview.stopped', exp_overview.tStopRefresh)
# check responses
if endInstructions_3.keys in ['', [], None]:  # No response was made
    endInstructions_3.keys = None
thisExp.addData('endInstructions_3.keys',endInstructions_3.keys)
if endInstructions_3.keys != None:  # we had a response
    thisExp.addData('endInstructions_3.rt', endInstructions_3.rt)
thisExp.addData('endInstructions_3.started', endInstructions_3.tStartRefresh)
thisExp.addData('endInstructions_3.stopped', endInstructions_3.tStopRefresh)
thisExp.nextEntry()
# the Routine "overview" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "fixation_2"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_4.keys = []
endInstructions_4.rt = []
_endInstructions_4_allKeys = []
# keep track of which components have finished
fixation_2Components = [fix_instr, Fix_Example, endInstructions_4]
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
if endInstructions_4.keys in ['', [], None]:  # No response was made
    endInstructions_4.keys = None
thisExp.addData('endInstructions_4.keys',endInstructions_4.keys)
if endInstructions_4.keys != None:  # we had a response
    thisExp.addData('endInstructions_4.rt', endInstructions_4.rt)
thisExp.addData('endInstructions_4.started', endInstructions_4.tStartRefresh)
thisExp.addData('endInstructions_4.stopped', endInstructions_4.tStopRefresh)
thisExp.nextEntry()
# the Routine "fixation_2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
exemps = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('exemp_list.xlsx'),
    seed=None, name='exemps')
thisExp.addLoop(exemps)  # add the loop to the experiment
thisExemp = exemps.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisExemp.rgb)
if thisExemp != None:
    for paramName in thisExemp:
        exec('{} = thisExemp[paramName]'.format(paramName))

for thisExemp in exemps:
    currentLoop = exemps
    # abbreviate parameter names if possible (e.g. rgb = thisExemp.rgb)
    if thisExemp != None:
        for paramName in thisExemp:
            exec('{} = thisExemp[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "Exemplars"-------
    continueRoutine = True
    # update component parameters for each repeat
    img = np.load(stim_img) 
    
    #Show exemplars are arbitrarily low contrast 
    exemplar = np.array(img  * params['filter'] * 0.35)
    
    stim_pres.setImage(exemplar)
    endInstructions_6.keys = []
    endInstructions_6.rt = []
    _endInstructions_6_allKeys = []
    # keep track of which components have finished
    ExemplarsComponents = [stim_fix_3, stim_pres, endInstructions_6, img_instr, img_cats]
    for thisComponent in ExemplarsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    ExemplarsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Exemplars"-------
    while continueRoutine:
        # get current time
        t = ExemplarsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=ExemplarsClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *stim_fix_3* updates
        if stim_fix_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            stim_fix_3.frameNStart = frameN  # exact frame index
            stim_fix_3.tStart = t  # local t and not account for scr refresh
            stim_fix_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stim_fix_3, 'tStartRefresh')  # time at next scr refresh
            stim_fix_3.setAutoDraw(True)
        
        # *stim_pres* updates
        if stim_pres.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            stim_pres.frameNStart = frameN  # exact frame index
            stim_pres.tStart = t  # local t and not account for scr refresh
            stim_pres.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stim_pres, 'tStartRefresh')  # time at next scr refresh
            stim_pres.setAutoDraw(True)
        
        # *endInstructions_6* updates
        waitOnFlip = False
        if endInstructions_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endInstructions_6.frameNStart = frameN  # exact frame index
            endInstructions_6.tStart = t  # local t and not account for scr refresh
            endInstructions_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endInstructions_6, 'tStartRefresh')  # time at next scr refresh
            endInstructions_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endInstructions_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endInstructions_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endInstructions_6.status == STARTED and not waitOnFlip:
            theseKeys = endInstructions_6.getKeys(keyList=None, waitRelease=False)
            _endInstructions_6_allKeys.extend(theseKeys)
            if len(_endInstructions_6_allKeys):
                endInstructions_6.keys = _endInstructions_6_allKeys[-1].name  # just the last key pressed
                endInstructions_6.rt = _endInstructions_6_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *img_instr* updates
        if img_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img_instr.frameNStart = frameN  # exact frame index
            img_instr.tStart = t  # local t and not account for scr refresh
            img_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img_instr, 'tStartRefresh')  # time at next scr refresh
            img_instr.setAutoDraw(True)
        
        # *img_cats* updates
        if img_cats.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img_cats.frameNStart = frameN  # exact frame index
            img_cats.tStart = t  # local t and not account for scr refresh
            img_cats.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img_cats, 'tStartRefresh')  # time at next scr refresh
            img_cats.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ExemplarsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Exemplars"-------
    for thisComponent in ExemplarsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    exemps.addData('stim_fix_3.started', stim_fix_3.tStartRefresh)
    exemps.addData('stim_fix_3.stopped', stim_fix_3.tStopRefresh)
    exemps.addData('stim_pres.started', stim_pres.tStartRefresh)
    exemps.addData('stim_pres.stopped', stim_pres.tStopRefresh)
    # check responses
    if endInstructions_6.keys in ['', [], None]:  # No response was made
        endInstructions_6.keys = None
    exemps.addData('endInstructions_6.keys',endInstructions_6.keys)
    if endInstructions_6.keys != None:  # we had a response
        exemps.addData('endInstructions_6.rt', endInstructions_6.rt)
    exemps.addData('endInstructions_6.started', endInstructions_6.tStartRefresh)
    exemps.addData('endInstructions_6.stopped', endInstructions_6.tStopRefresh)
    exemps.addData('img_instr.started', img_instr.tStartRefresh)
    exemps.addData('img_instr.stopped', img_instr.tStopRefresh)
    exemps.addData('img_cats.started', img_cats.tStartRefresh)
    exemps.addData('img_cats.stopped', img_cats.tStopRefresh)
    # the Routine "Exemplars" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1 repeats of 'exemps'


# ------Prepare to start Routine "instr_cues"-------
continueRoutine = True
# update component parameters for each repeat
instr_resp_cue.keys = []
instr_resp_cue.rt = []
_instr_resp_cue_allKeys = []
# keep track of which components have finished
instr_cuesComponents = [instr_txt_cue, instr_resp_cue]
for thisComponent in instr_cuesComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instr_cuesClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr_cues"-------
while continueRoutine:
    # get current time
    t = instr_cuesClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr_cuesClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instr_txt_cue* updates
    if instr_txt_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instr_txt_cue.frameNStart = frameN  # exact frame index
        instr_txt_cue.tStart = t  # local t and not account for scr refresh
        instr_txt_cue.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instr_txt_cue, 'tStartRefresh')  # time at next scr refresh
        instr_txt_cue.setAutoDraw(True)
    
    # *instr_resp_cue* updates
    waitOnFlip = False
    if instr_resp_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instr_resp_cue.frameNStart = frameN  # exact frame index
        instr_resp_cue.tStart = t  # local t and not account for scr refresh
        instr_resp_cue.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instr_resp_cue, 'tStartRefresh')  # time at next scr refresh
        instr_resp_cue.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(instr_resp_cue.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(instr_resp_cue.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if instr_resp_cue.status == STARTED and not waitOnFlip:
        theseKeys = instr_resp_cue.getKeys(keyList=None, waitRelease=False)
        _instr_resp_cue_allKeys.extend(theseKeys)
        if len(_instr_resp_cue_allKeys):
            instr_resp_cue.keys = _instr_resp_cue_allKeys[-1].name  # just the last key pressed
            instr_resp_cue.rt = _instr_resp_cue_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr_cuesComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr_cues"-------
for thisComponent in instr_cuesComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('instr_txt_cue.started', instr_txt_cue.tStartRefresh)
thisExp.addData('instr_txt_cue.stopped', instr_txt_cue.tStopRefresh)
# check responses
if instr_resp_cue.keys in ['', [], None]:  # No response was made
    instr_resp_cue.keys = None
thisExp.addData('instr_resp_cue.keys',instr_resp_cue.keys)
if instr_resp_cue.keys != None:  # we had a response
    thisExp.addData('instr_resp_cue.rt', instr_resp_cue.rt)
thisExp.addData('instr_resp_cue.started', instr_resp_cue.tStartRefresh)
thisExp.addData('instr_resp_cue.stopped', instr_resp_cue.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr_cues" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
placholder_example = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('exp_exemp.xlsx'),
    seed=None, name='placholder_example')
thisExp.addLoop(placholder_example)  # add the loop to the experiment
thisPlacholder_example = placholder_example.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisPlacholder_example.rgb)
if thisPlacholder_example != None:
    for paramName in thisPlacholder_example:
        exec('{} = thisPlacholder_example[paramName]'.format(paramName))

for thisPlacholder_example in placholder_example:
    currentLoop = placholder_example
    # abbreviate parameter names if possible (e.g. rgb = thisPlacholder_example.rgb)
    if thisPlacholder_example != None:
        for paramName in thisPlacholder_example:
            exec('{} = thisPlacholder_example[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "instr_exp_cue"-------
    continueRoutine = True
    # update component parameters for each repeat
    place_right.setLineColor(prob_rightX)
    place_left.setLineColor(prob_leftX)
    endInstructions_9.keys = []
    endInstructions_9.rt = []
    _endInstructions_9_allKeys = []
    # keep track of which components have finished
    instr_exp_cueComponents = [place_right, place_left, long_back_R, hi_back_R, hi_back_L, long_back_L, place_explan, endInstructions_9]
    for thisComponent in instr_exp_cueComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    instr_exp_cueClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "instr_exp_cue"-------
    while continueRoutine:
        # get current time
        t = instr_exp_cueClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=instr_exp_cueClock)
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
        
        # *place_left* updates
        if place_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            place_left.frameNStart = frameN  # exact frame index
            place_left.tStart = t  # local t and not account for scr refresh
            place_left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(place_left, 'tStartRefresh')  # time at next scr refresh
            place_left.setAutoDraw(True)
        
        # *long_back_R* updates
        if long_back_R.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            long_back_R.frameNStart = frameN  # exact frame index
            long_back_R.tStart = t  # local t and not account for scr refresh
            long_back_R.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(long_back_R, 'tStartRefresh')  # time at next scr refresh
            long_back_R.setAutoDraw(True)
        
        # *hi_back_R* updates
        if hi_back_R.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hi_back_R.frameNStart = frameN  # exact frame index
            hi_back_R.tStart = t  # local t and not account for scr refresh
            hi_back_R.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hi_back_R, 'tStartRefresh')  # time at next scr refresh
            hi_back_R.setAutoDraw(True)
        
        # *hi_back_L* updates
        if hi_back_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hi_back_L.frameNStart = frameN  # exact frame index
            hi_back_L.tStart = t  # local t and not account for scr refresh
            hi_back_L.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hi_back_L, 'tStartRefresh')  # time at next scr refresh
            hi_back_L.setAutoDraw(True)
        
        # *long_back_L* updates
        if long_back_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            long_back_L.frameNStart = frameN  # exact frame index
            long_back_L.tStart = t  # local t and not account for scr refresh
            long_back_L.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(long_back_L, 'tStartRefresh')  # time at next scr refresh
            long_back_L.setAutoDraw(True)
        
        # *place_explan* updates
        if place_explan.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            place_explan.frameNStart = frameN  # exact frame index
            place_explan.tStart = t  # local t and not account for scr refresh
            place_explan.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(place_explan, 'tStartRefresh')  # time at next scr refresh
            place_explan.setAutoDraw(True)
        
        # *endInstructions_9* updates
        waitOnFlip = False
        if endInstructions_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endInstructions_9.frameNStart = frameN  # exact frame index
            endInstructions_9.tStart = t  # local t and not account for scr refresh
            endInstructions_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endInstructions_9, 'tStartRefresh')  # time at next scr refresh
            endInstructions_9.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endInstructions_9.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endInstructions_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endInstructions_9.status == STARTED and not waitOnFlip:
            theseKeys = endInstructions_9.getKeys(keyList=None, waitRelease=False)
            _endInstructions_9_allKeys.extend(theseKeys)
            if len(_endInstructions_9_allKeys):
                endInstructions_9.keys = _endInstructions_9_allKeys[-1].name  # just the last key pressed
                endInstructions_9.rt = _endInstructions_9_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_exp_cueComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "instr_exp_cue"-------
    for thisComponent in instr_exp_cueComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    placholder_example.addData('place_right.started', place_right.tStartRefresh)
    placholder_example.addData('place_right.stopped', place_right.tStopRefresh)
    placholder_example.addData('place_left.started', place_left.tStartRefresh)
    placholder_example.addData('place_left.stopped', place_left.tStopRefresh)
    placholder_example.addData('long_back_R.started', long_back_R.tStartRefresh)
    placholder_example.addData('long_back_R.stopped', long_back_R.tStopRefresh)
    placholder_example.addData('hi_back_R.started', hi_back_R.tStartRefresh)
    placholder_example.addData('hi_back_R.stopped', hi_back_R.tStopRefresh)
    placholder_example.addData('hi_back_L.started', hi_back_L.tStartRefresh)
    placholder_example.addData('hi_back_L.stopped', hi_back_L.tStopRefresh)
    placholder_example.addData('long_back_L.started', long_back_L.tStartRefresh)
    placholder_example.addData('long_back_L.stopped', long_back_L.tStopRefresh)
    placholder_example.addData('place_explan.started', place_explan.tStartRefresh)
    placholder_example.addData('place_explan.stopped', place_explan.tStopRefresh)
    # check responses
    if endInstructions_9.keys in ['', [], None]:  # No response was made
        endInstructions_9.keys = None
    placholder_example.addData('endInstructions_9.keys',endInstructions_9.keys)
    if endInstructions_9.keys != None:  # we had a response
        placholder_example.addData('endInstructions_9.rt', endInstructions_9.rt)
    placholder_example.addData('endInstructions_9.started', endInstructions_9.tStartRefresh)
    placholder_example.addData('endInstructions_9.stopped', endInstructions_9.tStopRefresh)
    # the Routine "instr_exp_cue" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1 repeats of 'placholder_example'


# ------Prepare to start Routine "Q_cat"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_7.keys = []
endInstructions_7.rt = []
_endInstructions_7_allKeys = []
# keep track of which components have finished
Q_catComponents = [text_5, endInstructions_7]
for thisComponent in Q_catComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Q_catClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Q_cat"-------
while continueRoutine:
    # get current time
    t = Q_catClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Q_catClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_5* updates
    if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_5.frameNStart = frameN  # exact frame index
        text_5.tStart = t  # local t and not account for scr refresh
        text_5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
        text_5.setAutoDraw(True)
    
    # *endInstructions_7* updates
    waitOnFlip = False
    if endInstructions_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endInstructions_7.frameNStart = frameN  # exact frame index
        endInstructions_7.tStart = t  # local t and not account for scr refresh
        endInstructions_7.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endInstructions_7, 'tStartRefresh')  # time at next scr refresh
        endInstructions_7.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endInstructions_7.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endInstructions_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endInstructions_7.status == STARTED and not waitOnFlip:
        theseKeys = endInstructions_7.getKeys(keyList=None, waitRelease=False)
        _endInstructions_7_allKeys.extend(theseKeys)
        if len(_endInstructions_7_allKeys):
            endInstructions_7.keys = _endInstructions_7_allKeys[-1].name  # just the last key pressed
            endInstructions_7.rt = _endInstructions_7_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Q_catComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Q_cat"-------
for thisComponent in Q_catComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_5.started', text_5.tStartRefresh)
thisExp.addData('text_5.stopped', text_5.tStopRefresh)
# check responses
if endInstructions_7.keys in ['', [], None]:  # No response was made
    endInstructions_7.keys = None
thisExp.addData('endInstructions_7.keys',endInstructions_7.keys)
if endInstructions_7.keys != None:  # we had a response
    thisExp.addData('endInstructions_7.rt', endInstructions_7.rt)
thisExp.addData('endInstructions_7.started', endInstructions_7.tStartRefresh)
thisExp.addData('endInstructions_7.stopped', endInstructions_7.tStopRefresh)
thisExp.nextEntry()
# the Routine "Q_cat" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "q_cat_exemp"-------
continueRoutine = True
# update component parameters for each repeat
## Write Code to set cue to be valid or not valid 
#based on conditions file 


probe_cue_img = '../cues/cue_left.png'


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

q_category_2.setText(cat_question
)
cat_resp_2.keys = []
cat_resp_2.rt = []
_cat_resp_2_allKeys = []
probe_cue_cat_3.setImage(probe_cue_img)
# keep track of which components have finished
q_cat_exempComponents = [q_category_2, cat_resp_2, probe_cue_cat_3]
for thisComponent in q_cat_exempComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
q_cat_exempClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "q_cat_exemp"-------
while continueRoutine:
    # get current time
    t = q_cat_exempClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=q_cat_exempClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *q_category_2* updates
    if q_category_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        q_category_2.frameNStart = frameN  # exact frame index
        q_category_2.tStart = t  # local t and not account for scr refresh
        q_category_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(q_category_2, 'tStartRefresh')  # time at next scr refresh
        q_category_2.setAutoDraw(True)
    
    # *cat_resp_2* updates
    waitOnFlip = False
    if cat_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        cat_resp_2.frameNStart = frameN  # exact frame index
        cat_resp_2.tStart = t  # local t and not account for scr refresh
        cat_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(cat_resp_2, 'tStartRefresh')  # time at next scr refresh
        cat_resp_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(cat_resp_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(cat_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if cat_resp_2.status == STARTED and not waitOnFlip:
        theseKeys = cat_resp_2.getKeys(keyList=['6', '7', '8', '9'], waitRelease=False)
        _cat_resp_2_allKeys.extend(theseKeys)
        if len(_cat_resp_2_allKeys):
            cat_resp_2.keys = _cat_resp_2_allKeys[-1].name  # just the last key pressed
            cat_resp_2.rt = _cat_resp_2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *probe_cue_cat_3* updates
    if probe_cue_cat_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        probe_cue_cat_3.frameNStart = frameN  # exact frame index
        probe_cue_cat_3.tStart = t  # local t and not account for scr refresh
        probe_cue_cat_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(probe_cue_cat_3, 'tStartRefresh')  # time at next scr refresh
        probe_cue_cat_3.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in q_cat_exempComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "q_cat_exemp"-------
for thisComponent in q_cat_exempComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('q_category_2.started', q_category_2.tStartRefresh)
thisExp.addData('q_category_2.stopped', q_category_2.tStopRefresh)
# check responses
if cat_resp_2.keys in ['', [], None]:  # No response was made
    cat_resp_2.keys = None
thisExp.addData('cat_resp_2.keys',cat_resp_2.keys)
if cat_resp_2.keys != None:  # we had a response
    thisExp.addData('cat_resp_2.rt', cat_resp_2.rt)
thisExp.addData('cat_resp_2.started', cat_resp_2.tStartRefresh)
thisExp.addData('cat_resp_2.stopped', cat_resp_2.tStopRefresh)
thisExp.nextEntry()
thisExp.addData('probe_cue_cat_3.started', probe_cue_cat_3.tStartRefresh)
thisExp.addData('probe_cue_cat_3.stopped', probe_cue_cat_3.tStopRefresh)
# the Routine "q_cat_exemp" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "Q_Rec"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_8.keys = []
endInstructions_8.rt = []
_endInstructions_8_allKeys = []
# keep track of which components have finished
Q_RecComponents = [text_6, endInstructions_8]
for thisComponent in Q_RecComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Q_RecClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Q_Rec"-------
while continueRoutine:
    # get current time
    t = Q_RecClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Q_RecClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_6* updates
    if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_6.frameNStart = frameN  # exact frame index
        text_6.tStart = t  # local t and not account for scr refresh
        text_6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
        text_6.setAutoDraw(True)
    
    # *endInstructions_8* updates
    waitOnFlip = False
    if endInstructions_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endInstructions_8.frameNStart = frameN  # exact frame index
        endInstructions_8.tStart = t  # local t and not account for scr refresh
        endInstructions_8.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endInstructions_8, 'tStartRefresh')  # time at next scr refresh
        endInstructions_8.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endInstructions_8.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endInstructions_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endInstructions_8.status == STARTED and not waitOnFlip:
        theseKeys = endInstructions_8.getKeys(keyList=None, waitRelease=False)
        _endInstructions_8_allKeys.extend(theseKeys)
        if len(_endInstructions_8_allKeys):
            endInstructions_8.keys = _endInstructions_8_allKeys[-1].name  # just the last key pressed
            endInstructions_8.rt = _endInstructions_8_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Q_RecComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Q_Rec"-------
for thisComponent in Q_RecComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_6.started', text_6.tStartRefresh)
thisExp.addData('text_6.stopped', text_6.tStopRefresh)
# check responses
if endInstructions_8.keys in ['', [], None]:  # No response was made
    endInstructions_8.keys = None
thisExp.addData('endInstructions_8.keys',endInstructions_8.keys)
if endInstructions_8.keys != None:  # we had a response
    thisExp.addData('endInstructions_8.rt', endInstructions_8.rt)
thisExp.addData('endInstructions_8.started', endInstructions_8.tStartRefresh)
thisExp.addData('endInstructions_8.stopped', endInstructions_8.tStopRefresh)
thisExp.nextEntry()
# the Routine "Q_Rec" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "q_rec_exemp"-------
continueRoutine = True
# update component parameters for each repeat
## Write Code to set cue to be valid or not valid 
#based on conditions file 

probe_cue_img = '../cues/cue_left.png'


rec_resp.keys = []
rec_resp.rt = []
_rec_resp_allKeys = []
probe_cue_rec_2.setImage(probe_cue_img)
# keep track of which components have finished
q_rec_exempComponents = [q_recognition_3, rec_resp, probe_cue_rec_2]
for thisComponent in q_rec_exempComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
q_rec_exempClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "q_rec_exemp"-------
while continueRoutine:
    # get current time
    t = q_rec_exempClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=q_rec_exempClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *q_recognition_3* updates
    if q_recognition_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        q_recognition_3.frameNStart = frameN  # exact frame index
        q_recognition_3.tStart = t  # local t and not account for scr refresh
        q_recognition_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(q_recognition_3, 'tStartRefresh')  # time at next scr refresh
        q_recognition_3.setAutoDraw(True)
    
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
    if rec_resp.status == STARTED and not waitOnFlip:
        theseKeys = rec_resp.getKeys(keyList=['7', '8'], waitRelease=False)
        _rec_resp_allKeys.extend(theseKeys)
        if len(_rec_resp_allKeys):
            rec_resp.keys = _rec_resp_allKeys[-1].name  # just the last key pressed
            rec_resp.rt = _rec_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *probe_cue_rec_2* updates
    if probe_cue_rec_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        probe_cue_rec_2.frameNStart = frameN  # exact frame index
        probe_cue_rec_2.tStart = t  # local t and not account for scr refresh
        probe_cue_rec_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(probe_cue_rec_2, 'tStartRefresh')  # time at next scr refresh
        probe_cue_rec_2.setAutoDraw(True)
    if probe_cue_rec_2.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > probe_cue_rec_2.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            probe_cue_rec_2.tStop = t  # not accounting for scr refresh
            probe_cue_rec_2.frameNStop = frameN  # exact frame index
            win.timeOnFlip(probe_cue_rec_2, 'tStopRefresh')  # time at next scr refresh
            probe_cue_rec_2.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in q_rec_exempComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "q_rec_exemp"-------
for thisComponent in q_rec_exempComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('q_recognition_3.started', q_recognition_3.tStartRefresh)
thisExp.addData('q_recognition_3.stopped', q_recognition_3.tStopRefresh)
# check responses
if rec_resp.keys in ['', [], None]:  # No response was made
    rec_resp.keys = None
thisExp.addData('rec_resp.keys',rec_resp.keys)
if rec_resp.keys != None:  # we had a response
    thisExp.addData('rec_resp.rt', rec_resp.rt)
thisExp.addData('rec_resp.started', rec_resp.tStartRefresh)
thisExp.addData('rec_resp.stopped', rec_resp.tStopRefresh)
thisExp.nextEntry()
thisExp.addData('probe_cue_rec_2.started', probe_cue_rec_2.tStartRefresh)
thisExp.addData('probe_cue_rec_2.stopped', probe_cue_rec_2.tStopRefresh)
# the Routine "q_rec_exemp" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "Summary"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_10.keys = []
endInstructions_10.rt = []
_endInstructions_10_allKeys = []
# keep track of which components have finished
SummaryComponents = [text_8, endInstructions_10]
for thisComponent in SummaryComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
SummaryClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Summary"-------
while continueRoutine:
    # get current time
    t = SummaryClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=SummaryClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_8* updates
    if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_8.frameNStart = frameN  # exact frame index
        text_8.tStart = t  # local t and not account for scr refresh
        text_8.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
        text_8.setAutoDraw(True)
    
    # *endInstructions_10* updates
    waitOnFlip = False
    if endInstructions_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endInstructions_10.frameNStart = frameN  # exact frame index
        endInstructions_10.tStart = t  # local t and not account for scr refresh
        endInstructions_10.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endInstructions_10, 'tStartRefresh')  # time at next scr refresh
        endInstructions_10.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endInstructions_10.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endInstructions_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endInstructions_10.status == STARTED and not waitOnFlip:
        theseKeys = endInstructions_10.getKeys(keyList=None, waitRelease=False)
        _endInstructions_10_allKeys.extend(theseKeys)
        if len(_endInstructions_10_allKeys):
            endInstructions_10.keys = _endInstructions_10_allKeys[-1].name  # just the last key pressed
            endInstructions_10.rt = _endInstructions_10_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in SummaryComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Summary"-------
for thisComponent in SummaryComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_8.started', text_8.tStartRefresh)
thisExp.addData('text_8.stopped', text_8.tStopRefresh)
# check responses
if endInstructions_10.keys in ['', [], None]:  # No response was made
    endInstructions_10.keys = None
thisExp.addData('endInstructions_10.keys',endInstructions_10.keys)
if endInstructions_10.keys != None:  # we had a response
    thisExp.addData('endInstructions_10.rt', endInstructions_10.rt)
thisExp.addData('endInstructions_10.started', endInstructions_10.tStartRefresh)
thisExp.addData('endInstructions_10.stopped', endInstructions_10.tStopRefresh)
thisExp.nextEntry()
# the Routine "Summary" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "begin_practice"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_4.keys = []
key_resp_4.rt = []
_key_resp_4_allKeys = []
# keep track of which components have finished
begin_practiceComponents = [begin, key_resp_4]
for thisComponent in begin_practiceComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
begin_practiceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "begin_practice"-------
while continueRoutine:
    # get current time
    t = begin_practiceClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=begin_practiceClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *begin* updates
    if begin.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        begin.frameNStart = frameN  # exact frame index
        begin.tStart = t  # local t and not account for scr refresh
        begin.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(begin, 'tStartRefresh')  # time at next scr refresh
        begin.setAutoDraw(True)
    
    # *key_resp_4* updates
    waitOnFlip = False
    if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_4.frameNStart = frameN  # exact frame index
        key_resp_4.tStart = t  # local t and not account for scr refresh
        key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
        key_resp_4.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_4.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_4.getKeys(keyList=None, waitRelease=False)
        _key_resp_4_allKeys.extend(theseKeys)
        if len(_key_resp_4_allKeys):
            key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
            key_resp_4.rt = _key_resp_4_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in begin_practiceComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "begin_practice"-------
for thisComponent in begin_practiceComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('begin.started', begin.tStartRefresh)
thisExp.addData('begin.stopped', begin.tStopRefresh)
# check responses
if key_resp_4.keys in ['', [], None]:  # No response was made
    key_resp_4.keys = None
thisExp.addData('key_resp_4.keys',key_resp_4.keys)
if key_resp_4.keys != None:  # we had a response
    thisExp.addData('key_resp_4.rt', key_resp_4.rt)
thisExp.addData('key_resp_4.started', key_resp_4.tStartRefresh)
thisExp.addData('key_resp_4.stopped', key_resp_4.tStopRefresh)
thisExp.nextEntry()
# the Routine "begin_practice" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of trials etc
conditions = data.importConditions('..\\Quest\\Q_tracks.xlsx')
trials = data.MultiStairHandler(stairType='QUEST', name='trials',
    nTrials=4,
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
    break_freq = 32
    blocks_total = math.ceil((num_exemps*num_trials)/break_freq)
    block_complete = ''
    categorization_accuracy = ''
    if trials.totalTrials == 0 or trials.totalTrials %break_freq != 0:
        continueRoutine = False
    else:
        blocks_count += 1
        block_complete = '\n\n\n\n\n\nBlocks Completed: {num_done:n}/{num_total:n}'.format(num_done = blocks_count, num_total = blocks_total)
        categorization_accuracy = '\n\n\n You correctly categorized {percent_correct:.0%} images!'.format(percent_correct =  category_correct_count/trials.totalTrials)
    
    
    
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
    
    # ------Prepare to start Routine "init_fix"-------
    continueRoutine = True
    # update component parameters for each repeat
    #Get ITI for trial 
    ITI_duration = ITI_list.pop()
    
    thisExp.addData('ITI', ITI_duration)
    
    # Loading .npy files
    #Select which stimuli to present 
    # Randomly choose between stim presented left or right
    # Scrambled image is always the scrambled version of the real counterpart 
    
    left_or_right = random()
    
    if left_or_right>0.5:
        #Display stim on left and scrambled on right
        target_left = stim_img
        target_right = stim_img[:13] + 'scram.npy'
        probe_loc = 'L'
        prob_cue_L = [-1.0,-1.0,-1.0]
        prob_cue_R = [1.0,1.0,1.0]
    
    else:
        #Display image on right not on left
        target_left = stim_img[:13] + 'scram.npy'
        target_right = stim_img
        probe_loc = 'R'
        prob_cue_L= [1.0,1.0,1.0]
        prob_cue_R = [-1.0,-1.0,-1.0]
    
    thisExp.addData('stim_left', target_left)
    thisExp.addData('stim_right',target_right) 
    
    # Load Stimulus images
    
    img_left = np.load(target_left)  
    img_right = np.load(target_right) 
    
    
    #Ramp up and down stim contrast with peak 
    #contrast at 66.7 mS (4 frames assuming 60Hz)
    
    minC = 0.001 
    maxC = 10**level
    
    #Presenting Stimuli for 300 ms for practice instead of 66 ms
    if not expInfo['frameRate']: #Can't measure FR, e.g. dropping frames
        nC = 18
    elif np.isclose(expInfo['frameRate'], 120, rtol = .1):
        nC = 36 #Num frames of presentation
    elif np.isclose(expInfo['frameRate'], 60, rtol = .1):
        nC = 18
    else: 
        nC = 36
    
    
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
    trials.addOtherData('cat_correct', correct_category)
    
    if cat_resp.keys:
        category_response = cat_map[cat_resp.keys[-1]]
        trials.addOtherData('cat_select', category_response)
    
    else: #If no response
        category_response = None
        trials.addOtherData('cat_select', category_response)
    
    
    #Calculate categorization accuracy
    if correct_category == category_response: 
        category_correct_count += 1 
    
    
        
    
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
    if probe_loc == 'L':
        probe_cue_img_2 = '../cues/cue_left.png'
        
    elif probe_loc == 'R':
        probe_cue_img_2 = '../cues/cue_right.png'
    
    rec_resp_2.keys = []
    rec_resp_2.rt = []
    _rec_resp_2_allKeys = []
    probe_cue_rec.setImage(probe_cue_img_2)
    # keep track of which components have finished
    q_recComponents = [q_recognition, rec_resp_2, probe_cue_rec]
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
        
        # *rec_resp_2* updates
        waitOnFlip = False
        if rec_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rec_resp_2.frameNStart = frameN  # exact frame index
            rec_resp_2.tStart = t  # local t and not account for scr refresh
            rec_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rec_resp_2, 'tStartRefresh')  # time at next scr refresh
            rec_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(rec_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(rec_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if rec_resp_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rec_resp_2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rec_resp_2.tStop = t  # not accounting for scr refresh
                rec_resp_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rec_resp_2, 'tStopRefresh')  # time at next scr refresh
                rec_resp_2.status = FINISHED
        if rec_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = rec_resp_2.getKeys(keyList=['7', '8'], waitRelease=False)
            _rec_resp_2_allKeys.extend(theseKeys)
            if len(_rec_resp_2_allKeys):
                rec_resp_2.keys = _rec_resp_2_allKeys[-1].name  # just the last key pressed
                rec_resp_2.rt = _rec_resp_2_allKeys[-1].rt
                # was this correct?
                if (rec_resp_2.keys == str('7')) or (rec_resp_2.keys == '7'):
                    rec_resp_2.corr = 1
                else:
                    rec_resp_2.corr = 0
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
        trials.addOtherData('recognition', 'R')
    else: #For "No" or No Response
        trials.addOtherData('recognition', 'U')
    trials.addOtherData('q_recognition.started', q_recognition.tStartRefresh)
    trials.addOtherData('q_recognition.stopped', q_recognition.tStopRefresh)
    # check responses
    if rec_resp_2.keys in ['', [], None]:  # No response was made
        rec_resp_2.keys = None
        # was no response the correct answer?!
        if str('7').lower() == 'none':
           rec_resp_2.corr = 1;  # correct non-response
        else:
           rec_resp_2.corr = 0;  # failed to respond (incorrectly)
    # store data for trials (MultiStairHandler)
    trials.addResponse(rec_resp_2.corr)
    trials.addOtherData('rec_resp_2.rt', rec_resp_2.rt)
    trials.addOtherData('rec_resp_2.started', rec_resp_2.tStartRefresh)
    trials.addOtherData('rec_resp_2.stopped', rec_resp_2.tStopRefresh)
    trials.addOtherData('probe_cue_rec.started', probe_cue_rec.tStartRefresh)
    trials.addOtherData('probe_cue_rec.stopped', probe_cue_rec.tStopRefresh)
    thisExp.nextEntry()
    
# all staircases completed


# ------Prepare to start Routine "thanks"-------
continueRoutine = True
# update component parameters for each repeat
categorization_accuracy = '\n\n\n You correctly categorized {percent_correct:.0%} images!'.format(percent_correct =  category_correct_count/trials.totalTrials)
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
