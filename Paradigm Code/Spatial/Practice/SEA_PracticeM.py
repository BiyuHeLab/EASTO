#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.10),
    on Wed Jun 22 21:16:56 2022
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

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
import EASTO_funcs as ea
import pandas as pd

quest_file = gui.fileOpenDlg('.')
if not quest_file:
    #    core.quit()
    print('Using Arbitrary Contrasts for Tests')
    #Gets initiated with target stim loading
    quest_estimates = []
else:
    with open(quest_file[0], 'rb') as handle:
        quest_estimates = pickle.load(handle)
import math
from copy import deepcopy
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
expName = 'SEA_PracticeM'  # from the Builder filename that created this script
expInfo = {'subject': '', 'session': '001', 'HPColor': 'black', 'Control': False}
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
    originPath='/Users/brandonchen93/EASTO/Paradigm/Master/Spatial/Practice/SEA_PracticeM.py',
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

# Initialize components for Routine "overview"
overviewClock = core.Clock()
instr_txt_2 = visual.TextStim(win=win, name='instr_txt_2',
    text='In this task, similar to the previous, you will be asked to make judgements about simple images. The key differences between this task and the last is that\n\n1) Two images will be shown every trial\n\n2) You will be given two types of cues that provide information on:\n     a) how likely an image will have meaningful content \n\n     b) how likely that you will have to answer questions about one of the images presented\n\n\nPress any button to continue',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
instr_resp_2 = keyboard.Keyboard()

# Initialize components for Routine "fixation"
fixationClock = core.Clock()
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
endInstructions_2 = keyboard.Keyboard()

# Initialize components for Routine "instr_cues"
instr_cuesClock = core.Clock()
instr_txt_cue = visual.TextStim(win=win, name='instr_txt_cue',
    text='At the beginning of a block and trial you will first see cues that indicate how likely an image with meaningful content will appear on the left or right side of the screen. \n\nBefore the images are presented you will see a cue that indicates how likely you will be asked a question about the left or right image at the end of the trial. \n\nImportantly, the content of the left and right image are unrelated. e.g. The left image can be of a house and the right image can be of a animal. Said another way, the left image does not provide information about the right image and vice versa. \n\nNow you will see what these cues look like and how to interpret them\n\npress any button to continue\n',
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
    ori=0, pos=(4, 4),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
place_left = visual.Rect(
    win=win, name='place_left',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 4),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)
long_back_R = visual.Rect(
    win=win, name='long_back_R',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4, 4),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_R = visual.Rect(
    win=win, name='hi_back_R',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4, 4),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
hi_back_L = visual.Rect(
    win=win, name='hi_back_L',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4, 4),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
long_back_L = visual.Rect(
    win=win, name='long_back_L',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(-4, 4),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-6.0, interpolate=True)
place_explan = visual.TextStim(win=win, name='place_explan',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-7.0);
place_left_fill_2 = visual.Rect(
    win=win, name='place_left_fill_2',
    width=(4, 4)[0], height=(4, 4)[1],
    ori=0, pos=(-4,4),
    lineWidth=1, lineColor=None, lineColorSpace='rgb',
    fillColor=1.0, fillColorSpace='rgb',
    opacity=1, depth=-8.0, interpolate=True)
place_right_fill_2 = visual.Rect(
    win=win, name='place_right_fill_2',
    width=(4, 4)[0], height=(4, 4)[1],
    ori=0, pos=(4,4),
    lineWidth=1, lineColor=None, lineColorSpace='rgb',
    fillColor=1.0, fillColorSpace='rgb',
    opacity=1, depth=-9.0, interpolate=True)
endInstructions = keyboard.Keyboard()

# Initialize components for Routine "instr_att_cue"
instr_att_cueClock = core.Clock()
cue_instr = visual.TextStim(win=win, name='cue_instr',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
att_cue_fix_2 = visual.ShapeStim(
    win=win, name='att_cue_fix_2', vertices='cross',
    size=(0.4, 0.4),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)
endInstructions_9 = keyboard.Keyboard()
att_left_exemp = visual.ImageStim(
    win=win,
    name='att_left_exemp', 
    image='../cues/cue_left.png', mask=None,
    ori=0, pos=(-0.4, 0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-4.0)
att_right_exemp = visual.ImageStim(
    win=win,
    name='att_right_exemp', 
    image='sin', mask=None,
    ori=0, pos=(0.4, 0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)

# Initialize components for Routine "Q_cat"
Q_catClock = core.Clock()
text_5 = visual.TextStim(win=win, name='text_5',
    text='The images will be presented very briefly. \nThe questions will always be presented with a cue (left or right pointing arrow) that indicates which image you should answer the question about\n\nIf you do not see anything - this is normal, just try your best to guess the correct image category. \n\nIf you need to make a guess, please make a genuine guess. That is, do not use a systematic strategy for your guesses, \nlike always pressing "1" or always pressing the opposite of what you pressed last.\n\nNo matter your visual experience, please answer every question. \n\nPress any button to continue\n',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_6 = keyboard.Keyboard()

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
    text='You will also be asked if you had any meaningful visual experience.\n\nFor example: if you saw an animal, or only part of an animal, as long as you have a sense of the image\'s identity - please answer "yes".\n\nThere will be cases where you will see only a noisy glimpse of light or simply nothing.\n\nIf this is the case, the experience is not meaningful - please answer "no".\n\nPress any button to continue',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_7 = keyboard.Keyboard()

# Initialize components for Routine "q_rec_exemp"
q_rec_exempClock = core.Clock()
rec_question  = (u'Meaningful visual experience: \n\n\n\n\n '
            u'{Y:\u00a0^15} {N:\u00a0^15}''\n\n\n '
            u'{b2:\u00a0^15} {b3:\u00a0^15}'.format(Y = u'Yes',
                                               N = u'No',
                                               b2 = u'2',
                                               b3 = u'3 '))
q_recognition_2 = visual.TextStim(win=win, name='q_recognition_2',
    text=rec_question,
    font='Courier New',
    units='norm', pos=(0, 0), height=0.1, wrapWidth=500, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
rec_resp_2 = keyboard.Keyboard()
probe_cue_rec_2 = visual.ImageStim(
    win=win,
    name='probe_cue_rec_2', 
    image='sin', mask=None,
    ori=0, pos=(0,0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# Initialize components for Routine "Q_Conf"
Q_ConfClock = core.Clock()
conf_question_explain = visual.TextStim(win=win, name='conf_question_explain',
    text='Lastly you will be asked to report how confident you were about whether or not you recognized meaningful content in the image.\n\nYou will be presented with 4 options (1, 2, 3, 4) which indicate low confidence (1) to high confidence (4) \n\nFor example: if you responded "Yes" or "No" to the previous question and are certain in your response, select "4". If uncertain  select "1". If somewhere in between select "2" or "3" \n\n\n\n\nPress any button to continue',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
endInstructions_8 = keyboard.Keyboard()

# Initialize components for Routine "q_conf_exemp"
q_conf_exempClock = core.Clock()
q_confidence_2 = visual.TextStim(win=win, name='q_confidence_2',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.1, wrapWidth=500, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
conf_resp_2 = keyboard.Keyboard()
probe_cue_cat_4 = visual.ImageStim(
    win=win,
    name='probe_cue_cat_4', 
    image='sin', mask=None,
    ori=0, pos=(0,0), size=(0.75, 0.4),
    color=[-1,-1,-1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# Initialize components for Routine "Summary"
SummaryClock = core.Clock()
text_8 = visual.TextStim(win=win, name='text_8',
    text='Summary of task: \n\n0. Block Cue \n...\n1. Fixation\n2. Trial Cue \n3. Image Presentations\n4. Category Question: Guess if unsure \n5. Visual Experience: Yes or No \n6. Confidence on Visual Experience: 1-4\n7. Repeat 1-6 until new block cue is presented\n8. Always respond \n\nPress any button to continue',
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
key_resp_2 = keyboard.Keyboard()

# Initialize components for Routine "pract_type"
pract_typeClock = core.Clock()
if expInfo['Control']: 
    practice_sequence = 'pract_type_control.xlsx'
    if expInfo['HPColor'] == 'black':
        block_conditions = 'practice_blocks_black-HP_control.xlsx'
    elif expInfo['HPColor'] == 'white':
        block_conditions = 'practice_blocks_white-HP_control.xlsx'
else:
    practice_sequence = 'pract_type.xlsx'
    if expInfo['HPColor'] == 'black':
        block_conditions = 'practice_blocks_black-HP.xlsx'
    elif expInfo['HPColor'] == 'white':
        block_conditions = 'practice_blocks_white-HP.xlsx'

practice_type_txt = visual.TextStim(win=win, name='practice_type_txt',
    text='default text',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
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
key_resp_3 = keyboard.Keyboard()

# Initialize components for Routine "break_2"
break_2Clock = core.Clock()
blocks_count = 0
trial_counter = 0
category_correct_count = 0
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

# Initialize components for Routine "Block_Cue"
Block_CueClock = core.Clock()
#Set up image lists
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

expect_left_trials, expect_right_trials, expect_neutral_trials = ea.generate_stim(real_imgs, scram_imgs, 'practice_cond.xlsx', [2/3, 1/3], 'Spatial')

place_left_4 = visual.Rect(
    win=win, name='place_left_4',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4,0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
place_right_4 = visual.Rect(
    win=win, name='place_right_4',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4,0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)
long_back_R_4 = visual.Rect(
    win=win, name='long_back_R_4',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4,0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_R_4 = visual.Rect(
    win=win, name='hi_back_R_4',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4,0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
hi_back_L_4 = visual.Rect(
    win=win, name='hi_back_L_4',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4,0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
long_back_L_4 = visual.Rect(
    win=win, name='long_back_L_4',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(-4,0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-6.0, interpolate=True)

# Initialize components for Routine "init_fix"
init_fixClock = core.Clock()
place_left_6 = visual.Rect(
    win=win, name='place_left_6',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
place_right_6 = visual.Rect(
    win=win, name='place_right_6',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-2.0, interpolate=True)
long_back_R_6 = visual.Rect(
    win=win, name='long_back_R_6',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_R_6 = visual.Rect(
    win=win, name='hi_back_R_6',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
hi_back_L_6 = visual.Rect(
    win=win, name='hi_back_L_6',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
long_back_L_6 = visual.Rect(
    win=win, name='long_back_L_6',
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
place_right_3 = visual.Rect(
    win=win, name='place_right_3',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=0.0, interpolate=True)
place_left_3 = visual.Rect(
    win=win, name='place_left_3',
    width=(4.2, 4.2)[0], height=(4.2, 4.2)[1],
    ori=0, pos=(-4, 0),
    lineWidth=3, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=0.7, depth=-1.0, interpolate=True)
long_back_R_3 = visual.Rect(
    win=win, name='long_back_R_3',
    width=(4.8, 3.2)[0], height=(4.8, 3.2)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)
hi_back_R_3 = visual.Rect(
    win=win, name='hi_back_R_3',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
hi_back_L_3 = visual.Rect(
    win=win, name='hi_back_L_3',
    width=(3.2, 4.8)[0], height=(3.2, 4.8)[1],
    ori=0, pos=(-4, 0),
    lineWidth=1, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
long_back_L_3 = visual.Rect(
    win=win, name='long_back_L_3',
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

# Initialize components for Routine "Thanks_msg"
Thanks_msgClock = core.Clock()
thanks_txt = visual.TextStim(win=win, name='thanks_txt',
    text='Thank you! If you feel comfortable with the paradigm and have no other questions we will move forward to the main experiment',
    font='Courier New',
    units='norm', pos=(0, 0), height=0.07, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
EndExp = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "overview"-------
continueRoutine = True
# update component parameters for each repeat
instr_resp_2.keys = []
instr_resp_2.rt = []
_instr_resp_2_allKeys = []
# keep track of which components have finished
overviewComponents = [instr_txt_2, instr_resp_2]
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
    
    # *instr_txt_2* updates
    if instr_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instr_txt_2.frameNStart = frameN  # exact frame index
        instr_txt_2.tStart = t  # local t and not account for scr refresh
        instr_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instr_txt_2, 'tStartRefresh')  # time at next scr refresh
        instr_txt_2.setAutoDraw(True)
    
    # *instr_resp_2* updates
    waitOnFlip = False
    if instr_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instr_resp_2.frameNStart = frameN  # exact frame index
        instr_resp_2.tStart = t  # local t and not account for scr refresh
        instr_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instr_resp_2, 'tStartRefresh')  # time at next scr refresh
        instr_resp_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(instr_resp_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(instr_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if instr_resp_2.status == STARTED and not waitOnFlip:
        theseKeys = instr_resp_2.getKeys(keyList=None, waitRelease=False)
        _instr_resp_2_allKeys.extend(theseKeys)
        if len(_instr_resp_2_allKeys):
            instr_resp_2.keys = _instr_resp_2_allKeys[-1].name  # just the last key pressed
            instr_resp_2.rt = _instr_resp_2_allKeys[-1].rt
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
thisExp.addData('instr_txt_2.started', instr_txt_2.tStartRefresh)
thisExp.addData('instr_txt_2.stopped', instr_txt_2.tStopRefresh)
# check responses
if instr_resp_2.keys in ['', [], None]:  # No response was made
    instr_resp_2.keys = None
thisExp.addData('instr_resp_2.keys',instr_resp_2.keys)
if instr_resp_2.keys != None:  # we had a response
    thisExp.addData('instr_resp_2.rt', instr_resp_2.rt)
thisExp.addData('instr_resp_2.started', instr_resp_2.tStartRefresh)
thisExp.addData('instr_resp_2.stopped', instr_resp_2.tStopRefresh)
thisExp.nextEntry()
# the Routine "overview" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "fixation"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_2.keys = []
endInstructions_2.rt = []
_endInstructions_2_allKeys = []
# keep track of which components have finished
fixationComponents = [fix_instr, Fix_Example, endInstructions_2]
for thisComponent in fixationComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
fixationClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "fixation"-------
while continueRoutine:
    # get current time
    t = fixationClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=fixationClock)
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
    
    # *endInstructions_2* updates
    waitOnFlip = False
    if endInstructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        endInstructions_2.frameNStart = frameN  # exact frame index
        endInstructions_2.tStart = t  # local t and not account for scr refresh
        endInstructions_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(endInstructions_2, 'tStartRefresh')  # time at next scr refresh
        endInstructions_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(endInstructions_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(endInstructions_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if endInstructions_2.status == STARTED and not waitOnFlip:
        theseKeys = endInstructions_2.getKeys(keyList=None, waitRelease=False)
        _endInstructions_2_allKeys.extend(theseKeys)
        if len(_endInstructions_2_allKeys):
            endInstructions_2.keys = _endInstructions_2_allKeys[-1].name  # just the last key pressed
            endInstructions_2.rt = _endInstructions_2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in fixationComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "fixation"-------
for thisComponent in fixationComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('fix_instr.started', fix_instr.tStartRefresh)
thisExp.addData('fix_instr.stopped', fix_instr.tStopRefresh)
thisExp.addData('Fix_Example.started', Fix_Example.tStartRefresh)
thisExp.addData('Fix_Example.stopped', Fix_Example.tStopRefresh)
# check responses
if endInstructions_2.keys in ['', [], None]:  # No response was made
    endInstructions_2.keys = None
thisExp.addData('endInstructions_2.keys',endInstructions_2.keys)
if endInstructions_2.keys != None:  # we had a response
    thisExp.addData('endInstructions_2.rt', endInstructions_2.rt)
thisExp.addData('endInstructions_2.started', endInstructions_2.tStartRefresh)
thisExp.addData('endInstructions_2.stopped', endInstructions_2.tStopRefresh)
thisExp.nextEntry()
# the Routine "fixation" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

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
exp_exemps = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('exp_exemp.xlsx'),
    seed=None, name='exp_exemps')
thisExp.addLoop(exp_exemps)  # add the loop to the experiment
thisExp_exemp = exp_exemps.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisExp_exemp.rgb)
if thisExp_exemp != None:
    for paramName in thisExp_exemp:
        exec('{} = thisExp_exemp[paramName]'.format(paramName))

for thisExp_exemp in exp_exemps:
    currentLoop = exp_exemps
    # abbreviate parameter names if possible (e.g. rgb = thisExp_exemp.rgb)
    if thisExp_exemp != None:
        for paramName in thisExp_exemp:
            exec('{} = thisExp_exemp[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "instr_exp_cue"-------
    continueRoutine = True
    # update component parameters for each repeat
    if expInfo['HPColor'] == 'black':
        HP_color = 'BLACK'
        LP_color = 'WHITE'
    elif expInfo['HPColor'] == 'white':
        LP_color = 'BLACK'
        HP_color = 'WHITE'
        
    exp_instr_txt = ('''At the beginning of a block you will first see cues that indicate how likely an image with meaningful content will appear within the placeholder. 
    \n\n\n\n\n\n\n
    The {H_color} lined placeholder indicates that a meaningful image is MORE LIKELY to appear within the placeholder. The {L_color} lined placeholders indicates that a meaningful image is LESS LIKELY to appear within the placeholder. 
    If the placeholders are both gray, both have the same likelihood of a meaningful image appearing.
    \n
    press any button to continue'''.format(H_color = HP_color , L_color = LP_color))
    place_right.setLineColor(prob_rightX)
    place_left.setLineColor(prob_leftX)
    place_explan.setText(exp_instr_txt)
    place_left_fill_2.setFillColor(left_fillX)
    place_right_fill_2.setFillColor(right_fillX)
    endInstructions.keys = []
    endInstructions.rt = []
    _endInstructions_allKeys = []
    # keep track of which components have finished
    instr_exp_cueComponents = [place_right, place_left, long_back_R, hi_back_R, hi_back_L, long_back_L, place_explan, place_left_fill_2, place_right_fill_2, endInstructions]
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
        
        # *place_left_fill_2* updates
        if place_left_fill_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            place_left_fill_2.frameNStart = frameN  # exact frame index
            place_left_fill_2.tStart = t  # local t and not account for scr refresh
            place_left_fill_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(place_left_fill_2, 'tStartRefresh')  # time at next scr refresh
            place_left_fill_2.setAutoDraw(True)
        
        # *place_right_fill_2* updates
        if place_right_fill_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            place_right_fill_2.frameNStart = frameN  # exact frame index
            place_right_fill_2.tStart = t  # local t and not account for scr refresh
            place_right_fill_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(place_right_fill_2, 'tStartRefresh')  # time at next scr refresh
            place_right_fill_2.setAutoDraw(True)
        
        # *endInstructions* updates
        waitOnFlip = False
        if endInstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endInstructions.frameNStart = frameN  # exact frame index
            endInstructions.tStart = t  # local t and not account for scr refresh
            endInstructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endInstructions, 'tStartRefresh')  # time at next scr refresh
            endInstructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endInstructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endInstructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endInstructions.status == STARTED and not waitOnFlip:
            theseKeys = endInstructions.getKeys(keyList=None, waitRelease=False)
            _endInstructions_allKeys.extend(theseKeys)
            if len(_endInstructions_allKeys):
                endInstructions.keys = _endInstructions_allKeys[-1].name  # just the last key pressed
                endInstructions.rt = _endInstructions_allKeys[-1].rt
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
    exp_exemps.addData('place_right.started', place_right.tStartRefresh)
    exp_exemps.addData('place_right.stopped', place_right.tStopRefresh)
    exp_exemps.addData('place_left.started', place_left.tStartRefresh)
    exp_exemps.addData('place_left.stopped', place_left.tStopRefresh)
    exp_exemps.addData('long_back_R.started', long_back_R.tStartRefresh)
    exp_exemps.addData('long_back_R.stopped', long_back_R.tStopRefresh)
    exp_exemps.addData('hi_back_R.started', hi_back_R.tStartRefresh)
    exp_exemps.addData('hi_back_R.stopped', hi_back_R.tStopRefresh)
    exp_exemps.addData('hi_back_L.started', hi_back_L.tStartRefresh)
    exp_exemps.addData('hi_back_L.stopped', hi_back_L.tStopRefresh)
    exp_exemps.addData('long_back_L.started', long_back_L.tStartRefresh)
    exp_exemps.addData('long_back_L.stopped', long_back_L.tStopRefresh)
    exp_exemps.addData('place_explan.started', place_explan.tStartRefresh)
    exp_exemps.addData('place_explan.stopped', place_explan.tStopRefresh)
    exp_exemps.addData('place_left_fill_2.started', place_left_fill_2.tStartRefresh)
    exp_exemps.addData('place_left_fill_2.stopped', place_left_fill_2.tStopRefresh)
    exp_exemps.addData('place_right_fill_2.started', place_right_fill_2.tStartRefresh)
    exp_exemps.addData('place_right_fill_2.stopped', place_right_fill_2.tStopRefresh)
    # check responses
    if endInstructions.keys in ['', [], None]:  # No response was made
        endInstructions.keys = None
    exp_exemps.addData('endInstructions.keys',endInstructions.keys)
    if endInstructions.keys != None:  # we had a response
        exp_exemps.addData('endInstructions.rt', endInstructions.rt)
    exp_exemps.addData('endInstructions.started', endInstructions.tStartRefresh)
    exp_exemps.addData('endInstructions.stopped', endInstructions.tStopRefresh)
    # the Routine "instr_exp_cue" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1 repeats of 'exp_exemps'


# set up handler to look after randomisation of conditions etc
att_exemps = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('attcue_list.xlsx'),
    seed=None, name='att_exemps')
thisExp.addLoop(att_exemps)  # add the loop to the experiment
thisAtt_exemp = att_exemps.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisAtt_exemp.rgb)
if thisAtt_exemp != None:
    for paramName in thisAtt_exemp:
        exec('{} = thisAtt_exemp[paramName]'.format(paramName))

for thisAtt_exemp in att_exemps:
    currentLoop = att_exemps
    # abbreviate parameter names if possible (e.g. rgb = thisAtt_exemp.rgb)
    if thisAtt_exemp != None:
        for paramName in thisAtt_exemp:
            exec('{} = thisAtt_exemp[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "instr_att_cue"-------
    continueRoutine = True
    # update component parameters for each repeat
    if att_locX == 'L':
        att_left_opac = 1
        att_right_opac = 0
        att_text = 'More likely to be asked about left image '
        
    elif att_locX == 'R':
        att_left_opac = 0
        att_right_opac = 1
        att_text = 'More likely to be asked about right image '
        
    elif att_locX == 'N':
        att_left_opac = 1
        att_right_opac = 1
        att_text = 'Equally likely to be asked about left OR right image'
        
    att_cue_instr = (''' At the beginning of each trial you will see a cue that 
    indicates how likely you will answer the questions about the image on the right 
    or left at the end of the trial. \n\n\n\n
    {cue_instr} \n\n\n\n
    
    press any button to continue
    '''.format(cue_instr = att_text))
    cue_instr.setText(att_cue_instr






)
    endInstructions_9.keys = []
    endInstructions_9.rt = []
    _endInstructions_9_allKeys = []
    att_left_exemp.setOpacity(att_left_opac)
    att_right_exemp.setOpacity(att_right_opac)
    att_right_exemp.setImage('../cues/cue_right.png')
    # keep track of which components have finished
    instr_att_cueComponents = [cue_instr, att_cue_fix_2, endInstructions_9, att_left_exemp, att_right_exemp]
    for thisComponent in instr_att_cueComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    instr_att_cueClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "instr_att_cue"-------
    while continueRoutine:
        # get current time
        t = instr_att_cueClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=instr_att_cueClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cue_instr* updates
        if cue_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cue_instr.frameNStart = frameN  # exact frame index
            cue_instr.tStart = t  # local t and not account for scr refresh
            cue_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cue_instr, 'tStartRefresh')  # time at next scr refresh
            cue_instr.setAutoDraw(True)
        
        # *att_cue_fix_2* updates
        if att_cue_fix_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            att_cue_fix_2.frameNStart = frameN  # exact frame index
            att_cue_fix_2.tStart = t  # local t and not account for scr refresh
            att_cue_fix_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(att_cue_fix_2, 'tStartRefresh')  # time at next scr refresh
            att_cue_fix_2.setAutoDraw(True)
        
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
        
        # *att_left_exemp* updates
        if att_left_exemp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            att_left_exemp.frameNStart = frameN  # exact frame index
            att_left_exemp.tStart = t  # local t and not account for scr refresh
            att_left_exemp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(att_left_exemp, 'tStartRefresh')  # time at next scr refresh
            att_left_exemp.setAutoDraw(True)
        
        # *att_right_exemp* updates
        if att_right_exemp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            att_right_exemp.frameNStart = frameN  # exact frame index
            att_right_exemp.tStart = t  # local t and not account for scr refresh
            att_right_exemp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(att_right_exemp, 'tStartRefresh')  # time at next scr refresh
            att_right_exemp.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_att_cueComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "instr_att_cue"-------
    for thisComponent in instr_att_cueComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    att_exemps.addData('cue_instr.started', cue_instr.tStartRefresh)
    att_exemps.addData('cue_instr.stopped', cue_instr.tStopRefresh)
    att_exemps.addData('att_cue_fix_2.started', att_cue_fix_2.tStartRefresh)
    att_exemps.addData('att_cue_fix_2.stopped', att_cue_fix_2.tStopRefresh)
    # check responses
    if endInstructions_9.keys in ['', [], None]:  # No response was made
        endInstructions_9.keys = None
    att_exemps.addData('endInstructions_9.keys',endInstructions_9.keys)
    if endInstructions_9.keys != None:  # we had a response
        att_exemps.addData('endInstructions_9.rt', endInstructions_9.rt)
    att_exemps.addData('endInstructions_9.started', endInstructions_9.tStartRefresh)
    att_exemps.addData('endInstructions_9.stopped', endInstructions_9.tStopRefresh)
    att_exemps.addData('att_left_exemp.started', att_left_exemp.tStartRefresh)
    att_exemps.addData('att_left_exemp.stopped', att_left_exemp.tStopRefresh)
    att_exemps.addData('att_right_exemp.started', att_right_exemp.tStartRefresh)
    att_exemps.addData('att_right_exemp.stopped', att_right_exemp.tStopRefresh)
    # the Routine "instr_att_cue" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1 repeats of 'att_exemps'


# ------Prepare to start Routine "Q_cat"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_6.keys = []
endInstructions_6.rt = []
_endInstructions_6_allKeys = []
# keep track of which components have finished
Q_catComponents = [text_5, endInstructions_6]
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
if endInstructions_6.keys in ['', [], None]:  # No response was made
    endInstructions_6.keys = None
thisExp.addData('endInstructions_6.keys',endInstructions_6.keys)
if endInstructions_6.keys != None:  # we had a response
    thisExp.addData('endInstructions_6.rt', endInstructions_6.rt)
thisExp.addData('endInstructions_6.started', endInstructions_6.tStartRefresh)
thisExp.addData('endInstructions_6.stopped', endInstructions_6.tStopRefresh)
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
endInstructions_7.keys = []
endInstructions_7.rt = []
_endInstructions_7_allKeys = []
# keep track of which components have finished
Q_RecComponents = [text_6, endInstructions_7]
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
if endInstructions_7.keys in ['', [], None]:  # No response was made
    endInstructions_7.keys = None
thisExp.addData('endInstructions_7.keys',endInstructions_7.keys)
if endInstructions_7.keys != None:  # we had a response
    thisExp.addData('endInstructions_7.rt', endInstructions_7.rt)
thisExp.addData('endInstructions_7.started', endInstructions_7.tStartRefresh)
thisExp.addData('endInstructions_7.stopped', endInstructions_7.tStopRefresh)
thisExp.nextEntry()
# the Routine "Q_Rec" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "q_rec_exemp"-------
continueRoutine = True
# update component parameters for each repeat
## Write Code to set cue to be valid or not valid 
#based on conditions file 

probe_cue_img = '../cues/cue_left.png'


rec_resp_2.keys = []
rec_resp_2.rt = []
_rec_resp_2_allKeys = []
probe_cue_rec_2.setImage(probe_cue_img)
# keep track of which components have finished
q_rec_exempComponents = [q_recognition_2, rec_resp_2, probe_cue_rec_2]
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
    
    # *q_recognition_2* updates
    if q_recognition_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        q_recognition_2.frameNStart = frameN  # exact frame index
        q_recognition_2.tStart = t  # local t and not account for scr refresh
        q_recognition_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(q_recognition_2, 'tStartRefresh')  # time at next scr refresh
        q_recognition_2.setAutoDraw(True)
    
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
    if rec_resp_2.status == STARTED and not waitOnFlip:
        theseKeys = rec_resp_2.getKeys(keyList=['7', '8'], waitRelease=False)
        _rec_resp_2_allKeys.extend(theseKeys)
        if len(_rec_resp_2_allKeys):
            rec_resp_2.keys = _rec_resp_2_allKeys[-1].name  # just the last key pressed
            rec_resp_2.rt = _rec_resp_2_allKeys[-1].rt
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
thisExp.addData('q_recognition_2.started', q_recognition_2.tStartRefresh)
thisExp.addData('q_recognition_2.stopped', q_recognition_2.tStopRefresh)
# check responses
if rec_resp_2.keys in ['', [], None]:  # No response was made
    rec_resp_2.keys = None
thisExp.addData('rec_resp_2.keys',rec_resp_2.keys)
if rec_resp_2.keys != None:  # we had a response
    thisExp.addData('rec_resp_2.rt', rec_resp_2.rt)
thisExp.addData('rec_resp_2.started', rec_resp_2.tStartRefresh)
thisExp.addData('rec_resp_2.stopped', rec_resp_2.tStopRefresh)
thisExp.nextEntry()
thisExp.addData('probe_cue_rec_2.started', probe_cue_rec_2.tStartRefresh)
thisExp.addData('probe_cue_rec_2.stopped', probe_cue_rec_2.tStopRefresh)
# the Routine "q_rec_exemp" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "Q_Conf"-------
continueRoutine = True
# update component parameters for each repeat
endInstructions_8.keys = []
endInstructions_8.rt = []
_endInstructions_8_allKeys = []
# keep track of which components have finished
Q_ConfComponents = [conf_question_explain, endInstructions_8]
for thisComponent in Q_ConfComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Q_ConfClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Q_Conf"-------
while continueRoutine:
    # get current time
    t = Q_ConfClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Q_ConfClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *conf_question_explain* updates
    if conf_question_explain.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        conf_question_explain.frameNStart = frameN  # exact frame index
        conf_question_explain.tStart = t  # local t and not account for scr refresh
        conf_question_explain.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(conf_question_explain, 'tStartRefresh')  # time at next scr refresh
        conf_question_explain.setAutoDraw(True)
    
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
    for thisComponent in Q_ConfComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Q_Conf"-------
for thisComponent in Q_ConfComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('conf_question_explain.started', conf_question_explain.tStartRefresh)
thisExp.addData('conf_question_explain.stopped', conf_question_explain.tStopRefresh)
# check responses
if endInstructions_8.keys in ['', [], None]:  # No response was made
    endInstructions_8.keys = None
thisExp.addData('endInstructions_8.keys',endInstructions_8.keys)
if endInstructions_8.keys != None:  # we had a response
    thisExp.addData('endInstructions_8.rt', endInstructions_8.rt)
thisExp.addData('endInstructions_8.started', endInstructions_8.tStartRefresh)
thisExp.addData('endInstructions_8.stopped', endInstructions_8.tStopRefresh)
thisExp.nextEntry()
# the Routine "Q_Conf" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "q_conf_exemp"-------
continueRoutine = True
# update component parameters for each repeat

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

q_confidence_2.setText(conf_question)
conf_resp_2.keys = []
conf_resp_2.rt = []
_conf_resp_2_allKeys = []
probe_cue_cat_4.setImage(probe_cue_img)
# keep track of which components have finished
q_conf_exempComponents = [q_confidence_2, conf_resp_2, probe_cue_cat_4]
for thisComponent in q_conf_exempComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
q_conf_exempClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "q_conf_exemp"-------
while continueRoutine:
    # get current time
    t = q_conf_exempClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=q_conf_exempClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *q_confidence_2* updates
    if q_confidence_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        q_confidence_2.frameNStart = frameN  # exact frame index
        q_confidence_2.tStart = t  # local t and not account for scr refresh
        q_confidence_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(q_confidence_2, 'tStartRefresh')  # time at next scr refresh
        q_confidence_2.setAutoDraw(True)
    
    # *conf_resp_2* updates
    waitOnFlip = False
    if conf_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        conf_resp_2.frameNStart = frameN  # exact frame index
        conf_resp_2.tStart = t  # local t and not account for scr refresh
        conf_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(conf_resp_2, 'tStartRefresh')  # time at next scr refresh
        conf_resp_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(conf_resp_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(conf_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if conf_resp_2.status == STARTED and not waitOnFlip:
        theseKeys = conf_resp_2.getKeys(keyList=['6', '7', '8', '9'], waitRelease=False)
        _conf_resp_2_allKeys.extend(theseKeys)
        if len(_conf_resp_2_allKeys):
            conf_resp_2.keys = _conf_resp_2_allKeys[-1].name  # just the last key pressed
            conf_resp_2.rt = _conf_resp_2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *probe_cue_cat_4* updates
    if probe_cue_cat_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        probe_cue_cat_4.frameNStart = frameN  # exact frame index
        probe_cue_cat_4.tStart = t  # local t and not account for scr refresh
        probe_cue_cat_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(probe_cue_cat_4, 'tStartRefresh')  # time at next scr refresh
        probe_cue_cat_4.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in q_conf_exempComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "q_conf_exemp"-------
for thisComponent in q_conf_exempComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('q_confidence_2.started', q_confidence_2.tStartRefresh)
thisExp.addData('q_confidence_2.stopped', q_confidence_2.tStopRefresh)
# check responses
if conf_resp_2.keys in ['', [], None]:  # No response was made
    conf_resp_2.keys = None
thisExp.addData('conf_resp_2.keys',conf_resp_2.keys)
if conf_resp_2.keys != None:  # we had a response
    thisExp.addData('conf_resp_2.rt', conf_resp_2.rt)
thisExp.addData('conf_resp_2.started', conf_resp_2.tStartRefresh)
thisExp.addData('conf_resp_2.stopped', conf_resp_2.tStopRefresh)
thisExp.nextEntry()
thisExp.addData('probe_cue_cat_4.started', probe_cue_cat_4.tStartRefresh)
thisExp.addData('probe_cue_cat_4.stopped', probe_cue_cat_4.tStopRefresh)
# the Routine "q_conf_exemp" was not non-slip safe, so reset the non-slip timer
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
key_resp_2.keys = []
key_resp_2.rt = []
_key_resp_2_allKeys = []
# keep track of which components have finished
begin_practiceComponents = [begin, key_resp_2]
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
    
    # *key_resp_2* updates
    waitOnFlip = False
    if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_2.frameNStart = frameN  # exact frame index
        key_resp_2.tStart = t  # local t and not account for scr refresh
        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
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
if key_resp_2.keys in ['', [], None]:  # No response was made
    key_resp_2.keys = None
thisExp.addData('key_resp_2.keys',key_resp_2.keys)
if key_resp_2.keys != None:  # we had a response
    thisExp.addData('key_resp_2.rt', key_resp_2.rt)
thisExp.addData('key_resp_2.started', key_resp_2.tStartRefresh)
thisExp.addData('key_resp_2.stopped', key_resp_2.tStopRefresh)
thisExp.nextEntry()
# the Routine "begin_practice" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
practice_type = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions(practice_sequence),
    seed=None, name='practice_type')
thisExp.addLoop(practice_type)  # add the loop to the experiment
thisPractice_type = practice_type.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisPractice_type.rgb)
if thisPractice_type != None:
    for paramName in thisPractice_type:
        exec('{} = thisPractice_type[paramName]'.format(paramName))

for thisPractice_type in practice_type:
    currentLoop = practice_type
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_type.rgb)
    if thisPractice_type != None:
        for paramName in thisPractice_type:
            exec('{} = thisPractice_type[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "pract_type"-------
    continueRoutine = True
    # update component parameters for each repeat
    """
    Define which rows of the blocks condition file 
    should be selected given what the subject is practicing 
    """
    
    practice_conditions = pd.read_excel(block_conditions)
    # Create dict to store indices of conditions file by block
    practice_levels = {cond : '' for cond in practice_conditions.practice_level.unique() }
    for cond in practice_conditions.practice_level.unique():
        start_ind = practice_conditions.index[practice_conditions['practice_level'] == cond][0]
        end_ind = practice_conditions.index[practice_conditions['practice_level'] == cond][-1] + 1 #Adding +1 to include last index
        practice_levels[cond] = '{start}:{end}'.format(start = start_ind, end = end_ind)
    # Select the blocks to run 
    if pract_type == 'E':
        pract_conds = practice_levels['E']
        practice_type_msg = ('The following trials you will practice using the block cue''\n\n\n\n\n' 'Press any button to continue')
    elif pract_type == 'A':
        pract_conds = practice_levels['A']
        practice_type_msg = ('The following trials you will practice using the trial cue''\n\n\n\n\n' 'Press any button to continue')
    elif pract_type == 'B':
        pract_conds = practice_levels['B']
        practice_type_msg = ('The following trials you will practice using both cues together''\n\n\n\n\n' 'Press any button to continue')
    
    #Show categorization accuracy after each block 
    categorization_accuracy = ''
    if practice_type.thisN != 0:
        categorization_accuracy = '\n\n You correctly categorized {percent_correct:.0%} of the images!'.format(percent_correct =  category_correct_count/trial_counter)
        #Reset category count every block and trial count
        category_correct_count = 0
        trial_counter = 0 
    practice_type_txt.setText(practice_type_msg)
    cat_accuracy_2.setText(categorization_accuracy )
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # keep track of which components have finished
    pract_typeComponents = [practice_type_txt, cat_accuracy_2, key_resp_3]
    for thisComponent in pract_typeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    pract_typeClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "pract_type"-------
    while continueRoutine:
        # get current time
        t = pract_typeClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=pract_typeClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practice_type_txt* updates
        if practice_type_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_type_txt.frameNStart = frameN  # exact frame index
            practice_type_txt.tStart = t  # local t and not account for scr refresh
            practice_type_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_type_txt, 'tStartRefresh')  # time at next scr refresh
            practice_type_txt.setAutoDraw(True)
        
        # *cat_accuracy_2* updates
        if cat_accuracy_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cat_accuracy_2.frameNStart = frameN  # exact frame index
            cat_accuracy_2.tStart = t  # local t and not account for scr refresh
            cat_accuracy_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cat_accuracy_2, 'tStartRefresh')  # time at next scr refresh
            cat_accuracy_2.setAutoDraw(True)
        if cat_accuracy_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cat_accuracy_2.tStartRefresh + 120-frameTolerance:
                # keep track of stop time/frame for later
                cat_accuracy_2.tStop = t  # not accounting for scr refresh
                cat_accuracy_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cat_accuracy_2, 'tStopRefresh')  # time at next scr refresh
                cat_accuracy_2.setAutoDraw(False)
        
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
        for thisComponent in pract_typeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "pract_type"-------
    for thisComponent in pract_typeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_type.addData('practice_type_txt.started', practice_type_txt.tStartRefresh)
    practice_type.addData('practice_type_txt.stopped', practice_type_txt.tStopRefresh)
    practice_type.addData('cat_accuracy_2.started', cat_accuracy_2.tStartRefresh)
    practice_type.addData('cat_accuracy_2.stopped', cat_accuracy_2.tStopRefresh)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    practice_type.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        practice_type.addData('key_resp_3.rt', key_resp_3.rt)
    practice_type.addData('key_resp_3.started', key_resp_3.tStartRefresh)
    practice_type.addData('key_resp_3.stopped', key_resp_3.tStopRefresh)
    # the Routine "pract_type" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler(nReps=1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(block_conditions, selection=pract_conds),
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
        
        # ------Prepare to start Routine "Block_Cue"-------
        continueRoutine = True
        routineTimer.add(3.000000)
        # update component parameters for each repeat
        #Rows that refer to condition file set in loop
        conditions = pd.read_excel('practice_cond.xlsx')
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
            
        elif exp_block == 'right':
            bloc_type = trials['R']
            targets_left = expect_right_trials['left_stims'] # Higher prob real left
            targets_right = expect_right_trials['right_stims']
        
        elif exp_block == 'neutral':
            bloc_type = trials['N']
            targets_left = expect_neutral_trials['left_stims'] # Higher prob real left
            targets_right = expect_neutral_trials['right_stims']
        
        #randomize order of stimuli & freq of Real/Scram
        for exp_types in targets_left.keys(): #dict keys should be identical regardless of left vs right or 1st vs 2nd
            #Shuffle images
            shuffle(targets_left[exp_types])
            shuffle(targets_right[exp_types])
        
        # Define ITI Distribution per block 
        #Define possible ITI Vals & number of trials in a block
        
        numTrials = n_imgs * n_trial_types * n_block_types # num exemplars * num trials * blocks (Total trials in a block) (Extra to prevent empty lists)
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
        
        
        place_left_4.setLineColor(prob_cue_L)
        place_right_4.setLineColor(prob_cue_R)
        # keep track of which components have finished
        Block_CueComponents = [place_left_4, place_right_4, long_back_R_4, hi_back_R_4, hi_back_L_4, long_back_L_4]
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
                if tThisFlipGlobal > place_left_4.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    place_left_4.tStop = t  # not accounting for scr refresh
                    place_left_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(place_left_4, 'tStopRefresh')  # time at next scr refresh
                    place_left_4.setAutoDraw(False)
            
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
                if tThisFlipGlobal > place_right_4.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    place_right_4.tStop = t  # not accounting for scr refresh
                    place_right_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(place_right_4, 'tStopRefresh')  # time at next scr refresh
                    place_right_4.setAutoDraw(False)
            
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
                if tThisFlipGlobal > long_back_R_4.tStartRefresh + 3-frameTolerance:
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
                if tThisFlipGlobal > hi_back_R_4.tStartRefresh + 3-frameTolerance:
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
                if tThisFlipGlobal > hi_back_L_4.tStartRefresh + 3-frameTolerance:
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
                if tThisFlipGlobal > long_back_L_4.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    long_back_L_4.tStop = t  # not accounting for scr refresh
                    long_back_L_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(long_back_L_4, 'tStopRefresh')  # time at next scr refresh
                    long_back_L_4.setAutoDraw(False)
            
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
        blocks.addData('place_left_4.started', place_left_4.tStartRefresh)
        blocks.addData('place_left_4.stopped', place_left_4.tStopRefresh)
        blocks.addData('place_right_4.started', place_right_4.tStartRefresh)
        blocks.addData('place_right_4.stopped', place_right_4.tStopRefresh)
        blocks.addData('long_back_R_4.started', long_back_R_4.tStartRefresh)
        blocks.addData('long_back_R_4.stopped', long_back_R_4.tStopRefresh)
        blocks.addData('hi_back_R_4.started', hi_back_R_4.tStartRefresh)
        blocks.addData('hi_back_R_4.stopped', hi_back_R_4.tStopRefresh)
        blocks.addData('hi_back_L_4.started', hi_back_L_4.tStartRefresh)
        blocks.addData('hi_back_L_4.stopped', hi_back_L_4.tStopRefresh)
        blocks.addData('long_back_L_4.started', long_back_L_4.tStartRefresh)
        blocks.addData('long_back_L_4.stopped', long_back_L_4.tStopRefresh)
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler(nReps=1, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('practice_cond.xlsx', selection=bloc_type),
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
            trial_counter += 1 # Manual trial counter 
            #Get ITI for trial 
            #Initiation of ITI_list is in "Block_Cue" Routine
            ITI_duration = ITI_list.pop()
            
            thisExp.addData('ITI', ITI_duration)
            
            
            place_left_6.setLineColor(prob_cue_L)
            place_right_6.setLineColor(prob_cue_R)
            # keep track of which components have finished
            init_fixComponents = [place_left_6, place_right_6, long_back_R_6, hi_back_R_6, hi_back_L_6, long_back_L_6, pre_cue_fix_2]
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
                
                # *place_left_6* updates
                if place_left_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    place_left_6.frameNStart = frameN  # exact frame index
                    place_left_6.tStart = t  # local t and not account for scr refresh
                    place_left_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(place_left_6, 'tStartRefresh')  # time at next scr refresh
                    place_left_6.setAutoDraw(True)
                if place_left_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > place_left_6.tStartRefresh + ITI_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        place_left_6.tStop = t  # not accounting for scr refresh
                        place_left_6.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(place_left_6, 'tStopRefresh')  # time at next scr refresh
                        place_left_6.setAutoDraw(False)
                
                # *place_right_6* updates
                if place_right_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    place_right_6.frameNStart = frameN  # exact frame index
                    place_right_6.tStart = t  # local t and not account for scr refresh
                    place_right_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(place_right_6, 'tStartRefresh')  # time at next scr refresh
                    place_right_6.setAutoDraw(True)
                if place_right_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > place_right_6.tStartRefresh + ITI_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        place_right_6.tStop = t  # not accounting for scr refresh
                        place_right_6.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(place_right_6, 'tStopRefresh')  # time at next scr refresh
                        place_right_6.setAutoDraw(False)
                
                # *long_back_R_6* updates
                if long_back_R_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    long_back_R_6.frameNStart = frameN  # exact frame index
                    long_back_R_6.tStart = t  # local t and not account for scr refresh
                    long_back_R_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(long_back_R_6, 'tStartRefresh')  # time at next scr refresh
                    long_back_R_6.setAutoDraw(True)
                if long_back_R_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > long_back_R_6.tStartRefresh + ITI_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        long_back_R_6.tStop = t  # not accounting for scr refresh
                        long_back_R_6.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(long_back_R_6, 'tStopRefresh')  # time at next scr refresh
                        long_back_R_6.setAutoDraw(False)
                
                # *hi_back_R_6* updates
                if hi_back_R_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    hi_back_R_6.frameNStart = frameN  # exact frame index
                    hi_back_R_6.tStart = t  # local t and not account for scr refresh
                    hi_back_R_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(hi_back_R_6, 'tStartRefresh')  # time at next scr refresh
                    hi_back_R_6.setAutoDraw(True)
                if hi_back_R_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > hi_back_R_6.tStartRefresh + ITI_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        hi_back_R_6.tStop = t  # not accounting for scr refresh
                        hi_back_R_6.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(hi_back_R_6, 'tStopRefresh')  # time at next scr refresh
                        hi_back_R_6.setAutoDraw(False)
                
                # *hi_back_L_6* updates
                if hi_back_L_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    hi_back_L_6.frameNStart = frameN  # exact frame index
                    hi_back_L_6.tStart = t  # local t and not account for scr refresh
                    hi_back_L_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(hi_back_L_6, 'tStartRefresh')  # time at next scr refresh
                    hi_back_L_6.setAutoDraw(True)
                if hi_back_L_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > hi_back_L_6.tStartRefresh + ITI_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        hi_back_L_6.tStop = t  # not accounting for scr refresh
                        hi_back_L_6.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(hi_back_L_6, 'tStopRefresh')  # time at next scr refresh
                        hi_back_L_6.setAutoDraw(False)
                
                # *long_back_L_6* updates
                if long_back_L_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    long_back_L_6.frameNStart = frameN  # exact frame index
                    long_back_L_6.tStart = t  # local t and not account for scr refresh
                    long_back_L_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(long_back_L_6, 'tStartRefresh')  # time at next scr refresh
                    long_back_L_6.setAutoDraw(True)
                if long_back_L_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > long_back_L_6.tStartRefresh + ITI_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        long_back_L_6.tStop = t  # not accounting for scr refresh
                        long_back_L_6.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(long_back_L_6, 'tStopRefresh')  # time at next scr refresh
                        long_back_L_6.setAutoDraw(False)
                
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
            trials.addData('place_left_6.started', place_left_6.tStartRefresh)
            trials.addData('place_left_6.stopped', place_left_6.tStopRefresh)
            trials.addData('place_right_6.started', place_right_6.tStartRefresh)
            trials.addData('place_right_6.stopped', place_right_6.tStopRefresh)
            trials.addData('long_back_R_6.started', long_back_R_6.tStartRefresh)
            trials.addData('long_back_R_6.stopped', long_back_R_6.tStopRefresh)
            trials.addData('hi_back_R_6.started', hi_back_R_6.tStartRefresh)
            trials.addData('hi_back_R_6.stopped', hi_back_R_6.tStopRefresh)
            trials.addData('hi_back_L_6.started', hi_back_L_6.tStartRefresh)
            trials.addData('hi_back_L_6.stopped', hi_back_L_6.tStopRefresh)
            trials.addData('long_back_L_6.started', long_back_L_6.tStartRefresh)
            trials.addData('long_back_L_6.stopped', long_back_L_6.tStopRefresh)
            trials.addData('pre_cue_fix_2.started', pre_cue_fix_2.tStartRefresh)
            trials.addData('pre_cue_fix_2.stopped', pre_cue_fix_2.tStopRefresh)
            # the Routine "init_fix" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "att_cue"-------
            continueRoutine = True
            routineTimer.add(0.050000)
            # update component parameters for each repeat
            if pract_type == 'E':
                att_left_opac = 0
                att_right_opac = 0
            else:
                if att_loc == 'L':
                    att_left_opac = 1
                    att_right_opac = 0
                    
                elif att_loc == 'R':
                    att_left_opac = 0
                    att_right_opac = 1
                    
                elif att_loc == 'N':
                    att_left_opac = 0
                    att_right_opac = 0
            
            place_right_5.setLineColor(prob_cue_R)
            place_left_5.setLineColor(prob_cue_L)
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
            place_right_3.setLineColor(prob_cue_R)
            place_left_3.setLineColor(prob_cue_L)
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
            blank_preComponents = [place_right_3, place_left_3, long_back_R_3, hi_back_R_3, hi_back_L_3, long_back_L_3, post_cue_fix2]
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
                    if tThisFlipGlobal > place_right_3.tStartRefresh + 0.9-frameTolerance:
                        # keep track of stop time/frame for later
                        place_right_3.tStop = t  # not accounting for scr refresh
                        place_right_3.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(place_right_3, 'tStopRefresh')  # time at next scr refresh
                        place_right_3.setAutoDraw(False)
                
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
                    if tThisFlipGlobal > place_left_3.tStartRefresh + 0.9-frameTolerance:
                        # keep track of stop time/frame for later
                        place_left_3.tStop = t  # not accounting for scr refresh
                        place_left_3.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(place_left_3, 'tStopRefresh')  # time at next scr refresh
                        place_left_3.setAutoDraw(False)
                
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
                    if tThisFlipGlobal > long_back_R_3.tStartRefresh + 0.9-frameTolerance:
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
                    if tThisFlipGlobal > hi_back_R_3.tStartRefresh + 0.9-frameTolerance:
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
                    if tThisFlipGlobal > hi_back_L_3.tStartRefresh + 0.9-frameTolerance:
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
                    if tThisFlipGlobal > long_back_L_3.tStartRefresh + 0.9-frameTolerance:
                        # keep track of stop time/frame for later
                        long_back_L_3.tStop = t  # not accounting for scr refresh
                        long_back_L_3.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(long_back_L_3, 'tStopRefresh')  # time at next scr refresh
                        long_back_L_3.setAutoDraw(False)
                
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
            trials.addData('place_right_3.started', place_right_3.tStartRefresh)
            trials.addData('place_right_3.stopped', place_right_3.tStopRefresh)
            trials.addData('place_left_3.started', place_left_3.tStartRefresh)
            trials.addData('place_left_3.stopped', place_left_3.tStopRefresh)
            trials.addData('long_back_R_3.started', long_back_R_3.tStartRefresh)
            trials.addData('long_back_R_3.stopped', long_back_R_3.tStopRefresh)
            trials.addData('hi_back_R_3.started', hi_back_R_3.tStartRefresh)
            trials.addData('hi_back_R_3.stopped', hi_back_R_3.tStopRefresh)
            trials.addData('hi_back_L_3.started', hi_back_L_3.tStartRefresh)
            trials.addData('hi_back_L_3.stopped', hi_back_L_3.tStopRefresh)
            trials.addData('long_back_L_3.started', long_back_L_3.tStartRefresh)
            trials.addData('long_back_L_3.stopped', long_back_L_3.tStopRefresh)
            trials.addData('post_cue_fix2.started', post_cue_fix2.tStartRefresh)
            trials.addData('post_cue_fix2.stopped', post_cue_fix2.tStopRefresh)
            
            # ------Prepare to start Routine "stim"-------
            continueRoutine = True
            # update component parameters for each repeat
            place_right_2.setLineColor(prob_cue_R)
            place_left_2.setLineColor(prob_cue_L)
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
            #
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
            thisExp.nextEntry()
            
        # completed 1 repeats of 'trials'
        
    # completed 1 repeats of 'blocks'
    
# completed 1 repeats of 'practice_type'


# ------Prepare to start Routine "Thanks_msg"-------
continueRoutine = True
# update component parameters for each repeat
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
