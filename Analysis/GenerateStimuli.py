#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:15:46 2021

@author: brandonchen93
Translating PreProcStim.m from Ella/Max's HLTP Threshold Experiment 
Usings .npy to save image arrays for speed will potentially consider loading
as such as well into psychopy. 

What is missing:
    Currently there is no equivalent of SHINE toolbox in python (afaik), so 
    if matching needs to be done between the images, should be done in MATLAB
    first and then the saved SHINEd image files (.png or .jpg) should undergo 
    the second round of processing. 
    TBD: How much preprocessing of the image should be donen prior to SHINEing
    Might consider using SHINE-color adaptated toolbox https://osf.io/auzjy/

Real images undergo the following processing steps
1) Resized according to params['imsize']
2) Converted to Grayscale if applicable 
3) Pixels are z-scored 
4) Pixel values are Min-Max normalized between -1 and 1 
5) Image is gaussian blurred 7x7 px kernel, std of 1.5 
6) If preparing an exemplar, image is filtered such that edges fade to params['imbgrd']
7) Scrambled images are made by phase-scrambling real image arrays (prior to final filtering)
8) All images and image arrays are saved to their respective folders for future use in experiment

"""

from PIL import Image, ImageFilter
import cv2
import numpy as np 
import os 
import glob
import matplotlib.pyplot as plt 
from scipy.stats import zscore, multivariate_normal
import pickle 
import pandas as pd

# For Using Psychopy in IDE
# from psychopy import locale_setup
# from psychopy import prefs
# from psychopy import sound, gui, visual, core, data, event, logging, clock
# from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
#                                 STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

# from psychopy.hardware import keyboard

#%% Functions 
def proc_img(file, params, save_img = False):
    '''

    Parameters
    ----------
    file : path to img file (.png or .jpg)
        Image desired to be processed 
    params : dict
        Dictionary of parameters relevant for stimuli processing. Required keys
        are:
            params['imsize'] : desired stim image dimensions in pixels
            params['ardir'] : if save_img = True, the directory where the .npy 
            arrays should be saved 
    save_img : bool, optional
        Save image to params['ardir']. The default is False.

    Returns
    -------
    proced_img : .npy array object 
        Returns resized, normalized, blurred stimuli image

    '''
    img = Image.open(file)
    exemplar = os.path.splitext(os.path.basename(file))[0] #Get exemplar name without extension str
    img = img.resize(params['imsize']) #resize image 
    if img.mode != 'L': #if img is RGB convert to grayscale 
        img = img.convert('L') # L = grayscale
    img = zscore(img, axis = None, ddof = 1) # zscore across entire array, ddof=1 to keep consistency between MATLAB default
    proced_img = cv2.normalize(img, None, alpha = -1, beta = 1, norm_type=cv2.NORM_MINMAX) #equiv of mat2gray, normalize to values between -1 and 1
    # proced_img = cv2.GaussianBlur(img, (7,7),1.5) #Blur img 
    if save_img is True:
        np.save(params['arrdir']+ os.sep + exemplar, proced_img)
    return proced_img

def scramble_img(file, save_img = False):
    '''
    Phase scramble an image 
    

    Parameters
    ----------
    file : path to filename
        filename of .npy file of processed stim image to be scrambled
    save_img : bool, optional
        To save the resulting scrambled image array. The default is False.

    Returns
    -------
    scramed_img : .npy array object
        Phase scrambled image as a numpy array. If save_img = True, saves as .npy
        

    '''
    
    imarray = np.load(file)
    exemplar = os.path.splitext(os.path.basename(file))[0]
    imsize = imarray.shape
    random_phase = np.angle(np.fft.fft2(np.random.rand(imsize[0],imsize[1])))
    imfourier = np.fft.fft2(imarray)
    amplitude = np.absolute(imfourier)
    phase = np.angle(imfourier) + random_phase
    scramed_img = np.real(np.fft.ifft2(amplitude * np.exp(np.sqrt(-1+0j) * phase)))
    #DIFFERENT FROM HLTP: Need to Renormalize image after scrambling to be between -1 and 1
    scramed_img = cv2.normalize(scramed_img, None, alpha = -1, beta = 1, norm_type=cv2.NORM_MINMAX)
    if save_img is True:
        np.save(params['arrdir']+ os.sep + exemplar + 'Scram', scramed_img)
    return scramed_img 

#Make Filter Param
def make_filter(params):
    '''
    

    Parameters
    ----------
    params : dict
        Dictionary of parameters relevant for stimuli processing. Required keys
        are:
            params['imsize'] : desired stim image dimensions in pixels

    Returns
    -------
    F : numpy array
        Image 2-D gaussian filter to fade edges of the stimuli image to background
        but otherwise leave contrast of central portion unchanged

    '''
    
    x1 = np.linspace(-1,1,params['imsize'][0])
    x2 = x1
    X1,X2 = np.meshgrid(x1, x2)
    jX = np.asarray([X1.flatten(), X2.flatten()]).T
    F = multivariate_normal.pdf(jX, [0,0], [[.2, 0],[0, .2]])
    F = np.reshape(F, params['imsize'])
    F -= np.min(F)
    F /= np.max(F)
    return F
#%% 

#Dictionary of parameters
# stimpath  = '/Users/brandonchen93/Downloads/Attention-Expectation/stimuli/FinalStimuli/'
stimpath  = '/Users/brandonchen93/Downloads/Image Toolboxes/SHINEtoolbox/SHINE_OUTPUT/LumMatchWholeImagePreProc/'

categories = ['AM', 'AN', 'FF', 'FM', 'HB', 'HH', 'OH','ON']

#proc HLTP OG Images 
# stimpath = '/Users/brandonchen93/Downloads/Attention-Expectation/stimuli/HLTP_Stimuli'

categories = ['animal', 'face', 'house', 'object']

params = {'imsize' : [300,300],
          'imbgrd' : 127,
          'categories' : categories,
          'stimdir' : stimpath,
    }


#Change to stimuli directory
os.chdir(params['stimdir'])
files = glob.glob('*.jpg')
#Make new directories to hold image arrays and processed image exemplars 
os.mkdir('stim_arrays')
os.mkdir('stim_examples')

params['exdir'] = params['stimdir'] + 'stim_examples'
params['arrdir'] = params['stimdir'] + 'stim_arrays'

params['filter'] = make_filter(params) 


#Save img arrays for all images in folder 
for img in files:
    proc_img(img, params, save_img = True)


for imarray in glob.glob(params['arrdir'] + os.sep + '*.npy'):
    scramble_img(imarray, save_img = True)


# Save exemplars as image files (.png) 
stim_contrast= 1
for img in glob.glob(params['arrdir'] + os.sep + '*.npy'):
    exemp = np.load(img)
    exemplar = os.path.splitext(os.path.basename(img))[0]
    
    F = stim_contrast * params['filter']
    proc_exemp = np.array(params['imbgrd'] * exemp * F + params['imbgrd'], dtype='uint8' )
    proc_exemp = Image.fromarray(proc_exemp,'L')
    proc_exemp.save(params['exdir'] + os.sep + exemplar + '.jpg')

#%% Normalize Exemplars with Threshold Estimate 
from psychopy.tools.filetools import fromFile
#Set directory variables 
data_dir = '/isilon/LFMI/VMdrive/Brandon/EASTO-local'

save_dir = '/isilon/LFMI/VMdrive/Brandon/EASTO-local'

data_dir = '/Users/brandonchen93/Downloads/Data_local'
save_dir = '/Users/brandonchen93/Downloads/Data_local'

data_ext = 'csv'


expt_version = 'Paradigm_V3'
paradigm = 'Spatial'


flist = glob.glob(data_dir + os.sep + expt_version + os.sep + 'Subjects/*/' + paradigm + os.sep + 'Quest/*.' + data_ext, 
                      recursive = True)

# In case the csv file from the raw data uses the wrong delimiter. 
for fname in flist:
    with open(fname, 'r') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.readline())
    if dialect.delimiter != ',':
        print('Saved CSV does not use correct delimiter, resaving from original psydat')
        
        #Generate file list of .psydat with same parameters 
        data_ext = 'psydat'
        flist = glob.glob(data_dir + os.sep + expt_version + os.sep + 'Subjects/*/' + paradigm + os.sep + 'Quest/*.' + data_ext, 
                          recursive = True)
        #Resave .csv files, overwrite old .csv is default 
        ea.psydat_to_csv(flist)
        
        #Try to make a global_df again 
        data_ext = 'csv'
        flist = glob.glob(data_dir + os.sep + expt_version + os.sep + 'Subjects/*/' + paradigm + os.sep + 'Quest/*.' + data_ext, 
                          recursive = True)
        exp_df = pd.concat([pd.read_csv(fname) for fname in flist], ignore_index = True)
    else:
        exp_df = pd.concat([pd.read_csv(fname) for fname in flist], ignore_index = True)
        
exp_df = pd.concat([pd.read_csv(fname) for fname in flist], ignore_index = True )


if 'TEA_Quest' in exp_df['expName'].any():
    exp_df.rename(columns = {'stim_left' : 'stim_1st', 
                             'stim_right': 'stim_2nd',
                             'participant': 'subject'}, inplace = True)
    quest_df = pd.concat([exp_df['subject'],
                      exp_df['trials.label'].rename('exemplar'),
                      exp_df['trials.intensity'].rename('intensity'),
                      exp_df['stim_1st'],
                      exp_df['stim_2nd'],
                      exp_df['trials.thisRepN'].rename('RepN'),
                      exp_df['trials.response'].rename('response'),
                      exp_df['ITI']], axis=1).dropna(subset = ['exemplar'])
        # Weird Error where subject column is duplicated. 

    quest_df.columns = ['subject', 'subject1', 'exemplar', 'intensity', 'stim_1st', 'stim_1st',
       'stim_2nd', 'stim_2nd', 'RepN', 'response', 'ITI']
    quest_df.loc[quest_df.subject.isnull(), 'subject'] = quest_df.subject1
    quest_df.drop('subject1', axis = 1, inplace = True)
    quest_df.dropna(subset = ['subject'], inplace = True)
    
elif 'SEA_Quest' in exp_df['expName'].any():
    exp_df.rename(columns = {'participant': 'subject'}, inplace = True)
    quest_df = pd.concat([exp_df['subject'],
                      exp_df['trials.label'].rename('exemplar'),
                      exp_df['trials.intensity'].rename('intensity'),
                      exp_df['stim_left'],
                      exp_df['stim_right'],
                      exp_df['trials.thisRepN'].rename('RepN'),
                      exp_df['trials.response'].rename('response'),
                      exp_df['ITI']], axis=1).dropna(subset = ['exemplar'])
    

        
elif 'OEA_Quest' in exp_df['expName'].any():
    exp_df.rename(columns = {'participant': 'subject'}, inplace = True)
    quest_df = pd.concat([exp_df['subject'],
                      exp_df['trials.label'].rename('exemplar'),
                      exp_df['trials.intensity'].rename('intensity'),
                      exp_df['stim'],
                      exp_df['trials.thisRepN'].rename('RepN'),
                      exp_df['trials.response'].rename('response'),
                      exp_df['ITI']], axis=1).dropna(subset = ['exemplar'])
else: 
    print('ERROR: Check DataFrame for ExpName')
    


    
mean_recognition = quest_df.groupby([quest_df.exemplar, quest_df.subject]).mean().response * 100

mean_recognition.to_pickle(data_dir + os.sep + paradigm + expt_version+'_quest_rec.pkl')
#%% Normalize based on Highest Threshold Contrast 


#%% Normalize based off of mean recognition 



normalize_value = 50/(mean_recognition)

for exemp in mean_recognition.index.unique():
    if mean_recognition[exemp] >= 75:
        normalize_value[exemp] = 0.8
    elif mean_recognition[exemp] <= 25:
        normalize_value[exemp] = 1.2

for exemp in mean_recognition.index.unique():
    im_dir = '/home/bc1693/Downloads/Temporal_NoFeedback/images/'
    new_dir = '/home/bc1693/Downloads/Temporal_NoFeedback/images/norm'
    new_contrast = normalize_value[exemp]
    img = np.load(im_dir + exemp + '.npy')
    #Apply to scram
    imgS = np.load(im_dir + exemp + 'Scram.npy')
    norm_img = img * new_contrast
    norm_imgS = imgS * new_contrast 
    np.save(new_dir + os.sep + exemp, norm_img)
    np.save(new_dir + os.sep + exemp + 'Scram', norm_imgS)


#%% Test Out applying quest estimate to imarrays 

os.chdir('/Users/brandonchen93/Downloads/Attention-Expectation')

quest_file = os.getcwd() + '/SB_questimate.pkl'
with open(quest_file, 'rb') as handle:
    quest_estimates = pickle.load(handle)

#Apply quest contrast to images... Then figure the code to do this frame-by-frame
#Write code in frame by frame, updating the .npy to be displayed in image location
for k, v in quest_estimates.items():
    exemplar = os.path.split(k)[1]
    new_contrast = v 
    img = np.load(params['arrdir']+ os.sep + exemplar + '.npy')
    
    F = new_contrast * params['filter']
    proc_exemp = np.array(params['imbgrd'] * img * F + params['imbgrd'], dtype='uint8' )
    proc_exemp = Image.fromarray(proc_exemp)
    proc_exemp.save('stim_examples/%s.png'%(exemplar))
proc_scram = np.array(params['imbgrd'] * scramed_img * F + params['imbgrd'], dtype='uint8' )
proc_scram.save('stim_examples/%sScram.png' %(exemplar))

#%% Check if stim is correct 

ex_ex  = np.load('expAM18.npy')
print(np.max(ex_ex))

og_file = 'images/AM2.npy'
ex_og = np.load(og_file)
quest_ind = 'images/AM2'
print(np.max(ex_og*params['filter']*quest_estimates[quest_ind]))

grad_conntr = np.linspace(0.01, quest_estimates[quest_ind], 4)


#%% Using Psychopy to Examine Processing of images 
# Doing so in IDE so that can inspect the npy array 

#Make Filter Param
from scipy.stats import zscore, multivariate_normal


def make_filter(params):
    '''
    

    Parameters
    ----------
    params : dict
        Dictionary of parameters relevant for stimuli processing. Required keys
        are:
            params['imsize'] : desired stim image dimensions in pixels

    Returns
    -------
    F : numpy array
        Image 2-D gaussian filter to fade edges of the stimuli image to background
        but otherwise leave contrast of central portion unchanged

    '''
    
    x1 = np.linspace(-1,1,params['imsize'][0])
    x2 = x1
    X1,X2 = np.meshgrid(x1, x2)
    jX = np.asarray([X1.flatten(), X2.flatten()]).T
    F = multivariate_normal.pdf(jX, [0,0], [[.2, 0],[0, .2]])
    F = np.reshape(F, params['imsize'])
    F -= np.min(F)
    F /= np.max(F)
    return F
    
# Make param dict 
params = {'imsize' : [300,300],
      'imbgrd' : 127,
      'categories' : ['AM', 'AN', 'FF', 'FM', 'HB', 'HH', 'OH','ON']
}
#Make Filter store in params dict 
params['filter'] = make_filter(params) 

#Load Quest Estimate 

#Load Stimulus Lists 
norm_stimpath = '/Users/brandonchen93/Downloads/EASTO/Stimuli/Norm_Stimuli_P13'
stimpath = '/Users/brandonchen93/EASTO/Paradigm/Stimuli/P9-TBD/images'

os.chdir(stimpath)

imglist = glob.glob('*.npy')

# Stimulus Contrasts
minC = 0.001 
maxC = .7
nC = 8 # 8 for 120 Hz
# nC = 4 # 4 for 60 Hz 

grad_contr = np.linspace(minC,maxC,nC)


#Initiate a window 

win = visual.Window(
    size=[1024, 768], fullscr=False, 
    winType='pyglet', allowGUI=True, allowStencil=True,
    monitor='MEG', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='deg')
#Initiate Stim 
stim_exemp = visual.ImageStim(
    win=win,
    name='stim_exemp', 
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=(4, 4),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=True,
    texRes=512, interpolate=True, depth=-1.0)


# img = np.load(imglist[38])  

    
stimClock =core.Clock()
while not event.getKeys(): #Press any key to close 
    for exemp in imglist:
        img = np.load(exemp)
        for frameN in range(8):
            F = grad_contr[frameN] * params['filter'] 
            target = np.array(img* F)
                    
            stim_exemp.setImage(target)
            stim_exemp.draw()
            win.flip()
            core.wait(1) #Wait one second before continuing the loop 
    
win.close()
core.quit()




