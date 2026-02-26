#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:27:51 2021

@author: brandonchen93

To take individual subject experiment run from psychopy output in .csv
format and convert into a behavioral dataframe keeping all relevant information. 



"""
#%% Import packages 
import os
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import joblib 
import glob
import sys
sys.path.append('/Users/brandonchen93/EASTO-Behavior/Analysis/EASTO_funcs')
import EASTO_funcs as ea
import csv
import re


#%% Set directory variables 

data_ext = 'csv'
expt_version = 'Paradigm_expControl'
paradigm = 'Spatial'
local = True 

if local:
    data_dir = '/Users/brandonchen93/Downloads/Data_local'
    save_dir = '/Users/brandonchen93/Downloads'
else:
    data_dir = '/isilon/LFMI/VMdrive/Brandon/EASTO-local'
    save_dir = '/isilon/LFMI/VMdrive/Brandon/EASTO-local'
#%% Compile raw data files, check if delimiter is correct, resave if not, and concat into single dataframe

flist = glob.glob(data_dir + os.sep + expt_version + os.sep + 'Subjects/*/' + paradigm + os.sep + 'Main/*.' + data_ext, 
                      recursive = True)

# In case the csv file from the raw data uses the wrong delimiter. 
for fname in flist:
    with open(fname, 'r') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.readline())
    if dialect.delimiter != ',':
        print('Saved CSV does not use correct delimiter, resaving from original psydat')
        
        #Generate file list of .psydat with same parameters 
        data_ext = 'psydat'
        flist = glob.glob(data_dir + os.sep + expt_version + os.sep + 'Subjects/*/' + paradigm + os.sep + 'Main/*.' + data_ext, 
                          recursive = True)
        #Resave .csv files, overwrite old .csv is default 
        ea.psydat_to_csv(flist)
        
        #Try to make a global_df again 
        data_ext = 'csv'
        flist = glob.glob(data_dir + os.sep + expt_version + os.sep + 'Subjects/*/' + paradigm + os.sep + 'Main/*.' + data_ext, 
                          recursive = True)
        glob_df = pd.concat([pd.read_csv(fname) for fname in flist], ignore_index = True, sort = True)
    else:
        glob_df = pd.concat([pd.read_csv(fname) for fname in flist], ignore_index = True)

#%% Create Dataframe of processed behavioral data 

if paradigm == 'Spatial':
    #Select the Columns with relevant variables
    bhv_df = pd.concat([glob_df['subject'],
                       glob_df['blocks.thisN'].rename('block'),
                       glob_df['trials.thisN'].rename('trial'),
                       glob_df['exp_loc'],
                       glob_df['att_loc'],
                       glob_df['probe_loc'],
                       glob_df['trial_type'], # Potentially remove, use for validating psychopy conditions file 
                       glob_df['validity'], 
                       glob_df['stim_left'],
                       glob_df['stim_right'],
                       
                       #Responses 
                       glob_df['objective_category'],
                       glob_df['category_response'],
                       glob_df['prob_left'],
                       glob_df['prob_right'],
                       glob_df['rec_resp.keys'].rename('recognition_response'),
                       glob_df['conf_resp.keys'].rename('confidence_response'), #Confidence of recognition choice, not categorization
                       glob_df['cat_resp.rt'].rename('categorization_RT'),
                       glob_df['rec_resp.rt'].rename('recognition_RT'),
                       glob_df['conf_resp.rt'].rename('confidence_RT'),
                       
                       #More descriptives 
                       glob_df['EyeTrack'],
                       glob_df['ITI']], axis = 1).dropna(subset = ['exp_loc']) #Drop rows in which NaN in exp_block column
        
    # Remove path and extension from stimulus file stirng 
    bhv_df.stim_left = bhv_df.stim_left.str[10:-4]
    bhv_df.stim_right = bhv_df.stim_right.str[10:-4]
    
    #Dummy code whether stimulus was real or scrambled 
    bhv_df['stim_left_real'] = np.nan
    bhv_df['stim_right_real'] = np.nan
    
    bhv_df.loc[bhv_df.stim_right.str.contains('scram'), 'stim_right_real'] = 0
    bhv_df.loc[~bhv_df.stim_right.str.contains('scram'), 'stim_right_real'] = 1
    
    bhv_df.loc[bhv_df.stim_left.str.contains('scram'), 'stim_left_real'] = 0
    bhv_df.loc[~bhv_df.stim_left.str.contains('scram'), 'stim_left_real'] = 1
    
    #Determine whether the probed stimulus was real
    bhv_df['probeReal'] = np.nan 
    bhv_df.loc[(bhv_df.stim_left_real == 1) & (bhv_df.probe_loc == 'L'), 'probeReal'] = 1 
    bhv_df.loc[(bhv_df.stim_left_real == 0) & (bhv_df.probe_loc == 'L'), 'probeReal'] = 0
    
    bhv_df.loc[(bhv_df.stim_right_real == 1) & (bhv_df.probe_loc == 'R'), 'probeReal'] = 1 
    bhv_df.loc[(bhv_df.stim_right_real == 0) & (bhv_df.probe_loc == 'R'), 'probeReal'] = 0
    
    #Determine whether the non-probed stimulus was real 
    bhv_df['not_probe_real'] = np.nan 
    bhv_df.loc[(bhv_df.stim_left_real == 1) & (bhv_df.probe_loc == 'R'), 'not_probe_real'] = 1 
    bhv_df.loc[(bhv_df.stim_left_real == 0) & (bhv_df.probe_loc == 'R'), 'not_probe_real'] = 0
    
    bhv_df.loc[(bhv_df.stim_right_real == 1) & (bhv_df.probe_loc == 'L'), 'not_probe_real'] = 1 
    bhv_df.loc[(bhv_df.stim_right_real == 0) & (bhv_df.probe_loc == 'L'), 'not_probe_real'] = 0

    #Define the color of expected placeholder 
    bhv_df['exp_color'] = np.nan
    bhv_df.loc[(bhv_df.prob_left == '[-1.0, -1.0, -1.0]') & (bhv_df.exp_loc == 'L'), 'exp_color' ] = 'black'
    bhv_df.loc[(bhv_df.prob_left == '[1.0, 1.0, 1.0]') & (bhv_df.exp_loc == 'L'), 'exp_color' ] = 'white'
    bhv_df.loc[(bhv_df.prob_right == '[-1.0, -1.0, -1.0]') & (bhv_df.exp_loc == 'R'), 'exp_color' ] = 'black'
    bhv_df.loc[(bhv_df.prob_right == '[1.0, 1.0, 1.0]') & (bhv_df.exp_loc == 'R'), 'exp_color' ] = 'white'
    
    #Replace all other possibilities (attention blocks for control with neutral
    bhv_df['exp_color'].fillna('Neutral', inplace = True)
    
    #Define categories for stimuli
    categories = ['F','A','O','H']

    bhv_df['base_cat_left'] = np.nan
    bhv_df['base_cat_right'] = np.nan

    
    for cats in categories:
        bhv_df.loc[bhv_df.stim_left.str.startswith(cats), 'base_cat_left'] = cats
        bhv_df.loc[bhv_df.stim_right.str.startswith(cats), 'base_cat_right'] = cats

    
    #DEFINE WHICH STIMULUS WAS PROBED 
    bhv_df.loc[bhv_df.probe_loc == 'L', 'probed_stimulus'] = bhv_df.stim_left
    bhv_df.loc[bhv_df.probe_loc == 'R', 'probed_stimulus'] = bhv_df.stim_right

    
    # Recognition Response
    # Convert button press to (R)ec (1) and (U)nrec (0)
    # MEG System Buttons Responnse is 1-4; EEG Lab 6-9
    recognition_map = {7 : 1, 8 : 0,
                       2 : 1, 3 : 0,
                      '7' : 1, '8' : 0,
                      '2' : 1, '3' : 0,
                      'None' : np.nan}
    
    bhv_df['recognition'] = bhv_df['recognition_response'].map(recognition_map)
    
    #Confidence Response 
    confidence_map = {6 : 1, 7 : 2, 8 : 3, 9 : 4,
                '6' : 1, '7' : 2, '8' : 3, '9' : 4,
                1 : 1, 2 : 2, 3 : 3, 4 : 4,
                '1' : 1, '2' : 2, '3' : 3, '4' : 4,
                'None' : np.nan}
    bhv_df['confidence'] = bhv_df['confidence_response'].map(confidence_map)
    
    #Category Correctness 
    bhv_df['correct'] = np.nan
   
    bhv_df['category_response'].replace(to_replace=['None'], value=np.nan, inplace=True)
    
    bhv_df.loc[bhv_df['category_response'] == bhv_df['objective_category'], 'correct'] = 1 
    bhv_df.loc[(bhv_df['category_response'] != bhv_df['objective_category']) & 
               (~pd.isnull(bhv_df['category_response'])), 'correct'] = 0
    
    # Count number of non-responses 
    print(f'Number of Missing Responses for Recognition Question is {bhv_df.recognition.isnull().sum()}')
    print(f'Number of Missing Responses for Categorization Question is {bhv_df.category_response.isnull().sum()}')
    print(f'Number of Missing Responses for Confidence Question is {bhv_df.confidence.isnull().sum()}')


else: 
    print('Update this code to reflect new paradigm version')

#%% Add conditions (Expected, Unexpected, Attended, Unattended, Neutral)

"""
# Trial type in previous versions was incorrectly labeled. Redo the labels based on more verifiable
# experimental soruces. e.g. stimulus, probe_loc, and att_loc 

Also Expectation can be defined in two ways 
1) Expectation of whether the image was real or noise based on probability cues 
2) Expectation of whether the stimulus aligned with the probability cue. i.e. whether expectations were fulfilled 

In this experiment, either can make sense. Their task is to report their subjective experience. 
The former (1) makes sense if we think of them doing this, the latter (2) makes sense if they think of 
the recognition question as a discrimination. 
"""

if paradigm == 'Spatial':
    alt_1 = 'L'
    alt_2 = 'R'
elif paradigm == 'Temporal':
    alt_1 = '1st'
    alt_2 = '2nd'


#Expect Stimulus 1 (Left/1st) Trials
bhv_df.loc[(bhv_df.exp_loc == alt_1) & (bhv_df.probe_loc == alt_1), 'expectation_condition'] = 'Expect Real'
bhv_df.loc[(bhv_df.exp_loc == alt_1) & (bhv_df.probe_loc == alt_2), 'expectation_condition'] = 'Expect Scrambled'
#Expect Stimulus 2 (Right/2nd) Trials
bhv_df.loc[(bhv_df.exp_loc == alt_2) & (bhv_df.probe_loc == alt_2), 'expectation_condition'] = 'Expect Real'
bhv_df.loc[(bhv_df.exp_loc == alt_2) & (bhv_df.probe_loc == alt_1), 'expectation_condition'] = 'Expect Scrambled'

# For expectation control experiments 
bhv_df.loc[bhv_df.exp_loc == 'Lo', 'expectation_condition'] = 'Expect Scrambled' 
bhv_df.loc[bhv_df.exp_loc == 'Hi', 'expectation_condition'] = 'Expect Real'
    
#Expect Neutral Trials
bhv_df.loc[(bhv_df.exp_loc == 'N'), 'expectation_condition'] = 'Neutral'

#Define Expectation by whether probed stimulus was consistent with expectation cue 

#Expected if stimulus was expected real & real or expected scrambled & scrambled 
bhv_df.loc[(bhv_df.expectation_condition == 'Expect Real') & (bhv_df.probeReal == 1), 'expectation_validity'] = 'Expected'
bhv_df.loc[(bhv_df.expectation_condition == 'Expect Scrambled') & (bhv_df.probeReal == 0), 'expectation_validity'] = 'Expected'

#Unexpected if stimulus was expected real & scrambled or expected scrambled & real 
bhv_df.loc[(bhv_df.expectation_condition == 'Expect Real') & (bhv_df.probeReal == 0), 'expectation_validity'] = 'Unexpected'
bhv_df.loc[(bhv_df.expectation_condition == 'Expect Scrambled') & (bhv_df.probeReal == 1), 'expectation_validity'] = 'Unexpected'

#Attention Conditions 

#Attend Stimulus 1 (Left/1st) Trials
bhv_df.loc[(bhv_df.att_loc == alt_1) & (bhv_df.probe_loc == alt_1), 'attention_condition'] = 'Attended'
bhv_df.loc[(bhv_df.att_loc == alt_1) & (bhv_df.probe_loc == alt_2), 'attention_condition'] = 'Unattended'

#Attend Stimulus 2 (Right/2nd) Trials
bhv_df.loc[(bhv_df.att_loc == alt_2) & (bhv_df.probe_loc == alt_2), 'attention_condition'] = 'Attended'
bhv_df.loc[(bhv_df.att_loc == alt_2) & (bhv_df.probe_loc == alt_1), 'attention_condition'] = 'Unattended'

#Attend Neutral Trials
bhv_df.loc[(bhv_df.att_loc == 'N'), 'attention_condition'] = 'Neutral'

#%% Add contrast of each image

# Find files with contrast threshold estimates
questEstimates = glob.glob(data_dir + os.sep + expt_version + os.sep + 'Subjects/*/' + paradigm + os.sep + 'Main/*.pkl', 
                      recursive = True)

# Load dictionary of image contrasts for each subject into a single dataframe 

estimatesDF = pd.concat([pd.DataFrame.from_dict(pickle.load(open(subEstimate, 'rb')), orient = 'index', columns =['contrast']).assign(subject =re.search(r'P\d+', subEstimate).group()).reset_index(names = 'image')
                     for subEstimate in questEstimates], ignore_index = True)

# Clean up image names dataframe so merging is easier 
estimatesDF.image = estimatesDF.image.str[10:]
estimatesDF = pd.concat([estimatesDF, estimatesDF.assign(image = estimatesDF.image+'scram')])


# Probed Stimulus contrast 
bhv_df = pd.merge(bhv_df, estimatesDF, left_on = ['probed_stimulus', 'subject'], right_on = ['image', 'subject'])
bhv_df.rename(columns = {'contrast' : 'probedContrast'}, inplace = True)

# Left and right stimulus contrasts
bhv_df = pd.merge(bhv_df, estimatesDF, left_on = ['stim_left', 'subject'], right_on = ['image', 'subject'])
bhv_df.rename(columns = {'contrast' : 'contrastLeft'}, inplace = True)

bhv_df = pd.merge(bhv_df, estimatesDF, left_on = ['stim_right', 'subject'], right_on = ['image', 'subject'])
bhv_df.rename(columns = {'contrast' : 'contrastRight'}, inplace = True)

# Drop unnecessary columns 
bhv_df.drop(columns = ['image', 'image_x', 'image_y'], inplace = True)

# Calculate contrast difference (Probed - NotProbed)
bhv_df.loc[bhv_df.probe_loc == 'L', 'contrastDiff'] = bhv_df.contrastLeft - bhv_df.contrastRight
bhv_df.loc[bhv_df.probe_loc == 'R', 'contrastDiff'] = bhv_df.contrastRight - bhv_df.contrastLeft
#%% Re-order Columnns (not necessary, but nice to have)

col_titles = ['subject',
 'block',
 'trial',
 'attention_condition',
 'expectation_condition',
 'expectation_validity',
 'exp_loc',
 'att_loc',
 'exp_color',
 'probe_loc',
 'stim_left',
 'contrastLeft',
 'stim_right',
 'contrastRight',
 'probed_stimulus',
 'probedContrast',
 'probeReal',
 'not_probe_real',
 'contrastDiff',
 'recognition',
 'confidence',
 'correct',
 'objective_category',
 'categorization_RT',
 'recognition_RT',
 'confidence_RT',
 'base_cat_left',
 'base_cat_right',
 'stim_left_real',
 'stim_right_real',
 'category_response',
 'recognition_response',
 'confidence_response',
 'trial_type',
 'validity',
 'EyeTrack',
 'ITI',
]

bhv_df = bhv_df.reindex(columns = col_titles)

# Save Dataframe as .pkl 
bhv_df.to_pickle(data_dir + os.sep + paradigm + expt_version + '_bhv_df.pkl')

# %%
