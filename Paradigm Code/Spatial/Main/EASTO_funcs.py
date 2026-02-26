# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:46:37 2022

@author: bc1693

Functions for use in EASTO paradigm. Main things are to

1) Generate lists of stimuli to be presented with appropriate frequency 
2) Load stimuli and process for presentation 
3) 
"""
from copy import deepcopy 
import numpy as np
import pandas as pd 
from scipy.stats import zscore, multivariate_normal

#%% Generate stimuli lists 

def generate_stim(real_images, scrambled_images, conditions, probability_ratio, paradigm): 
    """
    Function generates dictionaries of stimuli lists for each paradigm. Currently number of repeats is rounded to nearest integer after multiplying with probability ratio
    There may be a better way to do it instead, but this should work. 

    Parameters
    ----------
    real_images : List
        List of real images to be used in the experiment 
    scrambled_images : List
        List of scrambled images to be used in the experiment
    conditions : str
            Path to conditions files
    probability_ratio : List
        First entry is high probability, second entry is low probability. Should always sum to 1
    paradigm : str
        Spatial, Temporal or Category for deciding what is returned 

    Raises
    ------
    Exception
       Paradigm not defined

    Returns
    -------
    Dicts
        Three dictionaries to be used in psychopy for picking stimuli 
    """

    #Load conditions file for experiment
    conditions_df = pd.read_excel(conditions)
    #Group by trial_type and expectation block type to get trial type frequencies based on conditions file 
    trial_type_counts = conditions_df.groupby(['exp_loc', 'trial_type'])['trial_type'].count().rename('trial_freqs', inplace= True).reset_index()

    
    # Initiate master stimulus lists at the beginning of every block and assign 
    #expectation condition accordingly at the beginning of the routine 
    
    #Expectation blocks should have identical trial types, so just pick one 
    if paradigm == 'Spatial':
        block_choice = 'L'
    elif paradigm == 'Temporal': 
        block_choice = '1st'
    elif paradigm == 'Category': 
        block_choice == 'Face'
    else: 
        raise Exception("Define which paradigm this should be for, choose between Spatial, Temporal, or Category")
        
    trials_df = trial_type_counts.loc[trial_type_counts['exp_loc'] == block_choice]
    # Generate dictionary of high probability condition stimuli 
    high_probability_blocks = {trial_type :
                                real_images * int(round(probability_ratio[0] * trials_df.loc[trials_df['trial_type'] == trial_type]['trial_freqs'].values[0])) +
                                scrambled_images * int(round(probability_ratio[1] * trials_df.loc[trials_df['trial_type'] == trial_type]['trial_freqs'].values[0]))
                                for trial_type in trials_df['trial_type'].unique()}
    
    low_probability_blocks = {trial_type :
                                real_images * int(round(probability_ratio[1] * trials_df.loc[trials_df['trial_type'] == trial_type]['trial_freqs'].values[0])) +
                                scrambled_images * int(round(probability_ratio[0] * trials_df.loc[trials_df['trial_type'] == trial_type]['trial_freqs'].values[0]))
                                for trial_type in trials_df['trial_type'].unique()}
    #For neutral block trials
    trials_df = trial_type_counts.loc[trial_type_counts['exp_loc'] == 'N']
    equal_probability_blocks = {trial_type :
                                real_images * int(round(0.5 * trials_df.loc[trials_df['trial_type'] == trial_type]['trial_freqs'].values[0])) +
                                scrambled_images * int(round(0.5 * trials_df.loc[trials_df['trial_type'] == trial_type]['trial_freqs'].values[0]))
                                for trial_type in trials_df['trial_type'].unique()}
        
    # Create deepcopies (not just reference varibles) of stimuli & Real vs. Scram 

    if paradigm == 'Spatial':
        expect_left_trials = {'left_stims' : deepcopy(high_probability_blocks),
                    'right_stims' : deepcopy(low_probability_blocks)}
                    
        expect_right_trials = {'left_stims' : deepcopy(low_probability_blocks),
                    'right_stims' : deepcopy(high_probability_blocks)}
                    
        expect_neutral_trials = {'left_stims' : deepcopy(equal_probability_blocks),
                    'right_stims' : deepcopy(equal_probability_blocks)}
        
        return expect_left_trials, expect_right_trials, expect_neutral_trials
    
    elif paradigm == 'Temporal': 
       expect_1st_trials = {'int1_stims' : deepcopy(high_probability_blocks),
                   'int2_stims' : deepcopy(low_probability_blocks)}
                   
       expect_2nd_trials = {'int1_stims' : deepcopy(low_probability_blocks),
                   'int2_stims' : deepcopy(high_probability_blocks)}
                   
       expect_neutral_trials = {'int1_stims' : deepcopy(equal_probability_blocks),
                   'int2_stims' : deepcopy(equal_probability_blocks)}
       
       return expect_1st_trials, expect_2nd_trials, expect_neutral_trials

    elif paradigm == 'Category': 
        
        return 'Not Ready Yet'
        
        
#%% 2D gaussian filter function for stimuli 

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
    


