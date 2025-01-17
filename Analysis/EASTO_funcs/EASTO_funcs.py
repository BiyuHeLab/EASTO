#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:08:01 2021

@author: brandonchen93


Functions for EASTO Analysis 

"""


import pandas as pd 
import numpy as np 
import scipy.stats
import os

#%% Behavioral Data Preprocessing 

def psydat_to_csv(data, overwrite = True):
    """
    Function to resave csv from original psydat datafile in the case that the csv is not readable
    i.e. using delimiter "auto" as opposed to ","

    Parameters
    ----------
    fname : psydat filename or list of filenames
        psydat from relevant experiment. Requires full path 
    overwrite: bool 
        if true, will overwrite any existing .csv file with same basename as .psydat 
        if false, will rename newly generated .csv file 

    Returns
    -------
    Replaces .csv file in present directory. Overwrites old .csv file 

    """
    from psychopy.misc import fromFile 
    
    if overwrite:
        replaceMethod = 'overwrite'
    else: 
        replaceMethod = 'rename'
        
    if type(data) is list:
        for fname in data: 
            
            psydata = fromFile(fname)
            # TO DO: Rename subject column to fit with the label on the folder 
            data_dir = os.path.dirname(fname)
            basename = os.path.basename(fname)
            
            psydata.saveAsWideText(data_dir + os.sep + os.path.splitext(basename)[0] + '.csv',
                                   fileCollisionMethod = replaceMethod)
    else: 
        psydata = fromFile(data)
            
        data_dir = os.path.dirname(data)
        basename = os.path.basename(data)
        
        psydata.saveAsWideText(data_dir + os.sep + os.path.splitext(basename)[0] + '.csv', 
                               fileCollisionMethod = replaceMethod)

        
    return

#%% Define Behavioral Functions 

def HLTP_get_bhv_vars(df):
    n_rec_real = sum(df[df.real == True].recognition == 1)
    n_real = len(df[(df.real == True) & (df.recognition != 0)])
    n_rec_scr = sum(df[df.real == False].recognition == 1)
    n_scr = len(df[(df.real == False) & (df.recognition != 0)])
    p_correct = df.correct.values.mean()   
    # catRT =  df.catRT.values.mean()
    # recRT =  df.recRT.values.mean()
    HR, FAR, d, c, d_var, c_var, lnb = get_sdt_msr(n_rec_real, n_real, n_rec_scr, n_scr)
    return HR, FAR, d, c, d_var, c_var, lnb, p_correct #, catRT, recRT

def get_bhv_vars(df, paradigm, version):
    """

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame with behavioral data that includes the following columns:
            probeReal: Whether the probed stimulus was real (1) or scrambled (0)
            recognition: Whether response was recognized (1) or unrecognized (0)
            Correct: Whether categorization was correct (1) or incorrect (0)
            qcat_RT: Reaction time for categorization response
            qrec_RT: Reaction time for detection/recognition response
    paradigm : string
        Whether the paradigm was temporal, spatial, or object. 

    Returns
    -------
    HR : Float
        Hit Rate, i.e. reported recognized on real images
    FAR : Float
        False-Alarm Rate, i.e. reported recognized on scrambled images
    d : Float
        d' (Sensitivity) calculated according to standard SDT
        (see function get_sdt_msr for implementation details)
    c : Float
        Criterion (response bias) calculated according to standard SDT 
        (see function get_sdt_msr for implementation details)
    p_correct : Float
        Proportion correct categorization 
        If paradigm is "Object" or "OEA" then returns a tuple where the first 
        value indicated the proportion correct for subcategorization task (fine),
        and second value indicates proportion correct for general categorization
        (coarse). e.g. (Fine, Coarse)
    catRT : Float
        Mean reaction time for answering categorization question
        If paradigm is "Object" or "OEA" then returns a tuple where the first 
        value indicated the reqction time for subcategorization task (fine),
        and second value indicates reaction time for general categorization
        (coarse). e.g. (Fine, Coarse)
    recRT : Float
        Mean reaction time for answering recognition question 

    """
    n_rec_real = sum(df[df.probeReal == True].recognition == 1)
    n_real = len(df[(df.probeReal == True) & (~pd.isnull(df.recognition))])
    
    n_rec_scr = sum(df[df.probeReal == False].recognition == 1)
    n_scr = len(df[(df.probeReal == False) & (~pd.isnull(df.recognition))])
    

    p_correct = np.nanmean(df.correct.values)
    catRT =  np.nanmean(df.categorization_RT.values)
        
    recRT =  np.nanmean(df.recognition_RT.values)
    if version in ['Paradigm_V1', 'Paradigm_V2']:
        confRT = np.nan
    else:
        confRT =  np.nanmean(df.confidence_RT.values)

    
    HR, FAR, d, c, d_var, c_var, lnb = get_sdt_msr(n_rec_real, n_real, n_rec_scr, n_scr)
    return HR, FAR, d, c, d_var, c_var, lnb, p_correct, catRT, recRT, confRT

def get_sdt_msr(n_rec_signal, n_signal, n_rec_noise, n_noise):
    """
    

    Parameters
    ----------
    n_rec_signal : TYPE
        DESCRIPTION.
    n_signal : TYPE
        DESCRIPTION.
    n_rec_noise : TYPE
        DESCRIPTION.
    n_noise : TYPE
        DESCRIPTION.

    Returns
    -------
    HR : TYPE
        DESCRIPTION.
    FAR : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    d_var : TYPE
        DESCRIPTION.
    c_var : TYPE
        DESCRIPTION.
    lnb : TYPE
        DESCRIPTION.

    """
    Z = scipy.stats.norm.ppf
    #Corrections for FAR = 0 and HR = 1
    if (n_noise == 0): FAR = np.nan
    else: FAR = max( float(n_rec_noise) / n_noise, 1. / (2 * n_noise) )
    if  n_signal == 0: HR = np.nan
    else: HR = min( float(n_rec_signal) / n_signal, 1 - (1. / (2 * n_signal)))
    # Calculate d' and criterion 
    d = Z(HR)- Z(FAR)
    c = -(Z(HR) + Z(FAR))/2
    #Optimal Criterion
    lnb = d * c 
    #Variance of d' and criterion estimates 
    d_var = HR*(1-HR)/(n_signal * scipy.stats.norm.pdf(Z(HR))**2) + FAR*(1-FAR)/(n_noise * scipy.stats.norm.pdf(Z(FAR))**2)
    c_var = 0.25 * d_var 
    # return nans instead of infs
    if np.abs(d) == np.inf: d = np.nan
    if np.abs(c) == np.inf: c = np.nan
    
    return HR, FAR, d, c, d_var, c_var, lnb 

def make_confusion(df, expectation_condition, attention_condition, recognition = 'recognized', img_type = 'all'):
    if recognition == 'recognized':
        use_rec = 1 
    elif recognition == 'unrecognized':
        use_rec = 0
    elif recognition == 'all':
        use_rec = df.recognition
        
    if img_type == 'all':
        df = df.loc[(df.expectation_condition == expectation_condition) & 
                    (df.attention_condition == attention_condition) &
                    (df.recognition == use_rec)].fillna(value=np.nan)
    elif img_type == 'real':
        use_real = 1
        # 1 is coded as real, 0 is coded as scrambled
        df = df.loc[(df.expectation_condition == expectation_condition) & 
                    (df.attention_condition == attention_condition) & 
                    (df.probeReal == use_real) & 
                    (df.recognition == use_rec)].fillna(value=np.nan)
    elif img_type == 'scrambled':
        use_real = 0
        # 1 is coded as real, 0 is coded as scrambled
        df = df.loc[(df.expectation_condition == expectation_condition) & 
                    (df.attention_condition == attention_condition) & 
                    (df.probeReal == use_real) &
                    (df.recognition == use_rec)].fillna(value=np.nan)


    df.replace(to_replace='None', value=np.nan, inplace=True)
    confusion_matrix = pd.crosstab(df['objective_category'], df['category_response'], 
                                   rownames=['Actual'], colnames=['Selected'],
                                   normalize = 'index')
    return confusion_matrix





