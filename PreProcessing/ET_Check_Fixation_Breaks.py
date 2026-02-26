# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:32:59 2025
1. Extracts and plots gaze positions between two time points (e.g., between attention cue onset and stimulus offset).
2. Determines whether fixation was maintained or broken in each trial 
   (criterion: deviation > 2 degrees of visual angle for more than 100 ms consecutively).
3. Stores fixation-related information in the behavioral (bhv) DataFrame.

@author: Yuan-hao Wu
"""
#%% Load Packages
import pandas as pd
from os.path import join
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

#%% PARAMETERS
width_cm = 50        # screen width in cm
height_cm = 35       # screen height in cm
res_x = 1920         # horizontal resolution (pixels)
res_y = 1080         # vertical resolution (pixels)
x_center = res_x / 2  # 960
y_center = res_y / 2  # 540
view_dist_cm = 55    # viewing distance in cm

#CONVERT PIXELS TO DEGREES OF VISUAL ANGLE
# step 1: pixel size in cm
pixel_size_cm = width_cm / res_x
# step 2: degrees per pixel
deg_per_pixel = 2 * math.degrees(math.atan((pixel_size_cm / 2) / view_dist_cm))
# step 3: pixels per degree (inverse)
pix_per_deg = 1 / deg_per_pixel

pixBuffer = 36.86135449077602*2; timeBuffer = 100
# %%
def px_to_dva(px, center_px=x_center, px_per_deg=pix_per_deg):
    """Convert pixel coordinates to deviation-from-center in DVA."""
    return (px - center_px) / px_per_deg

def load_ET_data(ET_Dir, ET_filename):
    #ET_Dir = join(DataDir, exp_version, subj, paradigm, 'Main', 'edfData') 
    ET_data = join(ET_Dir, ET_filename)
    with open(ET_data, "rb") as f:
                data = pickle.load(f)
                dfRec = data['dfRec']
                dfMsg = data['dfMsg']
                dfFix = data['dfFix']
                dfSacc = data['dfSacc']
                dfBlink = data['dfBlink']
                dfSamples = data['dfSamples']
    return dfRec, dfMsg, dfFix, dfSacc, dfBlink, dfSamples

def get_dominant_eye(dfFix):
    # Determine the dominant eye based on the number of fixation
    if len(dfFix['eye'].unique()) == 1:
        dominant_eye = dfFix['eye'].unique()[0]
    elif len(dfFix['eye'].unique()) == 2:
        # Determine the dominant eye based on the number of fixations
        dominant_eye = dfFix['eye'].value_counts()
        if dominant_eye['L'] > dominant_eye['R']:
            dominant_eye = "L"
        else:
            dominant_eye = "R"   
    return dominant_eye


def get_gaze_positions(subj_df_sorted, condition, dfMsg, dfSamples, dominant_eye):
    #if exp_version == 'Paradigm_Control':
    #block_types = ['attention', 'expectation']
    #for block in block_types: 
    if condition == 'LE':
        cond_mask = subj_df_sorted['exp_color']!='Neutral'
    elif condition =='GE':
        cond_mask = subj_df_sorted['exp_color']=='Neutral'
    elif condition == "":
        cond_mask = np.ones(len(subj_df_sorted), dtype=bool)
    cond_df = subj_df_sorted[cond_mask].reset_index()

    if dominant_eye =='R':
        x_pos = 'RX'
        y_pos = 'RY'
    elif dominant_eye =='L':
        x_pos = 'LX'
        y_pos = 'LY'
    
    # Determine the time winndow of interest (attention cue onset - stimulus offset
    onsetTime = dfMsg.loc[dfMsg.text.str.contains('attention_cue_onset')].reset_index(drop = True).rename(columns = {'time':'onset', 'text' : 'onset_event'})
    offsetTime = dfMsg.loc[dfMsg.text.str.contains('poststim_fix')].reset_index(drop = True).rename(columns = {'time':'offset', 'text' : 'offset_event'})
    checkFixationPeriod = pd.concat([onsetTime, offsetTime], axis = 1)
    checkFixationPeriod = checkFixationPeriod[cond_mask].reset_index()

    plt.figure(figsize=(9, 6))
    for idx in range(len(cond_df)):
        #Select the relevant samples to check for fixation 
        samples = dfSamples.loc[dfSamples.tSample.between(checkFixationPeriod.onset[idx], checkFixationPeriod.offset[idx])]
        #samples.reset_index   #(drop=True, inplace=True)
        X_SamplesFix = samples[x_pos].between(x_center - pixBuffer, x_center + pixBuffer)
        Y_SamplesFix = samples[y_pos].between(y_center - pixBuffer, y_center + pixBuffer)

        Fix = X_SamplesFix * Y_SamplesFix # Require fixation for each eye for both horizontal AND vertical axes
        
        fix_broken = ~Fix
        segments = (fix_broken != fix_broken.shift()).cumsum()
        durations = (fix_broken[fix_broken].groupby(segments[fix_broken]).size())
        
        if (durations >= timeBuffer).any():
        #if numSamplesBrokeFix <= len(samples) - timeBuffer: 
            cond_df.loc[idx, 'brokeFixation'] = 1
            plt.plot(np.arange(1, len(samples)+1, 1), 
                     px_to_dva(samples[x_pos]), color = 'red', alpha=0.5)
        else:
            cond_df.loc[idx, 'brokeFixation'] = 0
            plt.plot(np.arange(1, len(samples)+1, 1), 
                     px_to_dva(samples[x_pos]), color = 'blue')
    nBrokeFix = cond_df['brokeFixation'].sum()
    print(f"Subject {subj} broke fixation in {str(int(nBrokeFix))} trials")
    plt.vlines(x = 0, ymin=-7, ymax = 7, colors='black', linestyles='dashed')
    plt.vlines(x = 50, ymin=-7, ymax = 7, colors='black', linestyles='dashed')
    plt.vlines(x = 950, ymin=-7, ymax = 7, colors='black', linestyles='dashed')
    plt.vlines(x = 1016, ymin=-7, ymax = 7, colors='black', linestyles='dashed')
    plt.xlabel('Time relative to attention cue onset')
    plt.ylim([-7, 7])
    plt.ylabel(f'{x_pos} (dva)')
    plt.title(f'{condition} block: | Subject {subj}')
    plt.show()     
    return cond_df
        
        
DataDir = '/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior/Data'
exp_version = 'Paradigm_V3' #['Paradigm_V3', 'Paradigm_Control', 'Paradigm_expControl']
paradigm = 'Spatial'
sub_version = ''
bhv_df = pd.read_pickle(join(DataDir, f"{paradigm}{exp_version}_bhv_df.pkl")) 


for subj in ['P20', 'P21', 'P22', 'P30', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39',
             'P40', 'P41', 'P42', 'P43', 'P44', 'P45', 'P46','P50', 'P52']:
    ET_Dir = join(DataDir, exp_version, subj, paradigm, 'Main', 'edfData')
    ET_filename = f"{subj}_eye_tracking_data.pkl"
    subj_df = bhv_df[(bhv_df['subject']==subj)]
    subj_df_sorted = subj_df.sort_values(['block', 'trial'])
    #subj_df.reset_index(drop=True, inplace=True)

    dfRec, dfMsg, dfFix, dfSacc, dfBlink, dfSamples = load_ET_data(ET_Dir, ET_filename)
    dominant_eye = get_dominant_eye(dfFix)

    #for condition in ['GE', 'LE']:
    cond_df = get_gaze_positions(subj_df_sorted, "", dfMsg, dfSamples, dominant_eye)

    # Create the column if it doesn't exist yet
    if 'brokeFixation' not in bhv_df.columns:
        bhv_df['brokeFixation'] = np.nan  # or False if it's a boolean column

    # Update only the relevant rows
    bhv_df.loc[cond_df['index'], 'brokeFixation'] = cond_df['brokeFixation'].values

bhv_df.to_pickle(join(DataDir, f"{paradigm}{exp_version}_bhv_df.pkl"))
             