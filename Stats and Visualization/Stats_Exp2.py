#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

mainEffect = 'att' #['exp', 'att', 'None']
#%% Import packages
import os
from os.path import join
import seaborn as sns
import pandas as pd 
import numpy as np 
from scipy.stats import t
import pingouin as pg
import EASTO_funcs.EASTO_funcs as ea
import pingouin as pg


RootDir = '/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior'
AnalysisDir = join(RootDir, 'Analysis')
DataDir = join(RootDir, 'Data')
#Load processed data 
os.chdir(DataDir)
paradigm = 'Spatial'
expt_version = 'Paradigm_Control'
sub_version = ''
fname = f"{paradigm}{expt_version}{sub_version}_bhv_df.pkl"
bhv_df = pd.read_pickle(fname)

def CI(data, confidence_level = 0.95):
# Calculate sample mean and standard error
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))

    # Compute the t critical value for 95% CI
    degrees_of_freedom = len(data) - 1
    t_critical = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Compute the margin of error
    return t_critical * sem

#%% Load Data 
no_neutral = True

# Possible Conditions Depending on Main Effect
if mainEffect == 'exp':
    cond_combo = ['Expect Real', 'Neutral', 'Expect Scrambled']
    exp_type = 'expectation_condition'
elif mainEffect == 'att':
    cond_combo = ['Attended', 'Neutral', 'Unattended']
    exp_type = 'attention_condition'    
if no_neutral: 
    cond_combo.remove('Neutral')
#%% Slice Dataframe 

# Drops trials where fixation was broken in between cue onset -> stim offset
# bhv_df = bhv_df.loc[bhv_df.brokeFixation == 0]

# # Drop Subjects that have fewer than 100 trials remaining 
# subTrials = {sub : len(bhv_df.loc[bhv_df.subject == sub]) for sub in bhv_df.subject.unique()}
# subRemove = [sub for sub, numTrials in subTrials.items() if numTrials <100]
# bhv_df = bhv_df[~bhv_df.subject.isin(subRemove)]

# Drop trials in which subject did not respond to one of the questions
bhv_df = bhv_df.dropna(axis = 0, subset = ['recognition'])
bhv_df = bhv_df.dropna(axis = 0, subset = ['category_response'])
bhv_df = bhv_df.dropna(axis = 0, subset = ['confidence'])

# bhv_df = bhv_df.loc[bhv_df.exp_color == 'white']
#%% Add objective category of non probed image 

#DEFINE WHICH STIMULUS WAS NOT PROBED 
bhv_df.loc[bhv_df.probe_loc == 'L', 'notProbed_stimulusCat'] = bhv_df.base_cat_right
bhv_df.loc[bhv_df.probe_loc == 'R', 'notProbed_stimulusCat'] = bhv_df.base_cat_left

# Define Correct based on not probed stimulus 
#Category Correctness 
bhv_df['notProbeCorrect'] = np.nan

bhv_df.loc[bhv_df['category_response'] == bhv_df['notProbed_stimulusCat'], 'notProbeCorrect'] = 1 
bhv_df.loc[(bhv_df['category_response'] != bhv_df['notProbed_stimulusCat']) & 
            (~pd.isnull(bhv_df['category_response'])), 'notProbeCorrect'] = 0

# Potentially remove the trials where the category of probed and non probed are the same 
bhv_df.loc[bhv_df.base_cat_right == bhv_df.base_cat_left, 'sameCat'] = 1 
bhv_df.loc[bhv_df.base_cat_right != bhv_df.base_cat_left, 'sameCat'] = 0 

#%% ########################################################################################### 
# COMPUTE BEHAVIORAL MEASUREMENTS FOR INDIVIDUAL SUBJECTS AND STORE THEM IN A NEW DATAFRAME
# ##############################################################################################
df_bhv_vars = {sub : {cond : [] for cond in cond_combo} for sub in bhv_df.subject.unique()}
for s in bhv_df.subject.unique():
    subj_df = bhv_df[bhv_df.subject == s]
    for cond in cond_combo:
        
        if mainEffect == 'exp':
            df_bhv_vars[s][cond].append(
                ea.get_bhv_vars(subj_df.loc[(subj_df['expectation_condition'] == cond)], paradigm, expt_version ))            
            bhv_vars_df = [(sub, k, *t) for sub, df in df_bhv_vars.items() for k, v in df.items() for t in v]
            bhv_vars_df = pd.DataFrame(bhv_vars_df, columns = ['subject', 'expectation_condition',  'Hit', 'FA', 'd', 'c', 'd_var', 'c_var', 'lnb', 'p_correct', 'catRT', 'recRT', 'confRT'])
            bhv_vars_df.to_pickle(join(DataDir, f"Exp2_expectation_SDT.pkl"))
        elif mainEffect == 'att':
            df_bhv_vars[s][cond].append(
                ea.get_bhv_vars(subj_df.loc[(subj_df['attention_condition'] == cond)], paradigm, expt_version ))            
            bhv_vars_df = [(sub, k, *t) for sub, df in df_bhv_vars.items() for k, v in df.items() for t in v]
            bhv_vars_df = pd.DataFrame(bhv_vars_df, columns = ['subject','attention_condition', 'Hit', 'FA', 'd', 'c','d_var', 'c_var', 'lnb', 'p_correct', 'catRT', 'recRT', 'confRT'])
            bhv_vars_df.to_pickle(join(DataDir, f"Exp2_attention_SDT.pkl"))

for i, bhv in enumerate(["Hit", "FA", "d", "c"]): 
    cond1 = bhv_vars_df[bhv_vars_df[exp_type]==bhv_vars_df[exp_type][0]].reset_index()
    cond2 = bhv_vars_df[bhv_vars_df[exp_type]==bhv_vars_df[exp_type][1]].reset_index()
    print("")
    print("*********************") 
    print(f"{bhv}")
    print("*********************")
    print("MEAN ± 95% CI:")
    print(f"{cond1[exp_type][0]}: \
        {round(cond1[bhv].mean(),2)} ± {round(CI(cond1[bhv]),2)}")
    print(f"{cond2[exp_type][0]}: \
        {round(cond2[bhv].mean(),2)} ± {round(CI(cond2[bhv]),2)}")
    print("")
    print("Pair-wise t-test:")
    print(pg.ttest(cond1[bhv],cond2[bhv], paired = True, alternative='two-sided'))


#%% ################################################################ 
# Category Discrimination and Confidence
# ################################################################## 
group_choice = ['recognition', 'probeReal','subject', exp_type]
df = bhv_df.groupby(group_choice)['correct'].mean()
df = df.reset_index()

#Rename Variables for Titles
df.loc[df.recognition == 0,'recognition'] = 'Unrecognized'
df.loc[df.recognition == 1,'recognition'] = 'Recognized'
df.loc[df.probeReal == 0,'probeReal'] = 'Scrambled'
df.loc[df.probeReal == 1,'probeReal'] = 'Real'

if mainEffect == 'att':
    df.to_pickle(join(f'Exp2_attention_categorization.pkl'))
elif mainEffect == 'exp':
    df.to_pickle(join(f'Exp2_expectation_categorization.pkl'))

for rec_status in sorted(df.recognition.unique()):
    for probe_id in sorted(df.probeReal.unique()):
        data = df.loc[(df.probeReal == probe_id) &
                            (df.recognition == rec_status)]
        cond1 = data[data[exp_type]==np.unique(data[exp_type])[0]].reset_index()
        cond2 = data[data[exp_type]==np.unique(data[exp_type])[1]].reset_index()
        print("")
        print("**************************************")
        print(f'Stats for {rec_status} | {probe_id}')
        print("**************************************")
       
        print("MEAN ± 95% CI:")
        print(f"{cond1[exp_type][0]}: \
        {round(cond1['correct'].mean(),2)} ± {round(CI(cond1['correct']),2)}")
        print(f"{cond2[exp_type][0]}: \
        {round(cond2['correct'].mean(),2)} ± {round(CI(cond2['correct']),2)}")
        print("")
        print("Pair-wise t-test:")      
        print(pg.ttest(cond1['correct'], cond2['correct'],
                alternative = "two-sided", paired=True))
        
# %%