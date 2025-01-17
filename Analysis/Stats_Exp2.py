#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 16:06:39 2022

@author: brandonchen93
Creating descriptive plots for behavior of EASTO experiment
1) 
"""
#%% Import packages
import os
import sys
RootDir = '/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior'
AnalysisDir = os.path.join(RootDir, 'Analysis')
DataDir = os.path.join(RootDir, 'Data')
sys.path.append(AnalysisDir + '/EASTO_funcs/')

import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import stats
from scipy.stats import t
import pingouin as pg
import EASTO_funcs.EASTO_funcs as ea
import itertools 
#import scipy
import pingouin as pg
from matplotlib.ticker import MultipleLocator

#Load processed data 
os.chdir(DataDir)
paradigm = 'Spatial'
expt_version = 'Paradigm_Control'
sub_version = ''
fname = paradigm + expt_version + sub_version+ '_bhv_df.pkl'
bhv_df = pd.read_pickle(fname)

plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["font.weight"]="normal"
plt.rcParams["axes.labelsize"]=7
plt.rcParams["ytick.labelsize"]=7
plt.rcParams["xtick.labelsize"]=7
FigDir = '/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior/Figures/'

def sem(data):
    return np.std(data, ddof = 1) / np.sqrt(len(data))

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
mainEffect = 'exp' #['exp', 'att', 'None']
#exp_cue_or_valid = 'cue'
#subjectPlots = False

#Pick type of expectation, either whether stim matched cue, or probability of probe
# if exp_cue_or_valid == 'cue':
#     exp_type = 'expectation_condition'
# elif exp_cue_or_valid == 'stim':
#     exp_type = 'expectation_validity'

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

# #Remove trials where images share the same category

# bhv_df = bhv_df.loc[bhv_df.sameCat == 0]

# Look at non-swap error trials 

# bhv_df.loc[bhv_df.notProbeCorrect == 1, 'swapError'] = 1
# bhv_df.loc[bhv_df.notProbeCorrect == 0, 'swapError'] = 0

# bhv_df.loc[(bhv_df.notProbeCorrect == 1) & (bhv_df.recognition == 1), 'swapError'] = 1
# # bhv_df.loc[(bhv_df.notProbeCorrect == 0) & (bhv_df.recognition == 1), 'swapError'] = 0
# # bhv_df.loc[(bhv_df.notProbeCorrect == 1) & (bhv_df.recognition == 0), 'swapError'] = 1
# # bhv_df.loc[(bhv_df.notProbeCorrect == 0) & (bhv_df.recognition == 0), 'swapError'] = 0

# bhv_df = bhv_df.loc[bhv_df.swapError == 0]

# # Only look at trials in which the non-probed stimulus was real 
# bhv_df = bhv_df.loc[bhv_df.not_probe_real == 1]
# Only look at trials in which the non-probed stimulus was scrambled
# bhv_df = bhv_df.loc[bhv_df.not_probe_real == 0]


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

        elif mainEffect == 'att':
            df_bhv_vars[s][cond].append(
                ea.get_bhv_vars(subj_df.loc[(subj_df['attention_condition'] == cond)], paradigm, expt_version ))            
            bhv_vars_df = [(sub, k, *t) for sub, df in df_bhv_vars.items() for k, v in df.items() for t in v]
            bhv_vars_df = pd.DataFrame(bhv_vars_df, columns = ['subject','attention_condition', 'Hit', 'FA', 'd', 'c','d_var', 'c_var', 'lnb', 'p_correct', 'catRT', 'recRT', 'confRT'])
# %% SET FIGURE PARAMETERS

if no_neutral: 
    #Removes all Neutral trials
    bhv_vars_df = bhv_vars_df[(bhv_vars_df.iloc[:,:] != 'Neutral').all(axis=1)]
    if mainEffect == 'exp':
        cond_names = 'expectation_condition'
        odr_scheme = ['Expect Scrambled', 'Expect Real']
        x_ticks = ['Unexpected', 'Expected']
        color='purple'
    elif mainEffect == 'att':
        cond_names = "attention_condition"
        odr_scheme = ['Unattended', 'Attended']
        x_ticks = ['Unattended', 'Attended']
        color='green'
#%% PLOT HR, FAR, d', and c COLAPPSED ACROSS CONDITIONS AND COMPUTE GROUP STATISTICS

fig, axes = plt.subplots(1,4,figsize = (6,2))        
for i, bhv in enumerate(["Hit", "FA", "d", "c"]):
    
    print("")
    print("*********************") 
    print(f"{bhv}")
    print("*********************")
    print(bhv_vars_df.groupby(cond_names)[bhv].mean())
    
    print(bhv_vars_df.groupby(cond_names)[bhv].apply(CI)) 
    
    
#     sns.pointplot(x= cond_names, y= bhv, data = bhv_vars_df,
#                 errorbar = ('ci', 95),
#                 order = odr_scheme, color = color, linewidth=1,
#                 ax = axes[i], markersize = 5, zorder = 2)
#     #Plot individual subject data
#     sns.stripplot(x = cond_names, y = bhv, data = bhv_vars_df,
#                              #hue='attention_condition', order=['Expect Scrambled', 'Expect Real'],
#                              color = color, alpha = 0.3,
#                              ax = axes[i], size=3, zorder =1)

#     if bhv == "Hit" or bhv =="FA":
#          axes[i].set_yticks(np.arange(0, 1.2, 0.2))
#          axes[i].axhline(y=0.5, linestyle='--', color = 'black', linewidth = 1)
#     #elif bhv == "d":
#         #axes[i].set_yticks(np.arange(0, 3, 0.5))
#         #axes[i].spines['left'].set_bounds(0, 2.5)
#     #elif bhv =="c":
#         #axes[i].set_yticks(np.arange(-0.6, 0.6, 0.2))
#         #axes[i].spines['left'].set_bounds(-0.6, 0.4)    
#     #     axes[i].yaxis.set_major_locator(MultipleLocator(0.2))
#     axes[i].spines['top'].set_visible(False)
#     axes[i].spines['right'].set_visible(False)
#     axes[i].spines['left'].set_position(('outward',5))
#     axes[i].spines['bottom'].set_position(('outward', 5))
#     axes[i].legend("").remove()
#     axes[i].set_xlabel("")
#     axes[i].set_xticklabels(x_ticks, fontsize=7, rotation = 30)      
# plt.tight_layout()
# plt.savefig(FigDir + 'Exp2_' + mainEffect + '_SDT.svg', 
#                 dpi=300, bbox_inches='tight')
# plt.show()

# Paired t-tests on HR, FAR, d, c between conditions
cond1 = bhv_vars_df[bhv_vars_df[cond_names]==bhv_vars_df[cond_names][0]]
cond2 = bhv_vars_df[bhv_vars_df[cond_names]==bhv_vars_df[cond_names][1]]

print(pg.ttest(cond1['Hit'],cond2['Hit'], paired = True, alternative='two-sided'))
print(pg.ttest(cond1['FA'],cond2['FA'], paired = True,alternative='two-sided'))
print(pg.ttest(cond1['d'],cond2['d'], paired = True,alternative='two-sided'))
print(pg.ttest(cond1['c'],cond2['c'], paired = True,alternative='two-sided'))

#%% ########################################################################## 
# RECOGNITION RATE BY THE IDENTITY OF NON-TARGET STIMULI
# ############################################################################  
#Pick a main effect 
# if mainEffect == 'exp':
#     group_choice = ['not_probe_real', 'probeReal', 'subject', 'expectation_condition']
# elif mainEffect == 'att':
#     group_choice = ['not_probe_real', 'probeReal', 'subject', 'attention_condition']

# proportion_R = bhv_df.groupby(group_choice)['recognition'].mean()
# rec_df = proportion_R.reset_index()

# rec_df.loc[rec_df.not_probe_real == 0,'not_probe_real'] = '(Not-Probed) Scrambled'
# rec_df.loc[rec_df.not_probe_real == 1,'not_probe_real'] = '(Not-Probed) Real'    

# rec_df.loc[rec_df.probeReal == 0,'probeReal'] = '(Probed) Scrambled'
# rec_df.loc[rec_df.probeReal == 1,'probeReal'] = '(Probed) Real'

# if no_neutral:
#     rec_df = rec_df[rec_df[cond_names]!='Neutral']
    
# # Plot the results    
# fig, axes = plt.subplots(2,2,figsize = (6,6))
# counter = 0
# for not_probe_id in sorted(rec_df['not_probe_real'].unique()):
#     for probe_id in sorted(rec_df['probeReal'].unique()):
    
#         data = rec_df[(rec_df['not_probe_real'] == not_probe_id)
#                         & (rec_df['probeReal'] == probe_id)]

#         row, col=divmod(counter,2)
#         sns.pointplot(data = data, x = cond_names, y = data.columns[-1],
#                     order=odr_scheme, linewidth = 2,
#                     errorbar= ('ci', 95), dodge=True,
#                     ax = axes[row, col])
#         sns.swarmplot(data = data, x = cond_names, y = data.columns[-1],
#                 #order=odr_scheme, linewidth = 2,
#                 #errorbar= ('ci', 95), dodge=True,
#                 alpha = 0.3, ax = axes[row, col])
#         # for subject in bhv_vars_df.subject.unique():
#         #     sns.pointplot(x=cond_names, y = 'recognition', 
#         #                 data = data[data['subject'] == subject], 
#         #                 order=odr_scheme,
#         #             color=color,                      
#         #             alpha = .5,
#         #             markersize = .5,
#         #             linewidth= 1,ax = axes[row, col])
        
#         axes[row, col].set_yticks(np.arange(0, 1.2, 0.2))
#         axes[row, col].axhline(y=0.5, linestyle='--', color = 'black', linewidth = 1)
#         axes[row, col].yaxis.set_major_locator(MultipleLocator(0.2))
#         axes[row, col].spines['top'].set_visible(False)
#         axes[row, col].spines['right'].set_visible(False)
#         axes[row, col].spines['left'].set_position(('outward',5))
#         axes[row, col].spines['bottom'].set_position(('outward', 5))
#         axes[row, col].set_xlabel("")
#         axes[row, col].set_xticklabels(x_ticks, fontsize=7)
#         axes[row, col].set_title(not_probe_id + " | " + probe_id, fontsize = 7)        
                
#         counter = counter + 1    
# plt.tight_layout()
# plt.show()           

# Compute group statistics
# for not_probe_id in sorted(rec_df['not_probe_real'].unique()):
#     for probe_id in sorted(rec_df['probeReal'].unique()):
    
#         data = rec_df[(rec_df['not_probe_real'] == not_probe_id)
#                         & (rec_df['probeReal'] == probe_id)]
        
#         print('')
#         print('*******************************************************')
#         print(not_probe_id + " | " + probe_id)
#         print('')
#         cond1 = data[data[cond_names] == np.unique(data[cond_names])[0]]
#         cond2 = data[data[cond_names] == np.unique(data[cond_names])[1]]
#         print(pg.ttest(cond1['recognition'], cond2['recognition'],
#                        paired = True, alternative='two-sided'))

#%% ################################################################ 
# Category Discrimination and Confidence
# ################################################################## 
group_choice = ['recognition', 'probeReal','subject', cond_names]

#bhv_vars = ['correct', 'confidence']
for bhv_var in ['correct']:

    df = bhv_df.groupby(group_choice)[bhv_var].mean()
    df = df.reset_index()
    if no_neutral:
        df = df[df[cond_names]!='Neutral']
    
    #Rename Variables for Titles
    df.loc[df.recognition == 0,'recognition'] = 'Unrecognized'
    df.loc[df.recognition == 1,'recognition'] = 'Recognized'
    df.loc[df.probeReal == 0,'probeReal'] = 'Scrambled'
    df.loc[df.probeReal == 1,'probeReal'] = 'Real'

    for rec_status in sorted(df.recognition.unique()):
        for probe_id in sorted(df.probeReal.unique()):
            print("")
            print("**************************************")
            print(f'Stats for {rec_status} | {probe_id}')
            print("**************************************")
            data = df.loc[(df.probeReal == probe_id) &
                                (df.recognition == rec_status)]
            print('MEAN')
            print(data.groupby([cond_names])[bhv_var].mean())
            # print('******************************************')
            # print('SEM')
            # print(data.groupby([cond_names])[bhv_var].sem())
            print('******************************************')
            print('CI')
            print(data.groupby([cond_names])[bhv_var].apply(CI))
            cond1 = data[data[cond_names]==np.unique(data[cond_names])[0]]
            cond2 = data[data[cond_names]==np.unique(data[cond_names])[1]]
            
            print(pg.ttest(cond1[bhv_var], cond2[bhv_var],
                    alternative = "two-sided", paired=True))
        
# %%
counter = 0
fig, axes = plt.subplots(1,4,figsize = (6,2), sharey=True)

for rec_status in sorted(df.recognition.unique()):
    for probe_id in sorted(df.probeReal.unique()):
        titles = {0: "Hit trials", 1: "FA Trials",
                      2: "Miss trials", 3: "CR Trials"}
        data = df.loc[(df.probeReal == probe_id) &
                            (df.recognition == rec_status)]
        
#         sns.pointplot(data = data, x = cond_names, y = data.columns[-1],
#                     order=odr_scheme, linewidth = 1, markersize=5,
#                     errorbar= ('ci', 95), dodge=True, color=color, zorder=2,
#                     ax = axes[counter])
#         sns.swarmplot(data = data, x = cond_names, y = data.columns[-1],
#                     alpha = 0.3, size=3, ax = axes[counter], color=color, zorder=1)
        
#         if bhv_var == 'correct':
#             axes[counter].set_yticks(np.arange(0, 1.25, 0.25))
#             axes[counter].axhline(y=0.25, linestyle='--', color = 'black', linewidth = 1)
#             axes[counter].yaxis.set_major_locator(MultipleLocator(0.2))
#         elif bhv_var == 'confidence':
#             axes[counter].set_yticks(np.arange(1, 4.5, 0.5))
#             axes[counter].yaxis.set_major_locator(MultipleLocator(0.5))
#         axes[counter].spines['top'].set_visible(False)
#         axes[counter].spines['right'].set_visible(False)
#         axes[counter].spines['left'].set_position(('outward',5))
#         axes[counter].spines['bottom'].set_position(('outward', 5))
#         axes[counter].set_xlabel("")
#         axes[counter].set_ylabel("Categorization Accuracy")
#         if counter in titles:
#                 axes[counter].set_title(titles[counter], fontsize=7, fontweight="bold")
#         axes[counter].set_xticklabels(x_ticks, fontsize=7, rotation = 30)                 
#         counter += 1    
# plt.tight_layout()
# plt.savefig(FigDir + 'Exp2_' + mainEffect + '_categorization.svg', 
#                 dpi=300, bbox_inches='tight')
# plt.show()                   
    
#cor_comb.savefig(f'{paradigm}_correct.svg'.format(paradigm))
#%% Category Discrimination (Based on Not-Probed Image Category) NOT REPLICATED need 
#Pick a main effect 
# if mainEffect == 'exp':
#     group_choice = ['recognition', 'probeReal','subject', exp_type]
# elif mainEffect == 'att':
#     group_choice =  ['recognition', 'probeReal','subject', 'attention_condition']
# else: 
#     group_choice =  ['recognition', 'probeReal', 'not_probe_real', 'subject', exp_type, 'attention_condition']
    


# correct_pd_group = bhv_df.groupby(group_choice)['notProbeCorrect'].mean()

# cor_df = correct_pd_group.reset_index()

    
# #Rename Variables for Titles
# cor_df.loc[cor_df.recognition == 0,'recognition'] = 'Unrecognized'
# cor_df.loc[cor_df.recognition == 1,'recognition'] = 'Recognized'

# cor_df.loc[cor_df.not_probe_real == 0,'not_probe_real'] = 'NP Scrambled'
# cor_df.loc[cor_df.not_probe_real == 1,'not_probe_real'] = 'NP Real'

# #cor_df.loc[cor_df.probeReal == 0,'probeReal'] = 'Scrambled'
# #cor_df.loc[cor_df.probeReal == 1,'probeReal'] = 'Real'

# if mainEffect == 'exp':
#     cor_comb = sns.catplot(x=exp_type, y = 'notProbeCorrect',  row = 'recognition', col = 'probeReal',
#                 col_order = ['Real', 'Scrambled'], row_order = ['Recognized', 'Unrecognized'],
#                 data = cor_df, sharey = False, markers=mkr_scheme, order = odr_scheme, kind = 'point', ci= 'sd')
    
#     cor_comb.map(plt.axhline, y=0.25, color=".7", dashes=(2, 1), zorder=0)
#     cor_comb.set(xlabel = None, ylabel = 'Proportion Correct', ylim = [0,1])
#     cor_comb.set_titles(col_template = '{col_name}', row_template ='{row_name}', fontweight = 'bold')
#     # cor_comb.fig.suptitle('Prop. Correct by Exp.  Conditions')
#     #Plotting individual subject data 
#     for subject in bhv_vars_df.subject.unique():
#         for row_ind, rec_unrec in enumerate(np.flip(cor_df.recognition.unique())):
#             for col_ind, real_scram in enumerate(np.flip(cor_df.probeReal.unique())): 
#                 sns.pointplot( x=exp_type, y = 'notProbeCorrect',  
#                              data = cor_df.loc[(cor_df.subject == subject) & (cor_df.recognition == rec_unrec) & (cor_df.probeReal == real_scram)], 
#                              sharey = False, 
#                              order = odr_scheme,  
#                              ci = None, 
#                              scale = 0.5,
#                              ax = cor_comb.axes[row_ind][col_ind])
#                 #Setting alpha of specific plot elements 
#                 plt.setp(cor_comb.axes[row_ind][col_ind].collections[1:], alpha=.5) #for the markers
#                 plt.setp(cor_comb.axes[row_ind][col_ind].lines[3:], alpha=.5)       #for the lines
# elif mainEffect == 'att':
#     if no_neutral == True:
#         cor_df = cor_df[cor_df['attention_condition']!='Neutral']
#     cor_comb = sns.catplot(x='attention_condition', y = 'notProbeCorrect', row = 'recognition', col = 'probeReal',
#                col_order = ['Real', 'Scrambled'], row_order = ['Recognized', 'Unrecognized'],
#                 data = cor_df, sharey = False, markers=mkr_scheme,  kind = 'point', ci= 'sd')
    
#     cor_comb.map(plt.axhline, y=0.25, color=".7", dashes=(2, 1), zorder=0)
#     cor_comb.set(xlabel = None, ylabel = 'Proportion Correct', ylim = [0,1])
#     cor_comb.set_titles(col_template = '{col_name}', row_template ='{row_name}', fontweight = 'bold')
#     # cor_comb.fig.suptitle('Prop. Correct by Att. Conditions')
#     #Plotting individual subject data 
#     for subject in bhv_vars_df.subject.unique():
#         for row_ind, rec_unrec in enumerate(np.flip(cor_df.recognition.unique())):
#             for col_ind, real_scram in enumerate(np.flip(cor_df.probeReal.unique())): 
#                 sns.pointplot( x='attention_condition', y = 'notProbeCorrect',  
#                              data = cor_df.loc[(cor_df.subject == subject) & (cor_df.recognition == rec_unrec) & (cor_df.probeReal == real_scram)], 
                               
#                              ci = None, 
#                              scale = 0.5,
#                              ax = cor_comb.axes[row_ind][col_ind])
#                 #Setting alpha of specific plot elements 
#                 plt.setp(cor_comb.axes[row_ind][col_ind].collections[1:], alpha=.5) #for the markers
#                 plt.setp(cor_comb.axes[row_ind][col_ind].lines[3:], alpha=.5)       #for the lines

# else: 
#     cor_comb = sns.catplot(x=exp_type, y = 'notProbeCorrect', hue = 'attention_condition', 
#                 row = 'recognition', col = 'not_probe_real',
#                 col_order = ['NP Real', 'NP Scrambled'], row_order = ['Recognized', 'Unrecognized'],
#                 data = cor_df, sharey = False, 
#                 markers=mkr_scheme, order = odr_scheme,  
#                 palette = clr_scheme,
#                 kind = 'point', errorbar= ('ci', 95))
#     cor_comb.map_dataframe(sns.stripplot, x=exp_type, y = 'notProbeCorrect', 
#                            hue = 'attention_condition', alpha = 0.5, 
#                            dodge = False, legend = False,
#                            palette = clr_scheme, order = odr_scheme)
#     #Plotting individual subject data 
#     if subjectPlots: 
#         for subject in cor_df.subject.unique():
#             for row_ind, rec_unrec in enumerate(np.flip(cor_df.recognition.unique())):
#                 for col_ind, real_scram in enumerate(np.flip(cor_df.probeReal.unique())): 
#                     sns.pointplot( x=exp_type, y = 'notProbeCorrect', hue = 'attention_condition', 
#                                 data = cor_df.loc[(cor_df.subject == subject) & (cor_df.recognition == rec_unrec) & (cor_df.probeReal == real_scram)], 
#                                 sharey = False, 
#                                 markers=mkr_scheme, 
#                                 order = odr_scheme,  
#                                 palette= clr_scheme,
#                                 ci = None, 
#                                 scale = 0.5,
#                                 ax = cor_comb.axes[row_ind][col_ind])
#                     #Setting alpha of specific plot elements 
#                     plt.setp(cor_comb.axes[row_ind][col_ind].collections[2:], alpha=.5) #for the markers
#                     plt.setp(cor_comb.axes[row_ind][col_ind].lines[6:], alpha=.5)       #for the lines
#     cor_comb._legend.set_title(title = None)
# cor_comb.map(plt.axhline, y=0.25, color=".7", dashes=(2, 1), zorder=0)
# cor_comb.set(xlabel = None, ylabel = 'Proportion Correct (NP)', ylim = [0,1])
# cor_comb.set_titles(col_template = '{col_name}', row_template ='{row_name}', fontweight = 'bold')

# #cor_comb.savefig(f'{paradigm}_NPcorrect.svg'.format(paradigm))
#%% Categorization accuracy split by Probed & non Probed (REPLICATED 20231207 slide 50)
#Pick a main effect 
if mainEffect == 'exp':
    group_choice = ['not_probe_real','recognition', 'probeReal','subject', exp_type]
elif mainEffect == 'att':
    group_choice =  ['recognition', 'probeReal','subject', 'attention_condition']
else: 
    group_choice =  ['not_probe_real', 'probeReal', 'recognition', 'subject', exp_type, 'attention_condition']
 
 if mainEffect == 'exp':
    group_choice = ['not_probe_real', 'probeReal', 'subject', 'expectation_condition']
elif mainEffect == 'att':
    group_choice = ['not_probe_real', 'probeReal', 'subject', 'attention_condition']

 
 
    
correct_pd_group = bhv_df.groupby(group_choice)['notProbeCorrect'].mean()
#correct_pd_group = bhv_df.groupby(group_choice)['correct'].mean()

cor_df = correct_pd_group.reset_index()
    
#Rename Variables for Titles
cor_df.loc[cor_df.recognition == 0,'recognition'] = 'Unrecognized'
cor_df.loc[cor_df.recognition == 1,'recognition'] = 'Recognized'

cor_df.loc[cor_df.not_probe_real == 0,'not_probe_real'] = '(Not-Probed) Scrambled'
cor_df.loc[cor_df.not_probe_real == 1,'not_probe_real'] = '(Not-Probed) Real'

cor_df.loc[cor_df.probeReal == 0,'probeReal'] = '(Probed) Scrambled'
cor_df.loc[cor_df.probeReal == 1,'probeReal'] = '(Probed) Real'

for recStatus in cor_df.recognition.unique():
    cor_comb = sns.catplot(x=exp_type, y = 'notProbeCorrect', hue = 'attention_condition', 
                           col = 'not_probe_real', row = 'probeReal',
                row_order = ['(Probed) Real', '(Probed) Scrambled'], 
                col_order = ['(Not-Probed) Real', '(Not-Probed) Scrambled'],
                data = cor_df.loc[cor_df.recognition == recStatus],
                sharey = False,
                markers=mkr_scheme,
                order = odr_scheme, 
                palette = clr_scheme,
                kind = 'point', 
                errorbar = ('ci', 95))
    cor_comb.map_dataframe(sns.stripplot, x=exp_type, y = 'notProbeCorrect', 
                           hue = 'attention_condition', alpha = 0.5, 
                           dodge = False, legend = False,
                           palette = clr_scheme, order = odr_scheme)

    #Plotting individual subject data 
    if subjectPlots:
        for subject in cor_df.subject.unique():   
            for row_ind, real_scram in enumerate(np.flip(cor_df.probeReal.unique())):
                for col_ind, notreal_scram in enumerate(np.flip(cor_df.not_probe_real.unique())): 
                    sns.pointplot( x=exp_type, y = 'notProbeCorrect', hue = 'attention_condition', 
                                    data = cor_df.loc[(cor_df.subject == subject) &
                                                    (cor_df.probeReal == real_scram) & 
                                                    (cor_df.not_probe_real == notreal_scram) & 
                                                    (cor_df.recognition == recStatus)], 
                                    sharey = False, 
                                    markers=mkr_scheme, 
                                    order = odr_scheme,  
                                    palette= clr_scheme,
                                    ci = None, 
                                    scale = 0.5,
                                    ax = cor_comb.axes[row_ind][col_ind])
                    #Setting alpha of specific plot elements 
                    plt.setp(cor_comb.axes[row_ind][col_ind].collections[2:], alpha=.5) #for the markers
                    plt.setp(cor_comb.axes[row_ind][col_ind].lines[6:], alpha=.5)       #for the lines
            cor_comb._legend.set_title(title = None)
    cor_comb.map(plt.axhline, y=0.25, color=".7", dashes=(2, 1), zorder=0)
    cor_comb.set(xlabel = None, ylabel = 'Proportion Correct', ylim = [0,1])
    cor_comb.set_titles(col_template = '{col_name}', row_template ='{row_name}', fontweight = 'bold')
    cor_comb.fig.suptitle(f'Prop. Correct for {recStatus} Trials', y= 1.02)


    #cor_comb.savefig(f'{paradigm}_correct_{recStatus}.svg'.format(paradigm))

#%% ####################################### 
# Confidence Plots 
##########################################3
#Pick a main effect 
if mainEffect == 'exp':
    group_choice = ['recognition', 'probeReal','subject', exp_type]
elif mainEffect == 'att':
    group_choice =  ['recognition', 'probeReal','subject', exp_type]

confidence_pd_group = bhv_df.groupby(group_choice)['confidence'].mean()
conf_df = confidence_pd_group.reset_index()
if no_neutral == True:
    conf_df = conf_df[conf_df[exp_type]!='Neutral']

#conf_df = conf_df.loc[conf_df.notProbeCorrect == 1]
#conf_df = conf_df.loc[conf_df.correct == 0]
    
#Rename Variables for Titles
conf_df.loc[conf_df.recognition == 0,'recognition'] = 'Unrecognized'
conf_df.loc[conf_df.recognition == 1,'recognition'] = 'Recognized'

conf_df.loc[conf_df.probeReal == 0,'probeReal'] = 'Scrambled'
conf_df.loc[conf_df.probeReal == 1,'probeReal'] = 'Real'

trial_types = ['Hit', 'FA', 'Miss', 'CR']
counter = 0
for rec_status in sorted(conf_df.recognition.unique()):
    for probe_id in sorted(conf_df.probeReal.unique()):

            
        print("**************************************")
        print(f'Stats for {trial_types[counter]} Trials')
        print("**************************************")
        subset = conf_df[(conf_df['recognition'] == rec_status) & (conf_df['probeReal']== probe_id)]
        
        print('MEAN')
        print(subset.groupby(['expectation_condition'])['confidence'].mean())
        print('******************************************')
        print('CI')
        print(subset.groupby(['expectation_condition'])['confidence'].apply(CI))
            
        print("**************************************")
        print(f'Paired t-test between expect real and expect scrambled trials')
        print("**************************************")
        print(pg.ttest(subset[subset['expectation_condition'] == 'Expect Real']['confidence'],
                       subset[subset['expectation_condition'] == 'Expect Scrambled']['confidence'],
                    alternative = "two-sided", paired=True))
        counter += 1
del counter        
        
  



if mainEffect == 'exp':
    conf_comb = sns.catplot(x=exp_type, y = 'confidence', row = 'recognition', col = 'probeReal',
                col_order = ['Real', 'Scrambled'], row_order = ['Recognized', 'Unrecognized'],
                data = conf_df, sharey = False, order = odr_scheme, kind = 'point', ci= 'sd')
    conf_comb.set(xlabel = None, ylabel = 'Mean Confidence', ylim = [1,4])
    conf_comb.set_titles(col_template = '{col_name}', row_template ='{row_name}', fontweight = 'bold')
    #Plotting individual subject data 
    for subject in bhv_vars_df.subject.unique():
        for row_ind, rec_unrec in enumerate(np.flip(conf_df.recognition.unique())):
            for col_ind, real_scram in enumerate(np.flip(conf_df.probeReal.unique())): 
                sns.pointplot( x=exp_type, y = 'confidence', 
                             data = conf_df.loc[(conf_df.subject == subject) & (conf_df.recognition == rec_unrec) & (conf_df.probeReal == real_scram)], 
                             #sharey = False, 
                             order = odr_scheme,  
                             ci = None, 
                             scale = 0.5,
                             ax = conf_comb.axes[row_ind][col_ind])
                # conf_comb.axes[row_ind][col_ind].get_legend().remove()
                #Setting alpha of specific plot elements 
                plt.setp(conf_comb.axes[row_ind][col_ind].collections[1:], alpha=.5) #for the markers
                plt.setp(conf_comb.axes[row_ind][col_ind].lines[3:], alpha=.5)       #for the lines

elif mainEffect == 'att':
    if no_neutral == True:
        conf_df = conf_df[conf_df['attention_condition']!='Neutral']
    conf_comb = sns.catplot(x='attention_condition', y = 'confidence', row = 'recognition', col = 'probeReal',
                col_order = ['Real', 'Scrambled'], row_order = ['Recognized', 'Unrecognized'],
                order = odr_scheme,  data = conf_df, sharey = False,  kind = 'point', ci= 'sd')
    conf_comb.set(xlabel = None, ylabel = 'Mean Confidence', ylim = [1,4])
    conf_comb.set_titles(col_template = '{col_name}', row_template ='{row_name}', fontweight = 'bold')
    #Plotting individual subject data 
    #set color for each subject
    for subject in bhv_vars_df.subject.unique():
        for row_ind, rec_unrec in enumerate(np.flip(conf_df.recognition.unique())):
            for col_ind, real_scram in enumerate(np.flip(conf_df.probeReal.unique())): 
                sns.pointplot( x='attention_condition', y = 'confidence', 
                             data = conf_df.loc[(conf_df.subject == subject) & (conf_df.recognition == rec_unrec) & (conf_df.probeReal == real_scram)], 
                             order = odr_scheme,   
                             ci = None, 
                             scale = 0.5,
                             ax = conf_comb.axes[row_ind][col_ind])
                # conf_comb.axes[row_ind][col_ind].get_legend().remove()
                #Setting alpha of specific plot elements 
                plt.setp(conf_comb.axes[row_ind][col_ind].collections[1:], alpha=.5) #for the markers
                plt.setp(conf_comb.axes[row_ind][col_ind].lines[3:], alpha=.5) 
else: 
    conf_comb = sns.catplot(x=exp_type, y = 'confidence', hue = 'attention_condition', row = 'recognition', col = 'probeReal',
                col_order = ['Real', 'Scrambled'], row_order = ['Recognized', 'Unrecognized'],
                data = conf_df, sharey = False, markers=mkr_scheme, 
                order = odr_scheme,  palette = clr_scheme, 
                kind = 'point', errorbar = ('ci', 95))
    conf_comb.map_dataframe(sns.stripplot, x=exp_type, y = 'confidence', 
                           hue = 'attention_condition', alpha = 0.5, 
                           dodge = False, legend = False,
                           palette = clr_scheme, order = odr_scheme)
    #Plotting individual subject data 
    if subjectPlots:
        for subject in conf_df.subject.unique():
            for row_ind, rec_unrec in enumerate(np.flip(conf_df.recognition.unique())):
                for col_ind, real_scram in enumerate(np.flip(conf_df.probeReal.unique())): 
                    sns.pointplot( x=exp_type, y = 'confidence', hue = 'attention_condition', 
                                data = conf_df.loc[(conf_df.subject == subject) & (conf_df.recognition == rec_unrec) & (conf_df.probeReal == real_scram)], 
                                markers=mkr_scheme, 
                                order = odr_scheme,  
                                palette= clr_scheme,
                                errorbar = None, 
                                scale = 0.5,
                                ax = conf_comb.axes[row_ind][col_ind])
                    conf_comb.axes[row_ind][col_ind].get_legend().remove()
                    #Setting alpha of specific plot elements 
                    plt.setp(conf_comb.axes[row_ind][col_ind].collections[2:], alpha=.5) #for the markers
                    plt.setp(conf_comb.axes[row_ind][col_ind].lines[6:], alpha=.5)       #for the lines
                    
#conf_comb._legend.set_title(title = None)
conf_comb.set(xlabel = None, ylabel = 'Mean Confidence', ylim = [1,4])
conf_comb.set_titles(col_template = '{col_name}', row_template ='{row_name}', fontweight = 'bold')
    # conf_comb.fig.suptitle('Prop. confidence by Exp. & Att. Conditions')

#conf_comb.savefig(f'{paradigm}_confidence.svg'.format(paradigm))
#%% Confidence split by Probed & non Probed 
#Pick a main effect 
if mainEffect == 'exp':
    group_choice = ['recognition', 'probeReal','subject', exp_type]
elif mainEffect == 'att':
    group_choice =  ['recognition', 'probeReal','subject', 'attention_condition']
else: 
    group_choice =  ['correct','not_probe_real', 'probeReal', 'recognition', 'subject', exp_type, 'attention_condition']
    
confidence_pd_group = bhv_df.groupby(group_choice)['confidence'].mean()

conf_df = confidence_pd_group.reset_index()
    
conf_df = conf_df.loc[conf_df.correct == 0]

#Rename Variables for Titles
conf_df.loc[conf_df.recognition == 0,'recognition'] = 'Unrecognized'
conf_df.loc[conf_df.recognition == 1,'recognition'] = 'Recognized'

conf_df.loc[conf_df.not_probe_real == 0,'not_probe_real'] = '(Not-Probed) Scrambled'
conf_df.loc[conf_df.not_probe_real == 1,'not_probe_real'] = '(Not-Probed) Real'

conf_df.loc[conf_df.probeReal == 0,'probeReal'] = '(Probed) Scrambled'
conf_df.loc[conf_df.probeReal == 1,'probeReal'] = '(Probed) Real'

for recStatus in conf_df.recognition.unique():
    conf_comb = sns.catplot(x=exp_type, y = 'confidence', hue = 'attention_condition', 
                           col = 'not_probe_real', row = 'probeReal',
                row_order = ['(Probed) Real', '(Probed) Scrambled'], col_order = ['(Not-Probed) Real', '(Not-Probed) Scrambled'],
                data = conf_df.loc[conf_df.recognition == recStatus],
                sharey = False, 
                markers=mkr_scheme,
                order = odr_scheme, 
                palette = clr_scheme,
                kind = 'point', 
                errorbar = ('ci', 95))
    conf_comb.map_dataframe(sns.stripplot, x=exp_type, y = 'confidence', 
                           hue = 'attention_condition', alpha = 0.5, 
                           dodge = False, legend = False,
                           palette = clr_scheme, order = odr_scheme)
    # #Plotting individual subject data 
    # for subject in conf_df.subject.unique():   
    #     for row_ind, real_scram in enumerate(np.flip(conf_df.probeReal.unique())):
    #         for col_ind, notreal_scram in enumerate(np.flip(conf_df.not_probe_real.unique())): 
    #             sns.pointplot(x=exp_type, y = 'confidence', hue = 'attention_condition', 
    #                             data = conf_df.loc[(conf_df.subject == subject) &
    #                                             (conf_df.probeReal == real_scram) & 
    #                                             (conf_df.not_probe_real == notreal_scram) & 
    #                                             (conf_df.recognition == recStatus)], 
    #                             markers=mkr_scheme, 
    #                             order = odr_scheme,  
    #                             palette= clr_scheme,
    #                             errorbar = None, 
    #                             scale = 0.5,
    #                             ax = conf_comb.axes[row_ind][col_ind])
    #             conf_comb.axes[row_ind][col_ind].get_legend().remove()
    #             #Setting alpha of specific plot elements 
    #             plt.setp(conf_comb.axes[row_ind][col_ind].collections[2:], alpha=.5) #for the markers
    #             plt.setp(conf_comb.axes[row_ind][col_ind].lines[6:], alpha=.5)       #for the lines
                
    conf_comb._legend.set_title(title = None)
    conf_comb.set(xlabel = None, ylabel = 'Confidence', ylim = [1,4])
    conf_comb.set_titles(col_template = '{col_name}', row_template ='{row_name}', fontweight = 'bold')
    conf_comb.fig.suptitle(f'Mean Confidence for {recStatus} trials', y = 1.02)

    conf_comb.savefig(f'{paradigm}_confidence_{recStatus}.svg'.format(paradigm))
    
    

#%% Plots that need fixing #%% 
#%% Confusion Matrices
img_choice = 'scrambled'
rec_choice = 'recognized'
fig, ax = plt.subplots(2,2, sharey=False, sharex=False, figsize = (10,8))
cbar_ax = fig.add_axes([.91, .3, .03, .4])


#Change Dataframe to select trials specific to a given condition 

#Not probed is Scrambled
# conf_df = bhv_df.loc[bhv_df.not_probe_real == 0]
# Not probed is real 
conf_df = bhv_df.loc[bhv_df.not_probe_real == 1]
# Use all trials 
conf_df = bhv_df

for i, (axi, cond) in enumerate(zip(ax.flat,cond_combo)):
    exp = cond[0] #Expectation Condition
    att = cond[1] # Attention Condition 

    conf_mat = ea.make_confusion(conf_df, cond[0], cond[1], rec_choice,  img_type = img_choice)
    sns.heatmap(conf_mat, annot = True,
                fmt = '.1%',
                cbar = i == 0, 
                vmin = 0, vmax = 1,
                cbar_ax = None if i else cbar_ax, 
                ax = axi)
    axi.set_xlabel('')
    axi.set_ylabel('')
    axi.set_title(f'{exp} & {att}'.format(cond[0], cond[1]))
    cbar = ax.flat[0].collections[0].colorbar
    cbar.set_ticks([0, .5, 1])
    cbar.set_ticklabels(['0', '50%', '100%'])


oax = fig.add_subplot(111, frameon = False)

plt.tick_params(labelcolor = 'none', top =False, bottom = False, left = False, right = False)
oax.set_ylabel('Actual')
oax.set_xlabel('Selected')
fig.suptitle(f'{img_choice.capitalize()} Images')

# plt.tight_layout(rect=[0, 0, .9, 1]) 

# fig.savefig(f'{paradigm}_{img_choice}_conf_mat.svg', bbox_inches = 'tight')
#%% Confidence on X axis 

#Pick to show catgorization accuracy or recognition 
plot_choice = 'recognition'

#Pick a main effect 
if mainEffect == 'exp':
    group_choice = ['confidence', 'probeReal','subject', exp_type]
elif mainEffect == 'att':
    group_choice =  ['confidence', 'probeReal','subject', 'attention_condition']
else: 
    group_choice =  ['confidence', 'probeReal','subject', exp_type, 'attention_condition']
    
#Pooled or not 
if pooled: 
    group_choice.remove('subject')

confidence_pd_group = bhv_df.groupby(group_choice)[plot_choice].mean()

conf_df = confidence_pd_group.reset_index()

if no_neutral:
    conf_df = conf_df[(conf_df.iloc[:,:] != 'Neutral').all(axis=1)]
    clr_scheme = ['tab:purple','tab:orange', 'tab:pink', 'tab:brown']
    mkr_scheme  = ["o", 'd']
else: 
    clr_scheme = ['tab:orange', 'tab:gray', 'tab:purple']
    mkr_scheme  = ["o", "x", 'd']
    
#Rename Variables for Titles

conf_df.loc[conf_df.probeReal == 0,'probeReal'] = 'Scrambled'
conf_df.loc[conf_df.probeReal == 1,'probeReal'] = 'Real'
if mainEffect == 'exp':
    conf_comb = sns.catplot(x='confidence', y = plot_choice, hue = exp_type, col = 'probeReal',
                col_order = ['Real', 'Scrambled'], 
                palette = ['tab:brown', 'tab:pink'],
                data = conf_df, sharey = False, kind = 'point', ci= 'sd')
    conf_comb.set(xlabel = None, ylabel = f'Mean {plot_choice}', ylim = [0,1])
    conf_comb.set_titles(col_template ='{col_name}', fontweight = 'bold')

elif mainEffect == 'att':    
    conf_comb = sns.catplot(x='confidence', y = plot_choice, hue = 'attention_condition', col = 'probeReal',
                col_order = ['Real', 'Scrambled'], 
                palette = ['tab:orange', 'tab:purple'],
                data = conf_df, sharey = False, kind = 'point', ci= 'sd')
    conf_comb.set(xlabel = None, ylabel = f'Mean {plot_choice}', ylim = [0,1])
    conf_comb.set_titles(col_template ='{col_name}', fontweight = 'bold')

else: 
    conf_df['trial_cond'] = conf_df[[exp_type, 'attention_condition']].apply(tuple, axis=1)
    
    conf_comb = sns.catplot(x='confidence', y = plot_choice, hue = 'trial_cond', col = 'probeReal',
                col_order = ['Real', 'Scrambled'], 
                palette = clr_scheme, 
                data = conf_df, sharey = False, kind = 'point', ci= 'sd')
    conf_comb.set(xlabel = None, ylabel = f'Mean {plot_choice}', ylim = [0,1])
    conf_comb.set_titles(col_template ='{col_name}', fontweight = 'bold')
    conf_comb._legend.set_title(title = None)
    # conf_comb.fig.suptitle('Prop. confidence by Exp. & Att. Conditions')


conf_comb.savefig(f'{paradigm}_conf_by_{plot_choice}.svg'.format(paradigm))

#%% Event Plot for Trial Distributions 

trial_dist = []
for ttype in bhv_df.trial_type.unique():
    trial_dist.append(bhv_df[bhv_df.trial_type == ttype].index)

color_code = ['k','r','saddlebrown','darkorange',
              'y','olive','lawngreen','turquoise',
              'darkcyan','slategrey','b','darkmagenta']

plt.figure()
ax = plt.eventplot(trial_dist, 
              lineoffsets = np.linspace(0,4,len(trial_dist)), 
              linelengths = 0.25, 
              # colors =color_code, 
              label = bhv_df.trial_type.unique())
plt.legend(bhv_df.trial_type.unique(),bbox_to_anchor=(1.01,1), loc="upper left")
plt.title('Trial Type Sequence')
plt.yticks(np.linspace(0,4,len(trial_dist)),bhv_df.trial_type.unique())
plt.xlabel('Trials')

