#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   FreqStats_YH.py
@Time    :   2024/09/24 01:34:38
@Author  :   Yuan-hao Wu 
@Version :   1.0
@Contact :   yuanhao.wu@nyulangone.org, bc1693@nyu.edu
@License :   None
@Desc    :   None
'''
# %%
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
import scipy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.options.display.float_format = '{:.3f}'.format


#Load processed data 
os.chdir(DataDir)
paradigm = 'Spatial'
expt_version = 'Paradigm_V3'
sub_version = ''
fname = paradigm + expt_version + sub_version+ '_bhv_df.pkl'
bhv_df = pd.read_pickle(fname)

# Remove problematic subjects 
badSubjects = ['P40', 'P48']
badSubMask = bhv_df['subject'].isin(badSubjects)
bhv_df = bhv_df.loc[~badSubMask]   
del badSubMask, badSubjects


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

# %% Add objective category of non probed image 

#DEFINE WHICH STIMULUS CATEGORY WAS NOT PROBED 
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

exp_type = 'expectation_condition'
cond_combo = pd.MultiIndex.from_product([sorted(bhv_df.expectation_condition.unique()),
                                    sorted(bhv_df.attention_condition.unique())]).tolist()
#%% Slice DataFrame 

# # Drops trials where fixation was broken in between cue onset -> stim offset
# bhv_df = bhv_df.loc[bhv_df.brokeFixation == 0]

# # Drop Subjects that have fewer than 100 trials remaining 
# subTrials = {sub : len(bhv_df.loc[bhv_df.subject == sub]) for sub in bhv_df.subject.unique()}
# subRemove = [sub for sub, numTrials in subTrials.items() if numTrials <100]
# bhv_df = bhv_df[~bhv_df.subject.isin(subRemove)]

# # Drop trials in which subject did not respond to the recognition question
# bhv_df = bhv_df.dropna(axis = 0, subset = ['recognition'])
# bhv_df = bhv_df.dropna(axis = 0, subset = ['category_response'])

#Exclude subjects 
# bhv_df = bhv_df.loc[bhv_df.subject != 'P46']


# ## Create dataframe including each subject's SDT behavioral mesaures, categorization accuracy,
# RT for expectation and attention conditions
df_bhv_vars = {sub : {cond : [] for cond in cond_combo} for sub in bhv_df.subject.unique()}
for s in bhv_df.subject.unique():
    subj_df = bhv_df[bhv_df.subject == s]
    for cond in cond_combo:
        exp = cond[0] #Expectation Condition
        att = cond[1] # Attention Condition 

        df_bhv_vars[s][cond].append(
                    ea.get_bhv_vars(subj_df.loc[(subj_df[exp_type] == exp) & (subj_df.attention_condition == att)],
                    paradigm, expt_version))
               
        bhv_vars_df = [(sub, k[0],k[1], *t) for sub, df in df_bhv_vars.items() for k, v in df.items() for t in v]
        bhv_vars_df = pd.DataFrame(bhv_vars_df, columns = ['subject','expectation_condition', 'attention_condition', 'Hit', 'FA', 'd', 'c', 'd_var', 'c_var', 'lnb', 'p_correct', 'catRT', 'recRT', 'confRT'])

#bhv_vars_df.to_pickle(os.path.join(DataDir, "SDT_2x2conds.pkl"))

# %%
# COMPUTE STATS FOR SDT BEHAVIORAL MEASURES
print("")
print("***********************************************************************")
print("***********************************************************************")
print(f'                  STATS FOR SDT METRICS')
print("") 
for i in ['Hit', 'FA', 'd', 'c']:
    print("")
    print("*********************") 
    print(f"{i}")
    print("*********************")
    
    print("MEAN FOR EACH CONDITION")
    print('--------------------------')
    print(bhv_vars_df.groupby(['expectation_condition',
                            'attention_condition'])[i].mean())
    print('')  
    print("95% CI FOR EACH CONDITION")
    print('--------------------------')
    print(bhv_vars_df.groupby(['expectation_condition',
                            'attention_condition'])[i].apply(CI))  
    
      
    attended = bhv_vars_df[bhv_vars_df['attention_condition']=="Attended"]
    attended = attended.groupby('subject')[i].mean()
    unattended = bhv_vars_df[bhv_vars_df['attention_condition']=="Unattended"]
    unattended = unattended.groupby('subject')[i].mean() 
    exp_real = bhv_vars_df[bhv_vars_df['expectation_condition']=="Expect Real"]
    exp_real = exp_real.groupby('subject')[i].mean()
    exp_scr = bhv_vars_df[bhv_vars_df['expectation_condition']=="Expect Scrambled"]
    exp_scr = exp_scr.groupby('subject')[i].mean()
 
    print('')
    print('--------------------------')  
    print("MARGINAL MEAN")
    print('--------------------------')
    print('') 
    print('ATTENDED ')
    print('--------------------------')
    print(f"MEAN: {round(np.mean(attended),3)}")
    print(f"CI: {round(CI(attended),3)}")
    print('')
    print('UNATTENDED')
    print('--------------------------')
    print(f"MEAN: {round(np.mean(unattended),3)}")
    print(f"CI: {round(CI(unattended),3)}")
    print('')
    print('EXPECT REAL ')
    print('--------------------------')
    print(f"MEAN: {round(np.mean(exp_real),3)}")
    print(f"CI: {round(CI(exp_real),3)}")
    print('')
    print('EXPECT SCRAMBLED')
    print('--------------------------') 
    print(f"MEAN: {round(np.mean(exp_scr),3)}")
    print(f"CI: {round(CI(exp_scr),3)}")
    print('')
            
    print('')
    print('--------------------------')  
    print("REPEATED-MEASURES ANOVA")
    print('--------------------------')  
    print(pg.rm_anova(dv = i, 
                within = ['attention_condition', 'expectation_condition'],
                subject = 'subject', data = bhv_vars_df,
                detailed = True, effsize = "np2"))
#bhv_vars_df.to_pickle(os.path.join(DataDir, "Exp1_SDT.pkl"))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RECOGNITION RATE BY NON-TARGET STIMULUS IDENTITY
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("")
print("***********************************************************************")
print('STATS:   RECOGNITION RATE BY NON-TARGET STIMULUS IDENTITY')
print("***********************************************************************")
print("") 

group_choice = ['not_probe_real', 'probeReal', 'subject', exp_type, 'attention_condition']
proportion_R = bhv_df.groupby(group_choice)['recognition'].mean()

#Renaming variables for titles 
rec_df = proportion_R.reset_index()
rec_df.loc[rec_df.not_probe_real == 0,'not_probe_real'] = 'Scr Non-Target'
rec_df.loc[rec_df.not_probe_real == 1,'not_probe_real'] = 'Real Non-Target'

rec_df.loc[rec_df.probeReal == 0,'probeReal'] = 'Scr Target'
rec_df.loc[rec_df.probeReal == 1,'probeReal'] = 'Real Target'

for probe_id in sorted(rec_df.probeReal.unique()):
    for not_probe_id in sorted(rec_df.not_probe_real.unique()):
        print("")
        print("*******************************************")
        print(f'{probe_id} | {not_probe_id}')
        print("*******************************************")
        dataDF = rec_df.loc[(rec_df.probeReal == probe_id) &
                            (rec_df.not_probe_real == not_probe_id)]
        
        
        # COmpute the marginal mean for each factor        
        attended = dataDF[dataDF['attention_condition']=="Attended"]
        attended = attended.groupby('subject')['recognition'].mean()
        unattended = dataDF[dataDF['attention_condition']=="Unattended"]
        unattended = unattended.groupby('subject')['recognition'].mean()
        
        exp_real = dataDF[dataDF['expectation_condition']=="Expect Real"]
        exp_real = exp_real.groupby('subject')['recognition'].mean()
        exp_scr = dataDF[dataDF['expectation_condition']=="Expect Scrambled"]
        exp_scr = exp_scr.groupby('subject')['recognition'].mean()
        
        
        print("")
        print('MEAN FOR EACH CONDITION')
        print('------------------------')
        print("")
        print('95% CI FOR EACH CONDITION')
        print(dataDF.groupby(["expectation_condition", "attention_condition"])["recognition"].mean())
        print("-------------------------")   
        print(dataDF.groupby(["expectation_condition", "attention_condition"])["recognition"].apply(CI))
        
        print('')
        print('--------------------------')  
        print("MARGINAL MEAN")
        print('--------------------------')
        print('') 
        print('ATTENDED ')
        print(f"MEAN: {round(np.mean(attended),3)}")
        print(f"CI: {round(CI(attended),3)}")
        print('******************************************')
        print('UNATTENDED') 
        print(f"MEAN: {round(np.mean(unattended),3)}")
        print(f"CI: {round(CI(unattended),3)}")
        print('******************************************')
        print('EXPECT REAL ')
        print(f"MEAN: {round(np.mean(exp_real),3)}")
        print(f"CI: {round(CI(exp_real),3)}")
        print('******************************************')
        print('EXPECT SCRAMBLED') 
        print(f"MEAN: {round(np.mean(exp_scr),3)}")
        print(f"CI: {round(CI(exp_scr),3)}")
        print('******************************************')
               
        print('')
        print('--------------------------')  
        print("REPEATED-MEASURES ANOVA")
        print('--------------------------') 
        print(pg.rm_anova(dv = 'recognition', 
            within = ['attention_condition', 'expectation_condition'],
            subject = 'subject', data = dataDF,
            detailed = True, effsize = "np2"))

#rec_df.to_pickle(os.path.join(DataDir, "Exp1_Recognition_by_NonTarget.pkl"))
#%% 
# CATEGORIZATION ACCURACY BY HIT, FA, MISS, and CR
print("")
print("***********************************************************************")
print("***********************************************************************")
print(f'                  STATS FOR CATEGORIZATION ACCURACY')
print("***********************************************************************")
print("***********************************************************************")

group_choice =  ['recognition', 'probeReal', 'subject', exp_type, 'attention_condition']
accuracy_df = bhv_df.groupby(group_choice)['correct'].mean()
accuracy_df = accuracy_df.reset_index()
    
#Rename Variables for Titles
accuracy_df.loc[accuracy_df.recognition == 0,'recognition'] = 'Unrecognized'
accuracy_df.loc[accuracy_df.recognition == 1,'recognition'] = 'Recognized'

accuracy_df.loc[accuracy_df.probeReal == 0,'probeReal'] = 'Scrambled'
accuracy_df.loc[accuracy_df.probeReal == 1,'probeReal'] = 'Real'

for rec_status in sorted(accuracy_df.recognition.unique()):
    for probe_id in sorted(accuracy_df.probeReal.unique()):
        print("")
        print("**************************************")
        print(f'{rec_status} | {probe_id}')
        print("**************************************")
        dataDF = accuracy_df.loc[(accuracy_df.probeReal == probe_id) &
                            (accuracy_df.recognition == rec_status)]
        
        print(dataDF.groupby(["expectation_condition", "attention_condition"])["correct"].mean())
        print(dataDF.groupby(["expectation_condition", "attention_condition"])["correct"].apply(CI))
        
        attended = dataDF[dataDF['attention_condition']=="Attended"]
        attended = attended.groupby('subject')['correct'].mean()
        unattended = dataDF[dataDF['attention_condition']=="Unattended"]
        unattended = unattended.groupby('subject')['correct'].mean()
        
        exp_real = dataDF[dataDF['expectation_condition']=="Expect Real"]
        exp_real = exp_real.groupby('subject')['correct'].mean()
        exp_scr = dataDF[dataDF['expectation_condition']=="Expect Scrambled"]
        exp_scr = exp_scr.groupby('subject')['correct'].mean()
        
        print('ATTENDED ')
        print(f"MEAN: {np.mean(attended)}")
        print(f"CI: {CI(attended)}")
        print('******************************************')
        print('UNATTENDED') 
        print(f"MEAN: {np.mean(unattended)}")
        print(f"CI: {CI(unattended)}")
        print('******************************************')
        print('EXPECT REAL ')
        print(f"MEAN: {np.mean(exp_real)}")
        print(f"CI: {CI(exp_real)}")
        print('******************************************')
        print('EXPECT SCRAMBLED') 
        print(f"MEAN: {np.mean(exp_scr)}")
        print(f"CI: {CI(exp_scr)}")
        print('******************************************')
      
        
        print(pg.rm_anova(dv = 'correct', 
            within = ['attention_condition', 'expectation_condition'],
            subject = 'subject',
            data = dataDF,
            detailed = True,
            effsize = "np2"))
        
accuracy_df.to_pickle(os.path.join(DataDir, "Exp1_Categorization.pkl"))

# %%
# CATEGORIZATION ACCURACY BY NON-TARGET STIMULUS IDENTITY
print("")
print("***********************************************************************")
print("***********************************************************************")
print(f'CATEGORIZATION ACCURACY BY NON-TARGET STIMULUS IDENTITY')
print("***********************************************************************")
print("***********************************************************************")

group_choice =  ['not_probe_real', 'probeReal', 'recognition', 'subject', exp_type, 'attention_condition']
correct_pd_group = bhv_df.groupby(group_choice)['correct'].mean()

accuracy_df = correct_pd_group.reset_index()
    
#Rename Variables for Titles
accuracy_df.loc[accuracy_df.recognition == 0,'recognition'] = 'Unrecognized'
accuracy_df.loc[accuracy_df.recognition == 1,'recognition'] = 'Recognized'

accuracy_df.loc[accuracy_df.not_probe_real == 0,'not_probe_real'] = 'Scr Non-Target'
accuracy_df.loc[accuracy_df.not_probe_real == 1,'not_probe_real'] = 'Real Non-Target'

accuracy_df.loc[accuracy_df.probeReal == 0,'probeReal'] = 'Scr Target'
accuracy_df.loc[accuracy_df.probeReal == 1,'probeReal'] = 'Real Target'

recognition_status = 'Recognized'
rec_accuracy_df = accuracy_df.loc[accuracy_df.recognition == recognition_status]

for probe_id in sorted(rec_accuracy_df.probeReal.unique()):
    for not_probe_id in sorted(rec_accuracy_df.not_probe_real.unique()):
        print("")
        print("**************************************")
        print(f'{probe_id} | {not_probe_id}')
        print("**************************************")
        dataDF = rec_accuracy_df.loc[(rec_accuracy_df.probeReal == probe_id) &
                            (rec_accuracy_df.not_probe_real == not_probe_id)]
        
        attended = dataDF[dataDF['attention_condition']=="Attended"]
        attended = attended.groupby('subject')['correct'].mean()
        unattended = dataDF[dataDF['attention_condition']=="Unattended"]
        unattended = unattended.groupby('subject')['correct'].mean()
        
        
        
        print('ATTENDED ')
        print(f"MEAN: {np.mean(attended)}")
        print(f"CI: {CI(attended)}")
        print('******************************************')
        print('UNATTENDED') 
        print(f"MEAN: {np.mean(unattended)}")
        print(f"CI: {CI(unattended)}")
        print('******************************************')
        
        
        #print(dataDF.groupby(["expectation_condition", "attention_condition"])['correct'].mean())
        print(pg.rm_anova(dv = 'correct', 
            within = ['attention_condition', 'expectation_condition'],
            subject = 'subject',
            data = dataDF,
            detailed = True,
            effsize = "np2"))
rec_accuracy_df.to_pickle(os.path.join(DataDir, "Exp1_Categorization_by_NonTarget.pkl"))
#%% Confidence
group_choice =  ['recognition', 'probeReal','subject', exp_type, 'attention_condition']    
conf_df = bhv_df.groupby(group_choice)['confidence'].mean()

conf_df = conf_df.reset_index()

#Rename Variables for Titles
conf_df.loc[conf_df.recognition == 0,'recognition'] = 'Unrecognized'
conf_df.loc[conf_df.recognition == 1,'recognition'] = 'Recognized'

conf_df.loc[conf_df.probeReal == 0,'probeReal'] = 'Scrambled'
conf_df.loc[conf_df.probeReal == 1,'probeReal'] = 'Real'

for rec_status in conf_df.recognition.unique():
    for probe_id in conf_df.probeReal.unique():
        print("")
        print("**************************************")
        print(f'Stats for {rec_status} | {probe_id}')
        print("**************************************")
        
        dataDF = conf_df.loc[(conf_df.recognition == rec_status) &
                            (conf_df.probeReal == probe_id)]
        
        print(dataDF.groupby(["expectation_condition", "attention_condition"])['confidence'].mean())
        print(dataDF.groupby(["expectation_condition", "attention_condition"])['confidence'].apply(CI))
        
        attended = dataDF[dataDF['attention_condition']=="Attended"]
        attended = attended.groupby('subject')['confidence'].mean()
        unattended = dataDF[dataDF['attention_condition']=="Unattended"]
        unattended = unattended.groupby('subject')['confidence'].mean()
        
        exp_real = dataDF[dataDF['expectation_condition']=="Expect Real"]
        exp_real = exp_real.groupby('subject')['confidence'].mean()
        exp_scr = dataDF[dataDF['expectation_condition']=="Expect Scrambled"]
        exp_scr = exp_scr.groupby('subject')['confidence'].mean()
        
        print('ATTENDED ')
        print(f"MEAN: {np.mean(attended)}")
        print(f"CI: {CI(attended)}")
        print('******************************************')
        print('UNATTENDED') 
        print(f"MEAN: {np.mean(unattended)}")
        print(f"CI: {CI(unattended)}")
        print('******************************************')
        print('EXPECT REAL ')
        print(f"MEAN: {np.mean(exp_real)}")
        print(f"CI: {CI(exp_real)}")
        print('******************************************')
        print('EXPECT SCRAMBLED') 
        print(f"MEAN: {np.mean(exp_scr)}")
        print(f"CI: {CI(exp_scr)}")
        print('******************************************')

              
        print(pg.rm_anova(dv = 'confidence', 
            within = ['attention_condition', 'expectation_condition'],
            subject = 'subject', data = dataDF,
            detailed = True, effsize = "np2"))

conf_df.to_pickle(os.path.join(DataDir, "Exp1_Confidence.pkl"))
#%% Confidence Split by probed and not probed
group_choice =  ['not_probe_real', 'probeReal', 'recognition', 'subject', exp_type, 'attention_condition']
conf_df = bhv_df.groupby(group_choice)['confidence'].mean()

conf_df = conf_df.reset_index()
    
#Rename Variables for Titles
conf_df.loc[conf_df.recognition == 0,'recognition'] = 'Unrecognized'
conf_df.loc[conf_df.recognition == 1,'recognition'] = 'Recognized'

conf_df.loc[conf_df.not_probe_real == 0,'not_probe_real'] = 'Scr Non-Target'
conf_df.loc[conf_df.not_probe_real == 1,'not_probe_real'] = 'Real Non-Target'

conf_df.loc[conf_df.probeReal == 0,'probeReal'] = 'Scrambled Target'
conf_df.loc[conf_df.probeReal == 1,'probeReal'] = 'Real Target'

recognition_status = 'Recognized'
rec_conf_df = conf_df.loc[conf_df.recognition == recognition_status]

for not_probe_id in rec_conf_df.not_probe_real.unique():
    for probe_id in rec_conf_df.probeReal.unique():
        print("")
        print('**************************************')
        print(f'Stats for {not_probe_id} | {probe_id}')
        print('**************************************')
        dataDF = rec_conf_df.loc[(rec_conf_df.probeReal == probe_id) &
                            (rec_conf_df.not_probe_real == not_probe_id)]
        print(dataDF.groupby(['expectation_condition', 'attention_condition'])['confidence'].mean())
        print(dataDF.groupby(['expectation_condition', 'attention_condition'])['confidence'].apply(CI))
        
        
        print(pg.rm_anova(dv = 'confidence', 
            within = ['attention_condition', 'expectation_condition'],
            subject = 'subject', data = dataDF, detailed = True,
            correction = True, effsize = "np2"))       
rec_conf_df.to_pickle(os.path.join(DataDir, 'Confidence_2x2conds_split.pkl'))

# %%
# CATEGORIZATION ACCURACY BY NON-TARGET STIMULUS IDENTITY
print("")
print("***********************************************************************")
print("***********************************************************************")
print(f'NON-TARGET CATEGORIZATION ACCURACY BY NON-TARGET STIMULUS IDENTITY')
print("***********************************************************************")
print("***********************************************************************")

group_choice =  ['not_probe_real', 'probeReal', 'recognition', 'subject', exp_type, 'attention_condition']
correct_pd_group = bhv_df.groupby(group_choice)['notProbeCorrect'].mean()

accuracy_df = correct_pd_group.reset_index()
    
#Rename Variables for Titles
accuracy_df.loc[accuracy_df.recognition == 0,'recognition'] = 'Unrecognized'
accuracy_df.loc[accuracy_df.recognition == 1,'recognition'] = 'Recognized'

accuracy_df.loc[accuracy_df.not_probe_real == 0,'not_probe_real'] = 'Scr Non-Target'
accuracy_df.loc[accuracy_df.not_probe_real == 1,'not_probe_real'] = 'Real Non-Target'

accuracy_df.loc[accuracy_df.probeReal == 0,'probeReal'] = 'Scr Target'
accuracy_df.loc[accuracy_df.probeReal == 1,'probeReal'] = 'Real Target'

recognition_status = 'Recognized'
rec_accuracy_df = accuracy_df.loc[accuracy_df.recognition == recognition_status]

for probe_id in sorted(rec_accuracy_df.probeReal.unique()):
    for not_probe_id in sorted(rec_accuracy_df.not_probe_real.unique()):
        print("")
        print("**************************************")
        print(f'{probe_id} | {not_probe_id}')
        print("**************************************")
        
        dataDF = rec_accuracy_df.loc[(rec_accuracy_df.probeReal == probe_id) &
                            (rec_accuracy_df.not_probe_real == not_probe_id)]
        
        attended = dataDF[dataDF['attention_condition']=="Attended"]
        attended = attended.groupby('subject')['notProbeCorrect'].mean()
        unattended = dataDF[dataDF['attention_condition']=="Unattended"]
        unattended = unattended.groupby('subject')['notProbeCorrect'].mean()
        
        
        exp_real = dataDF[dataDF['expectation_condition']=="Expect Real"]
        exp_real = exp_real.groupby('subject')['notProbeCorrect'].mean()
        exp_scr = dataDF[dataDF['expectation_condition']=="Expect Scrambled"]
        exp_scr = exp_scr.groupby('subject')['notProbeCorrect'].mean()
        
        
        
        print(dataDF.groupby(["expectation_condition", "attention_condition"])["notProbeCorrect"].mean())   
        print(dataDF.groupby(["expectation_condition", "attention_condition"])["notProbeCorrect"].apply(CI))
        
        print('ATTENDED ')
        print(f"MEAN: {np.mean(attended)}")
        print(f"CI: {CI(attended)}")
        print(pg.ttest(attended,0.25, alternative='two-sided'))
        print('******************************************')
        print('UNATTENDED') 
        print(f"MEAN: {np.mean(unattended)}")
        print(f"CI: {CI(unattended)}")
        print(pg.ttest(unattended,0.25, alternative='two-sided'))
        print('******************************************')
        print('EXPECT REAL ')
        print(f"MEAN: {np.mean(exp_real)}")
        print(f"CI: {CI(exp_real)}")
        print('******************************************')
        print('EXPECT SCRAMBLED') 
        print(f"MEAN: {np.mean(exp_scr)}")
        print(f"CI: {CI(exp_scr)}")
        print('******************************************')
        
        print(pg.rm_anova(dv = 'notProbeCorrect', 
            within = ['attention_condition', 'expectation_condition'],
            subject = 'subject',
            data = dataDF,
            detailed = True,
            effsize = "np2"))
           
rec_accuracy_df.to_pickle(os.path.join(DataDir, "Exp1_Non-Target_Categorization_by_NonTarget.pkl"))