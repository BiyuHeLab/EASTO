#!/usr/bin/env python3
# -*- coding: utf-8 -*-

mainEffect = 'attention'
#%% Import packages
import os
from os.path import join
#import sys
RootDir = '/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior'
AnalysisDir = join(RootDir, 'Analysis')
DataDir = join(RootDir, 'Data')
#FigDir = join(RootDir, 'Figures')

#sys.path.append(join(AnalysisDir, 'EASTO_funcs'))

import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import t
import pingouin as pg
from matplotlib.ticker import MultipleLocator
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["font.weight"]="normal"
plt.rcParams["axes.labelsize"]=7
plt.rcParams["ytick.labelsize"]=7
plt.rcParams["xtick.labelsize"]=7

# functions for generating SEM and CI

def CI(data, confidence_level = 0.95):
# Calculate sample mean and standard error
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))

    # Compute the t critical value for 95% CI
    degrees_of_freedom = len(data) - 1
    t_critical = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Compute the margin of error
    return t_critical * sem

# SET FIGURE PARAMETERS
bhv_vars_df = pd.read_pickle(os.path.join(DataDir, f"Exp2_{mainEffect}_SDT.pkl"))

if mainEffect == 'expectation':
    cond_names = 'expectation_condition'
    odr_scheme = ['Expect Scrambled', 'Expect Real']
    x_ticks = ['Expect scr', 'Expect real']
    color= ['purple', 'green']
elif mainEffect == 'attention':
    cond_names = "attention_condition"
    odr_scheme = ['Unattended', 'Attended']
    x_ticks = ['Unattended', 'Attended']
    palette = sns.color_palette()
    color = palette[1], palette[0]
    
    
#%% PLOT HR, FAR, d', and c COLAPPSED ACROSS CONDITIONS AND COMPUTE GROUP STATISTICS   
fig, axes = plt.subplots(1,4,figsize = (6,2))        
for i, bhv in enumerate(["Hit", "FA", "d", "c"]):   
    # Plot mean and CI
    sns.pointplot(x=cond_names, y=bhv, data=bhv_vars_df,
                  errorbar=('ci', 95), palette=color,
                  order=odr_scheme, linewidth=1,
                  ax=axes[i], markersize=4, zorder=2)

    #Plot individual subject data
    sns.stripplot(x = cond_names, y = bhv, data = bhv_vars_df,
                order = odr_scheme, dodge=True,
                palette = color, alpha = 0.3,
                ax = axes[i], size=3, zorder =1)

    if bhv == "Hit":
        axes[i].set_ylabel('HR')
        axes[i].set_yticks(np.arange(0, 1.2, 0.2))
        axes[i].axhline(y=0.5, linestyle='--', color = 'black', linewidth = 1)
    elif bhv == 'FA':
        axes[i].set_yticks(np.arange(0, 1.2, 0.2))
    elif bhv == "d":
        axes[i].set_ylabel("d'")
    
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_position(('outward',5))
    axes[i].spines['bottom'].set_position(('outward', 5))
    axes[i].legend("").remove()
    axes[i].set_xlabel("")
    axes[i].set_xticklabels(x_ticks, fontsize=7, rotation = 30)      
plt.tight_layout()
plt.show()

#%% ################################################################ 
# Category Discrimination
# ################################################################## 
df = pd.read_pickle(os.path.join(DataDir, f"Exp2_{mainEffect}_categorization.pkl"))

# Plot categorization accuracy by SDT metrics (Fig 5 A-ii and B-ii)
counter = 0
fig, axes = plt.subplots(1,4,figsize = (6,2), sharey=True)

for rec_status in sorted(df.recognition.unique()):
    for probe_id in sorted(df.probeReal.unique()):
        titles = {0: "Hit trials", 1: "FA Trials",
                      2: "Miss trials", 3: "CR Trials"}
        data = df.loc[(df.probeReal == probe_id) &
                            (df.recognition == rec_status)]
       
        sns.pointplot(data = data, x = cond_names, y = data.columns[-1],
                    order=odr_scheme, linewidth = 1, markersize=5, palette = color, 
                    errorbar= ('ci', 95), dodge=True, zorder=2,
                    ax = axes[counter])
        
        sns.swarmplot(data = data, x = cond_names, y = data.columns[-1],
                      palette = color, order=odr_scheme, alpha = 0.3, size=3, dodge=True,
                      ax = axes[counter], zorder=1)
        
       
        axes[counter].set_yticks(np.arange(0, 1.25, 0.25))
        axes[counter].axhline(y=0.25, linestyle='--', color = 'black', linewidth = 1)
        axes[counter].yaxis.set_major_locator(MultipleLocator(0.2))
        axes[counter].spines['top'].set_visible(False)
        axes[counter].spines['right'].set_visible(False)
        axes[counter].spines['left'].set_position(('outward',5))
        axes[counter].spines['bottom'].set_position(('outward', 5))
        axes[counter].set_xlabel("")
        axes[counter].set_ylabel("Categorization Accuracy")
        if counter in titles:
                axes[counter].set_title(titles[counter], fontsize=7, fontweight="bold")
        axes[counter].set_xticklabels(x_ticks, fontsize=7, rotation = 30)                 
        counter += 1    
plt.tight_layout()
plt.show()                   
    