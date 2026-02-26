#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Author: Brandon Chen 

Script for taking results from interleaved quest staircases in psychopy (.psydat) files
Accomplishes: 
    1) Checks for convergence of threshold estimates for each exemplar
    2) Retrieves final threshold estimate (mean of posterior pdf) for each exemplar
    3) Puts together list of paired contrast threshold estimates and images to be sent to 
        experiment
"""

from psychopy import data, gui, core
from psychopy.tools.filetools import fromFile
import pylab
import os
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
from copy import deepcopy


#%% Quest Analysis. 

#Get PsyDat File Path which has both Trial data (saved to csv), and multistairhandlers
#for Quest estimates 

def getThresholdEstimate(paradigm):
    """
    

    Returns
    -------
    Psychopy object file with all staircases for extracting estimate 
    Dictionary with image paths as keys and estimate from QUEST staircases as values

    """
    quest_file = gui.fileOpenDlg('.')
    if not quest_file:
        core.quit()
            
    thisDat = fromFile(quest_file[0])
    #Turn trial entries into dataframe 
    quest_df = pd.DataFrame(thisDat.entries)
    #Was a typo in earlier version of paradigm so need to rename a column before indexing just in case
    #Will only apply to older files 
    quest_df.rename(columns = {'objetive_category':'objective_category'}, inplace = True)
    if paradigm == 'SEA':
        quest_df = pd.concat([quest_df['trials.label'].rename('exemplar'),
                          quest_df['trials.intensity'].rename('intensity'),
                          quest_df['stim_left'],
                          quest_df['stim_right'],
                          quest_df['trials.thisRepN'].rename('RepN'),
                          quest_df['trials.response'].rename('response'),
                          quest_df['category_correct'],
                          quest_df['objective_category'], #Need to correct typo in script 
                          quest_df['category_response'],
                          quest_df['participant'],
                          quest_df['expName'],
                          quest_df['ITI']], axis=1).dropna(subset = ['exemplar'])
        
    elif paradigm == 'TEA':
        #Because of old version of experiment, stim_1st = stim_left, stim_2nd = stim_right
        #rename should not impact the df or throw error if the column label does not exist
        quest_df.rename(columns = {'stim_left' : 'stim_1st', 'stim_right': 'stim_2nd'})
        quest_df = pd.concat([quest_df['trials.label'].rename('exemplar'),
                          quest_df['trials.intensity'].rename('intensity'),
                          quest_df['stim_1st'],
                          quest_df['stim_2nd'],
                          quest_df['trials.thisRepN'].rename('RepN'),
                          quest_df['trials.response'].rename('response'),
                          quest_df['category_correct'],
                          quest_df['objective_category'],
                          quest_df['category_response'],
                          quest_df['participant'],
                          quest_df['expName'],
                          quest_df['ITI']], axis=1).dropna(subset = ['exemplar'])

    elif paradigm == 'OEA':
        quest_df = pd.concat([quest_df['trials.label'].rename('exemplar'),
                              quest_df['trials.intensity'].rename('intensity'),
                              quest_df['stim'],
                              quest_df['trials.thisRepN'].rename('RepN'),
                              quest_df['trials.response'].rename('response'),
                              quest_df['category_correct'],
                              quest_df['objective_category'],
                              quest_df['category_response'],
                              quest_df['participant'],
                              quest_df['expName'],
                              quest_df['ITI']], axis=1).dropna(subset = ['exemplar'])
    else: 
        print('ERROR: Check DataFrame for ExpName ')
        
    # Not entirely necessary, but create addition series in dataframe 
    #for broad categories which may be useful for plotting/later analysis 

    #Replace Single letter with full category name 
    quest_df.objective_category.replace({'F': 'Face', 'H': 'House', 'A': 'Animal', 'O': 'Object'}, inplace=True)
    quest_df.sort_values('exemplar', inplace=True)
    
    #Turn log contrast to linear contrast
    quest_df['intensity'] = 10**quest_df['intensity']

    return thisDat, quest_df


def exportQuestEstimate(thisDat, quest_df, participant, paradigm, save = True):

    #Get list of tuples of img & estimated thresh
    
    quest_estimates=[]
    qMini_estimate = []
    for stair in range(len(thisDat.loops[1].staircases)):
        exemplar = thisDat.loops[1].staircases[stair].condition['stim_img'][:-4] #Take first part of exemplar name to set contrast same for scram vs real
        thresh_est = 10**thisDat.loops[1].staircases[stair].mean() 
        if 'q' in exemplar: 
            qMini_estimate.append(thresh_est)
        else:
            quest_estimates.append((exemplar,thresh_est))
            
    quest_estimates = dict(quest_estimates)
    
    if bool(qMini_estimate):
        qMini_estimate = np.mean(qMini_estimate)
    
    
    if save and bool(quest_estimates): 
        with open(f"{participant}_{paradigm}_questimate.pkl", 'wb') as fp:
            pickle.dump(quest_estimates, fp, protocol=2)
    if save and bool(qMini_estimate): 
        with open(f"{participant}_{paradigm}_qMiniestimate.pkl", 'wb') as fp:
            pickle.dump(qMini_estimate, fp, protocol=2)
    return [estimate for estimate in [quest_estimates, qMini_estimate] if bool(estimate)]

def updateQuestEstimate(old_estimate, new_estimate, participant, paradigm, qMini = False, save = True):

    
    updated_estimate = {k: new_estimate.get(k, v) for k, v in old_estimate.items()}
    
    if qMini: 
        # Open Quest Mini File 
        quest_file = gui.fileOpenDlg('.')
        if not quest_file:
                core.quit()
        with open(quest_file[0], 'rb') as handle:
            qMini_estimate = pickle.load(handle)
            
        updated_estimate = {k : v * qMini_estimate for k, v in updated_estimate.items()}
    
    if save: 
        with open(f"{participant}_{paradigm}_questimate.pkl", 'wb') as fp:
            pickle.dump(updated_estimate, fp, protocol=2)
    return updated_estimate

                                                        ##### PLOTS #####
# Plot Each exemplar staircase to check for convergence 
def plotStaircase(quest_df, participant, paradigm, save = True):
    g = sns.FacetGrid(quest_df, col="objective_category")
    for col_var, facet_df in quest_df.groupby(["objective_category"]):
        sns.lineplot(x="RepN", y="intensity", hue="exemplar", data=facet_df,
                      ax=g.axes_dict[col_var])
    for col_val, ax in g.axes_dict.items(): #Format Legend
        ax.legend(ncol=2, title='Exemplar')
    g.set_titles('{col_name}')
    # g.set(ylim = [0,0.3])
    g.fig.suptitle('Quest Convergence', y = 1)
    if save:
        g.savefig(f"{participant}_{paradigm}_convergence_plots.png" )
    return

# Plot recognition for each exemplar 
def plotQuestRecognition(quest_df, participant, paradigm,save = True):
    rec = sns.FacetGrid(quest_df,col = 'objective_category', sharey=False, sharex=False)
    for col_var, facet_df in quest_df.groupby(['objective_category']):
        sns.barplot(x='exemplar', y= 'response', data = facet_df, alpha= 0.5,
                     ax=rec.axes_dict[col_var], ci=95)
    rec.fig.suptitle('Quest Recognition Rate', y = 1)
    (rec.map(plt.axhline, y=0.50, color="k", dashes=(2, 1), zorder=0)
     .set(ylim = [0, 1.05])
      .set_axis_labels("Exemplar", "Recognition")
      .set_titles("{col_name}", weight='bold')
     .tight_layout(w_pad=0))
    if save:
        rec.savefig(f"{participant}_{paradigm}_recognition.png" )
    return

#Plot Categorization accuracy for each category
def plotQuestCategorization(quest_df, participant, paradigm,save = True):
    cat = sns.FacetGrid(quest_df,col = 'objective_category', row = 'response', sharey=False, sharex=False)
    for (col_var, row_var), facet_df in quest_df.groupby(['objective_category', 'response']):
        sns.barplot(x='exemplar', y= 'category_correct', data = facet_df, alpha= 0.5,
                     ax=cat.axes_dict[row_var,col_var], ci=95)
    cat.fig.suptitle('Quest Categorization Accuracy', y = 1)
    (cat.map(plt.axhline, y=0.25, color="k", dashes=(2, 1), zorder=0) 
     .set(ylim = [0, 1.05])
      .set_axis_labels("Exemplar", "Categorization Accuracy")
      .set_titles("{col_name}", weight='bold')
     .tight_layout(w_pad=0))
    if save:
        cat.savefig(f"{participant}_{paradigm}_categorization.png" )
    return

def plotCategoryRecDiff(quest_df, participant, paradigm,save = True):
    # There should be a way to do this using groupby, but I just don't know so I'm making a new dataframe of the differences
    catDiff = {exemp : []for exemp in quest_df.exemplar.unique()}
    for exemp in quest_df.exemplar.unique(): 
        catDiff[exemp] = (quest_df.loc[(quest_df.exemplar == exemp) & (quest_df.response == 1)].category_correct.mean() - 
                            quest_df.loc[(quest_df.exemplar == exemp) & (quest_df.response == 0)].category_correct.mean())
    catDiff_df = pd.DataFrame.from_dict(catDiff, orient = 'index', columns = ['category_correct'])
    catDiff_df.reset_index(inplace = True)
    catDiff_df.rename(columns = {'index' :'exemplar'}, inplace = True)
    categories = ['F','A','O','H'] 
    # #Add column to dataframe for base categories
    catDiff_df['objective_category'] = np.nan
    for cats in categories:
        catDiff_df.loc[catDiff_df.exemplar.str.startswith(cats), 'objective_category'] = cats

    #Replace Single letter with full category name 
    catDiff_df.objective_category.replace({'F': 'Face', 'H': 'House', 'A': 'Animal', 'O': 'Object'}, inplace=True)
    catDiff_df.sort_values('exemplar', inplace=True)

    
    cat = sns.FacetGrid(catDiff_df,col = 'objective_category',  sharey=False, sharex=False)
    for col_var, facet_df in catDiff_df.groupby(['objective_category']):
        sns.barplot(x='exemplar', y= 'category_correct', data = facet_df, alpha= 0.5,
                     ax=cat.axes_dict[col_var], ci=95)
        cat.fig.suptitle('Quest Categorization Accuracy Difference (Rec - Unrec)', y = 1)
        (cat.map(plt.axhline, y=0.3, color="k", dashes=(2, 1), zorder=0) 
         .set(ylim = [0, 1.05])
          .set_axis_labels("Exemplar", "Accuracy Difference (Rec - Unrec)")
          .set_titles("{col_name}", weight='bold')
         .tight_layout(w_pad=0))
        if save:
            cat.savefig(f"{participant}_{paradigm}_categorizationDifference.png" )
    return

#%% Main Functions
#Change to different directory
os.chdir('/Users/brandonchen93/')


participant = 'PBC'
paradigm = 'SEA'
qMini = False

#Load in Data and reformat for plotting
rawData, quest_df = getThresholdEstimate(paradigm)

# Plot figures
plotStaircase(quest_df, participant, paradigm, save = True)
plotQuestRecognition(quest_df, participant, paradigm, save = True)
plotQuestCategorization(quest_df, participant, paradigm, save = True)
plotCategoryRecDiff(quest_df, participant, paradigm, save = True)


# export Quest Estimate
quest_estimate = exportQuestEstimate(rawData, quest_df, participant, paradigm)

|#%% Updating with new staircase values 


origStaircase, origDf = getThresholdEstimate(paradigm)
old_quest = exportQuestEstimate(origStaircase, origDf, participant, paradigm)

newStaircase, newDf = getThresholdEstimate(paradigm)
new_quest = exportQuestEstimate(newStaircase, newDf, participant, paradigm)

if qMini:
    qMiniStaircase, qMiniDf = getThresholdEstimate(paradigm)
    qMiniEst = exportQuestEstimate(qMiniStaircase,qMiniDf, participant, paradigm)
#%%

updateEst = updateQuestEstimate(old_quest[0], new_quest[0], participant, paradigm, qMini = qMini, save = True)


# %%
