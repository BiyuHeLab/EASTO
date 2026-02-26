# This script generates figures related to results obtained from Exp1/Exp 3 
# %% Import packages and set figure parameters
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["font.weight"]="normal"
plt.rcParams["axes.labelsize"]=7
plt.rcParams["ytick.labelsize"]=7
plt.rcParams["xtick.labelsize"]=7
FigDir = '/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior/Figures/'
# %% 
def plot_SDT_measures_by_conditions(data):
    ''' 
    This function generates figures displaying SDT behavioral measures
    including HR, FAR, criterion and sensitivity under different
    expectation and attention conditions. 
    '''
    fig, axes = plt.subplots(1,4,figsize = (6,2))        
    for i, bhv in enumerate(["Hit", "FA", "d", "c"]):
        #row, col=divmod(i,2)
        sns.stripplot(data = df, x = 'expectation_condition', y = bhv,
                        hue='attention_condition', order=['Expect Scrambled', 'Expect Real'],
                        alpha = 0.3, size = 3, dodge=True, ax = axes[i], zorder = 1)
        sns.pointplot(data = df, x = 'expectation_condition', y = bhv,
                        hue='attention_condition', order=['Expect Scrambled', 'Expect Real'],
                        errorbar= ('ci', 95), ax = axes[i],
                        linewidth=1, markersize=5, dodge = True, zorder=2)
          
        if bhv in ["Hit", "FA"]:
            axes[i].set_yticks(np.arange(0, 1.2, 0.2))
        if bhv == 'Hit':    
            axes[i].axhline(y=0.5, linestyle='--', color = 'black', linewidth=1)
            axes[i].set_ylabel("HR")
        elif bhv == "FA":
            axes[i].set_ylabel("FAR")
        elif bhv == "d":
            axes[i].set_ylabel("d'")
        elif bhv == "c":
            axes[i].set_ylabel("c")
        else:
            pass
       
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_position(('outward',5))
        axes[i].spines['bottom'].set_position(('outward', 5))
        axes[i].set_xticklabels(["Expect scr", "Expect real"], 
                                       rotation = 45)
        axes[i].set_xlabel("")
         # Hide legend for subplots
        axes[i].get_legend().remove()
    plt.tight_layout()
    return fig, axes
    #plt.show()
    
    
def plot_bhv_by_conditions(data):
    '''
    This function generates figures displaying behavioral measurements including
    discrimination rate and confidence under different
    expectation and attention conditions.
    '''
     
    fig, axes = plt.subplots(1,4,figsize = (6, 2), sharey=True, sharex=True)
    counter = 0
    for rec_status in sorted(df['recognition'].unique()):
        for img_type in sorted(df['probeReal'].unique()):
            titles = {0: "Hit trials", 1: "FA Trials",
                      2: "Miss trials", 3: "CR Trials"}
            data = df[(df['recognition'] == rec_status) & (df['probeReal'] == img_type)]
            
            #row, col=divmod(counter,2)
            sns.stripplot(data = data, x = 'expectation_condition', y = data.columns[-1],
                            hue='attention_condition', order=['Expect Scrambled', 'Expect Real'],
                            alpha = 0.3, size = 3, dodge=True, ax = axes[counter], zorder = 1)
            sns.pointplot(data = data, x = 'expectation_condition', y = data.columns[-1],
                            hue='attention_condition', order=['Expect Scrambled', 'Expect Real'],
                            errorbar= ('ci', 95), ax = axes[counter], dodge=True,
                            linewidth=1, markersize=5, zorder=2 )
             
            if (df.columns[-1] == 'correct') or df.columns[-1] == ('notProbeCorrect'):
                axes[counter].set_yticks(np.arange(0, 1.25, 0.25))
                axes[counter].axhline(y=0.25, linestyle='--', color = 'black', linewidth=1)
                axes[counter].spines['left'].set_bounds(0, 1)
                axes[counter].set_ylabel("Categorization Accuracy")
            elif df.columns[-1] == 'confidence':
                axes[counter].set_yticks(np.arange(1, 5, 1))
                axes[counter].spines['left'].set_bounds(1, 4)
                axes[counter].set_ylabel("Confidence")
            axes[counter].spines['left'].set_position(('outward',5))
            axes[counter].spines['bottom'].set_position(('outward', 5))
            
            axes[counter].spines['top'].set_visible(False)
            axes[counter].spines['right'].set_visible(False)
            if counter in titles:
                axes[counter].set_title(titles[counter], fontsize=8, fontweight="bold")
            axes[counter].legend("").remove()
            axes[counter].set_xlabel("")
            axes[counter].set_xticklabels(["Expect scr", "Expect real"], 
                                       rotation = 20)
            counter +=1    
    plt.tight_layout()
    #plt.show()

def plot_bhv_by_conditions_split(data):
    '''
    This function generates figures displaying behavioral measurements including
    recognition rate as well as discrimination rate and confidence (for recognized trials)
    under different expectation and attention conditions, split by the type of not-probed-stimuli.
    '''
    fig, axes = plt.subplots(1,4,figsize = (6,2), sharex=True, sharey=True)
    counter = 0
    for probe_id in sorted(df['probeReal'].unique()):
        for not_probe_id in sorted(df['not_probe_real'].unique()):
            
            data = df[(df['not_probe_real'] == not_probe_id) & (df['probeReal'] == probe_id)]
            
            sns.stripplot(data = data, x = 'expectation_condition', y = data.columns[-1],
                            hue='attention_condition', order=['Expect Scrambled', 'Expect Real'],
                            alpha = 0.3, size=3, dodge=True, ax = axes[counter], zorder =1)
            sns.pointplot(data = data, x = 'expectation_condition', y = data.columns[-1],
                            hue='attention_condition', order=['Expect Scrambled', 'Expect Real'],
                            errorbar= ('ci', 95), ax = axes[counter], 
                            dodge=True, markersize=5, linewidth=1, zorder =2)
                  
            if df.columns[-1] == 'correct':
                axes[counter].set_yticks(np.arange(0, 1.25, 0.25))
                axes[counter].axhline(y=0.25, linestyle='--', color = 'black', linewidth=1)
                axes[counter].spines['left'].set_bounds(0, 1)
                axes[counter].set_ylabel("Cat Accuracy")
            elif df.columns[-1] == 'recognition':
                axes[counter].set_yticks(np.arange(0, 1.25, 0.25))
                axes[counter].axhline(y=0.5, linestyle='--', color = 'black', linewidth=1)
                axes[counter].spines['left'].set_bounds(0, 1)
                axes[counter].set_ylabel("P('yes')")
            elif df.columns[-1] == 'notProbeCorrect':
                axes[counter].set_yticks(np.arange(0, 1.25, 0.25))
                axes[counter].axhline(y=0.25, linestyle='--', color = 'black', linewidth=1)
                axes[counter].spines['left'].set_bounds(0, 1)
                axes[counter].set_ylabel("Nontarget Cat Acc")             
            elif df.columns[-1] == 'confidence':
                pass
            axes[counter].spines['left'].set_position(('outward',5))
            axes[counter].spines['bottom'].set_position(('outward', 0))
            
            axes[counter].spines['top'].set_visible(False)
            axes[counter].spines['right'].set_visible(False)
            axes[counter].set_title(f"{probe_id} | \n{not_probe_id}",
                                     fontsize=7, fontweight="normal")
            axes[counter].legend("").remove()
            axes[counter].set_xlabel("")
            axes[counter].set_xticklabels(["Expect scr", "Expect real"], 
                                       rotation = 20)
            counter +=1    
    plt.tight_layout()
    return fig, axes
    #plt.show()

# %%
RootDir = "/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior" 
DataDir = os.path.join(RootDir, "Data")
exp_name = 'Exp1_'

#Plot SDT-related measures
df = pd.read_pickle(os.path.join(DataDir, exp_name + "SDT.pkl"))
fig, axes = plot_SDT_measures_by_conditions(df)   
plt.savefig(FigDir + exp_name + "SDT.svg", dpi=300, bbox_inches='tight')
plt.show()

#Plot Categrization behavior
bhv_vars = ["Categorization", "Confidence"]
for i in bhv_vars:
    df = pd.read_pickle(os.path.join(DataDir, exp_name + i + ".pkl")) 
    plot_bhv_by_conditions(df)
    #plt.savefig(FigDir + exp_name + i + '.svg', dpi=300, bbox_inches='tight')

#Plot behavior collapsed across target-nontarget pairing 
bhv_vars = ["Recognition", "Categorization"]
for i in bhv_vars:
    df = pd.read_pickle(os.path.join(DataDir, exp_name + i + "_by_NonTarget.pkl"))
    df.loc[df['not_probe_real'].str.contains('Scr Non-Target', na=False), 'not_probe_real'] = 'Scr Nontarget'
    df.loc[df['not_probe_real'].str.contains('Real Non-Target', na=False), 'not_probe_real'] = 'Real Nontarget'
    
    plot_bhv_by_conditions_split(df)
    # plt.savefig(FigDir + exp_name + i + '_by_NonTarget.svg', 
    #            dpi=300, bbox_inches='tight')