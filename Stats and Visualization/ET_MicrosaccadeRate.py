# This code computes microssacade rate in Exp 2, separately for attention anf expection blocks.
# In attention blocks, we compare microsaccade rate between trials where the gaze is direct
# torward vs away from the target.  
# In expectation block, we compare microsaccade ra te between trials where gaze is direct 
# toward vs away from the location where real images are expected.
# Microsaccade rate was estimated following Engbert & Kliegl
 

# Load Packages
import pandas as pd
from os.path import join
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
RootDir = '/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior'
from EngbertMicrosaccadeToolbox import microsac_detection

#########################################################################
def load_ET_data(ET_Dir, ET_filename):
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


def px_to_dva(px, center_px=960, px_per_deg=36.86135449077602):
    """Convert pixel coordinates to deviation-from-center in DVA."""
    return (px - center_px) / px_per_deg
############################################################################################

DataDir = '/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior/Data'
exp_version = 'Paradigm_Control' #['Paradigm_V3', 'Paradigm_Control', 'Paradigm_expControl']
paradigm = 'Spatial'
sub_version = ''
bhv_df = pd.read_pickle(join(DataDir, f"{paradigm}{exp_version}_bhv_df.pkl")) 

rate_all_trials = []
win_size = 100
step_size = 10

for subj in ['P25', 'P26', 'P28']:
    ET_Dir = join(DataDir, exp_version, subj, paradigm, 'Main', 'edfData')
    ET_filename = f"{subj}_eye_tracking_data.pkl"
    subj_df = bhv_df[(bhv_df['subject']==subj)]
    subj_df_sorted = subj_df.sort_values(['block', 'trial'])

    dfRec, dfMsg, dfFix, dfSacc, dfBlink, dfSamples = load_ET_data(ET_Dir, ET_filename)

    for block in ['attention', 'expectation']:
        #for location in ['L', 'R']: 
        # --- build block-specific trial data ---
        cond_mask = (subj_df_sorted[f'{block}_condition']!='Neutral') \
                    & (subj_df_sorted['brokeFixation']==0) #\
                    #& (subj_df_sorted[f'{block[:3]}_loc']==location)
        cond_df = subj_df_sorted[cond_mask].reset_index(drop=True)
        
        # 'attention_cue_onst', "init_fix"
        stim_onsets = dfMsg.loc[dfMsg.text.str.contains('stim_onset')].reset_index(drop=True).rename(columns={'time':'onset','text':'onset_event'})
        onsetTime = stim_onsets['onset']-1950
        offsetTime = stim_onsets['onset']+700; offsetTime = offsetTime.rename('offset')

        Period_Interest = pd.concat([onsetTime, offsetTime], axis=1)
        Period_Interest = Period_Interest[cond_mask.to_numpy()].reset_index(drop=True)

        for idx in range(len(cond_df)):
            onset = Period_Interest['onset'][idx]
            offset = Period_Interest['offset'][idx]    
            if pd.isna(onset) or pd.isna(offset):
                continue
            #trial_dur = offset -onset
            win_onsets = np.arange(-1950, 500, step_size)
            
            # Extract gaze samples for the current trial and convert pixel coordinates
            # to degrees of visual angle (DVA).
            trial_samples = dfSamples.loc[dfSamples['tSample'].between(onset, offset)]
            trial_samples['LX'] = px_to_dva(trial_samples['LX'],center_px=960)
            trial_samples['LY'] = px_to_dva(trial_samples['LY'],center_px=540)
            trial_samples['RX'] = px_to_dva(trial_samples['RX'],center_px=960)
            trial_samples['RY'] = px_to_dva(trial_samples['RY'],center_px=540)
            
            
            # microsaccade detection based on Engbert and Kliegl (2003)
            right_eye = np.vstack([np.asarray(trial_samples.RX), np.asarray(trial_samples.RY)]).T
            left_eye = np.vstack([np.asarray(trial_samples.LX), np.asarray(trial_samples.LY)]).T
            
            # Skip if trial has too few samples
            if len(right_eye) < 5 or len(left_eye) < 5:
                continue
            
            ms_r, rad_r = microsac_detection.microsacc(right_eye, vfac=6, mindur=12, sampling=1000)
            ms_l, rad_l = microsac_detection.microsacc(left_eye, vfac=6, mindur=12, sampling=1000)
            
            if len(ms_r) == 0 or len(ms_l) == 0:
                bino, monol, monor = [], [], []
            else:
                bino, monol, monor = microsac_detection.binsacc(ms_r, ms_l)
            bino = pd.DataFrame(bino, columns=["onset_l", "end_l",
                                               "peakvelocity_l", "horizontalcomponent_l",
                                               "verticalcomponent_l","horizontalamplitude_l",
                                               "verticalamplitude_l", "onset_r", "end_r",
                                               "peakvelocity_r", "horizontalcomponent_r",
                                               "verticalcomponent_r","horizontalamplitude_r",
                                               "verticalamplitude_r"])
            if len(bino) == 0:
                continue
            bino["onset_l"] = bino["onset_l"]-1950
            bino["end_l"] = bino["end_l"]-1950
            
            bino["direction"] = np.nan  # initialize

            for j in range(len(bino)):
                target_side = cond_df.loc[idx, f"{block[:3]}_loc"]  # e.g., 'L' or 'R'
                horiz_amp = bino.loc[j, "horizontalamplitude_l"]

                if (target_side == "L" and horiz_amp < 0) or (target_side == "R" and horiz_amp > 0):
                    bino.loc[j, "direction"] = "Toward"
                else:
                    bino.loc[j, "direction"] = "Away"
            
            # onset times of detected binocular microsaccades (relative to attention cue onset)
            micro_onsets = np.array(bino['onset_l'])
            micro_dirs   = bino["direction"].to_numpy()
            
            
            # store per-window rate
            rate_rows = []
                        
            for t in win_onsets:
                start_t = t
                end_t   = t + win_size - 1
                in_window = (micro_onsets >= start_t) & (micro_onsets < end_t)

                for direction in ["Toward", "Away"]:
                    n_dir = np.sum(in_window & (micro_dirs == direction))
                    rate_hz = n_dir / (win_size / 1000)
                    rate_rows.append({
                        "trial": idx,
                        "subject": subj,
                        "block type": block,
                        "time_ms": t,
                        "direction": direction,
                        "micro_rate_Hz": rate_hz
                    })
            # after all windows for this trial:
            trial_rate_df = pd.DataFrame(rate_rows)

            # append to global list
            rate_all_trials.append(trial_rate_df)

rate_all_trials = pd.concat(rate_all_trials, ignore_index=True)            

                    
# average across trials within each subject & condition
plot_df = (rate_all_trials.groupby(["subject", "block type", "direction", "time_ms"])["micro_rate_Hz"]
    .mean().reset_index())



for block in ["attention", "expectation"]:
    block_df = plot_df[plot_df['block type']==block]
    plt.figure(figsize=(4,2.2))

    # draw line with SEM ribbon
    ax = sns.lineplot(data=block_df, x="time_ms", y="micro_rate_Hz",
       hue='direction', errorbar="se", lw=1)
    ax.set_ylim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward',5))
    ax.spines['bottom'].set_position(('outward', 5))
    plt.axvline(0, color="gray", linestyle="dotted", lw=1)  # optional event marker
    plt.axvline(466, color="gray", linestyle="dotted", lw=1)  # optional event marker
  
    if block == 'attention':
        plt.axvline(-950, color="gray", linestyle="--", lw=1)
        plt.text(-955,                # x-position (ms)
            plt.ylim()[1]*0.4,  # y-position (90 % of top axis)
            "Attention cue",
            ha="right", va="bottom",
            rotation=0, fontsize=8, color="black")

    plt.text(0,
        plt.ylim()[1]*0.8,
        "Stimulus onset",
        ha="right", va="bottom",
        rotation=0, fontsize=8, color="black")
    plt.text(466,
        plt.ylim()[1]*0.7,
        "Probe Cue",
        ha="right", va="bottom",
        rotation=0, fontsize=8, color="black")
    plt.xticks([-2000, -1500, -1000, -500, 0, 500],
               ["-2000", "-1500", "-1000", "-500", "0", "500"], fontsize=8)
    plt.yticks([0.0, 1.0, 2.0, 3.0],
               ["0.0", "1.0", "2.0", "3.0"], fontsize=8)
    plt.xlabel("Time relative to Stimulus Onset (ms)", fontsize= 8)
    plt.ylabel("Microsaccade Rate (Hz)", fontsize= 8)
    plt.title(f"{block} blocks", fontweight='bold', fontsize=10)
    if block == "attention":
        plt.legend(title="Direction", fontsize=8, title_fontsize=8, loc='upper left') 
    else:   
        plt.legend(title="Direction", fontsize=8, title_fontsize=8)
    plt.tight_layout()
    plt.show()
    
    