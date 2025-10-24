# Load Packages
import pandas as pd
from os.path import join
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
RootDir = '/isilon/LFMI/VMdrive/YuanHao/EASTO-Behavior'
sys.path.append(join(RootDir, 'EngbertMicrosaccadeToolbox'))
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
exp_version = 'Paradigm_expControl' #['Paradigm_V3', 'Paradigm_Control', 'Paradigm_expControl']
paradigm = 'Spatial'
sub_version = ''
bhv_df = pd.read_pickle(join(DataDir, f"{paradigm}{exp_version}_bhv_df.pkl")) 

rate_all_trials = []
win_size = 100
step_size = 10

for subj in ['P57', 'P58', 'P59', 'P63']:
    ET_Dir = join(DataDir, exp_version, subj, paradigm, 'Main', 'edfData')
    ET_filename = f"{subj}_eye_tracking_data.pkl"
    subj_df = bhv_df[(bhv_df['subject']==subj)]
    subj_df_sorted = subj_df.sort_values(['block', 'trial'])
   
    dfRec, dfMsg, dfFix, dfSacc, dfBlink, dfSamples = load_ET_data(ET_Dir, ET_filename)
    eye = get_dominant_eye(dfFix)

     # 'attention_cue_onst', "init_fix"
    stim_onsets = dfMsg.loc[dfMsg.text.str.contains('stim_onset')].reset_index(drop=True).rename(columns={'time':'onset','text':'onset_event'})
    onsetTime = stim_onsets['onset']-1950
    offsetTime = stim_onsets['onset']+700; offsetTime = offsetTime.rename('offset')
    Period_Interest = pd.concat([onsetTime, offsetTime], axis=1)


    for block in ['LE', 'GE']:
        # --- build block-specific trial data ---
        if block =='LE':
            cond_mask = (subj_df_sorted['exp_color']!='Neutral') & (subj_df_sorted['brokeFixation']==0)
        elif block =='GE':
            cond_mask = (subj_df_sorted['exp_color']=='Neutral') & (subj_df_sorted['brokeFixation']==0)
        
        cond_df = subj_df_sorted[cond_mask].reset_index(drop=True)
        timestamp = Period_Interest[cond_mask.to_numpy()].reset_index(drop=True)

    
        for idx in range(len(cond_df)):
            onset = timestamp['onset'][idx]
            offset = timestamp['offset'][idx]    
            if pd.isna(onset) or pd.isna(offset):
                continue
            #trial_dur = offset -onset
            win_onsets = np.arange(-1950, 500, step_size)
            
            trial_samples = dfSamples.loc[dfSamples['tSample'].between(onset, offset)]
            pupil_z = trial_samples[f'{eye}Pupil'].transform(lambda x: (x - x.mean()) / x.std())
            pupil_z = pupil_z.to_frame()
            pupil_z['time to stim'] = np.arange(-1950, 701,1) 
            # onset times of detected binocular microsaccades (relative to attention cue onset)
            
            # store per-window rate
            rate_rows = []
                        
            for t in win_onsets:
                start_t = t
                end_t   = t + win_size - 1
                window_mask = pupil_z['time to stim'].between(start_t, end_t)
                window_pupil_z = pupil_z.loc[window_mask, f'{eye}Pupil']
 
                rate_rows.append({
                        "trial": idx,
                        "subject": subj,
                        "block type": block,
                        "time_ms": t,
                        "pupil size": window_pupil_z.mean()
                    })
            # after all windows for this trial:
            trial_rate_df = pd.DataFrame(rate_rows)

            # append to global list
            rate_all_trials.append(trial_rate_df)

rate_all_trials = pd.concat(rate_all_trials, ignore_index=True)            

                    
# average across trials within each subject & condition
plot_df = (rate_all_trials.groupby(["subject", "block type", "time_ms"])["pupil size"]
    .mean().reset_index())


plt.figure(figsize=(8,4))
# draw line with SEM ribbon
ax = sns.lineplot(data=plot_df, x="time_ms", y="pupil size",
    hue='block type', errorbar="se", lw=2)
#ax.set_ylim([0, 3])
plt.axvline(-950, color="gray", linestyle="dotted", lw=1)
plt.axvline(0, color="gray", linestyle="dotted", lw=1)  # optional event marker
plt.axvline(466, color="gray", linestyle="dotted", lw=1)  # optional event marker

plt.text(-960,                # x-position (ms)
    plt.ylim()[1]*0.7,  # y-position (90 % of top axis)
    "Attention cue",
    ha="right", va="bottom",
    rotation=0, fontsize=10, color="black")

plt.text(-10,
    plt.ylim()[1]*0.7,
    "Stimulus onset",
    ha="right", va="bottom",
    rotation=0, fontsize=10, color="black")
plt.text(456,
    plt.ylim()[1]*0.8,
    "Probe Cue",
    ha="right", va="bottom",
    rotation=0, fontsize=10, color="black")
plt.xlabel("Time relative to Stimulus Onset (ms)")
plt.ylabel("Normalized Pupil dilation")

plt.tight_layout()
plt.show()
    
    