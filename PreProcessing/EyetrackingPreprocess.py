# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:32:59 2021

@author: bc1693
"""
#%% Load Packages
import pandas as pd 
import numpy as np 
import time
import os 
import cloudpickle as pickle
import glob
import itertools

#%% Main Function
def checkBrokenFixation(df, subjects, datadir, pixBuffer = 60, timeBuffer = 20, save = True):
    nBrokeFix = {sub : [] for sub in subjects}
    for subject in subjects: 

        #Load subject eyetracking datafile 
        elFilename = glob.glob(datadir + os.sep + 'Data_local/' + expt_version + os.sep + f'Subjects/{subject}/' + paradigm + os.sep + 'Main/edfData/*.asc')[0]

        print('Reading in EyeLink file %s...'%elFilename)
        t = time.time()
        f = open(elFilename,'r')
        fileTxt0 = f.read().splitlines(True) # split into lines
        fileTxt0 = list(filter(None, fileTxt0)) #  remove emptys
        fileTxt0 = np.array(fileTxt0) # convert to np array for simpler indexing
        f.close()
        print('Done! Took %f seconds.'%(time.time()-t))

        nLines = len(fileTxt0)

        # Separate lines into samples and messages
        print('Sorting lines...')
        nLines = len(fileTxt0)
        lineType = np.array(['OTHER']*nLines,dtype='object')
        iStartRec = None
        t = time.time()
        for iLine in range(nLines):
            if len(fileTxt0[iLine])<3:
                lineType[iLine] = 'EMPTY'
            elif fileTxt0[iLine].startswith('*') or fileTxt0[iLine].startswith('>>>>>'):
                lineType[iLine] = 'COMMENT'
            # each proper sample should have same # of entries(cols) may not be the same if asc is not generated the same way 
            elif fileTxt0[iLine].split()[0][0].isdigit() and len(fileTxt0[iLine].split()) >= 10: 
                lineType[iLine] = 'SAMPLE'
            else:
                lineType[iLine] = fileTxt0[iLine].split()[0]
            if '!CAL' in fileTxt0[iLine]: # TODO: Find more general way of determining if recording has started
                iStartRec = iLine+1
        print('Done! Took %f seconds.'%(time.time()-t))

    
        # ===== PARSE EYELINK FILE ===== #
        t = time.time()
        # Trials
        print('Parsing recording markers...')
        iNotStart = np.nonzero(lineType!='START')[0]
        dfRecStart = pd.read_csv(elFilename,skiprows=iNotStart,header=None,delim_whitespace=True,usecols=[1])
        dfRecStart.columns = ['tStart']
        iNotEnd = np.nonzero(lineType!='END')[0]
        dfRecEnd = pd.read_csv(elFilename,skiprows=iNotEnd,header=None,delim_whitespace=True,usecols=[1,5,6])
        dfRecEnd.columns = ['tEnd','xRes','yRes']
        # combine trial info
        dfRec = pd.concat([dfRecStart,dfRecEnd],axis=1)
        nRec = dfRec.shape[0]
        print('%d recording periods found.'%nRec)

        # Import Messages
        print('Parsing stimulus messages...')
        t = time.time()
        iMsg = np.nonzero(lineType=='MSG')[0]
        # set up
        tMsg = []
        txtMsg = []
        t = time.time()
        for i in range(len(iMsg)):
            # separate MSG prefix and timestamp from rest of message
            info = fileTxt0[iMsg[i]].split()
            # extract info
            tMsg.append(int(float(info[1])))
            txtMsg.append(' '.join(info[2:]))
        # Convert dict to dataframe
        dfMsg = pd.DataFrame({'time':tMsg, 'text':txtMsg})
        print('Done! Took %f seconds.'%(time.time()-t))
        
        # Import Fixations
        print('Parsing fixations...')
        t = time.time()
        iNotEfix = np.nonzero(lineType!='EFIX')[0]
        dfFix = pd.read_csv(elFilename,skiprows=iNotEfix,header=None,delim_whitespace=True,usecols=range(1,8))
        dfFix.columns = ['eye','tStart','tEnd','duration','xAvg','yAvg','pupilAvg']
        nFix = dfFix.shape[0]
        print('Done! Took %f seconds.'%(time.time()-t))

        # # Saccades
        # print('Parsing saccades...')
        # t = time.time()
        # iNotEsacc = np.nonzero(lineType!='ESACC')[0]
        # dfSacc = pd.read_csv(elFilename,skiprows=iNotEsacc,header=None,delim_whitespace=True,usecols=range(1,11))
        # dfSacc.columns = ['eye','tStart','tEnd','duration','xStart','yStart','xEnd','yEnd','ampDeg','vPeak']
        # print('Done! Took %f seconds.'%(time.time()-t))
        
        # # Blinks
        # print('Parsing blinks...')
        # iNotEblink = np.nonzero(lineType!='EBLINK')[0]
        # dfBlink = pd.read_csv(elFilename,skiprows=iNotEblink,header=None,delim_whitespace=True,usecols=range(1,5))
        # dfBlink.columns = ['eye','tStart','tEnd','duration']
        # print('Done! Took %f seconds.'%(time.time()-t))
        
        # determine sample columns based on eyes recorded in file
        eyesInFile = np.unique(dfFix.eye)
        if eyesInFile.size==2:
            print('binocular data detected.')
            cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
        else:
            eye = eyesInFile[0]
            print('monocular data detected (%c eye).'%eye)
            cols = ['tSample', '%cX'%eye, '%cY'%eye, '%cPupil'%eye]
        # Import samples    
        print('Parsing samples...')
        t = time.time()
        iNotSample = np.nonzero(lineType!='SAMPLE')[0]
        dfSamples = pd.read_csv(elFilename,skiprows=iNotSample,header=None,delim_whitespace=True,
                                usecols=range(0,len(cols)))
        dfSamples.columns = cols
        # Convert values to numbers
        for eye in ['L','R']:
            if eye in eyesInFile:
                dfSamples['%cX'%eye] = pd.to_numeric(dfSamples['%cX'%eye],errors='coerce')
                dfSamples['%cY'%eye] = pd.to_numeric(dfSamples['%cY'%eye],errors='coerce')
                dfSamples['%cPupil'%eye] = pd.to_numeric(dfSamples['%cPupil'%eye],errors='coerce')
            else:
                dfSamples['%cX'%eye] = np.nan
                dfSamples['%cY'%eye] = np.nan
                dfSamples['%cPupil'%eye] = np.nan
                
        print('Done! Took %.1f seconds.'%(time.time()-t))
    

        onsetTime = dfMsg.loc[dfMsg.text.str.contains('attention_cue_onset')].reset_index(drop = True).rename(columns = {'time':'onset', 'text' : 'onset_event'})
        offsetTime = dfMsg.loc[dfMsg.text.str.contains('poststim_fix')].reset_index(drop = True).rename(columns = {'time':'offset', 'text' : 'offset_event'})
        checkFixationPeriod = pd.concat([onsetTime, offsetTime], axis = 1)

        # brokeFixation = pd.DataFrame({
        #     'subject' : subject,
        #     'trial': range(len(onsetTime)),
        #     'dropTrial' : np.nan})
        print(f'Checking Broken Fixations for Subject {subject}')
        blockTrials = itertools.product(df.loc[df.subject == subject].block.unique(), df[df.subject == subject].trial.unique())
        for iter, (block, trial) in enumerate(blockTrials):
            #Select the relevant samples to check for fixation 
            samples = dfSamples.loc[dfSamples.tSample.between(checkFixationPeriod.onset[iter], checkFixationPeriod.offset[iter])]
            LXSamplesFix = samples.LX.between(960 - pixBuffer, 960 + pixBuffer)
            LYSamplesFix = samples.LY.between(540 - pixBuffer, 540 + pixBuffer)
            RXSamplesFix = samples.RX.between(960 - pixBuffer, 960 + pixBuffer)
            RYSamplesFix = samples.RY.between(540 - pixBuffer, 540 + pixBuffer)
            
            LFix = LXSamplesFix * LYSamplesFix # Require fixation for each eye for both horizontal AND vertical axes
            RFix = RXSamplesFix * RYSamplesFix
            
            # Using OR for the cases that one eye drops out due to poor eyetracking. 
            numSamplesBrokeFix = sum(LFix + RFix)
            
            if numSamplesBrokeFix <= len(samples) - timeBuffer: 
                df.loc[(df.subject == subject) & (df.block == block) & (df.trial == trial), 'brokeFixation'] = 1               
            # #Check for blinks anytime between onset of att_cue and offset of stimulus
            # if dfBlink.tStart.between(checkFixationPeriod.iloc[iter].onset, checkFixationPeriod.iloc[iter].offset).any():
                # df.loc[(df.subject == subject) & (df.block == block) & (df.trial == trial), 'brokeFixation'] = 1               
                # continue
            # # Check for saccades during the same time * eyelink nests blink events within in saccade events
            # elif dfSacc.tStart.between(checkFixationPeriod.iloc[iter].onset, checkFixationPeriod.iloc[iter].offset).any():
            #     df.loc[(df.subject == subject) & (df.block == block) & (df.trial == trial), 'brokeFixation'] = 1      
            #     continue
            # Check if fixating at center of the screen *Maybe add later
            # Can only go based on the avg X & Y position on the screen. Best done online during experiment
            else:
                df.loc[(df.subject == subject) & (df.block == block) & (df.trial == trial), 'brokeFixation'] = 0       
        nBrokeFix[subject] = df.loc[df.subject == subject].brokeFixation.sum()
        print(f'Finished: Subject {subject} broke fixation {nBrokeFix[subject]} times')
    if save:         
        # Save Dataframe as .pkl 
        df.to_pickle(datadir + os.sep + paradigm + expt_version + '_bhv_df.pkl')

    return df, nBrokeFix


#%% Load Behavioral Data 
curComp = 'Mac'
if curComp == 'VM':
    datadir = '/isilon/LFMI/VMdrive/Brandon/EASTO-local/'
else:
    datadir = '/Users/brandonchen93/Downloads'
os.chdir(datadir)

paradigm = 'Spatial'
expt_version = 'Paradigm_V3'
sub_version = ''
fname = paradigm + expt_version + sub_version+ '_bhv_df.pkl'

with open(fname, "rb") as input_file:
    bhv_df = pickle.load(input_file)

#%%Add new column for whether subject broke fixation  
bhv_df['brokeFixation'] = np.nan
subjects = bhv_df.subject.unique()

beye_df,nBrokeFixations = checkBrokenFixation(bhv_df, subjects, datadir, save = True)

