# Expectation Exerts Flexible and Context-Dependent Influence on Conscious Object Recognition

This repository contains code to reproduce the analyses and visualizations presented in the manuscript:  
**"Expectation Exerts Flexible and Context-Dependent Influence on Conscious Object Recognition"**.

Analyses were originally performed on a machine running **RedHat Linux v9.6** and **Python 3.12**.

---

## 🧪 Analysis

### Scripts for reproducing statistical results and generating figures in the paper:
- `BehaviorPreprocess.py`:      Converts raw PsychoPy output files into analysis-ready DataFrames.
-  `Stats_Exp1.py`:             Statistical analyses and source data generation for Experiment 1  
- `Stats_Exp2.py`:              Statistical analyses and source data generation for Experiment 2  
- `Stats_Exp3.py`:              Statistical analyses and source data generation for Experiment 3  
- `Plot_Exp1n3.py`:             Visualizations for Experiments 1 and 3  
- `Plot_Exp2.py`:               Visualizations for Experiment 2

### Scripts for reproducing oculomotor behavior reported in response letter to the reviewers
- `ET_ParseEyeLinkAsc.py`:      Converts raw Eyelink .edf files into analysis-ready DataFrames.
- `ET_Check_Fixation_Breaks.py`:Identify fixation breaks
- `ET_MicrosaccadeRate.py`:     Visualization of microsaccadid dynamics in Experiment 2
- `ET_PupilSize.py`:            Visualizagion of pipul diameter changes in Experiment 3

### Others:
- `GeneratStimuli`:                 Generate stimuli
- `Quest_Estimate.py`:              Obtain threshold estimate for each stimulus exemplar 
-  `EASTO_funcs`:                   Helper functions
- `EngbertMicrosaccadeToolbox`:     Toolbox for oculomotor behavior analysis
- `edf2asc`:                        edf-asc conversion function

---

## 📊 Data

This folder contains processed and source data used to generate all figures reported in the manuscript, all stored in pickle format
### Processed data:
- `SpatialParadigm_V3_bhv_df.pkl`:              Exp1 
- `SpatialParadigm_Control_bhv_df.pkl`:         Exp2 
- `SpatialParadigm_expControl_bhv_df.pkl`:      Exp3

### Source data: 
- `Exp1_SDT.pkl`:                               Source data corresponding to Fig 2A
- `Exp1_Categorization.pkl`:                    Source data correspoding to Fig 2B
- `Exp1_Confidence.pkl`:                        Source data corresponding to Fig 2C

- `Exp1_Recognition_By_NonTarget.pkl`:          Source data corresponding to Fig 3A              
- `Exp1_Categorization_by_NonTarget.pkl`:       Source data corresponding to Fig 3B

- `Exp2_attention_SDT.pkl`:                     Source data corresponding to Fig 5A-i
- `Exp2_attention_categorization.pkl`:          Source data corresponding to Fig 5A-ii
- `Exp2_expectation_SDT.pkl`:                   Source data corresponding to Fig 5B-i
- `Exp2_expectation_categorization.pkl`:        Source data corresponding to Fig 5B-ii

- `Exp3_non-neutral_SDT.pkl`:                   Source data corresponding to Fig 7A-i
- `Exp3_non-neutral_Categorization.pkl`:        Source data corresponding to Fig 7A-ii     
- `Exp3_neutral_SDT.pkl`:                       Source data corresponding to Fig 7B-i
- `Exp3_neutral_Categorization.pkl`:            Source data corresponding to Fig 7B-ii

- `Exp3_non-neutral_Recognition_By_NonTarget.pkl`:      Source data corresponding to Fig 8A-i              
- `Exp3_non-neutral_Categorization_by_NonTarget.pkl`:   Source data corresponding to Fig 8A-ii
- `Exp3_neutral_Recognition_By_NonTarget.pkl`:          Source data corresponding to Fig 8B-i              
- `Exp3_neutral_Categorization_by_NonTarget.pkl`:       Source data corresponding to Fig 8B-ii

- `Hit_contingent table.csv`:                           Table S1 in supplementary data 
- `FA_contingent table.csv`:                            Table S2 in supplementary data

- `JASP files`:                                         data stored in .csv formatted for JASP                           

---

## Paradigm Code
contain everything needed to run the behavioral experiments for the Exp_1 (Main), Exp_2 (Control), and Exp_3 (expControl) paradigms.
- `Spatial`:                   Psychopy Code necessary to run Experiment 1 and 2
- `Spatial_ExpControl`:        Psychopy code necessary to run Experiment 3
