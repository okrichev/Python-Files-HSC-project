#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:18:50 2019
s
@author: Oleg Krichevsky okrichev@bgu.ac.il
"""


import sys
sys.path.append('/Users/oleg/Documents/Python programming/NYU/')

import pandas as pd
import numpy as np
#import matplotlib as mpl
#mpl.use('GTKAgg') # to use GTK UI
import matplotlib.pyplot as plt
#import random
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.decomposition import PCA 
#from scipy.special import binom 
#from scipy.special import erf
from scipy.signal import medfilt #median filter 
from scipy.signal import savgol_filter # savitsky golay filter
import scipy.io as sio
import pickle

#from CellTrackModule import CellTrackClass
from CellTrackModule import CellTrackArrayClass
#from CellTrackModule import DoContourWritheCalc
#from CellTrackModule import CellTrackMatchClass
#from CellTrackModule import ContourLength
from CellTrackModule import saveObject
from CellTrackModule import loadObject
from CellTrackModule import GetStats
#import importlib
#import FitToolModule #import the module here, so that it can be reloaded.
#from FitToolModule import FitTool
#import FitToolModule.FitTool

#%%
ExcelFilePath = '/Users/oleg/Documents/Experiments/NYU/HSC data summary/intravital measurements_041319.xlsx'
trks = CellTrackArrayClass()
trks.DoLoadDataFromExcelFile(ExcelFilePath)
trks.DoSortTracksByParam('dt_s') 
HSC = trks.DoGetCellTrackSubarrayByParamValue({'cellType' : 'hsc'})
MPHG = trks.DoGetCellTrackSubarrayByParamValue({'cellType' : 'macrophage'})

#%%
HSC.DoCalculateMSD()
MPHG.DoCalculateMSD()


#%% first macrophages
# Find preliminary stretches
Rerror = np.sqrt(0.34**2 + 0.25**2) #um from the analysis of immobile cells imaaged at high rate
#Rerror = 1
MPHG.DoFindDirectedMotion(maxProbability = 0.05, minSteps = 3, 
                    #Rerror = np.sqrt(0.34**2 + 0.25**2),
                    Rerror = Rerror,
                       showPlots = False, RefineProb = False)

#%%
MPHG.DoShowTrackStretches()

#%%
MPHG.DoFindConfinements(MinStretchLength = 10, maxProbability = 0.05,
                     #Rerror = np.sqrt(0.34**2 + 0.25**2), 
                     Rerror = Rerror,
                     showPlots = True)


#%%
MPHG.DoShowConfinements(MinStretchLength = 10, maxProbability = 0.05,
                     Rerror = Rerror, showPlots = True)


#%% get statistics on D estimation
max_dt_s = 200
D = []
for trk in MPHG.tracks:
  #  print(trk.CompleteMetaName)
  if (trk.dt_s < max_dt_s) :
    if len(trk.ConfinementList) > 0:
        D.append(trk.ConfinementList[0]['Average Diff Coeff (um^2/min)'])
        
D = np.array(D)
print(D.size)
print(np.median(D))
#%%
H, bins = np.histogram(D, bins = 15, range=(0, 3))

plt.cla()
plt.plot((bins[:-1]+bins[1:])/2, H, '-o', label = 'Diff coeff')
plt.xlabel('D (um^2/min)')
plt.ylabel('Histogram')
plt.legend(loc="best")
plt.show()
 
#%% now with Dlims
medianD = np.median(D)
limFactor = 2
MPHG.DoFindDirectedMotion(maxProbability = 0.005, minSteps = 3, 
                    Rerror = Rerror,
                       showPlots = False, RefineProb = False,
                       Dsource = 'confinements', Dlims = (medianD/limFactor, medianD, medianD*limFactor))


#%%
MPHG.DoShowTrackStretches()

#%%
MPHG.DoFindConfinements(MinStretchLength = 10, maxProbability = 0.05,
                     Rerror = Rerror, showPlots = True,
                       Dlims = (medianD/limFactor, medianD, medianD*limFactor))


#%%
MPHG.DoShowConfinements(MinStretchLength = 10, maxProbability = 0.05,
                     Rerror = np.sqrt(0.34**2 + 0.25**2), showPlots = True,
                      Dlims = (medianD/limFactor, medianD, medianD*limFactor))

#%%
MPHG.DoShowTrackDirectedAndConfined()
#%% save macrophage object

SaveFolder = '/Users/oleg/Documents/Experiments/NYU/HSC data summary/'
fname = SaveFolder +'MPHG.pkl'
saveObject(MPHG, fname)

#%%
print(len(MPHG.tracks))

#%% example of loading
MPHG = loadObject(MPHGfname)

#%% run the routine with all HSC then separate before and after drug
#%Find preliminary stretches
HSC.DoFindDirectedMotion(maxProbability = 0.05, minSteps = 3, 
                    Rerror = Rerror,
                       showPlots = False, RefineProb = False)

#%%
HSC.DoShowTrackStretches()

#%%
HSC.DoFindConfinements(MinStretchLength = 10, maxProbability = 0.05,
                     Rerror = Rerror, showPlots = True)


#%%
HSC.DoShowConfinements(MinStretchLength = 10, maxProbability = 0.05,
                     Rerror = np.sqrt(0.34**2 + 0.25**2), showPlots = True)


#%% get statistics on D estimation
max_dt_s = 1000
D = []
for trk in HSC.tracks:
  #  print(trk.CompleteMetaName)
  if (trk.dt_s < max_dt_s) :
    if len(trk.ConfinementList) > 0:
        D.append(trk.ConfinementList[0]['Average Diff Coeff (um^2/min)'])
        
D = np.array(D)
print(D.size)
print(np.median(D))
#%%
H, bins = np.histogram(D, bins = 15, range=(0, 3))

plt.cla()
plt.plot((bins[:-1]+bins[1:])/2, H, '-o', label = 'Diff coeff')
plt.xlabel('D (um^2/min)')
plt.ylabel('Histogram')
plt.legend(loc="best")
plt.show()
 
#%% now with Dlims
medianD = np.median(D)
limFactor = 2
HSC.DoFindDirectedMotion(maxProbability = 0.005, minSteps = 3, 
                    Rerror = np.sqrt(0.34**2 + 0.25**2),
                       showPlots = False, RefineProb = False,
                       Dsource = 'confinements', Dlims = (medianD/limFactor, medianD, medianD*limFactor))


#%%
HSC.DoShowTrackStretches()

#%%
HSC.DoFindConfinements(MinStretchLength = 10, maxProbability = 0.05,
                     Rerror = np.sqrt(0.34**2 + 0.25**2), showPlots = True,
                       Dlims = (medianD/limFactor, medianD, medianD*limFactor))


#%%
HSC.DoShowConfinements(MinStretchLength = 10, maxProbability = 0.05,
                     Rerror = np.sqrt(0.34**2 + 0.25**2), showPlots = True,
                      Dlims = (medianD/limFactor, medianD, medianD*limFactor))

#%%
HSC.DoShowTrackDirectedAndConfined()

#%%
SaveFolder = '/Users/oleg/Documents/Experiments/NYU/HSC data summary/'
fname = SaveFolder +'HSC.pkl'
saveObject(HSC, fname)


#%%
SaveFolder = '/Users/oleg/Documents/Experiments/NYU/HSC data summary/'
fname = SaveFolder +'HSC.pkl'
HSC = loadObject(fname)
#%% how many have directed strethces, how many have confined and how many both

#%%
HSCbeforedrug = HSC.DoGetCellTrackSubarrayBySubstring({'metaname' : ['PreRx', 'preRx', 'Pre']})
#%%

fname = SaveFolder +'HSCbeforedrug.pkl'
saveObject(HSCbeforedrug, fname)

#%%
HSCafterdrug = HSC.DoGetCellTrackSubarrayBySubstring({'metaname' : ['PostRx', 'postRx', 'Post']})
#%%
fname = SaveFolder +'HSCafterdrug.pkl'
saveObject(HSCafterdrug, fname)

#%% exclude post Rx data
print(len(HSC.tracks))
HSCnodrug= HSC.DoExcludeCellTracksBySubstring({'metaname' : 'PostRx'})
HSCnodrug= HSCnodrug.DoExcludeCellTracksBySubstring({'metaname' : 'postRx'})
HSCnodrug= HSCnodrug.DoExcludeCellTracksBySubstring({'metaname' : 'Post'})
print(len(HSCnodrug.tracks))
fname = SaveFolder +'HSCnodrug.pkl'
saveObject(HSCnodrug, fname)

#%%
HSCnodrug_motile = HSC.DoGetCellTrackSubarrayByParamValue({'maxDist': 20}, operation = 'greater')
#%%
fname = SaveFolder +'HSCnodrug_motile.pkl'
saveObject(HSCnodrug_motile, fname)
#%%
print(len(HSCnodrug_motile.tracks))

#%%
maxDist = np.array([trk.maxDist for trk in HSCnodrug_motile.tracks])
print(min(maxDist))

#%%
HSCnodrug.DoCalculateMSD()

#%%
CellArray = HSCnodrug
print(len(CellArray.tracks))
HD = [len(trk.DirectStretchList) > 0 for trk in CellArray.tracks]
HC = [len(trk.ConfinementList) > 0 for trk in CellArray.tracks]
HaveDirected = np.array(HD).sum()
print(HaveDirected)
HaveConfined = np.array(HC).sum()
print(HaveConfined)
HaveBoth = (np.array(HC) & np.array(HD)).sum()
print(HaveBoth)
print(HaveBoth/HaveDirected)
print(HaveBoth/HaveConfined)

#%%
distanceThreshold = 20
dispacementAboveThreshold = [(trk.maxDist > distanceThreshold) for trk in CellArray.tracks]
print(dispacementAboveThreshold)
#%%
HaveDirectedAndDisplacement = (np.array(dispacementAboveThreshold) & np.array(HD)).sum()
print(HaveDirectedAndDisplacement/HaveDirected)
print(HaveDirectedAndDisplacement/np.array(dispacementAboveThreshold).sum())

#%% 
print(len(dispacementAboveThreshold))
#
#%%
HSCbeforedrug.DoShowTrackDirectedAndConfined()


#%%
HSCafterdrug.DoShowTrackDirectedAndConfined()


#%% load population data
D = pd.read_excel('/Users/oleg/Documents/Experiments/NYU/HSC data summary/tomato+ cell counts_intravital biopsiesNumeric names.xlsx')
#%%

fracAboveThresh = [] 
meanVar = []
HSC = [] 
STHSC = []
MPP2 = []
MPP34 = []
MyP = []
MEP = []
ScaPcKitN = []
ScaNcKitN = []
Lin = []
AnimalID = []


animalIDs = [trk.animalID for trk in HSCnodrug.tracks]
animalIDs = np.array(animalIDs)

expl_var = attr_HSCnodrug

print(len(animalIDs))

animalIDunique= set(animalIDs)

varByID = dict()
for aID in animalIDunique:
    varByID[aID] = expl_var[animalIDs == aID]    
    
for aID in varByID:
    AnimalID.append(aID)
    meanVar.append(np.mean(varByID[aID]))
    fracAboveThresh.append(np.sum(varByID[aID] > Thresh)/varByID[aID].size)
    HSC.append(D.loc[D['Animal ID'] == aID]['HSC'].values[0])
    STHSC.append(D.loc[D['Animal ID'] == aID]['ST-HSC'].values[0])
    MPP2.append(D.loc[D['Animal ID'] == aID]['MPP2'].values[0])
    MPP34.append(D.loc[D['Animal ID'] == aID]['MPP3/4'].values[0])
    MyP.append(D.loc[D['Animal ID'] == aID]['MyP'].values[0])
    MEP.append(D.loc[D['Animal ID'] == aID]['MEP'].values[0])
    ScaPcKitN.append(D.loc[D['Animal ID'] == aID]['Sca+ cKit-'].values[0])
    ScaNcKitN.append(D.loc[D['Animal ID'] == aID]['Sca- cKit-'].values[0])
    Lin.append(D.loc[D['Animal ID'] == aID]['Lin+'].values[0])

meanVar = np.array(meanVar)  
fracAboveThresh = np.array(fracAboveThresh) 
HSC = np.array(HSC)
STHSC = np.array(STHSC)
MPP2 = np.array(MPP2)
MPP34 = np.array(MPP34)
MyP = np.array(MyP)
MEP = np.array(MEP)
ScaPcKitN = np.array(ScaPcKitN)
ScaNcKitN = np.array(ScaNcKitN)
Lin = np.array(Lin)
AnimalID = np.array(AnimalID)

AllPops = HSC + STHSC + MPP2 + MPP34 + MyP + MEP + ScaPcKitN + ScaNcKitN + Lin
#%%
plt.cla()
plt.plot(HSC/AllPops, meanVar, 'o')
plt.ylabel('Mean Fraction of contour in directed motion')
plt.xlabel('HSC Fraction')
plt.show()
np.corrcoef(HSC/AllPops, meanVar)

#%%
HSCfrac = HSC/AllPops
plt.cla()
plt.plot(HSCfrac, fracAboveThresh, 'o')
plt.show()
np.corrcoef(HSCfrac, fracAboveThresh)
#%%
print(np.corrcoef(STHSC/AllPops, meanVar))
#%%
print(np.corrcoef(MPP2/AllPops, meanVar))

#%%
print(np.corrcoef(MPP34/AllPops, meanVar))

#%%
print(np.corrcoef(MyP/AllPops, meanVar))

#%%
print(np.corrcoef(MEP/AllPops, meanVar))

#%%
print(np.corrcoef(ScaPcKitN/AllPops, meanVar))

#%%
print(np.corrcoef(ScaNcKitN/AllPops, meanVar))


#%% prepare coordinates for output to matlab
metanameTmpl = '2454_a5'
HSC2454a5 = HSC.DoGetCellTrackSubarrayBySubstring({'CompleteMetaName' : metanameTmpl})
print(HSC2454a5.metanameList)

#%%
pre = HSC2454a5.DoGetCellTrackSubarrayBySubstring({'CompleteMetaName' : ['PreRx', 'preRx', 'Pre']})
print(pre.metanameList)
post = HSC2454a5.DoGetCellTrackSubarrayBySubstring({'CompleteMetaName' : ['PostRx', 'postRx', 'Post']})
print(post.metanameList)

#%% save relevant data to matlab file
trk = pre.tracks[0]
trk.DoSaveDataToMatlabFile('/Users/oleg/Documents/Experiments/NYU/2454a5/2454a5_pre_data/')
#%%
trk = post.tracks[1]
trk.DoSaveDataToMatlabFile('/Users/oleg/Documents/Experiments/NYU/2454a5/2454a5_post_data')


#%% prepare coordinates for output to matlab 2454a3
metanameTmpl = '2454_a3'
HSCanimal = HSC.DoGetCellTrackSubarrayBySubstring({'CompleteMetaName' : metanameTmpl})
print(HSCanimal.metanameList)

#%%
pre = HSCanimal.DoGetCellTrackSubarrayBySubstring({'CompleteMetaName' : ['PreRx', 'preRx', 'Pre']})
print(pre.metanameList)
post = HSCanimal.DoGetCellTrackSubarrayBySubstring({'CompleteMetaName' : ['PostRx', 'postRx', 'Post']})
print(post.metanameList)

#%% save relevant data to matlab file
trk = pre.tracks[0]
trk.DoSaveDataToMatlabFile('/Users/oleg/Documents/Experiments/NYU/2454a3/2454a3_pre_data')
#%%
trk = post.tracks[1]
trk.DoSaveDataToMatlabFile('/Users/oleg/Documents/Experiments/NYU/2454a3/2454a3_post_data')

#%% 
print(len(pre.tracks[0].DirectStretchList))

#%%
pre.DoShowTrackDirectedAndConfined()

#%% Get statistics on those that do not exhibit processive motion
fname = '/Users/oleg/Documents/Experiments/NYU/HSC data summary/HSCnodrug.pkl'
HSCnodrug = loadObject(fname)
HSCnoPM = CellTrackArrayClass()
HSCnoPM.tracks = [trk for trk in HSCnodrug.tracks if len(trk.DirectStretchList) == 0]
print(len(HSCnoPM.tracks))

#%%
HSCnoPM.DoShowTrackDirectedAndConfined()

#%% Use GetStats function
maxDist = np.array([trk.maxDist for trk in HSCnoPM.tracks])
stats = GetStats(maxDist)
print(stats)



#%%
Vxydispl_um_min = np.array([trk.Vxydispl_um_min  for trk in HSCnoPM.tracks])
stats = GetStats(Vxydispl_um_min)
print(stats)


#%% statistics for those with PM
HSC_PM = CellTrackArrayClass()
HSC_PM.tracks = [trk for trk in HSCnodrug.tracks if len(trk.DirectStretchList) > 0]
print(len(HSC_PM.tracks))


#%%
maxDist = np.array([trk.maxDist for trk in HSC_PM.tracks])
stats = GetStats(maxDist)
print(stats)


#%%
Vxydispl_um_min = np.array([trk.Vxydispl_um_min  for trk in HSC_PM.tracks])
stats = GetStats(Vxydispl_um_min)
print(stats)

#%%
V = []
for trk in HSC_PM.tracks:
    for DS in trk.DirectStretchList:
        V.append(DS['Displacement Speed (um/min)'])

stats = GetStats(V)
print(stats)


#%%
PMlength = []
for trk in HSC_PM.tracks:
    for DS in trk.DirectStretchList:
        PMlength.append(DS['Stretch Length (um)'])

stats = GetStats(PMlength)
print(stats)

#%%
PMtime = []
for trk in HSC_PM.tracks:
    for DS in trk.DirectStretchList:
        PMtime.append(DS['Stretch Time (s)']/60)

stats = GetStats(PMtime)
print(stats)


#%%
CRWtime = []
for trk in HSCnodrug.tracks:
    for CL in trk.ConfinementList:
        CRWtime.append(CL['Stretch Time (s)']/60)

stats = GetStats(CRWtime)
print(stats)

#%%
CRWdisp = []
for trk in HSCnodrug.tracks:
    for CL in trk.ConfinementList:
        CRWdisp.append(CL['Max Displacement (um)'])

stats = GetStats(CRWdisp)
print(stats)
#%% all HSCs

Vxydispl_um_min = np.array([trk.Vxydispl_um_min  for trk in HSCnodrug.tracks])
stats = GetStats(Vxydispl_um_min)
print(stats)


