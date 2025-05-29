#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:06:59 2024

@author: isabella
"""


import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scienceplots
import seaborn as sns
import matplotlib as mpl
plt.style.use(['science','nature']) # sans-serif font
plt.close('all')

plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('axes', labelsize=10)
sizeL=10
params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}

plt.rcParams.update(params)


mpl.rcParams['text.usetex'] = False  # Disable LaTeX
mpl.rcParams['axes.unicode_minus'] = True  # Ensure minus sign is rendered correctly


cmap=sns.color_palette("colorblind")


track_all = np.zeros((10,20))

for dm in range(10):
        
    df_Ori = pd.read_csv('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/timeseries/10_lines_stack/GraFT/10_lines_0{0}/0_df_line_comparison.csv'.format(dm))
    
    df_Ori.columns
    
    uframe = np.unique(df_Ori['frame'])
    trackList = np.zeros(len(uframe))
    
    # remove all places where we look at the true that has not been matched with defined
    dflInesI = df_Ori[pd.notna(df_Ori['match index'])]

    overlapList = np.zeros(len(uframe))
    
    for n in range(len(uframe)):
        
        dflInesICov = dflInesI[dflInesI['frame']== n].copy()
        dflInesICov[['match index','overlap ratio']]

     
        dflInesICov['FS_coverage'] = dflInesICov['overlap ratio']#/dflInesICov['true_len']
        
        dflInesICov['FS_coverage'] = dflInesICov['FS_coverage'].fillna(0)
        
        overlapList[n] = np.median(dflInesICov['overlap ratio'])
        
        
    
        # for the first frame we need to save the correct matches
        if(n==0):
            matchedVals=[]
            matchVal = dflInesICov['match index'].dropna()    
            for kl,val in zip(range(len(matchVal)),matchVal):
                vlas = dflInesICov[['match index','overlap ratio']][(dflInesICov['match index'] == val)]
                if(vlas['overlap ratio']>0.8).all():
                    matchedVals.append(vlas['match index'].item())
            trackList[n] = len(matchedVals)/5
                    
        else:
            if(len(matchedVals)!=5):
                
                matchedValsInter=[]
                matchValInter = dflInesICov['match index'].dropna()    
                for kl,val in zip(range(len(matchValInter)),matchValInter):
                    vlas = dflInesICov[['match index','overlap ratio']][(dflInesICov['match index'] == val)]
                    if(vlas['overlap ratio']>0.8).all():
                        matchedValsInter.append(vlas['match index'].item())
                
                # search for the new values appearing
                mvi = set(matchedVals + matchedValsInter)
                if(len(mvi)>=5):
                    matchedVals=list(mvi)
                else:
                    print('error', n)
                    
            result = np.zeros(len(matchedVals))
            for kl,val in zip(range(len(matchedVals)),matchedVals):
                # Apply the condition: 'best_IDmatch_line1_id' equals the current value and 'overlap ratio' > 0.8
                if(dflInesICov['match index'] == val).any():
                    result[kl] = float(dflInesICov['overlap ratio'][(dflInesICov['match index'] == val)].iloc[0])
                else:
                    result[kl] =0
         
            trackList[n] = (result[(result >0.8)]).size/5
            
    track_all[dm] = trackList


plt.figure(figsize=(8.27,5))
for i in range(10):
    plt.plot(track_all[i], color='red', alpha=0.1)  # Plot each array with alpha=0.3

mean_array = track_all.mean(axis=0)

# Plot the mean array with alpha=1
plt.plot(mean_array, color='red', alpha=1, label='GraFT Mean')

plt.savefig('/home/isabella/Documents/PLEN/dfs/others_code/TSOAX/timeseries/10_lines_stack/figs/graft_performance.png')


track_all_TSOAX = np.load('/home/isabella/Documents/PLEN/dfs/others_code/TSOAX/timeseries/10_lines_stack/TSOAX_lines_track.csv.npy')

for i in range(10):
    plt.plot(track_all_TSOAX[i], color='green', alpha=0.1)  # Plot each array with alpha=0.3

mean_arrayTSOAX = track_all_TSOAX.mean(axis=0)
plt.plot(mean_arrayTSOAX, color='green', alpha=1, label='TSOAX Mean')

plt.xlabel('Frame number')
plt.ylabel('Ratio tracked correctly')
plt.legend()
plt.tight_layout()

plt.savefig('/home/isabella/Documents/PLEN/dfs/others_code/TSOAX/timeseries/10_lines_stack/figs/GRAFT_TSOAX_performance.png')

df_tsoax = pd.DataFrame()
for nm in range(10):
    df = pd.DataFrame({
        'value TSOAX': track_all_TSOAX[nm],
        'value graft':track_all[nm],
        'no lines': nm,
        'frame': np.arange(len(track_all_TSOAX[nm]))})
    
    df_tsoax = pd.concat([df_tsoax, df], ignore_index=True)
df_tsoax['frame']= df_tsoax['frame']+1

#plt.close('all')

plt.figure(figsize=(8.27,5))

sns.lineplot(data=df_tsoax, y='value graft', x='frame',estimator='mean', ci=90,label='GraFT') #for 95% confidence interval
sns.lineplot(data=df_tsoax, y='value TSOAX', x='frame',estimator='mean', ci=90,label='TSOAX',color= (0.8705882352941177, 0.5607843137254902, 0.0196078431372549)) #for 95% confidence interval
plt.xlabel('Frame number')
plt.ylabel('Ratio tracked correctly')
plt.ylim(0.2, 1.02)
plt.xlim(1, 20)
plt.xticks(np.append(np.arange(1, 20, 2),20))  
plt.legend(loc='lower left')
plt.tight_layout()

plt.savefig('/home/isabella/Documents/PLEN/dfs/figures_article/figs/ci_GRAFT_TSOAX_performance.pdf')
