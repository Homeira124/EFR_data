#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 01:32:39 2019

@author: homeira
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 02:42:49 2020

@author: homeira
"""

import mne
from mne import find_events
import numpy as np
import matplotlib.pyplot as plt
from anlffr import spectral
from scipy import io
import os
import fnmatch
import pylab as pl
import pandas as pd


froot='/home/homeira/Desktop/EEG/useful data/'
subjlist= ['S166',]

condlist = [1, 2, 4, 8]
condnames = ['Lm4dB', 'L0dB', 'Rm4dB',  'R0dB']
overwriteOld = False
for subj in subjlist:
    for k, cond in enumerate(condlist):
        fpath = froot + subj + '/'
        respath = fpath + 'RES/'
        condname = condnames[k]
        print 'Running Subject', subj, 'Condition', condname
	
        save_raw_name = subj + '_' + condname + '_alltrial.mat'
        if os.path.isfile(respath + save_raw_name) and not overwriteOld:
            print 'Epoched data is already available on disk!'
            print 'Loading data from:', respath + save_raw_name
            x = io.loadmat(respath + save_raw_name)['x']
            fs = 16384.0
        else:
            bdfs = fnmatch.filter(os.listdir(fpath), subj + '*EFR*.bdf')
            print 'No pre-epoched data found, looking for BDF files'
            print 'Viola!', len(bdfs), 'files found!'
	
         	
        for k, edfname in enumerate(bdfs):
# Load data and read event channel
            raw= mne.io.read_raw_edf(fpath+edfname)
            eves= find_events(raw, shortest_event=1, mask= 255)
            raw.set_channel_types({'EXG3': 'eeg', 'EXG4': 'eeg'})
            #raw.info['bads'] += ['EXG5', 'A4', 'A5', 'A8', 'A9', 'A10',
             #                   'A12', 'A13', 'A19', 'A21', 'A22', 'A23',
                #                'A26', 'A31', 'A32',
                 #               'EXG6', 'EXG7']
            #picks = mne.pick_types(raw.info, exclude='bads')
            raw.load_data()
            raw.filter(l_freq=80, h_freq=1000)
# MAYBE use w/ filtering picks=np.arange(0, 17, 1))
            
         

# raw.apply_proj()
            fs = raw.info['sfreq']

# Epoching events of type
            epochs = mne.Epochs(
                        raw, eves, cond, tmin=-0.025, proj=False,
                        tmax=1.025, baseline=(-0.025, 0.0),
                        reject=dict(eeg=200e-6)) # 200 regular, 50 strict

            xtemp = epochs.get_data()

# Reshaping to the format needed by spectral.mtcpca() and# calling it
            if(xtemp.shape[0] > 0):
                xtemp = xtemp.transpose((1, 0, 2))
                xtemp = xtemp[:15, :, :]
                if(k == 0):
                    x = xtemp
                else: 
                    x = np.concatenate((x, xtemp), axis=1)
            else: 
                continue
            
        nPerDraw = 400
        nDraws = 100
        params = dict(Fs=fs, fpass=[5, 1000], tapers=[1, 1], Npairs=2000,
                      itc=1, nfft=32768)
                      
        Ntrials = x.shape[1]

        print 'Running Mean Spectrum Estimation'
        (S, N, f) = spectral.mtspec(x, params, verbose=True)
#calculating combined phase locking value
        print 'Running CPCA PLV Estimation'
        (cplv, f) = spectral.mtcpca(x, params, verbose=True)
#this is most important, looking at each channel individual plv
        print 'Running channel by channel PLV Estimation'
        (plv, f) = spectral.mtplv(x, params, verbose=True)
#after doing pca in each channel the combined channels power estimation which is basically without noise
        print 'Running CPCA Power Estimation'
        (cpow, f) = spectral.mtcspec(x, params, verbose=True)
#running raw power estimation which is basically noise power in this case
        print 'Running raw spectrum estimation'
        (Sraw, f) = spectral.mtspecraw(x, params, verbose=True)

# Saving Results
        res = dict(cpow=cpow, plv=plv, cplv=cplv, Sraw=Sraw,
                   f=f, S=S, N=N, Ntrials=Ntrials)

        save_name = subj + '_' + condname + '_results.mat'
        
        index=0
            
        p=0
        q=0
        r=0
        for val in f:
            if val==163:
                print index
                p=index
           
            if val==223:
                print index
                q=index
            
            if val==283:
                print index
                r=index
                 
            index=index+1  
        
            if cond==1:
                z1=(cpow[p],cpow[q],cpow[r])
                
            if cond==2:
                z2=(cpow[p],cpow[q],cpow[r])
                
            if cond==4:
                z3=(cpow[p],cpow[q],cpow[r])
                
            if cond==8:
                z4=(cpow[p],cpow[q],cpow[r])
                
        
            
            
            
         
                
        
    
    m1=list(z1)
    m2=list(z2)
    m3=list(z3)
    m4=list(z4)
    m=[m1, m2, m3, m4]
        
        #z = z + ', '.join([str(z) for cond in condlist])
    df=pd.DataFrame(m)
    print (m)
  
    writer=pd.ExcelWriter('trial.xlsx',engine='xlsxwriter')
    df.to_excel(writer,sheet_name='sheet1')
    writer.save()
    
        
        #dB = 20*np.log10(np.sqrt(cpow))
    plotStuff= True
    if plotStuff:
        
            pl.plot(f, cpow, linewidth=2)
            pl.xlabel('Frequency (Hz)', fontsize=16)
            pl.ylabel('Phase Locking', fontsize=16)
            ax = pl.gca()
            ax.set_xlim([100, 500])
        
            ax.tick_params(labelsize=16)
            pl.show()
           
                            
 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            