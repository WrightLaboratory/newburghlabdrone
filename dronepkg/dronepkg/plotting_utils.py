#       _       _   _   _                      _   _ _                   
#      | |     | | | | (_)                    | | (_) |                  
# _ __ | | ___ | |_| |_ _ _ __   __ _    _   _| |_ _| |___   _ __  _   _ 
#| '_ \| |/ _ \| __| __| | '_ \ / _` |  | | | | __| | / __| | '_ \| | | |
#| |_) | | (_) | |_| |_| | | | | (_| |  | |_| | |_| | \__ \_| |_) | |_| |
#| .__/|_|\___/ \__|\__|_|_| |_|\__, |   \__,_|\__|_|_|___(_) .__/ \__, |
#| |                             __/ |_____                 | |     __/ |
#|_|                            |___/______|                |_|    |___/ 

## 20211110 WT - The module structure is being completely refactored...
## This is the brand new plotting_utils.py file
## Plotting functions will be written here in a more general format... hopefully! ;)

import os
import glob
import pandas
import csv
import datetime
import pytz
from matplotlib.pyplot import *
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import h5py
      
def Plot_Waterfalls(data_class):
    ## Express bounds for the plot axes
    wfbounds=[data_class.freq[-1],data_class.freq[0],data_class.t[-1]-data_class.t[0],0.0]
    ## This should plot waterfalls for the imported gain calibrated data:
    fig1=figure(figsize=(16,int(4*data_class.n_channels/2)))
    ## Plotting the individual waterfall plots (note freq ind is reversed!)
    for i in range(data_class.n_channels):
        ax=fig1.add_subplot(int(data_class.n_channels/2),2,i+1)
        im=ax.imshow(data_class.V[:,::-1,data_class.chmap[i]].real,extent=wfbounds,cmap='gnuplot2',aspect='auto',norm=LogNorm())
        ax.set_title('Auto-Corr: Channel {}x{} - Ind {}'.format(data_class.chmap[i],data_class.chmap[i],data_class.automap[i]))
        ax.set_xlabel('Frequency, [$MHz$]')
        ax.set_ylabel('$\Delta$Time [$s$]')
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig1.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
    tight_layout()

def Plot_Saturation_Maps(data_class):
    ## Express bounds for the plot axes
    wfbounds=[data_class.freq[-1],data_class.freq[0],data_class.t[-1]-data_class.t[0],0.0]
    ## This should plot waterfalls for the imported gain calibrated data:
    fig1=figure(figsize=(16,int(4*data_class.n_channels/2)))
    ## Plotting the individual waterfall plots (note freq ind is reversed!)
    for i in range(data_class.n_channels):
        ax=fig1.add_subplot(int(data_class.n_channels/2),2,i+1)
        im=ax.imshow(data_class.sat[:,::-1,data_class.chmap[i]].real,extent=wfbounds,cmap='gnuplot2',aspect='auto')
        ax.set_title('Auto-Corr: Channel {}x{} - Ind {}'.format(data_class.chmap[i],data_class.chmap[i],data_class.automap[i]))
        ax.set_xlabel('Frequency, [$MHz$]')
        ax.set_ylabel('$\Delta$Time [$s$]')
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig1.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
    tight_layout()

def Plot_Time_Series(data_class,tbounds=[0,-1],freqlist=[100,700,900]):
    fig1=figure(figsize=(16,int(4*data_class.n_channels/2)))
    for i in range(data_class.n_channels):
        ax=fig1.add_subplot(int(data_class.n_channels/2),2,i+1)
        for k in freqlist:
            ax.plot(data_class.t_index[tbounds[0]:tbounds[1]],data_class.V[tbounds[0]:tbounds[1],k,data_class.chmap[i]].real,'.',label="F={}".format(data_class.freq[k]))
        ax.set_title('Time Series: Channel {}x{} - Ind {}'.format(data_class.chmap[i],data_class.chmap[i],data_class.automap[i]))
        ax.set_ylabel('Power [$ADU^2$]')
        ax.set_xlabel('Time Index')
        ax.legend()
    tight_layout()

def Plot_Spectra(data_class,tbounds=[5,-5],tstep=2000):
    fig1=figure(figsize=(16,4*int(data_class.n_channels/2)))
    CNorm=colors.Normalize()
    CNorm.autoscale(np.arange(len(data_class.t))[tbounds[0]:tbounds[1]:tstep])
    CM=cm.gnuplot2
    CM=cm.magma
    for i in range(data_class.n_channels):
        ax=fig1.add_subplot(int(data_class.n_channels/2),2,i+1)
        for k,t_ind in enumerate(np.arange(len(data_class.t))[tbounds[0]:tbounds[1]:tstep]):
            ax.semilogy(data_class.freq,data_class.V[t_ind,:,data_class.chmap[i]],'.',c=CM(CNorm(t_ind)),label='t = {:.2f}'.format(float(data_class.t[t_ind]-data_class.t[0])))
        ax.set_title('Spectra: Channel {}x{} - Ind {}'.format(data_class.chmap[i],data_class.chmap[i],data_class.automap[i]))
        ax.set_ylabel('Log Power [$ADU^2$]')
        ax.set_xlabel('Frequency [MHz]')
        ax.legend(fontsize='small')
    tight_layout()

def Plot_Gains_vs_Data(data_class,tind=1):
    ## Let's plot the calculated gain solution for the data:
    fig1=figure(figsize=(16,int(4*data_class.n_channels/2)))
    for i in range(data_class.n_channels):
        ax=fig1.add_subplot(int(data_class.n_channels/2),4,int(2*i)+1)
        ax.plot(data_class.freq,data_class.gain[:,data_class.chmap[i]],'.',label="Gain Exp = {}".format(data_class.gain_exp[data_class.chmap[i]]))
        ax.set_title("Gain: Channel {}x{} - Ind {}".format(data_class.chmap[i],data_class.chmap[i],data_class.automap[i]))
        ax.set_ylabel('Gain')            
        ax.set_xlabel('Frequency [MHz]')
        ax.legend()
        ax=fig1.add_subplot(int(data_class.n_channels/2),4,int(2*i)+2)
        ax.plot(data_class.freq,data_class.V[tind,:,data_class.chmap[i]],'.')
        ax.set_title("Spectra: Channel {}x{} - Ind {}".format(data_class.chmap[i],data_class.chmap[i],data_class.automap[i]))
        ax.set_ylabel('Power [$ADU^2$]')            
        ax.set_xlabel('Frequency [MHz]')
    tight_layout()
