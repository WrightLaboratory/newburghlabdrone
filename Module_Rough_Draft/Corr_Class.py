## I'm moving the correlator machine back up to LFOP on Wednesday, July 28, 2021
## This script will be used for the commissioning process of setting gains, using calculated gains and testing
## The class appears to work, so that should be a quick way to generate data/plots to assess gain levels.

##From loadD3Adata_Dallas.py:
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import h5py
##From WT:
import os
import glob
from matplotlib import colors
import pandas
import csv
import datetime
import pytz

class Corr_Data:
    def __init__(self,Data_Directory="",Load_Gains=True,Gain_Directory="",Fix_Gains=False,Gain_Params=[1.0,24.0],flb=0,fub=-1):
        ## Get data files
        self.Data_Directory=Data_Directory
        self.Gain_Directory=Gain_Directory
        os.chdir(self.Data_Directory)
        self.filenames=np.sort(glob.glob('*[!.lock]'))[0:-1]
        os.chdir('/Users/wct9/python')
        ## Load first data file to get array dimensions for V,t,f,prod:
        fd=h5py.File(self.Data_Directory+self.filenames[0], 'r')
        #vis=fd['vis'][:] # Visibility matrix
        vis=fd['vis'][:,flb:fub,:]
        tm=np.array([i[3] for i in fd['index_map']['time'][:]]) # time axis
        ## Declare freq and prod variables for the class:
        self.freq=np.array([i[0] for i in fd['index_map']['freq'][flb:fub]]) # frequency axis
        self.prod=fd['index_map']['prod'][:] # product axis
        ## Initialize Visibility and Time data products:
        self.V_full=np.zeros((len(self.filenames),vis.shape[0],vis.shape[1],vis.shape[2]))
        self.t_full=np.zeros((len(self.filenames),vis.shape[0]))
        self.sat_full=np.zeros((len(self.filenames),vis.shape[0],vis.shape[1],vis.shape[2]))
        ## Get gain file (for all data files) if it exists...
        if Load_Gains==True:
            os.chdir(Gain_Directory)
            self.gainfile=str(glob.glob('*')[0])
            os.chdir('/Users/wct9/python')
            fg=h5py.File(self.Gain_Directory+self.gainfile)
            self.gain_coeffs=fg['gain_coeff'][0] 
            self.gain_exp=fg['gain_exp'][0]
            digital_gain=fg['gain_coeff'][0] 
            digital_gain*=np.power(2,fg['gain_exp'][0])[np.newaxis,:]
        elif Fix_Gains==True:
            digital_gain=Gain_Params[0]*np.array((2**Gain_Params[1])*np.ones((1024,20))).astype(complex)
            self.gain_coeffs=Gain_Params[0]*np.ones(len(self.freq))
            self.gain_exp=Gain_Params[1]*np.ones(vis.shape[2])
        self.gain=digital_gain.real[flb:fub,:]
        fd.close()
        fg.close()
        ## Loop over all files to populate V_full,t_full
        for i,file in enumerate(self.filenames):
            try:
                fd_n=h5py.File(self.Data_Directory+self.filenames[i], 'r')
                vis=fd_n['vis'][:,flb:fub,:] # Visibility matrix
                tm=np.array([i[3] for i in fd_n['index_map']['time'][:]]) # time axis
                freq=np.array([i[0] for i in fd_n['index_map']['freq'][flb:fub]]) # frequency axis
                prod=fd_n['index_map']['prod'][:] # product axis
                ## gain calibrate visibilities:
                for ii,pp in enumerate(prod):                
                    vis[:,:,ii]/=(self.gain[:,pp[0]]*self.gain[:,pp[1]])[np.newaxis,:]
                self.V_full[i,:,:,:]=vis.real
                self.sat_full[i,:,:,:]=fd_n['sat'][:,flb:fub,:].real
                self.t_full[i,:]=tm      
            except OSError:
                print('Skipping file: {}'.format(file))
        ## reshape these arrays
        self.V_full=self.V_full.reshape((len(self.filenames)*vis.shape[0],vis.shape[1],vis.shape[2]))
        self.t_full=self.t_full.reshape(len(self.filenames)*vis.shape[0])
        self.sat_full=self.sat_full.reshape((len(self.filenames)*vis.shape[0],vis.shape[1],vis.shape[2]))
        self.t_arr_datetime=np.array([datetime.datetime.fromtimestamp(tt,pytz.timezone('America/Montreal')).astimezone(pytz.utc) for tt in self.t_full])
        self.t_index=np.arange(len(self.t_arr_datetime))

    def Plot_Waterfalls(self,):
        ## Express bounds for the plot axes
        wfbounds=[self.freq[-1],self.freq[0],self.t_full[-1]-self.t_full[0],0.0]
        ## This should plot waterfalls for the imported gain calibrated data:
        fig1,[[ax1,ax2,ax3],[ax4,ax5,ax6]]=subplots(nrows=2,ncols=3,figsize=(14,8))
        ## Plotting the individual waterfall plots (note freq ind is reversed!)
        for i,ax in enumerate([ax1]):
            im=ax.imshow(self.V_full[:,::-1,0].real,extent=wfbounds,cmap='gnuplot2',aspect='auto',norm=LogNorm())
            ax.set_title('Correlator Waterfall Plot - Channel 1 Auto')
            ax.set_xlabel('Frequency, [$MHz$]')
            ax.set_ylabel('$\Delta$Time [$s$]')
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right", size="5%", pad=0.05)
            cbar=fig1.colorbar(im,cax=cax)
            cbar.set_label('Power [$ADU^2$]')
        for i,ax in enumerate([ax2]):
            im=ax.imshow(self.V_full[:,::-1,2].real,extent=wfbounds,cmap='gnuplot2',aspect='auto',norm=LogNorm())
            ax.set_title('Correlator Waterfall Plot - Channel 2 Auto')
            ax.set_xlabel('Frequency, [$MHz$]')
            ax.set_ylabel('$\Delta$Time [$s$]')
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right", size="5%", pad=0.05)
            cbar=fig1.colorbar(im,cax=cax)
            cbar.set_label('Power [$ADU^2$]')
        for i,ax in enumerate([ax3]):
            im=ax.imshow(self.V_full[:,::-1,1].real,extent=wfbounds,cmap='gnuplot2',aspect='auto',norm=LogNorm())
            ax.set_title('Correlator Waterfall Plot - Cross')
            ax.set_xlabel('Frequency, [$MHz$]')
            ax.set_ylabel('$\Delta$Time [$s$]')
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right", size="5%", pad=0.05)
            cbar=fig1.colorbar(im,cax=cax)
            cbar.set_label('Power [$ADU^2$]')
        for i,ax in enumerate([ax4]):
            im=ax.imshow(self.V_full[:,::-1,0].real-np.nanmean(self.V_full[:,::-1,0].real,axis=0),extent=wfbounds,cmap='gnuplot2',aspect='auto')
            ax.set_title('Correlator Waterfall Plot - Channel 1 Auto')
            ax.set_xlabel('Frequency, [$MHz$]')
            ax.set_ylabel('$\Delta$Time [$s$]')
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right", size="5%", pad=0.05)
            cbar=fig1.colorbar(im,cax=cax)
            cbar.set_label('Mean Subtracted Power [$ADU^2$]')
        for i,ax in enumerate([ax5]):
            im=ax.imshow(self.V_full[:,::-1,2].real-np.nanmean(self.V_full[:,::-1,2].real,axis=0),extent=wfbounds,cmap='gnuplot2',aspect='auto')
            ax.set_title('Correlator Waterfall Plot - Channel 2 Auto')
            ax.set_xlabel('Frequency, [$MHz$]')
            ax.set_ylabel('$\Delta$Time [$s$]')
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right", size="5%", pad=0.05)
            cbar=fig1.colorbar(im,cax=cax)
            cbar.set_label('Mean Subtracted Power [$ADU^2$]')
        for i,ax in enumerate([ax6]):
            im=ax.imshow(self.V_full[:,::-1,1].real-np.nanmean(self.V_full[:,::-1,1].real,axis=0),extent=wfbounds,cmap='gnuplot2',aspect='auto')
            ax.set_title('Correlator Waterfall Plot - Cross')
            ax.set_xlabel('Frequency, [$MHz$]')
            ax.set_ylabel('$\Delta$Time [$s$]')
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right", size="5%", pad=0.05)
            cbar=fig1.colorbar(im,cax=cax)
            cbar.set_label('Mean Subtracted Power [$ADU^2$]')
        tight_layout()
        
    def Plot_Spectra(self,tbounds=[5,-5],tstep=200):
        ## Looking at spectra for specified times on both channels:
        fig1,[ax1,ax2]=subplots(nrows=1,ncols=2,figsize=(16,9),sharex=True,sharey=True)
        prod_keys=['0x0','1x1']
        chan_ind=[0,2]
        t_cut=tbounds
        t_step=tstep
        CNorm=colors.Normalize()
        CNorm.autoscale(np.arange(len(self.t_full))[t_cut[0]:t_cut[1]:t_step])
        CM=cm.gnuplot2
        CM=cm.magma
        for i,ax in enumerate([ax1,ax2]):
            for j,t_ind in enumerate(np.arange(len(self.t_full))[t_cut[0]:t_cut[1]:t_step]):
                ## AXIS 1: Let's plot channel i in log for specified time slices:
                ax.semilogy(self.freq,self.V_full[t_ind,:,chan_ind[i]],'.',c=CM(CNorm(t_ind)),label='t = {:.2f}'.format(float(self.t_full[t_ind]-self.t_full[0])))
            ax.set_title('Gain Corrected Data [{}]'.format(prod_keys[i]))
            ax.set_ylabel('Linear Data')
            ax.set_xlabel('Frequency [MHz]')
            ax.legend(fontsize='small')
        tight_layout()
        
    def Plot_Gains_vs_Data(self,):
        ## Let's plot the calculated gain solution for the data:
        fig1,[[ax1,ax2],[ax3,ax4]]=subplots(nrows=2,ncols=2,figsize=(16,9))
        for i,ax in enumerate([ax1,ax3]):
            ax1.plot(self.freq,self.gain[:,i],'.',label="Gain Exp = {}".format(self.gain_exp[i]))
            ax3.semilogy(self.freq,self.gain[:,i],'.',label="Gain Exp = {}".format(self.gain_exp[i]))
            ax.set_xlabel('Frequency [MHz]')
        chan_ind=[0,2]
        for i,ax in enumerate([ax2,ax4]):
            ax2.plot(self.freq,self.V_full[2,:,chan_ind[i]],'.',label="AutoCorr{}".format(str(i)))
            ax4.semilogy(self.freq,self.V_full[2,:,chan_ind[i]],'.',label="AutoCorr{}".format(str(i)))
            ax.set_xlabel('Frequency [MHz]')
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax1.set_title('Gains')
        ax2.set_title('Gain Corrected Data')
        ax1.set_ylabel('Linear Gain')
        ax2.set_ylabel('Linear Data')
        ax3.set_ylabel('Log Gain')
        ax4.set_ylabel('Log Data')
        tight_layout()