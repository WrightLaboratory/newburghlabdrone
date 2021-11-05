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
    def __init__(self,n_channels,chmap,Data_Directory="",Working_Directory="",Data_File_Index=None,Load_Gains=True,Gain_Directory="",Fix_Gains=False,Gain_Params=[1.0,24.0],flb=0,fub=-1,Apply_Gains=True):
        ## Get data files
        self.Data_Directory=Data_Directory
        self.Gain_Directory=Gain_Directory
        self.Working_Directory=Working_Directory
        os.chdir(self.Data_Directory)
        self.filenames=np.sort(glob.glob('*[!.lock]'))[0:-1]
        os.chdir(Working_Directory)
        print('Initializing Correlator Class using:')
        print(" --> "+self.Data_Directory)
        ## Load first data file to get array dimensions for V,t,f,prod:
        fd=h5py.File(self.Data_Directory+self.filenames[0], 'r')
        #vis=fd['vis'][:] # Visibility matrix
        vis=fd['vis'][:,flb:fub,:]
        ##distinguish bw processed and unprocessed files
        if 'processed' in Data_Directory: 
            tm = fd['tm']
            self.freq = fd['freq'][flb:fub]
            self.prod = fd['prod'][:]   
            self.n_channels=len(self.prod)
        else: 
            tm=np.array([i[3] for i in fd['index_map']['time'][:]]) # time axis
            self.freq=np.array([i[0] for i in fd['index_map']['freq'][flb:fub]]) # frequency axis
            self.prod=fd['index_map']['prod'][:] # product axis
            self.n_channels=int(n_channels)
        self.chmap=np.array(chmap[:self.n_channels]).astype(int)
        self.automap=np.zeros(self.n_channels).astype(int)
        prodmat=np.array([element for tupl in self.prod for element in tupl]).reshape(len(self.prod),2)
        for i,j in enumerate(self.chmap):
            self.automap[i]=np.intersect1d(np.where(prodmat[:,0]==j),np.where(prodmat[:,1]==j))
        if Data_File_Index is None:
            Data_File_Index=np.arange(len(self.filenames)).tolist()
        ## Initialize Visibility and Time data products:
        self.V_full=np.zeros((len(self.filenames[Data_File_Index]),vis.shape[0],vis.shape[1],self.n_channels))
        self.t_full=np.zeros((len(self.filenames[Data_File_Index]),vis.shape[0]))
        self.sat_full=np.zeros((len(self.filenames[Data_File_Index]),vis.shape[0],vis.shape[1],self.n_channels))
        # Get gain file (for all data files) if it exists...
        if Load_Gains==True:
            os.chdir(Gain_Directory)
            self.gainfile=str(glob.glob('*')[0])
            os.chdir(Working_Directory)
            try:
                fg=h5py.File(self.Gain_Directory+self.gainfile)
                self.gain_coeffs=fg['gain_coeff'][0] 
                self.gain_exp=fg['gain_exp'][0]
                digital_gain=fg['gain_coeff'][0] 
                digital_gain*=np.power(2,fg['gain_exp'][0])[np.newaxis,:]
            except OSError:
                print(" --> ERROR: Gain file not found in specified directory!")
        elif Fix_Gains==True:
            digital_gain=Gain_Params[0]*np.array((2**Gain_Params[1])*np.ones((len(self.freq),self.n_channels))).astype(complex)
            self.gain_coeffs=Gain_Params[0]*np.ones(len(self.freq))
            self.gain_exp=Gain_Params[1]*np.ones(vis.shape[2])
        elif Apply_Gains==False:
            digital_gain=np.ones((len(self.freq),self.n_channels))
            self.gain_coeffs=np.ones(len(self.freq))
            self.gain_exp=np.ones(vis.shape[2])
        self.gain=digital_gain.real[flb:fub,:]
        fd.close()
        fg.close()
        ## Loop over all files to populate V_full,t_full
        print(" --> Arrays initialized with shape {}".format(self.V_full.shape))
        print("Assigning array values by reading in data files:")
        for i,file in enumerate(self.filenames[Data_File_Index]):
            try:
                print("\r --> Loading File: {}/{}".format(self.filenames[i],self.filenames[-1]),end="")
                fd_n=h5py.File(self.Data_Directory+self.filenames[i], 'r')
                vis=fd_n['vis'][:,flb:fub,:] # Visibility matrix

                ##distinguish bw processed and unprocessed files
                if 'processed' in Data_Directory:
                    tm=fd_n['tm'][:] # time axis
                    freq=fd_n['freq'][flb:fub] # frequency axis
                    prod=fd_n['prod'][:] # product axis
                    for ii in range(len(prod)):
                        vis[:,:,ii]/=(self.gain[:,ii]*self.gain[:,ii])[np.newaxis,:]
                else:
                    tm=np.array([i[3] for i in fd_n['index_map']['time'][:]]) # time axis
                    freq=np.array([i[0] for i in fd_n['index_map']['freq'][flb:fub]]) # frequency axis
                    prod=fd_n['index_map']['prod'][:] # product axis
                    ## gain calibrate visibilities:
                    for ii,pp in enumerate(prod):
                        vis[:,:,ii]/=(self.gain[:,pp[0]]*self.gain[:,pp[1]])[np.newaxis,:]

                for j,k in enumerate(self.automap):
                    self.V_full[i,:,:,j]=vis[:,:,k].real
                    self.sat_full[i,:,:,j]=fd_n['sat'][:,flb:fub,k].real
                self.t_full[i,:]=tm
                fd.close()
                fg.close()
            except OSError:
                print('Skipping file: {}'.format(file))    

        ## reshape these arrays
        self.V_full=self.V_full.reshape((len(self.filenames[Data_File_Index])*vis.shape[0],vis.shape[1],self.n_channels))
        self.t_full=self.t_full.reshape(len(self.filenames[Data_File_Index])*vis.shape[0])
        self.sat_full=self.sat_full.reshape((len(self.filenames[Data_File_Index])*vis.shape[0],vis.shape[1],self.n_channels))
        self.t_arr_datetime=np.array([datetime.datetime.fromtimestamp(tt,pytz.timezone('America/Montreal')).astimezone(pytz.utc) for tt in self.t_full])
        self.t_index=np.arange(len(self.t_arr_datetime))
    def Plot_Auto_Corr_Waterfalls(self):
        ## Express bounds for the plot axes
        wfbounds=[self.freq[-1],self.freq[0],self.t_full[-1]-self.t_full[0],0.0]
        ## This should plot waterfalls for the imported gain calibrated data:
        fig1=figure(figsize=(16,int(4*self.n_channels/2)))
        ## Plotting the individual waterfall plots (note freq ind is reversed!)
        for i in range(self.n_channels):
            ax=fig1.add_subplot(int(self.n_channels/2),2,i+1)
            im=ax.imshow(self.V_full[:,::-1,self.chmap[i]].real,extent=wfbounds,cmap='gnuplot2',aspect='auto',norm=LogNorm())
            ax.set_title('Auto-Corr: Channel {}x{} - Ind {}'.format(self.chmap[i],self.chmap[i],self.automap[i]))
            ax.set_xlabel('Frequency, [$MHz$]')
            ax.set_ylabel('$\Delta$Time [$s$]')
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right", size="5%", pad=0.05)
            cbar=fig1.colorbar(im,cax=cax)
            cbar.set_label('Power [$ADU^2$]')
        tight_layout()
            
    def Plot_Auto_Corr_Saturation_Maps(self):
        ## Express bounds for the plot axes
        wfbounds=[self.freq[-1],self.freq[0],self.t_full[-1]-self.t_full[0],0.0]
        ## This should plot waterfalls for the imported gain calibrated data:
        fig1=figure(figsize=(16,int(4*self.n_channels/2)))
        ## Plotting the individual waterfall plots (note freq ind is reversed!)
        for i in range(self.n_channels):
            ax=fig1.add_subplot(int(self.n_channels/2),2,i+1)
            im=ax.imshow(self.sat_full[:,::-1,self.chmap[i]].real,extent=wfbounds,cmap='gnuplot2',aspect='auto')
            ax.set_title('Auto-Corr: Channel {}x{} - Ind {}'.format(self.chmap[i],self.chmap[i],self.automap[i]))
            ax.set_xlabel('Frequency, [$MHz$]')
            ax.set_ylabel('$\Delta$Time [$s$]')
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right", size="5%", pad=0.05)
            cbar=fig1.colorbar(im,cax=cax)
            cbar.set_label('Power [$ADU^2$]')
        tight_layout()
        
    def Plot_Auto_Corr_Time_Series(self,tbounds=[0,-1],freqlist=[100,700,900]):
        fig1=figure(figsize=(16,int(4*self.n_channels/2)))
        for i in range(self.n_channels):
            ax=fig1.add_subplot(int(self.n_channels/2),2,i+1)
            for k in freqlist:
                ax.plot(self.t_index[tbounds[0]:tbounds[1]],self.V_full[tbounds[0]:tbounds[1],k,self.chmap[i]].real,'.',label="F={}".format(self.freq[k]))
            ax.set_title('Time Series: Channel {}x{} - Ind {}'.format(self.chmap[i],self.chmap[i],self.automap[i]))
            ax.set_ylabel('Power [$ADU^2$]')
            ax.set_xlabel('Time Index')
            ax.legend()
        tight_layout()
        
    def Plot_Auto_Corr_Spectra(self,tbounds=[5,-5],tstep=2000):
        fig1=figure(figsize=(16,4*int(self.n_channels/2)))
        CNorm=colors.Normalize()
        CNorm.autoscale(np.arange(len(self.t_full))[tbounds[0]:tbounds[1]:tstep])
        CM=cm.gnuplot2
        CM=cm.magma
        for i in range(self.n_channels):
            ax=fig1.add_subplot(int(self.n_channels/2),2,i+1)
            for k,t_ind in enumerate(np.arange(len(self.t_full))[tbounds[0]:tbounds[1]:tstep]):
                ax.semilogy(self.freq,self.V_full[t_ind,:,self.chmap[i]],'.',c=CM(CNorm(t_ind)),label='t = {:.2f}'.format(float(self.t_full[t_ind]-self.t_full[0])))
            ax.set_title('Spectra: Channel {}x{} - Ind {}'.format(self.chmap[i],self.chmap[i],self.automap[i]))
            ax.set_ylabel('Log Power [$ADU^2$]')
            ax.set_xlabel('Frequency [MHz]')
            ax.legend(fontsize='small')
        tight_layout()
        
    def Plot_Gains_vs_Data(self,tind=1):
        ## Let's plot the calculated gain solution for the data:
        fig1=figure(figsize=(16,int(4*self.n_channels/2)))
        for i in range(self.n_channels):
            ax=fig1.add_subplot(int(self.n_channels/2),4,int(2*i)+1)
            ax.plot(self.freq,self.gain[:,self.chmap[i]],'.',label="Gain Exp = {}".format(self.gain_exp[self.chmap[i]]))
            ax.set_title("Gain: Channel {}x{} - Ind {}".format(self.chmap[i],self.chmap[i],self.automap[i]))
            ax.set_ylabel('Gain')            
            ax.set_xlabel('Frequency [MHz]')
            ax.legend()
            ax=fig1.add_subplot(int(self.n_channels/2),4,int(2*i)+2)
            ax.plot(self.freq,self.V_full[tind,:,self.chmap[i]],'.')
            ax.set_title("Spectra: Channel {}x{} - Ind {}".format(self.chmap[i],self.chmap[i],self.automap[i]))
            ax.set_ylabel('Power [$ADU^2$]')            
            ax.set_xlabel('Frequency [MHz]')
        tight_layout()
