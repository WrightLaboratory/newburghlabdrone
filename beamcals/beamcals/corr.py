# _____ _________________               
#/  __ \  _  | ___ \ ___ \              
#| /  \/ | | | |_/ / |_/ /  _ __  _   _ 
#| |   | | | |    /|    /  | '_ \| | | |
#| \__/\ \_/ / |\ \| |\ \ _| |_) | |_| |
# \____/\___/\_| \_\_| \_(_) .__/ \__, |
#                          | |     __/ |
#                          |_|    |___/ 

## 20211101 WT - The module structure is being completely refactored...
## This is the new correlator data class, improved, and all plotting functions relocated to plotting_utils!
## Timing fixed using irigb_time call explicitly, and doing the datetime after 1e-9 multiplication

##From loadD3Adata_Dallas.py:
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import h5py
import hdf5plugin
##From WT:
import os
import glob
from matplotlib import colors
import pandas
import csv
import datetime
import pytz

class Corr_Data:
    def __init__(self,Data_Directory,Gain_Directory,site_class,Data_File_Index=None,Load_Gains=True,Fix_Gains=False,Apply_Gains=True,Gain_Params=[1.0,24.0],fbounds=[0,1024],use_ctime=False,crossmap=None):
        ## Get data files using os instead of git:
        self.Data_Directory=Data_Directory
        self.Gain_Directory=Gain_Directory
        self.filenames=np.sort([x for x in os.listdir(self.Data_Directory) if ".lock" not in x])[:-1]
        print('Initializing Correlator Class using:')
        print("  --> "+self.Data_Directory)
        ## Load first data file to get array dimensions for V,t,f,prod:
        fd=h5py.File(self.Data_Directory+self.filenames[0], 'r')
        self.fbounds=fbounds
        flb=fbounds[0]
        fub=fbounds[1]
        vis=fd['vis'][:,flb:fub,:] ## This is the visibility matrix (the data)
        if 'CHIME' in site_class.name:
            vis=np.array(fd['vis']).transpose(2,0,1)[:,flb:fub,:]
        ##distinguish bw processed and unprocessed files (EK)
        if 'processed' in Data_Directory and 'new' not in Data_Directory: 
            tm = fd['tm']
            self.freq = fd['freq'][flb:fub]
            self.prod = fd['prod'][:]   
            self.n_channels=len(self.prod)
        if 'processed_new' in Data_Directory: 
            if use_ctime==False:
                tm=np.array(fd['tm']['irigb_time']) # time axis
            if use_ctime==True:
                tm=np.array(fd['tm']['ctime']) # time axis
            self.freq = fd['freq'][flb:fub]
            self.prod = fd['prod'][:]   
            self.n_channels=len(self.prod)
            self.n_dishes=int(self.n_channels/2)
        else: 
            if use_ctime==False:
                self.t0=1e-9*fd['index_map']['time']['irigb_time'][0]
            if use_ctime==True:
                self.t0=fd['index_map']['time']['ctime'][0]
            self.freq=np.array([i[0] for i in fd['index_map']['freq'][flb:fub]]) # frequency axis
            self.prod=fd['index_map']['prod'][:] # product axis
            self.n_channels=min(len(site_class.chmap),int(fd['index_map']['prod'][:][-1][0]+1))
            self.n_dishes=int(self.n_channels/2)
        self.chmap=np.array(site_class.chmap[:self.n_channels]).astype(int)
        self.automap=np.zeros(self.n_channels).astype(int)
        self.crossmap=crossmap
        prodmat=np.array([element for tupl in self.prod for element in tupl]).reshape(len(self.prod),2)
        for i,j in enumerate(self.chmap):
            self.automap[i]=np.intersect1d(np.where(prodmat[:,0]==j),np.where(prodmat[:,1]==j))
        if Data_File_Index is None:
            Data_File_Index=np.arange(len(self.filenames)).astype(int)
        ## Initialize Visibility and Time data products:
        self.V=np.zeros((len(Data_File_Index),vis.shape[0],vis.shape[1],self.n_channels))
        self.t=np.zeros((len(Data_File_Index),vis.shape[0]))
        self.sat=np.zeros((len(Data_File_Index),vis.shape[0],vis.shape[1],self.n_channels))
        self.V_cross=np.zeros((len(Data_File_Index),vis.shape[0],vis.shape[1],len(self.crossmap))).astype(complex)
        # Get gain file (for all data files) if it exists...
        if Load_Gains==True:
            self.gainfile=os.listdir(self.Gain_Directory)[0]
            try:
                fg=h5py.File(self.Gain_Directory+self.gainfile,'r')
                self.gain_coeffs=fg['gain_coeff'][0] 
                self.gain_exp=fg['gain_exp'][0]
                digital_gain=fg['gain_coeff'][0] 
                digital_gain*=np.power(2,fg['gain_exp'][0])[np.newaxis,:]
                fg.close()
            except OSError:
                print("  --> ERROR: Gain file not found in specified directory!")
        elif Fix_Gains==True:
            digital_gain=Gain_Params[0]*np.array((2**Gain_Params[1])*np.ones((vis.shape[1],self.n_channels))).astype(complex)
            self.gain_coeffs=Gain_Params[0]*np.ones(len(self.freq))
            self.gain_exp=Gain_Params[1]*np.ones(vis.shape[2])
        elif Apply_Gains==False:
            digital_gain=np.ones((len(self.freq),self.n_channels))
            self.gain_coeffs=np.ones(len(self.freq))
            self.gain_exp=np.ones(vis.shape[2])
        self.gain=digital_gain.real[flb:fub,:]
        fd.close()
        ## Loop over all files to populate V_full,t_full
        print("  --> Arrays initialized with shape {}".format(self.V.shape))
        print("Assigning array values by reading in data files:")
        for i,file in enumerate(self.filenames[Data_File_Index]):
            try:
                print("\r  --> Loading File: {}/{}".format(self.filenames[i],self.filenames[-1]),end="")
                fd_n=h5py.File(self.Data_Directory+self.filenames[i], 'r')
                vis=fd_n['vis'][:,flb:fub,:] # Visibility matrix
                if 'CHIME' in site_class.name:
                    vis=np.array(fd_n['vis']).transpose(2,0,1)[:,flb:fub,:]
                ##distinguish bw processed and unprocessed files
                if 'processed' in Data_Directory and 'new' not in Data_Directory:
                    tm=fd_n['tm'][:] # time axis
                    freq=fd_n['freq'][flb:fub] # frequency axis
                    prod=fd_n['prod'][:] # product axis
                    for ii in range(len(prod)):
                        vis[:,:,ii]/=(self.gain[:,ii]*self.gain[:,ii])[np.newaxis,:]
                if 'processed_new' in Data_Directory: 
                    if use_ctime==False:
                        tm=np.array(fd_n['tm']['irigb_time']) # time axis
                    if use_ctime==True:
                        tm=np.array(fd_n['tm']['ctime']) # time axis
                    freq=fd_n['freq'][flb:fub] # frequency axis
                    prod=fd_n['prod'][:] # product axis
                    for ii in range(len(prod)):
                        vis[:,:,ii]/=(self.gain[:,ii]*self.gain[:,ii])[np.newaxis,:]
                else:
                    if use_ctime==False:
                        tm=np.array(fd_n['index_map']['time']['irigb_time']) # time axis
                        if 'GBO' in site_class.name:
                            tm=(2.56e-6)*np.array(fd_n['index_map']['time']['fpga_count']) #construct time from fpga_counts
                    if use_ctime==True:
                        tm=np.array(fd_n['index_map']['time']['ctime']) # time axis
                    freq=np.array([i[0] for i in fd_n['index_map']['freq'][flb:fub]]) # frequency axis
                    prod=fd_n['index_map']['prod'][:] # product axis
                    ## gain calibrate visibilities:
                    for ii,pp in enumerate(prod):
                        vis[:,:,ii]/=(self.gain[:,pp[0]]*self.gain[:,pp[1]])[np.newaxis,:]
                ## Populate the automap array indices into V:
                for j,k in enumerate(self.automap):
                    self.V[i,:,:,j]=vis[:,:,k].real
                    try:
                        self.sat[i,:,:,j]=fd_n['sat'][:,flb:fub,k].real
                    except KeyError:
                        pass
                ## Populate the crossmap array indices into V_cross:
                for j,k in enumerate(self.crossmap):
                    self.V_cross[i,:,:,j]=vis[:,:,k]             
                self.t[i,:]=tm
                fd.close()
            except OSError:
                print('\nSkipping file: {}'.format(file))    
        print("\n  --> Finished. Reshaping arrays.")
        ## reshape these arrays
        self.V=self.V.reshape((len(Data_File_Index)*vis.shape[0],vis.shape[1],self.n_channels))
        self.V_cross=self.V_cross.reshape((len(Data_File_Index)*vis.shape[0],vis.shape[1],len(self.crossmap)))
        self.t=self.t.reshape(len(Data_File_Index)*vis.shape[0])
        self.sat=self.sat.reshape((len(Data_File_Index)*vis.shape[0],vis.shape[1],self.n_channels))
        if 'GBO' in site_class.name:
            timedeltas=np.array([datetime.timedelta(seconds=x) for x in self.t])
            dt0=datetime.datetime.fromtimestamp(self.t0,pytz.timezone('America/Montreal')).astimezone(pytz.utc)
            self.t_arr_datetime=dt0+timedeltas
        if 'CHIME' in site_class.name:
            self.t_arr_datetime=np.array([datetime.datetime.fromtimestamp(x,pytz.utc) for x in self.t])
        self.t_index=np.arange(len(self.t_arr_datetime))