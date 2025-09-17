## CHANGELOG:
## 02/25 Synchronization tests were successful, function added to synchronize data from drone and correlator
## 03/01 yaml config file tests were successful, need to write config saving script
## 03/08 Writing yaml config files to speed up loading times and reuse parameters found via iterative fitting...
## 03/09 Writing yaml interpreter that searches for the existing params when an init is called

from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import h5py
import hdf5plugin
import os
import glob
from matplotlib import colors
import pandas
import csv
import datetime
import pytz
import bisect
import pygeodesy
import yaml
from scipy.signal import square
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

## Import packages from our own module:
from beamcals import corr
from beamcals import drone
from beamcals import bicolog
import beamcals.plotting_utils as pu
import beamcals.fitting_utils as fu
import beamcals.geometry_utils as gu
import beamcals.time_utils as tu
## What could go wrong? ...
from beamcals import concat

class CONCAT:
    def __init__(self,CORRDATCLASS,DRONEDATCLASS,config_directory="/hirax/GBO_Analysis_Outputs/concat_config_files/",output_directory='/hirax/GBO_Analysis_Outputs/',load_yaml=True,traceback=True,save_traceback=True, t_drone_offset = 0):
        ## Decide whether or not we want a traceback for print statements/verification plots:
        self.traceback=traceback
        ## Decide whether or not we want to save traceback output plots:
        self.save_traceback=save_traceback
        ## Import file information from both corr and drone classes:
        self.name=DRONEDATCLASS.name
        self.Data_Directory=CORRDATCLASS.Data_Directory
        self.Gain_Directory=CORRDATCLASS.Gain_Directory
        self.filenames=CORRDATCLASS.filenames
        self.gainfile=CORRDATCLASS.gainfile
        self.Drone_Directory=DRONEDATCLASS.Drone_Directory
        self.FLYTAG=DRONEDATCLASS.FLYTAG
        ## YAML configuration variables and config parameter loading:
        self.Config_Directory=config_directory
        self.load_yaml=load_yaml
        if self.traceback==True:
            print('Initializing CONCAT CLASS with active traceback using:')
            print("  --> "+CORRDATCLASS.Data_Directory)
            print("  --> "+DRONEDATCLASS.FLYTAG)
            if self.save_traceback==True:
                print('Creating directory for saving traceback and analysis outputs:')
                if 'TONE_ACQ' in self.Data_Directory:
                    tmpcorrdir=self.Data_Directory.split("_yale")[0].split("TONE_ACQ/")[1]
                elif 'NFandFF' in self.Data_Directory:
                    tmpcorrdir=self.Data_Directory.split("_Suit")[0].split("NFandFF/")[1]
                tmpdronedir=self.FLYTAG.split('.')[0]
                tmpoutputdir=output_directory+'{}_{}'.format(tmpdronedir,tmpcorrdir)+'/'
                if os.path.exists(tmpoutputdir)==False:                
                    self.Output_Directory=tmpoutputdir
                    self.Output_Prefix='{}_{}'.format(tmpdronedir,tmpcorrdir)
                if os.path.exists(tmpoutputdir)==True:
                    suff=datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                    self.Output_Prefix='{}_{}_ver_{}'.format(tmpdronedir,tmpcorrdir,suff)
                    self.Output_Directory=output_directory+'{}_{}_ver_{}'.format(tmpdronedir,tmpcorrdir,suff)+'/'
                os.makedirs(self.Output_Directory)
                print("  --> "+self.Output_Directory)
            if self.save_traceback==False:
                print("  --> Traceback outputs will not be saved...")
        elif self.traceback==False:
            pass
        ## If we don't want to load previous yaml config files, we skip the yaml interpreter:
        if self.load_yaml==False:
            if self.traceback==True:
                print('Concat initialized without previous config file...')
            if traceback==False:
                pass
        ## If we DO want to load previous config files, we enter this loop:
        if self.load_yaml==True:
            ## First we check if a config has previously been saved using these two data sets:
            if 'TONE_ACQ' in self.Data_Directory:
                tmpcorrdir=self.Data_Directory.split("_yale")[0].split("TONE_ACQ/")[1]
            elif 'NFandFF' in self.Data_Directory:
                tmpcorrdir=self.Data_Directory.split("_Suit")[0].split("NFandFF/")[1]
            tmpdronedir=self.FLYTAG.split('.')[0]
            tmpconfigpath=self.Config_Directory+'config_{}_{}.yaml'.format(tmpdronedir,tmpcorrdir)
            self.yaml_exists=os.path.exists(tmpconfigpath)
            ## If the previous config file doesn't exist, we will tell the user and move on:
            if self.yaml_exists==False:
                ## Check to see if we are going to use some print statements:
                if self.traceback==True:
                    print('Searching for previous config file:')
                    print('  --> Checking directory for config_{}_{}.yaml:'.format(tmpdronedir,tmpcorrdir))
                    print('    --> FILE NOT FOUND')
                    print('    --> Parameters can not be imported...')
                if self.traceback==False:
                    pass
            ## If the previous config file exists, we enter this loop to load parameters:
            if self.yaml_exists==True:
                ymlfile=open(self.Config_Directory+'config_{}_{}.yaml'.format(tmpdronedir,tmpcorrdir))
                #documents=yaml.full_load(ymlfile)
                with open(ymlfile, 'r') as fff:
                    documents = yaml.safe_load(fff)
                ## Check to see if the variables exist for function: Extract_Source_Pulses
                if 'pulse_params' in documents.keys():
                    self.pulse_period=np.float64(documents["pulse_params"]["pulse_period"])
                    self.pulse_dutycycle=np.float64(documents["pulse_params"]["pulse_dutycycle"])
                    self.t_delta_pulse=np.float64(documents["pulse_params"]["t_delta_pulse"])
                ## Check to see if the variables exist for function: Synchronization_Function
                if 'timing_params' in documents.keys():
                    self.t_delta_dji=np.float64(documents["timing_params"]["t_delta_dji"])
                    self.copolchan=documents["timing_params"]["copolchan"]
                ymlfile.close() # close the config file
                if self.traceback==True:
                    print('Searching for previous config file:')
                    print('  --> Checking directory for config_{}_{}.yaml:'.format(tmpdronedir,tmpcorrdir))
                    print('    --> FILE EXISTS')
                    print('    --> Loading existing config file parameters...')
                    if hasattr(self,"pulse_period")==True:
                        print('      --> pulse_period = {}'.format(self.pulse_period))
                    if hasattr(self,"pulse_dutycycle")==True:
                        print('      --> pulse_dutycycle = {}'.format(self.pulse_dutycycle))
                    if hasattr(self,"t_delta_pulse")==True:
                        print('      --> t_delta_pulse = {}'.format(self.t_delta_pulse))                        
                    if hasattr(self,"t_delta_dji")==True:
                        print('      --> t_delta_dji = {}'.format(self.t_delta_dji)) 
                    if hasattr(self,"copolchan")==True:
                        print('      --> copolchan = {}'.format(self.copolchan))
                if self.traceback==False:
                    pass
        ## Import vars from corr and drone classes:
        self.n_dishes=CORRDATCLASS.n_dishes
        self.n_channels=CORRDATCLASS.n_channels
        self.chmap=CORRDATCLASS.chmap
        self.automap=CORRDATCLASS.automap
        self.crossmap=CORRDATCLASS.crossmap
        self.origin=DRONEDATCLASS.origin
        self.prime_origin=DRONEDATCLASS.prime_origin
        self.dish_keystrings=DRONEDATCLASS.dish_keystrings
        self.dish_coords=DRONEDATCLASS.dish_coords
        self.dish_pointings=DRONEDATCLASS.dish_pointings
        self.dish_polarizations=DRONEDATCLASS.dish_polarizations
        ## Time dimensions of all arrays must be concat with receiver data. Time index is therefore defined wrt telescope data.
        self.freq=CORRDATCLASS.freq
        self.t=CORRDATCLASS.t
        self.t_index=CORRDATCLASS.t_index
        self.t_arr_datetime=CORRDATCLASS.t_arr_datetime
        self.V=CORRDATCLASS.V
        self.V_cross=CORRDATCLASS.V_cross
        ## Define lb and ub t_index corresponding to drone data start/stop times:
        drone_t_min=DRONEDATCLASS.t_arr_datetime[0]
        drone_t_max=DRONEDATCLASS.t_arr_datetime[-1]
        CORR_t_ind_lb=bisect.bisect_right(self.t_arr_datetime, drone_t_min)
        CORR_t_ind_ub=bisect.bisect_left(self.t_arr_datetime, drone_t_max)
        ## Define interpolation time vectors for drone and corr data:
        tsepoch=datetime.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.UTC)
        ds_CORR=np.array([(np.datetime64(ts).astype(datetime.datetime).replace(tzinfo=pytz.UTC)-tsepoch).total_seconds() for ts in self.t_arr_datetime[CORR_t_ind_lb:CORR_t_ind_ub]])
        ds_drone=np.array([(np.datetime64(ts).astype(datetime.datetime).replace(tzinfo=pytz.UTC)-tsepoch).total_seconds() for ts in DRONEDATCLASS.t_arr_datetime])
        ds_drone = ds_drone+t_drone_offset
        #ds_CORR=[(self.t_arr_datetime[n]-self.t_arr_datetime[CORR_t_ind_lb]).total_seconds() for n in self.t_index[CORR_t_ind_lb:CORR_t_ind_ub]]
        #ds_drone=[(DRONEDATCLASS.t_arr_datetime[m]-drone_t_min).total_seconds() for m in DRONEDATCLASS.t_index]
        if self.traceback==True:
            print("Interpolating drone coordinates for each correlator timestamp:")
            print("  --> correlator timestamp axis contains {} elements".format(len(ds_CORR)))
            print("  --> drone timestamp axis contains {} elements".format(len(ds_drone)))
        elif self.traceback==False:
            pass
        ## Create useful drone coordinate arrays which we must interp, NAN non valued elements:
        self.drone_llh_interp=np.nan*np.ones((self.t_arr_datetime.shape[0],3))
        self.drone_xyz_LC_interp=np.nan*np.ones((self.t_arr_datetime.shape[0],3))
        self.drone_rpt_interp=np.nan*np.ones((self.t_arr_datetime.shape[0],3))
        self.drone_yaw_interp=np.nan*np.ones(self.t_arr_datetime.shape[0])
        self.drone_xyz_per_dish_interp=np.nan*np.ones((DRONEDATCLASS.xyz_per_dish.shape[0],self.t_arr_datetime.shape[0],3))
        self.drone_rpt_r_per_dish_interp=np.nan*np.ones((DRONEDATCLASS.rpt_r_per_dish.shape[0],self.t_arr_datetime.shape[0],3))
        self.drone_rpt_t_per_dish_interp=np.nan*np.ones((DRONEDATCLASS.rpt_r_per_dish.shape[0],self.t_arr_datetime.shape[0],3))
        ## Interp Drone variables:
        for i in [0,1,2]:
            self.drone_llh_interp[CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.coords_llh[:,i])
            self.drone_xyz_LC_interp[CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.coords_xyz_LC[:,i])
            self.drone_rpt_interp[CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.coords_rpt[:,i])
            for j in range(DRONEDATCLASS.rpt_r_per_dish.shape[0]):            
                self.drone_xyz_per_dish_interp[j,CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.xyz_per_dish[j,:,i])
                self.drone_rpt_r_per_dish_interp[j,CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.rpt_r_per_dish[j,:,i])
                self.drone_rpt_t_per_dish_interp[j,CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.rpt_t_per_dish[j,:,i])
        self.drone_yaw_interp[CORR_t_ind_lb:CORR_t_ind_ub]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.yaw[:])
        self.tstep=1e-9*np.nanmedian(np.diff(self.t))

    def Extract_Source_Pulses(self,Period=0.4e6,Dutycycle=0.2e6,t_bounds=[0,-1],f_ind=900,minmaxpercents=[10.0,99.5]):
        ## Search for all three timing variables that must be loaded from config:
        if hasattr(self,"pulse_period")==True and hasattr(self,"pulse_dutycycle")==True and hasattr(self,"t_delta_pulse")==True:
            if self.traceback==True:
                print("Extracting Source Pulses using parameters loaded from config file:")
                print('  --> pulse_period = {}'.format(self.pulse_period))                        
                print('  --> pulse_dutycycle = {}'.format(self.pulse_dutycycle))                        
                print('  --> t_delta_pulse = {}'.format(self.t_delta_pulse))
            if self.traceback==False:
                pass
            concat_duration=int(np.ceil((self.t_arr_datetime[-1]-self.t_arr_datetime[0]).total_seconds()))
            time_s,time_dt,switch=tu.Pulsed_Data_Waveform(total_duration=concat_duration,period=self.pulse_period,duty_cycle_on=self.pulse_dutycycle)
        ## If we don't have any variables, then we haven't loaded a yaml yet... and must run the function:
        if hasattr(self,"t_delta_pulse")==False:
            ## Create Switch Signal
            self.pulse_period=Period
            self.pulse_dutycycle=Dutycycle
            concat_duration=int(np.ceil((self.t_arr_datetime[-1]-self.t_arr_datetime[0]).total_seconds()))
            time_s,time_dt,switch=tu.Pulsed_Data_Waveform(total_duration=concat_duration,period=self.pulse_period,duty_cycle_on=self.pulse_dutycycle)
            ## Create t_offset range (1 period) and Pearson_r vars:
            t_offset_dist=np.arange(-1.0*self.pulse_period*1e-6,0.0,0.001)
            Pr_arr=np.NaN*np.ones((self.n_channels,t_offset_dist.shape[0]))
            Pr_max_ind_per_channel=np.NaN*np.ones(self.n_channels)
            Pr_max_t_0_per_channel=np.NaN*np.ones(self.n_channels)
            t_full=np.array([(m-self.t_arr_datetime[0]).total_seconds() for m in self.t_arr_datetime[:]])
            ## Loop over channels to find/plot a time offset solution with some clever fitting:
            if self.traceback==True:
                fig1,ax1=subplots(nrows=1,ncols=1,figsize=(16,4))
            elif self.traceback==False:
                pass
            for i in range(self.n_channels):
                ## If we use a mean subtracted data cut we can find where power exceeds zero to find signal
                minsubdata=self.V[:,f_ind,i]-np.nanpercentile(self.V[:,f_ind,i],minmaxpercents[0])
                normminsubdata=minsubdata/np.nanpercentile(minsubdata,minmaxpercents[1])
                clipnormminsubdata=normminsubdata.clip(0,1)
                stepped_func=interp1d(t_full,clipnormminsubdata,kind='previous',fill_value='extrapolate')
                sniparr=np.where(time_s[np.where(time_s<=t_full[t_bounds[1]])[0]]>=t_full[t_bounds[0]])[0]
                t_restrict=np.intersect1d(np.arange(len(time_s))[~np.isnan(stepped_func(time_s))],sniparr)
                ## Loop over all time offsets in t_offset_dist to find maximum correlation between squarewave and data:
                for j,t_offset in enumerate(t_offset_dist):
                    shiftedswitch=np.interp(time_s,time_s+t_offset,switch)
                    try:
                        Pr_arr[i,j]=pearsonr(stepped_func(time_s[t_restrict]),shiftedswitch[t_restrict])[0]
                    except ValueError:
                        Pr_arr[i,j]=np.nan
                if self.traceback==True:
                    ax1.plot(t_offset_dist,Pr_arr[i,:],'.')
                elif self.traceback==False:
                    pass
                try:
                    maxPrind=np.where(Pr_arr[i,:]==np.nanmax(Pr_arr[i,:]))[0][0]
                    if self.traceback==True:
                        ax1.plot(t_offset_dist[maxPrind],Pr_arr[i,maxPrind],'ro')
                    elif self.traceback==False:
                        pass
                    Pr_max_ind_per_channel[i]=maxPrind
                    Pr_max_t_0_per_channel[i]=t_offset_dist[maxPrind]
                except IndexError:
                    Pr_max_ind_per_channel[i]=np.nan
                    Pr_max_t_0_per_channel[i]=np.nan            
            self.t_delta_pulse=np.nanmedian(Pr_max_t_0_per_channel)
            if self.traceback==True:
                ax1.axvline(self.t_delta_pulse,label="selected t_offset")
                ax1.legend(loc=1)
                tight_layout()
                print("Maximum Pearson_R Correlations between data and square wave function:") 
                print("  --> t_indices = {}".format(Pr_max_ind_per_channel))
                print("  --> t_deltas = {}".format(np.around(Pr_max_t_0_per_channel,decimals=3)))
                print("Selecting square wave function time offset:")
                print("  --> t_delta_pulse = {:.10f}".format(self.t_delta_pulse))
                if self.save_traceback==True:
                    savefig(self.Output_Directory+self.Output_Prefix+"_t_delta_pulse_Pearson_R.png")
                if self.save_traceback==False:
                    pass
            elif self.traceback==False:
                pass
        ## Interpolate the switching function with the concat timestamps using either input or found t_delta_pulse:
        t_for_interp_out=np.array([(m-self.t_arr_datetime[0]).total_seconds() for m in self.t_arr_datetime[:]])
        t_for_interp_in=np.array([m.total_seconds() for m in time_dt])
        switch_interp_f=np.interp(t_for_interp_out,t_for_interp_in+self.t_delta_pulse,switch)
        self.switch_signal=switch
        self.switch_time=t_for_interp_in
        self.switch_signal_interp=switch_interp_f
        ## Once we have our time offset, we must extract indices where the source is on/off/rising:
        self.inds_span=np.union1d(list(set(np.where(np.diff(np.sign(switch_interp_f-0.5)))[0])),\
                                  np.intersect1d(np.where(1.0>switch_interp_f),np.where(switch_interp_f>0.0))).tolist()
        self.inds_on=list(set(np.where(switch_interp_f==1.0)[0])-set(self.inds_span))
        self.inds_off=list(set(np.where(switch_interp_f==0.0)[0])-set(self.inds_span))
        ## Each of these lists of indices should also have no overlap. Let's print to see:
        if self.traceback==True:
            print("Finding relevant pulsing indices and checking for overlaps:")        
            print("  --> on/off ind intersection:",np.intersect1d(self.inds_on,self.inds_off))
            print("  --> on/span ind intersection:",np.intersect1d(self.inds_on,self.inds_span))
            print("  --> off/span ind intersection:",np.intersect1d(self.inds_off,self.inds_span))
            ## Let's plot the on/off/rising index groups:
            fig3=figure(figsize=(16,int(4*self.n_channels/2)))
            for i in range(self.n_channels):
                ax=fig3.add_subplot(int(self.n_channels/2),2,i+1)   
                ax.semilogy(self.t_arr_datetime[:],self.V[:,f_ind,i],'k.',label='all')
                ax.semilogy(self.t_arr_datetime[self.inds_on],self.V[self.inds_on,f_ind,i],'.',label='on')
                ax.semilogy(self.t_arr_datetime[self.inds_off],self.V[self.inds_off,f_ind,i],'.',label='off')   
                ax.semilogy(self.t_arr_datetime[self.inds_span],self.V[self.inds_span,f_ind,i],'x',label='span')
                ax.semilogy(self.t_arr_datetime[:],(np.nanmax(self.V[self.inds_on,f_ind,i])*switch_interp_f)+np.nanmin(self.V[self.inds_on,f_ind,i]),'--',alpha=0.1,label='switch, t_offset={:.2f}'.format(self.t_delta_pulse))
                ax.set_ylabel("Log Power Received [$ADU^2$]")
                ax.set_xlabel("Datetime")
                ax.set_title("Channel {}".format(self.chmap[i]))
                ax.legend(loc=2)
                ax.set_xlim(self.t_arr_datetime[t_bounds[0]],self.t_arr_datetime[t_bounds[1]])
            tight_layout()
            if self.save_traceback==True:
                savefig(self.Output_Directory+self.Output_Prefix+"_t_delta_pulse_index_solution.png")
            if self.save_traceback==False:
                pass            
        if self.traceback==False:
            pass
        
    def Perform_Background_Subtraction(self,window_size=5):
        ## BACKGROUND SUBTRACTED SPECTRA: ##
        if self.traceback==True:
            print("Calculating background spectra from indices where the noise source is off.")
        if self.traceback==False:
            pass
        self.V_bg=np.zeros(self.V.shape)
        self.V_bgsub=np.zeros(self.V.shape)
        if self.crossmap!=None:
            self.V_cross_bg=np.zeros(self.V_cross.shape).astype(complex)
            self.V_cross_bgsub=np.zeros(self.V_cross.shape).astype(complex)
        ## Loop over all indices and construct the V_bg array:
        for k,ind in enumerate(self.t_index):
            ## If ind is an off spectra, use this off spectra for the background:
            if k in self.inds_off:
                self.V_bg[k,:,:]=self.V[k,:,:]
                if self.crossmap!=None:
                    self.V_cross_bg[k,:,:]=self.V_cross[k,:,:]
            ## If ind is an on spectra, create an off spectra by averaging the before/after off spectra:
            elif k in np.union1d(self.inds_on,self.inds_span):
                t_window=np.intersect1d(np.arange(k-window_size,k+window_size),self.inds_off)
                self.V_bg[k,:,:]=np.nanmean(self.V[t_window,:,:],axis=0)
                if self.crossmap!=None:
                    self.V_cross_bg[k,:,:]=np.nanmean(self.V_cross[t_window,:,:],axis=0)
        self.V_bgsub=self.V-self.V_bg
        if self.crossmap!=None:
            self.V_cross_bgsub=self.V_cross-self.V_cross_bg
        if self.traceback==True:
            print("  --> Background subtraction completed using window_size = {}".format(window_size))
        if self.traceback==False:
            pass
        
    def Synchronization_Function(self,inputcorr,inputdrone,coarse_params=[-10.0,10.0,0.2],fine_params=[-0.5,0.5,0.01],chans=np.array([0,1]),freqs=np.arange(100,1024,150),FMB_coordbounds=[30.0,30.0,150.0],FMB_ampbound=0.999):
        if self.traceback==True:
            print("Synchronizing data from correlator and drone:")
        if self.traceback==False:
            pass
        ## Begin by checking if the t_delta_dji parameter was already loaded from the config file:
        if hasattr(self,"t_delta_dji")==True:
            if self.traceback==True:
                print('  --> Loading parameter from existing configuration file: t_delta_dji = {}'.format(self.t_delta_dji))
                print("Applying a time correction of {:.2f} seconds using Channel {} fits.".format(self.t_delta_dji,self.copolchan))
            if self.traceback==False:
                pass
        if hasattr(self,"t_delta_dji")==False:
            if self.traceback==True:
                print("  --> Previous t_delta_dji not found") 
                print("  --> Calculating via 2DGauss fitting routine:")
            if self.traceback==False:
                pass
            ## Begin with specifying time axis for iteration:
            t_coarse=np.arange(coarse_params[0],coarse_params[1],coarse_params[2])
            ## Define output products from fits:
            AFit_f_params=np.zeros((len(t_coarse),len(chans),len(freqs),5))
            APRarr=np.zeros((len(t_coarse),len(chans),len(freqs)))
            GFit_f_params=np.zeros((len(t_coarse),len(chans),len(freqs),7))
            GPRarr=np.zeros((len(t_coarse),len(chans),len(freqs)))
            ## Begin iterative loop for coarse time axis:
            for i,ttry in enumerate(t_coarse):
                origtaxis=inputdrone.t_arr_datetime[:]
                tempdrone=inputdrone
                tempdrone.t_arr_datetime=inputdrone.t_arr_datetime+datetime.timedelta(seconds=ttry)
                tempconcat=concat.CONCAT(CORRDATCLASS=inputcorr,DRONEDATCLASS=tempdrone,load_yaml=False,traceback=False,save_traceback=False)
                try:
                    tempconcat.inds_on=self.inds_on
                except AttributeError:
                    pass
                inputdrone.t_arr_datetime=origtaxis
                ## Run fits:
                result=fu.Fit_Main_Beam(tempconcat,chans,freqs,theta_solve=False,coordbounds=FMB_coordbounds,ampbound=FMB_ampbound)
                AFit_f_params[i]=result[0]
                APRarr[i]=result[1]
                GFit_f_params[i]=result[2]
                GPRarr[i]=result[3]
            ## Loop over the channels and frequencies to find the index that maximizes the Pearson R Array:
            GPRmax=np.zeros((len(chans),len(freqs)))
            GPRval=np.zeros((len(chans),len(freqs)))
            for i in range(len(chans)):
                for j in range(len(freqs)):
                    try:
                        GPRmax[i,j]=np.where(GPRarr[:,i,j]==np.nanmax(GPRarr[:,i,j]))[0][0]
                        GPRval[i,j]=GPRarr[int(GPRmax[i,j]),i,j]
                    except IndexError:
                        GPRmax[i,j]=np.nan
                        GPRval[i,j]=np.nan
            copolchan=np.where(np.nanmean(GPRval,axis=1)==np.nanmax(np.nanmean(GPRval,axis=1)))[0][0]
            ## Redefine time axis for fine resolution:
            tfine0=t_coarse[int(np.nanmedian(GPRmax[copolchan,:]))]
            t_fine=np.arange(tfine0+fine_params[0],tfine0+fine_params[1],fine_params[2])
            ## Define output products from fine resolution fits:
            AFit_f_params_fine=np.zeros((len(t_fine),len(chans),len(freqs),5))
            APRarr_fine=np.zeros((len(t_fine),len(chans),len(freqs)))
            GFit_f_params_fine=np.zeros((len(t_fine),len(chans),len(freqs),7))
            GPRarr_fine=np.zeros((len(t_fine),len(chans),len(freqs)))
            ## Begin iterative loop for coarse time axis:    
            for i,ttry in enumerate(t_fine):
                origtaxis=inputdrone.t_arr_datetime[:]
                tempdrone=inputdrone
                tempdrone.t_arr_datetime=inputdrone.t_arr_datetime+datetime.timedelta(seconds=ttry)
                tempconcat=concat.CONCAT(CORRDATCLASS=inputcorr,DRONEDATCLASS=tempdrone,load_yaml=False,traceback=False,save_traceback=False)
                try:
                    tempconcat.inds_on=self.inds_on
                except AttributeError:
                    pass
                inputdrone.t_arr_datetime=origtaxis
                ## Run fits:
                result=fu.Fit_Main_Beam(tempconcat,chans,freqs,theta_solve=False,coordbounds=FMB_coordbounds,ampbound=FMB_ampbound)
                AFit_f_params_fine[i]=result[0]
                APRarr_fine[i]=result[1]
                GFit_f_params_fine[i]=result[2]
                GPRarr_fine[i]=result[3]
            ## Find the maximal Pearson R values for the fine time axis:
            GPRmax_fine=np.zeros((len(chans),len(freqs)))
            GPRval_fine=np.zeros((len(chans),len(freqs)))
            for i in range(len(chans)):
                for j in range(len(freqs)):
                    try:
                        GPRmax_fine[i,j]=np.where(GPRarr_fine[:,i,j]==np.nanmax(GPRarr_fine[:,i,j]))[0][0]
                        GPRval_fine[i,j]=GPRarr_fine[int(GPRmax_fine[i,j]),i,j]
                    except IndexError:
                        GPRmax_fine[i,j]=np.nan
                        GPRval_fine[i,j]=np.nan
            ## Which channel has the best gaussianity, and what time offset does that channel suggest?
            copolchan_fine=np.where(np.nanmean(GPRval_fine,axis=1)==np.nanmax(np.nanmean(GPRval_fine,axis=1)))[0][0]
            self.t_delta_dji=t_fine[int(np.nanmedian(GPRmax_fine[copolchan_fine,:]))]
            self.copolchan=copolchan_fine
            ## Now make plots if desired:
            if self.traceback==True:
                print("Applying a time correction of {:.2f} seconds using Channel {} fits.".format(self.t_delta_dji,self.copolchan))
                fig0,[[ax1,ax2],[ax3,ax4]]=subplots(nrows=2,ncols=2,figsize=(16,12))
                for i,ax in enumerate([ax1,ax2]):
                    for k,find in enumerate(freqs):
                        ax.plot(t_coarse,GPRarr[:,i,k],label='{:.2f}MHz'.format(tempconcat.freq[find]))
                        ax.plot(t_coarse[int(GPRmax[i,k])],GPRarr[int(GPRmax[i,k]),i,k],'r.')
                    ax.axvline(t_coarse[int(np.nanmedian(GPRmax[i]))],c='r',label='median t = {:.2f}'.format(t_coarse[int(np.nanmedian(GPRmax[i]))]))
                    ax.set_title('Channel {} Coarse Offset Correlation'.format(chans[i]))
                    ax.set_xlabel('$\Delta$t $[sec]$')
                    ax.set_ylabel('Pearson R Value')
                    ax.legend(loc=1,fontsize='small')
                for i,ax in enumerate([ax3,ax4]):
                    for k,find in enumerate(freqs):
                        ax.plot(t_fine,GPRarr_fine[:,i,k],label='{:.2f}MHz'.format(tempconcat.freq[find]))
                        ax.plot(t_fine[int(GPRmax_fine[i,k])],GPRarr_fine[int(GPRmax_fine[i,k]),i,k],'r.')
                    ax.axvline(t_fine[int(np.nanmedian(GPRmax_fine[i]))],c='r',label='median t = {:.2f}'.format(t_fine[int(np.nanmedian(GPRmax_fine[i]))]))
                    ax.set_title('Channel {} Coarse Offset Correlation'.format(chans[i]))
                    ax.set_xlabel('$\Delta$t $[sec]$')
                    ax.set_ylabel('Pearson R Value')
                    ax.legend(loc=1,fontsize='small')
                tight_layout()
                if self.save_traceback==True:
                    savefig(self.Output_Directory+self.Output_Prefix+"_t_delta_dji_Pearson_R.png")
                if self.save_traceback==False:
                    pass
            elif self.traceback==False:
                pass
        ## END of config interpretation loop!
        ## Now shift the drone time axis by t_delta_dji and reinterpolate the coordinates, then reassign the variables:
        origtaxis=inputdrone.t_arr_datetime[:]
        tempdrone=inputdrone
        tempdrone.t_arr_datetime=inputdrone.t_arr_datetime+datetime.timedelta(seconds=self.t_delta_dji)
        tempconcat=concat.CONCAT(CORRDATCLASS=inputcorr,DRONEDATCLASS=tempdrone,load_yaml=False,traceback=False,save_traceback=False)
        self.drone_llh_interp=tempconcat.drone_llh_interp
        self.drone_xyz_LC_interp=tempconcat.drone_xyz_LC_interp
        self.drone_rpt_interp=tempconcat.drone_rpt_interp
        self.drone_yaw_interp=tempconcat.drone_yaw_interp
        self.drone_xyz_per_dish_interp=tempconcat.drone_xyz_per_dish_interp
        self.drone_rpt_r_per_dish_interp=tempconcat.drone_rpt_r_per_dish_interp
        self.drone_rpt_t_per_dish_interp=tempconcat.drone_rpt_t_per_dish_interp
        inputdrone.t_arr_datetime=origtaxis
        if self.traceback==True:
            print('  --> Variable synchronation successful, generating output plots:')
            channels=chans
            pu.Synchronization_Verification_Plots(inputconcat=self,chans=channels,find=freqs[-1],coordbounds=FMB_coordbounds,ampbound=FMB_ampbound)
            if self.save_traceback==True:
                print('  --> Saving output plot.')
                savefig(self.Output_Directory+self.Output_Prefix+"_Synchronization_Verification.png")
            if self.save_traceback==False:
                pass
        if self.traceback==False:
            pass
        ## This should result in reassigned drone coordinates, and eliminate a lot of computation time...
        
    def Export_yaml(self,):
        ## Create the file on disk at the specified path and write an initialization comment:
        tmpcorrdir=(self.Data_Directory.split("Z_")[0]+'Z').split("/")[-1]
        tmpdronedir=self.FLYTAG.split('.')[0]
        tmpconfigpath=self.Config_Directory+'config_{}_{}.yaml'.format(tmpdronedir,tmpcorrdir)
        if self.traceback==True:
            print('Preparing to export configuration file:')
            print('  --> Checking directory for config_{}_{}.yaml:'.format(tmpdronedir,tmpcorrdir))
            if os.path.exists(tmpconfigpath)==False:
                print('    --> FILE NOT FOUND')
                print('    --> preparing to write a new configuration file...')
            if os.path.exists(tmpconfigpath)==True:
                print('    --> FILE EXISTS')
                print('    --> preparing to write a new versioned configuration file...')
        if self.traceback==False:
            pass
        ## Create the configuration file at the specified location on disk:
        if os.path.exists(tmpconfigpath)==False:                
            ymlfile=open(self.Config_Directory+'config_{}_{}.yaml'.format(tmpdronedir,tmpcorrdir), 'w')
        if os.path.exists(tmpconfigpath)==True:
            try:
                ymlfile=open(self.Config_Directory+'config_'+self.Output_Prefix+'.yaml', 'w')
            except AttributeError:
                suff=datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                ymlfile=open(self.Config_Directory+'config_{}_{}_ver_{}.yaml'.format(tmpdronedir,tmpcorrdir,suff), 'w')
        ymlfile.write('#YAML_CONFIG_FILE for {} and {}\n'.format(tmpdronedir,tmpcorrdir))
        ## (0) Write comments that indicate whether or not certain functions worked on this data set:
        ymlfile.write('#LIST OF SUCCESFULLY RUN CONCAT CLASS FUNCTIONS:\n')
        if hasattr(self,'V')==True:
            ymlfile.write('# --> [X] concat.__init__()\n')
        if hasattr(self,'switch_signal_interp')==True:
            ymlfile.write('# --> [X] concat.Extract_Source_Pulses()\n')
        if hasattr(self,'V_bgsub')==True:
            ymlfile.write('# --> [X] concat.Perform_Background_Subtraction()\n')
        if hasattr(self,'t_delta_dji')==True:
            ymlfile.write('# --> [X] concat.Synchronization_Function()\n')
        ymlfile.write('\n') # new line
        ## (1) These file info parameters will by definition always exist:
        file_info={'file_info':{'name':self.name,
                                'Data_Directory':self.Data_Directory,
                                'filenames':self.filenames.astype(str).tolist(),
                                'Drone_Directory':self.Drone_Directory,
                                'FLYTAG':self.FLYTAG,
                                'Gain_Directory':self.Gain_Directory,
                                'gainfile':self.gainfile}}
        ## Write a comment about and insert the file info section into the .yaml:
        ymlfile.write('#FILENAMES AND PATHS ON DISK TO CONCATENATED FILES:\n')
        yaml.dump(file_info, ymlfile, default_flow_style=None)
        ymlfile.write('\n') # new line
        ## (2) These site geometry parameters will by definition always exist:
        site_info={'site_geometry_i':{'n_channels':self.n_channels,
                                    'n_dishes':self.n_dishes,
                                    'automap':self.automap.tolist(),
                                    'chmap':self.chmap.tolist(),
                                    'dish_keystrings':self.dish_keystrings.tolist(),
                                    'dish_coords':self.dish_coords.tolist(),
                                    'dish_pointings':self.dish_pointings.tolist(),
                                    'dish_polarizations':self.dish_polarizations.tolist()}}
        ## Write a comment about and insert the site info section into the .yaml:
        ymlfile.write('#INITIAL SITE GEOMETRY PARAMETERS AND VARIABLES USED:\n')
        yaml.dump(site_info, ymlfile, default_flow_style=None)
        ymlfile.write('\n') # new line
        ## (3) These correlator parameters will by definition always exist:
        corr_info={'corr_params':{'tstep':float(self.tstep),
                                  'fstep':float(np.nanmedian(np.diff(self.freq))),
                                  'dimensions':list(self.V.shape)}}
        ## Write a comment about and insert the site info section into the .yaml:
        ymlfile.write('#CORRELATOR PARAMETERS AND VARIABLES USED:\n')
        yaml.dump(corr_info, ymlfile, default_flow_style=None)
        ymlfile.write('\n') # new line
        ## (4) These pulse parameters will only exist if the source was pulsed and Extract_Source_Pulses() was run:
        try:
            pulse_info={'pulse_params':{'pulse_dutycycle':self.pulse_dutycycle,
                                        'pulse_period':self.pulse_period,
                                        't_delta_pulse':float(self.t_delta_pulse)}}
            ymlfile.write('#SOURCE PULSING PARAMETERS (microsec):\n')
            yaml.dump(pulse_info, ymlfile)
        except AttributeError:
            ymlfile.write('#SOURCE PULSING PARAMETERS NOT FOUND -- REQUIRES Extract_Source_Pulses()\n')
        ## Write a comment about and insert the pulse info section into the .yaml:
        ymlfile.write('\n') # new line
        ## (5) These timing parameters will only exist if Synchronization_Function was run:
        try:
            timing_info={'timing_params':{'t_delta_dji':float(self.t_delta_dji),
                                          'copolchan':float(self.copolchan)}}
            ymlfile.write('#BEST-FIT DRONE TIMESTAMP TIMING OFFSET (seconds):\n')
            yaml.dump(timing_info, ymlfile)
        except AttributeError:
            ymlfile.write('#BEST-FIT DRONE TIMESTAMP TIMING OFFSET NOT FOUND: REQUIRES Synchronization_Function()\n')
        ## Write a comment about and insert the file info section into the .yaml:
        ymlfile.close()
        if self.traceback==True:
            print('    --> file saved successfully')
        if self.traceback==False:
            pass
    
    def Main_Beam_Fitting(self,fit_param_directory='/hirax/GBO_Analysis_Outputs/main_beam_fits/',freqs=np.arange(1024),theta_solve=False,FMB_ampbound=0.999,coordbounds=[50.0,50.0,150.0],Vargs='None'):
        if self.traceback==True:
            print('Performing 2DGauss and Airy fits for [{}]chans x [{}]freqs:'.format(self.n_channels,len(freqs)))
        if self.traceback==False:
            pass
        A_popt,A_PR,G_popt,G_PR=fu.Fit_Main_Beam(inputconcat=self,chans=range(self.n_channels),freqs=freqs,coordbounds=coordbounds,theta_solve=theta_solve,ampbound=FMB_ampbound,Vargs=Vargs)
        self.A_popt=A_popt
        self.A_PR=A_PR
        self.G_popt=G_popt
        self.G_PR=G_PR
        if self.traceback==True:
            if self.save_traceback==True:
                print('  --> Saving output fit parameters as an .npz filetype:')
                tmpfitpath=fit_param_directory+self.Output_Prefix+'_2dGauss_and_Airy_Params.npz'
                np.savez(tmpfitpath,A_popt=A_popt,A_PR=A_PR,G_popt=G_popt,G_PR=G_PR)      
                print('  --> {}'.format(self.Output_Prefix+'_2dGauss_and_Airy_Params.npz'))
        if self.traceback==False:
            pass
        
    def Distance_Compensation(self,f_ind=900,plot_channels=[0]):
        if hasattr(self,'V_bgsub'):
            if hasattr(self,'drone_xyz_per_dish_interp'):
                if self.traceback==True:
                    print('Applying correction to V_bgsub using concat.Distance_Compensation() Function:')
        r2coeffmatrix=np.NaN*np.ones((int(self.V.shape[0]),1,int(self.V.shape[2])))
        rmin=np.zeros(self.n_channels).astype(int)
        r0=np.zeros(self.n_channels)
        for i in range(self.n_channels):
            rdist=np.abs(np.sqrt(((self.drone_xyz_per_dish_interp[i,:,0]**2.0)+(self.drone_xyz_per_dish_interp[i,:,1]**2.0)+(self.drone_xyz_per_dish_interp[i,:,2]**2.0))))
            rmin[i]=int(np.where(rdist==np.nanmin(rdist))[0][0])
            r0[i]=rdist[rmin[i]]
            r2coeffmatrix[:,:,i]=((rdist**2.0)/(r0[i]**2.0)).reshape((len(rdist),1))
        ## Multiply V_bgsub by the distance compensation matrix 
        compensated_V_bgsub=r2coeffmatrix*self.V_bgsub
        ## Plot with the traceback:
        if self.traceback==True:
            fig,[[ax1,ax2],[ax3,ax4],[ax5,ax6]]=subplots(nrows=3,ncols=2,figsize=(18,12))
            mininds=np.zeros(self.n_channels).astype(int)
            for k in plot_channels:
                ## Grab minima for plotting
                mininds[k]=np.where(r2coeffmatrix[:,0,k]==np.nanmin(r2coeffmatrix[:,0,k]))[0][0]
                ## Plot r2coeffmatrix vs position
                ax1.plot(self.drone_xyz_per_dish_interp[k,:,0],r2coeffmatrix[:,0,k],'.',alpha=0.05,label='Channel {}'.format(k))
                ax1.plot(self.drone_xyz_per_dish_interp[k,mininds[k],0],r2coeffmatrix[mininds[k],0,k],'rx')
                ax2.plot(self.drone_xyz_per_dish_interp[k,:,1],r2coeffmatrix[:,0,k],'.',alpha=0.05,label='Channel {}'.format(k))
                ax2.plot(self.drone_xyz_per_dish_interp[k,mininds[k],1],r2coeffmatrix[mininds[k],0,k],'rx')         
                ## Plot before/after spectra vs x
                ax3.plot(self.drone_xyz_per_dish_interp[k,:,0][self.inds_on],self.V_bgsub[:,f_ind,k][self.inds_on],'.',alpha=0.25,label='Ch{} Before'.format(k))
                ax3.plot(self.drone_xyz_per_dish_interp[k,:,0][self.inds_on],compensated_V_bgsub[:,f_ind,k][self.inds_on],'.',alpha=0.25,label='Ch{} After'.format(k))
                ax3.axvline(self.drone_xyz_per_dish_interp[k,mininds[k],0])
                ax4.plot(self.drone_xyz_per_dish_interp[k,:,0][self.inds_on],self.V_bgsub[:,f_ind,k+1][self.inds_on],'.',alpha=0.25,label='Ch{} Before'.format(k+1))
                ax4.plot(self.drone_xyz_per_dish_interp[k,:,0][self.inds_on],compensated_V_bgsub[:,f_ind,k+1][self.inds_on],'.',alpha=0.25,label='Ch{} After'.format(k+1))
                ax4.axvline(self.drone_xyz_per_dish_interp[k,mininds[k],0])
                ## Plot before/after spectra vs y
                ax5.plot(self.drone_xyz_per_dish_interp[k,:,1][self.inds_on],self.V_bgsub[:,f_ind,k][self.inds_on],'.',alpha=0.25,label='Ch{} Before'.format(k))
                ax5.plot(self.drone_xyz_per_dish_interp[k,:,1][self.inds_on],compensated_V_bgsub[:,f_ind,k][self.inds_on],'.',alpha=0.25,label='Ch{} After'.format(k))
                ax5.axvline(self.drone_xyz_per_dish_interp[k,mininds[k],1])
                ax6.plot(self.drone_xyz_per_dish_interp[k,:,1][self.inds_on],self.V_bgsub[:,f_ind,k+1][self.inds_on],'.',alpha=0.25,label='Ch{} Before'.format(k+1))
                ax6.plot(self.drone_xyz_per_dish_interp[k,:,1][self.inds_on],compensated_V_bgsub[:,f_ind,k+1][self.inds_on],'.',alpha=0.25,label='Ch{} After'.format(k+1))
                ax6.axvline(self.drone_xyz_per_dish_interp[k,mininds[k],1])
            for ax in [ax1,ax2]:
                ax.plot([],[],'rx',label='minima')
            titles=['Distance_Compensation Matrix','Distance_Compensation Matrix','Xpol','Ypol','Xpol','Ypol']
            xlabels=['x','y','x','x','y','y']
            ylabels=['$DCM[r]$','$DCM[r]$','$ADU^2$','$ADU^2$','$ADU^2$','$ADU^2$']
            for ai,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
                ax.set_title(titles[ai])
                ax.set_xlabel(xlabels[ai])
                ax.set_ylabel(ylabels[ai])
                ax.legend(loc=1)
            tight_layout()
            ## Save figure?
            if self.save_traceback==True:
                print('  --> Saving output plot.')
                savefig(self.Output_Directory+self.Output_Prefix+"_Distance_Compensation_Verification.png")
            ## Redefine variable V_bgsub:
            print('  --> Complete.')
            del self.V_bgsub
            self.V_bgsub=compensated_V_bgsub

