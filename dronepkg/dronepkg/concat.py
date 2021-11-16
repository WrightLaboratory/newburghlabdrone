## Time array isn't valid because of interpolation bug, since drone data starts before telescope data...
## Fix interpolation axes of ds_CORR and ds_drone to solve this issue...?

from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import h5py
import os
import glob
from matplotlib import colors
import pandas
import csv
import datetime
import pytz
import bisect
import pygeodesy
from scipy.signal import square
from scipy.stats import pearsonr

## Import packages from our own module:
import dronepkg.plotting_utils as pu
import dronepkg.fitting_utils as fu
import dronepkg.geometry_utils as gu
import dronepkg.time_utils as tu

class CONCAT:
    def __init__(self,CORRDATCLASS,DRONEDATCLASS):
        print('Initializing CONCAT CLASS using:')
        print(" --> "+CORRDATCLASS.Data_Directory)
        print(" --> "+DRONEDATCLASS.FLYTAG)
        ## Import file information from both corr and drone classes:
        self.name=DRONEDATCLASS.name
        self.Data_Directory=CORRDATCLASS.Data_Directory
        self.Gain_Directory=CORRDATCLASS.Gain_Directory
        self.filenames=CORRDATCLASS.filenames
        self.gainfile=CORRDATCLASS.gainfile
        self.Drone_Directory=DRONEDATCLASS.Drone_Directory
        self.FLYTAG=DRONEDATCLASS.FLYTAG
        ## Import vars from corr and drone classes:
        self.n_dishes=CORRDATCLASS.n_dishes
        self.n_channels=CORRDATCLASS.n_channels
        self.chmap=CORRDATCLASS.chmap
        self.automap=CORRDATCLASS.automap
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
        ## Define lb and ub t_index corresponding to drone data start/stop times:
        drone_t_min=DRONEDATCLASS.t_arr_datetime[0]
        drone_t_max=DRONEDATCLASS.t_arr_datetime[-1]
        CORR_t_ind_lb=bisect.bisect_right(self.t_arr_datetime, drone_t_min)
        CORR_t_ind_ub=bisect.bisect_left(self.t_arr_datetime, drone_t_max)
        ## Define interpolation time vectors for drone and corr data:
        tsepoch=datetime.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.UTC)
        ds_CORR=np.array([(np.datetime64(ts).astype(datetime.datetime).replace(tzinfo=pytz.UTC)-tsepoch).total_seconds() for ts in self.t_arr_datetime[CORR_t_ind_lb:CORR_t_ind_ub]])
        ds_drone=np.array([(np.datetime64(ts).astype(datetime.datetime).replace(tzinfo=pytz.UTC)-tsepoch).total_seconds() for ts in DRONEDATCLASS.t_arr_datetime])
        #ds_CORR=[(self.t_arr_datetime[n]-self.t_arr_datetime[CORR_t_ind_lb]).total_seconds() for n in self.t_index[CORR_t_ind_lb:CORR_t_ind_ub]]
        #ds_drone=[(DRONEDATCLASS.t_arr_datetime[m]-drone_t_min).total_seconds() for m in DRONEDATCLASS.t_index]
        print("Interpolating drone coordinates for each correlator timestamp:")
        print("  --> correlator timestamp axis contains {} elements".format(len(ds_CORR)))
        print("  --> drone timestamp axis contains {} elements".format(len(ds_drone)))
        ## Create useful drone coordinate arrays which we must interp, NAN non valued elements:
        self.drone_llh_interp=np.NAN*np.ones((self.t_arr_datetime.shape[0],3))
        self.drone_xyz_LC_interp=np.NAN*np.ones((self.t_arr_datetime.shape[0],3))
        self.drone_rpt_interp=np.NAN*np.ones((self.t_arr_datetime.shape[0],3))
        self.drone_yaw_interp=np.NAN*np.ones(self.t_arr_datetime.shape[0])
        self.drone_rpt_r_per_dish_interp=np.NAN*np.ones((DRONEDATCLASS.rpt_r_per_dish.shape[0],self.t_arr_datetime.shape[0],3))
        self.drone_rpt_t_per_dish_interp=np.NAN*np.ones((DRONEDATCLASS.rpt_r_per_dish.shape[0],self.t_arr_datetime.shape[0],3))
        ## Interp Drone variables:
        for i in [0,1,2]:
            self.drone_llh_interp[CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.coords_llh[:,i])
            self.drone_xyz_LC_interp[CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.coords_xyz_LC[:,i])
            self.drone_rpt_interp[CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.coords_rpt[:,i])
            for j in range(DRONEDATCLASS.rpt_r_per_dish.shape[0]):            
                self.drone_rpt_r_per_dish_interp[j,CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.rpt_r_per_dish[j,:,i])
                self.drone_rpt_t_per_dish_interp[j,CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.rpt_t_per_dish[j,:,i])
        self.drone_yaw_interp[CORR_t_ind_lb:CORR_t_ind_ub]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.yaw[:])

    def Extract_Source_Pulses(self,Period=0.4e6,Dutycycle=0.2e6,t_bounds=[0,-1],f_ind=[900],half_int_period=0.021):
        ## Create Switch Signal
        self.pulse_period=Period
        self.pulse_dutycycle=Dutycycle
        concat_duration=int(np.ceil((self.t_arr_datetime[-1]-self.t_arr_datetime[0]).total_seconds()))
        time_s,time_dt,switch=tu.Pulsed_Data_Waveform(total_duration=concat_duration,period=self.pulse_period,duty_cycle_on=self.pulse_dutycycle)
        ## Create t_offset range (1 period) and Pearson_r vars:
        t_offset_dist=np.linspace(-1.0*self.pulse_period*1e-6,0.0,1000)
        Pr_arr=np.zeros((self.n_channels,t_offset_dist.shape[0]))
        Pr_max_ind_per_channel=np.zeros(self.n_channels)
        Pr_max_t_0_per_channel=np.zeros(self.n_channels)
        ## Define bounds for plotting later on:
        cdtlb,cdtub=t_bounds
        ## Loop over channels to find/plot a time offset solution with some clever fitting:
        fig1,ax1=subplots(nrows=1,ncols=1,figsize=(16,4))
        for i in range(self.n_channels):
            ## If we use a mean subtracted data cut we can find where power exceeds zero to find signal
            minsubdata=self.V[:,f_ind,i]-np.nanmin(self.V[:,f_ind,i])
            normminsubdata=minsubdata/np.nanmax(minsubdata)
            t_full=np.array([(m-self.t_arr_datetime[0]).total_seconds() for m in self.t_arr_datetime[:]])
            ## Loop over all time offsets in t_offset_dist to find maximum correlation between squarewave and data:
            for j,t_offset in enumerate(t_offset_dist):
                shiftedswitch=np.interp(t_full,time_s+t_offset,switch)
                try:
                    Pr_arr[i,j]=pearsonr(normminsubdata.flatten(),shiftedswitch.flatten())[0]
                except ValueError:
                    Pr_arr[i,j]=np.NAN
            ax1.plot(t_offset_dist,Pr_arr[i,:],'.')
            try:
                maxPrind=np.where(Pr_arr[i,:]==np.nanmax(Pr_arr[i,:]))[0][0]
                ax1.plot(t_offset_dist[maxPrind],Pr_arr[i,maxPrind],'ro')
                Pr_max_ind_per_channel[i]=maxPrind
                Pr_max_t_0_per_channel[i]=t_offset_dist[maxPrind]
            except IndexError:
                Pr_max_ind_per_channel[i]=np.NAN
                Pr_max_t_0_per_channel[i]=np.NAN
        t_offset_global=np.nanmedian(Pr_max_t_0_per_channel)+half_int_period # 1/2 integration period
        ax1.axvline(t_offset_global,label="t_offset with half-int-period")
        ax1.axvline(t_offset_global-half_int_period,label="t_offset without half-int-period")
        ax1.legend(loc=1)
        print("Maximum Pearson_R Correlations:") 
        print("  --> t_indices = {}".format(Pr_max_ind_per_channel))
        print("  --> t_offsets = {}".format(Pr_max_t_0_per_channel))
        print("Selecting global time offset:")
        print("  --> global_t_offset = {:.10f}".format(t_offset_global))
        ## Interpolate the switching function with the concat timestamps:
        t_for_interp_out=np.array([(m-self.t_arr_datetime[0]).total_seconds() for m in self.t_arr_datetime])
        t_for_interp_in=np.array([m.total_seconds() for m in time_dt])
        switch_interp_f=np.interp(t_for_interp_out,t_for_interp_in+t_offset_global,switch)
        self.switch_signal=switch
        self.switch_time=t_for_interp_in
        self.switch_signal_interp=switch_interp_f
        ## Once we have our time offset, we must extract indices where the source is on/off/rising:
        print("Finding relevant pulsing indices and checking for overlaps:")
        self.inds_span=np.union1d(list(set(np.where(np.diff(np.sign(switch_interp_f-0.5)))[0])),\
                                  np.intersect1d(np.where(1.0>switch_interp_f),np.where(switch_interp_f>0.0))).tolist()
        self.inds_on=list(set(np.where(switch_interp_f==1.0)[0])-set(self.inds_span))
        self.inds_off=list(set(np.where(switch_interp_f==0.0)[0])-set(self.inds_span))
        ## Each of these lists of indices should also have no overlap. Let's print to see:
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
            ax.semilogy(self.t_arr_datetime[:],(np.nanmax(self.V[self.inds_on,f_ind,i])*switch_interp_f)+np.nanmin(self.V[self.inds_on,f_ind,i]),'--',alpha=0.1,label='switch, t_offset={:.2f}'.format(t_offset_dist[maxPrind]))
            ax.set_ylabel("Log Power Received [$ADU^2$]")
            ax.set_xlabel("Datetime")
            ax.set_title("Channel {}".format(self.chmap[i]))
            ax.legend(loc=2)
            ax.set_xlim(self.t_arr_datetime[cdtlb],self.t_arr_datetime[cdtub])
        tight_layout()
        
    def Perform_Background_Subtraction(self,window_size=5):
        ## BACKGROUND SUBTRACTED SPECTRA: ##
        self.V_bg=np.zeros(self.V.shape)
        self.V_bgsub=np.zeros(self.V.shape)
        ## Loop over all indices and construct the V_bg array:
        for k,ind in enumerate(self.t_index):
            ## If ind is an off spectra, use this off spectra for the background:
            if k in self.inds_off:
                #print(k,"poo")
                self.V_bg[k,:,:]=self.V[k,:,:]
            ## If ind is an on spectra, create an off spectra by averaging the before/after off spectra:
            elif k in np.union1d(self.inds_on,self.inds_span):
                t_window=np.intersect1d(np.arange(k-window_size,k+window_size),self.inds_off)
                #print(k,t_window)
                self.V_bg[k,:,:]=np.nanmean(self.V[t_window,:,:],axis=0)
        self.V_bgsub=self.V-self.V_bg