#______ _           _              ______                       ___  ___          _       _      
#| ___ (_)         | |             | ___ \                      |  \/  |         | |     | |     
#| |_/ /_  ___ ___ | | ___   __ _  | |_/ / ___  __ _ _ __ ___   | .  . | ___   __| |_   _| | ___ 
#| ___ \ |/ __/ _ \| |/ _ \ / _` | | ___ \/ _ \/ _` | '_ ` _ \  | |\/| |/ _ \ / _` | | | | |/ _ \
#| |_/ / | (_| (_) | | (_) | (_| | | |_/ /  __/ (_| | | | | | | | |  | | (_) | (_| | |_| | |  __/
#\____/|_|\___\___/|_|\___/ \__, | \____/ \___|\__,_|_| |_| |_| \_|  |_/\___/ \__,_|\__,_|_|\___|
#                            __/ |                                                               
#                           |___/                                                                
      
#######################################################
##                    INSTRUCTIONS                   ##
#######################################################

## Import the module and create a Bicolog_Beam() class
    #import Bicolog_Beam_Model
    #data=Bicolog_Beam_Model.Bicolog_Beam()
## Test the Interpolation Function:
    #tAZ=np.linspace(3,243,250)
    #tAL=np.linspace(0.2,13,250)
    #tFF=np.linspace(455e6,777e6,1000)
    #data.Interpolate_D_ptf(Az_arr=tAZ,Alt_arr=tAL,Freq_arr=tFF)

#######################################################
##                  Module Contents                  ##
#######################################################

## Import packages used in analysis script:
import numpy as np
from matplotlib.pyplot import *
import glob
import os
import datetime
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from random import sample
from astropy.time import Time
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import pytz
import bisect
import pygeodesy
from mpl_toolkits import mplot3d
import pandas
from scipy.interpolate import interpn


## Define directories that contain modules/files of interest:
working_directory=u"/Users/wct9/python/"
drone_mod_directory=u'/Users/wct9/python/dronehacks/'
bico_beam_ECo_directory='/Users/wct9/python/Bicolog_Beam_EK_ECo/'
bico_beam_HCo_directory='/Users/wct9/python/Bicolog_Beam_EK_HCo/'

class Bicolog_Beam:
    def __init__(self,SParameter='S21'):
        print('Initialized Antenna Mapping Class: "Beam" loading file "data_{}.csv"'.format(SParameter))
        ## Load pandas DataFrame from .csv files stored in their respective directories for E,H polarizations:
        self.ECo_DF=pandas.read_csv(bico_beam_ECo_directory+'data_{}.csv'.format(SParameter),delimiter=',',skiprows=[62])
        self.HCo_DF=pandas.read_csv(bico_beam_HCo_directory+'data_{}.csv'.format(SParameter),delimiter=',',skiprows=[62])
        ## Extract freq and angle arrays from the dataframes:
        self.df_angle=np.append(np.array(self.ECo_DF['Angle'])[:61],np.array(self.ECo_DF['Angle'])[61:]+360.0)
        self.df_freq=np.array(self.ECo_DF.columns[1:]).astype(float)
        ## Create ndarrays of complex E,H polarization beams:
        self.complex_E_beam=np.array(self.ECo_DF,dtype=complex)[:,1:]
        self.complex_H_beam=np.array(self.HCo_DF,dtype=complex)[:,1:]
        ## Create array for dependent vars (phi[0,2pi](azimuth),theta[0,pi](altitude)) and frequency: 
        self.azimuth=np.linspace(0,360,360) # [0,2pi] every step 1 deg
        self.altitude=np.linspace(0,180,180)  # [0,pi] every step 1 deg
        self.D_ptf_t=np.zeros((len(self.azimuth),len(self.altitude),len(self.df_freq))) # az, alt, freq axes
        ## INTERPOLATE S21 for the E beam over our new 'altitude' array:
        ## Also want to normalize the values at 0,0 in order to avoid discontinuities:
        HtoEnorm_f=np.absolute(self.complex_E_beam[0,:])/np.absolute(self.complex_H_beam[0,:])
        alt_out=np.outer(self.altitude,np.ones(len(self.df_freq)))
        ## What we want to get out after looping over frequencies: 
        E_plus_out=np.zeros(alt_out.shape)
        H_plus_out=np.zeros(alt_out.shape)
        E_minus_out=np.zeros(alt_out.shape)
        H_minus_out=np.zeros(alt_out.shape)
        ## For E_plus and H_plus, the angle ranges are straightforward: [0,180]
        alt_plus_in=np.outer(self.df_angle[:61],np.ones(len(self.df_freq)))
        E_plus_in=np.absolute(self.complex_E_beam[:61,:])
        H_plus_in=np.absolute(self.complex_H_beam[:61,:])*HtoEnorm_f # now zero degeneracy for zenith pointing
        ## For E_minus and H_minus, the angle ranges change: [180,360]
        alt_minus_in=np.outer(np.append(self.df_angle[60:],self.df_angle[:1]+360.0),np.ones(len(self.df_freq)))
        E_minus_in=np.append(np.absolute(self.complex_E_beam[60:,:]),np.absolute(self.complex_E_beam[:1,:]),axis=0)
        H_minus_in=np.append(np.absolute(self.complex_H_beam[60:,:]),np.absolute(self.complex_H_beam[:1,:]),axis=0)*HtoEnorm_f
        ## Loop over frequencies so that the interpolation works properly:
        for i,f in enumerate(self.df_freq):
            E_plus_out[:,i]=np.interp(alt_out[:,i],alt_plus_in[:,i],E_plus_in[:,i])
            H_plus_out[:,i]=np.interp(alt_out[:,i],alt_plus_in[:,i],H_plus_in[:,i])
            E_minus_out[:,i]=np.interp(alt_out[::-1,i]+180.0,alt_minus_in[:,i],E_minus_in[:,i])
            H_minus_out[:,i]=np.interp(alt_out[::-1,i]+180.0,alt_minus_in[:,i],H_minus_in[:,i])
        ## What rotations should we use to map these data into our (L,T,Down)=(xyz) coordinate system?
            # These measurements are all the same E polarization! E/H is the theta axis convention.
            # Each previously computed slice in altitude corresponds to a fixed azimuth:
        self.D_ptf_t[0,:,:]=E_plus_out
        self.D_ptf_t[90,:,:]=H_plus_out
        self.D_ptf_t[180,:,:]=E_minus_out
        self.D_ptf_t[270,:,:]=H_minus_out
        ## Use cos and sin squared to sum the different slices for different azimuthal angles:
        for i,ang in enumerate(self.azimuth):
            EP=(np.cos(np.pi/180.0*ang).clip(min=0.0)**2.0)*E_plus_out
            HP=(np.sin(np.pi/180.0*ang).clip(min=0.0)**2.0)*H_plus_out
            EM=(np.cos(np.pi/180.0*(ang+180.0)).clip(min=0.0)**2.0)*E_minus_out
            HM=(np.sin(np.pi/180.0*(ang+180.0)).clip(min=0.0)**2.0)*H_minus_out
            self.D_ptf_t[i,:,:]=EP+HP+EM+HM
            
    def Interpolate_D_ptf(self,Az_arr,Alt_arr,Freq_arr):
        ## Assign coordinates for interpolation:
        AZ=Az_arr
        AL=Alt_arr
        FF=Freq_arr
        ## flatten so each interpolated point is (az,alt,freq) repeated as necessary:
        AZ_flat=np.outer(AZ,np.ones(len(FF))).flatten()
        AL_flat=np.outer(AL,np.ones(len(FF))).flatten()
        FF_flat=np.outer(np.ones(len(AZ)),FF).flatten()
        ## Define points, values, and interp point?
        points=(self.azimuth,self.altitude,self.df_freq) # tuple of the three axes [points]
        values=self.D_ptf_t                      # data values corresponding to points tuple with [az,alt,freq]
        ## Return the interpolated Directivity for input data:
        self.D_ptf_interp=interpn(points,values,(AZ_flat,AL_flat,FF_flat)).reshape((len(AL),len(FF)))

    def Plot_Beam_pcolormesh(self,Freq_ind=500):
        ##Define Coordinate Ranges for pcolormesh calls:
        X=np.outer(np.cos(np.pi/180.0*self.azimuth),np.sin(np.pi/180.0*self.altitude))
        Y=np.outer(np.sin(np.pi/180.0*self.azimuth),np.sin(np.pi/180.0*self.altitude))
        R=np.outer(np.ones(len(self.azimuth)),np.sin(np.pi/180.0*self.altitude))
        TH=np.outer(np.pi/180.0*self.azimuth,np.ones(len(self.altitude)))
        fig,[ax1,ax2]=subplots(nrows=1,ncols=2,figsize=(16,7),subplot_kw=dict(projection="polar"))
        im1=ax1.pcolormesh(TH[:,0:60],R[:,0:60],self.D_ptf_t[:,0:60,Freq_ind],shading='auto')
        ax1.set_title(r'Front Facing Lobes [$\theta < \pi$][f={} MHz]'.format(float(self.df_freq[Freq_ind])/1e6))
        ax1.set_yticklabels([])
        fig.colorbar(im1,ax=ax1)
        im2=ax2.pcolormesh(TH[:,60:],R[:,60:],self.D_ptf_t[:,60:,Freq_ind],shading='auto')
        ax2.set_title(r'Back Facing Lobes [$\theta > \pi$][f={} MHz]'.format(float(self.df_freq[Freq_ind])/1e6))
        ax2.set_yticklabels([])
        fig.colorbar(im2,ax=ax2)
        tight_layout()

        
    def Plot_Beam_Profiles(self,Polarization='E',lb_ub_step_list=[0,-1,33],Overplot_R_and_I=True):
        ## Initialize Polar Figure:
        fig=figure(figsize=(17,10))
        ax1=fig.add_subplot(111,projection='polar')
        fig.suptitle('Bicolog Antenna Azimuthal Beam Profiles -- {} Polarization'.format(Polarization),fontsize='x-large')
        ax1.set_title('Magnitude')
        ## Choose the polarization data that is going to be plotted:
        if Polarization=='E':
            ddd=self.complex_E_beam
        elif Polarization=='H':
            ddd=self.complex_H_beam
        ## Choose the frequency indices to plot:
        [blb,bub,bstep]=lb_ub_step_list
        ## Create color array for looped plots of beam contours: ##
        CNorm=colors.Normalize()
        CNorm.autoscale(np.arange(len(self.df_freq))[blb:bub:bstep])
        CM=cm.gnuplot2
        ## Plot several contours for various frequencies: ##
        for ind in np.arange(len(self.df_freq))[blb:bub:bstep][1:-1]:
            ax1.plot(np.pi/180.0*np.append(self.df_angle,self.df_angle[0]),np.absolute(np.append(ddd[:,ind],ddd[:1,ind])),'.-',c=CM(CNorm(ind)),label='{} MHz'.format(float(self.df_freq[ind])/1e6))
        ax1.plot(np.linspace(0.0,2.0*np.pi,100),1.0*np.ones(100),'k--',label='Isotropic Radiator')
        ax1.legend(loc=4,ncol=3,fontsize='small',framealpha=1.0)
        ax1.set_rgrids([0,1,2], angle=90.0)
        ax1.set_thetagrids([0,90,180,270])
        ax1.set_rlim(0,2.75)
        ax1.set_xticklabels([])
        if Overplot_R_and_I==True:
            ax2=fig.add_subplot(221,projection='polar')
            ax3=fig.add_subplot(223,projection='polar')
            ax2.set_ylabel('Real Component')
            ax3.set_ylabel('Imaginary Component')
            for ind in np.arange(len(self.df_freq))[blb:bub:bstep][1:-1]:
                ax2.plot(np.pi/180.0*self.df_angle,np.real(ddd[:,ind]),'.',c=CM(CNorm(ind)))
                ax3.plot(np.pi/180.0*self.df_angle,np.imag(ddd[:,ind]),'.',c=CM(CNorm(ind)))
            for ax in [ax1,ax2,ax3]:
                ax.set_rgrids([0,1,2], angle=90.0)
                ax.set_thetagrids([0,90,180,270])
                ax.set_rlim(0,2.75)
                ax.set_xticklabels([])
        tight_layout()