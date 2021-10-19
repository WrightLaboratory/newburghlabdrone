##  ____________ _____ _   _  _____  ___  ______________ _   _ _      _____  
## |  _  \ ___ \  _  | \ | ||  ___| |  \/  |  _  |  _  \ | | | |    |  ___| 
## | | | | |_/ / | | |  \| || |__   | .  . | | | | | | | | | | |    | |__   
## | | | |    /| | | | . ` ||  __|  | |\/| | | | | | | | | | | |    |  __|  
## | |/ /| |\ \\ \_/ / |\  || |___  | |  | \ \_/ / |/ /| |_| | |____| |___  
## |___/ \_| \_|\___/\_| \_/\____/  \_|  |_/\___/|___/  \___/\_____/\____/  
##  _____ _____ _____ _____  ____________  ___   _   _ _____  _   _         
## |_   _|  ___/  ___|_   _| | ___ \ ___ \/ _ \ | \ | /  __ \| | | |        
##   | | | |__ \ `--.  | |   | |_/ / |_/ / /_\ \|  \| | /  \/| |_| |        
##   | | |  __| `--. \ | |   | ___ \    /|  _  || . ` | |    |  _  |        
##   | | | |___/\__/ / | |   | |_/ / |\ \| | | || |\  | \__/\| | | |        
##   \_/ \____/\____/  \_/   \____/\_| \_\_| |_/\_| \_/\____/\_| |_/        
                                                                         
#######################################################
##                        Notes                      ##
#######################################################

## 20210706 -- WT
## This is the test branch for the drone module I'm trying to create for the Newburgh Lab drone analysis pipeline
## This will hopefully become a python module that can be easily used by the group with a lot of copy-pasta-ing
## I will comment as heavily as possible, and pull code from several previously used analysis scripts:
    ##     191014_OVRO_Flight_Processing.ipynb
    ##     20210409_LFOP_SpecAn_and_VNA_Work.ipynb
    ##     20210414_LFOP_Hacking.ipynb
    ##     BMX_Beam_Map.ipynb
    ##     Drone_Class_from_processed_CSV.ipynb
    ##     OVRO_Timestamp_Sync_Tests.ipynb
    ##     OVRO_data_vis.ipynb
    
## 20210801 -- WT
## ADDITIONAL VECTORIZATION CODING/PROOFS ##
    # CHANGES to drone class following this work:
    # 1. Want to initialize with additional receiver/array variables in dimensioned arrays:
            ##############################################################################################
            # I  Variable      # Dimension # description
            ##############################################################################################
            # A. Keys          # n dishes  # (string with name or channel index?)
            # B. Coordinates   # n by 3vec # (Vector position in local cartesian (E,N,U) relative origin)
            # C. Pointings     # n by 3vec # (Unit Vector in local cartesian (E,N,U))
            # D. Polarizations # n by 3vec # (Unit Vector in local cartesian (E,N,U))
            ##############################################################################################
    # 2. Want to calculate drone coordinates on per-dish basis, for xyz and rpt from origin based arrays
    
## 20210801 -- WT
## Code is updated to import FULL datcon_csv files, while still being able to import old processed files
    # to get around some NAN and some other issues, the first 500 rows of the datcon_csv files are cut


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

## Specify relevant coordinates in llh:
VECT_Drone_Start_LOC=pygeodesy.ellipsoidalNvector.LatLon(40.87031876496191, -72.86561763277804, 23.964228339399998).to3llh()
VECT_BMX_E_LOC=pygeodesy.ellipsoidalNvector.LatLon(40.86995317295864, -72.86603925418495, 19.464228339399998).to3llh()

## DEFN the Gauss Fit function:
def Gauss(x,a,x0,sigma,k):
    return a*np.exp(-(x-x0)**2.0/(2.0*sigma**2.0))+k

## Rotation Matrix for Yaw,Pitch,Roll rotations about z,y,x axes:
def RotMat(ypr_arr):
    [a,b,c]=(np.pi/180.0)*ypr_arr
    RM=np.ndarray((3,3))
    RM[0,:]=[np.cos(a)*np.cos(b),np.cos(a)*np.sin(b)*np.sin(c)-np.sin(a)*np.cos(c),np.cos(a)*np.sin(b)*np.cos(c)+np.sin(a)*np.sin(c)]
    RM[1,:]=[np.sin(a)*np.cos(b),np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c),np.sin(a)*np.sin(b)*np.cos(c)-np.cos(a)*np.sin(c)]
    RM[2,:]=[-1*np.sin(b),np.cos(b)*np.sin(c),np.cos(b)*np.cos(c)]
    return RM

## Convert from cartesian to polar (r,phi,theta):
def xyz_to_rpt(xyz):
    r_prime=np.sqrt(xyz[0]**2.0+xyz[1]**2.0+xyz[2]**2.0)
    phi_prime=np.arctan2(xyz[1],xyz[0])
    if phi_prime<0:
        phi_prime=phi_prime+(2.0*np.pi)
    theta_prime=np.arccos(xyz[2]/r_prime)
    rpt=[r_prime,phi_prime,theta_prime]
    return rpt

## Annie's function for fixing the time axis:
# (9/28/2021) function for adding sub-second accuracy to DJI timestamps
# now detects and eliminates >1s errors
def interp_time(df_in):
    # find where the GPS turns on
    gps_idx = df_in[df_in.gpsUsed == True].index[0]
    # interpolate the time and see if it works out!
    while (gps_idx < len(df_in)):
        # look for where the datetimestamp ticks
        first_dts = df_in["GPS:dateTimeStamp"][gps_idx]
        start_sec = int(first_dts[-3:-1])
        while(int(df_in["GPS:dateTimeStamp"][gps_idx][-3:-1]) == start_sec):
            gps_idx = gps_idx + 1
        # use this reference timestamp to convert the offsetTime column into proper datetimes
        start_dt = pandas.to_datetime(df_in["GPS:dateTimeStamp"][gps_idx])
        offsets = np.array(df_in["offsetTime"]-df_in["offsetTime"][gps_idx])
        offsets = pandas.to_timedelta(offsets, unit='s')
        timestamps = start_dt + offsets
        # put them in the dataframe
        df_in = df_in.assign(timestamp = timestamps)
        df_in = df_in.assign(UTC = timestamps)
        # check for excessive error by comparing the interpolated and uninterpolated timestamp columns
        gps_dts = pandas.to_datetime(df_in["GPS:dateTimeStamp"][gps_idx:-20]).values
        interp_dts = pandas.to_datetime(df_in["timestamp"][gps_idx:-20]).values
        if (np.mean(np.abs(gps_dts - interp_dts)/np.timedelta64(1,'ms')) < 1000):
            print("Timestamp interpolation succeeded")
            break
        else:
            print("Detected >1s error, retrying")
            gps_idx += 10 # increment the start timestamp index by an arbitrary amount and retry
    return df_in

## Make the colors cute and shit:
colorsarr=cm.gnuplot2(np.linspace(0,1,11))

class Drone_Data:
    def __init__(self,dronedir,FLYTAG,Origin_llh,Origin_key,dkeys,dcoords,dpointings,dpols,skip_rows=np.arange(1,500).tolist()):
        self.fn=FLYTAG
        self.Origin_llh=Origin_llh
        self.prime_origin=pygeodesy.EcefCartesian(latlonh0=Origin_llh[0],lon0=Origin_llh[1],height0=Origin_llh[2],name=Origin_key)
        ## Define/declare array variables in class:
        self.dish_keystrings=dkeys
        self.dish_coords_LC=dcoords
        self.dish_pointings_LC=dpointings
        self.dish_pols_LC=dpols
        ## Read Drone RTK Data
        drone_data=pandas.read_csv(dronedir+FLYTAG,skiprows=skip_rows,low_memory=False)
        ## Assign Drone RTK Data to class variables:
        if "_processed" in FLYTAG:
            print("Initializing drone data via processed_csv routine: {}".format(FLYTAG))
            print(" --> Skipping rows {} to {} to eliminate NAN values".format(skip_rows[0],skip_rows[-1]))
            self.latitude=np.array(drone_data.Lat)
            self.longitude=np.array(drone_data.Lon)
            self.pitch=np.array(drone_data.pitch)
            self.roll=np.array(drone_data.roll)
            self.yaw=np.array(drone_data.yaw)
            self.velocity=np.array(drone_data.vel)
            self.hmsl=np.array(drone_data.hmsl)
            self.altitude=np.array(drone_data.hmsl)[:]-Origin_llh[2]
            try:
                self.t_arr_timestamp=np.array(drone_data.timestamp)
            except AttributeError:
                self.t_arr_timestamp=np.array(drone_data.datetimestamp)
            self.t_index=np.arange(len(self.t_arr_timestamp))
            #self.t_arr_datetime=np.array([np.datetime64(self.t_arr_timestamp[m]).astype(datetime.datetime).replace(tzinfo=pytz.UTC) for m in range(self.t_index.shape[0])])
            self.t_arr_datetime=np.array(drone_data.assign(UTC=pandas.to_datetime(drone_data.UTC)).UTC)
            #self.t_arr_datetime=np.array([np.datetime64(self.t_arr_timestamp[m]).astype(datetime.datetime).replace(tzinfo=pytz.UTC) for m in range(self.t_index.shape[0])])
        else:
            print("Initializing drone data via datcon_csv routine: {}".format(FLYTAG))
            print(" --> Skipping rows {} to {} to eliminate NAN values".format(skip_rows[0],skip_rows[-1]))
            self.latitude=np.array(drone_data["RTKdata:Lat_P"])
            self.longitude=np.array(drone_data["RTKdata:Lon_P"])
            self.pitch=np.array(drone_data["IMU_ATTI(0):pitch"])
            self.roll=np.array(drone_data["IMU_ATTI(0):roll"])
            self.yaw=np.array(drone_data["RTKdata:YAW"])
            self.velocity=np.array(drone_data["IMU_ATTI(0):velComposite"])
            self.hmsl=np.array(drone_data["RTKdata:Hmsl_P"])
            self.t_arr_timestamp=np.array(drone_data["GPS:dateTimeStamp"])
            self.t_index=np.arange(len(self.t_arr_timestamp))
            self.t_arr_datetime=np.array(interp_time(drone_data)["UTC"],dtype='object')
            self.altitude=np.array(drone_data["RTKdata:Hmsl_P"])[:]-Origin_llh[2]
        ## Define coordinate systems we will eventually want to use:
        print(" --> generating llh, geocentric cartesian, local cartesian, and local spherical coordinates.")
        self.coords_llh=np.NAN*np.ones((self.t_index.shape[0],3))     ## Lat,Lon,hmsl from drone/RTK
        self.coords_xyz_GC=np.NAN*np.ones((self.t_index.shape[0],3))  ## x,y,z in meters in geocentric cartesian
        self.coords_xyz_LC=np.NAN*np.ones((self.t_index.shape[0],3))  ## x,y,z cartesian wrt a chosen origin (x=E,y=N,z=up)
        self.coords_rpt=np.NAN*np.ones((self.t_index.shape[0],3))     ## r,theta,phi wrt a chosen origin
        ## Populate and calculate these coordinate systems:
        for i in self.t_index[np.where(np.isnan(self.latitude)==False)]:
            try:
                ## Create LatLon point for each recorded drone position:
                p_t=pygeodesy.ellipsoidalNvector.LatLon(self.latitude[i],lon=self.longitude[i],height=self.hmsl[i])
            except RangeError:
                print("     --> RangeError for index{}".format(i))
            ## Assign llh, xyz, xyz_prime, rpt_prime coordinates, pointwise:
            self.coords_llh[i]=p_t.to3llh()
            self.coords_xyz_GC[i]=p_t.to3xyz()
            self.coords_xyz_LC[i]=self.prime_origin.forward(p_t).toVector()        
            r_prime=np.sqrt(self.coords_xyz_LC[i,0]**2.0+self.coords_xyz_LC[i,1]**2.0+self.coords_xyz_LC[i,2]**2.0)      
            phi_prime=np.arctan2(self.coords_xyz_LC[i,1],self.coords_xyz_LC[i,0])
            if phi_prime<0:
                phi_prime=phi_prime+(2.0*np.pi)
            theta_prime=np.arccos(self.coords_xyz_LC[i,2]/r_prime)
            self.coords_rpt[i]=[r_prime,phi_prime,theta_prime]
                
        print(" --> generating dish and receiver line of sight coordinates.")
        ## Calculate per-dish polar coordinates for drone/receiver in each other's beams as fxn of time:
        self.rpt_r_per_dish=np.zeros((len(self.dish_keystrings),len(self.t_index),3)) # drone posn wrt receiver
        self.rpt_t_per_dish=np.zeros((len(self.dish_keystrings),len(self.t_index),3)) # receiver posn wrt drone
        for i in range(len(self.dish_keystrings)):
            ## Receiver RPT after TRANS and ROT: (from receiver N to Drone in Receiver Cartesian "RC" coords)
            drone_xyz_RC=self.coords_xyz_LC-self.dish_coords_LC[i] # translate LC to receiver i position
            ## Rotate coord system by dish pointing with rotation matrix (constant in t):
            rec_pointing_rot=RotMat(np.array([xyz_to_rpt(self.dish_pointings_LC[i])[2],0.0,xyz_to_rpt(self.dish_pointings_LC[i])[1]]))
            ## Populate receiver position wrt drone:
            self.rpt_r_per_dish[i,:,:]=np.array([xyz_to_rpt(rec_pointing_rot@drone_xyz_RC[k]) for k in range(len(self.coords_xyz_LC))])
            ## Transmitter RPT after TRANS and ROT: (from Drone to receiver N in Drone coords)
            rec_xyz_LC=-1.0*(self.coords_xyz_LC)+self.dish_coords_LC[i] # in LC relative to drone, without rotation (yet)
            ## Rotate coord system by drone pointing with rotation matrix (varies with yaw,pitch,roll as fxns of t):
            ypr=np.ndarray((len(self.t_index),3))
            ypr[:,0]=self.yaw
            ypr[:,1]=self.pitch
            ypr[:,2]=self.roll
            self.rpt_t_per_dish[i,:,:]=np.array([xyz_to_rpt(RotMat(ypr[m,:])@(RotMat(np.array([90.0,0.0,180.0]))@rec_xyz_LC[m])) for m in range(len(self.coords_xyz_LC))])
        
    def Plot_Drone_Coordinates(self,t_cut=False,t_bounds=[0,-1]):
        print('plotting drone coordinates for all time samples:')
        fig1,[[ax1,ax2,ax3],[ax4,ax5,ax6]]=subplots(nrows=2,ncols=3,figsize=(15,9))
        ## Plot p0 coordinate origin:
        ax1.plot(self.Origin_llh[0],self.Origin_llh[1],'ro')
        ax2.axhline(self.Origin_llh[0],c='b')
        ax3.axhline(self.Origin_llh[1],c='b')
        ## Title each coordinate subplot:        
        ax1.set_title('Lat vs Lon')
        ax2.set_title('Lat vs Time')
        ax3.set_title('Lon vs Time')
        ax4.set_title('Velocity vs Time')
        ax5.set_title('Altitude vs Time')
        ax6.set_title('Yaw vs Time')
        ## Specify arrays/vectors to plot in 1,3,4 coordinate subplot
        xqtys=[self.latitude,self.t_index,self.t_index,self.t_index,self.t_index,self.t_index]
        yqtys=[self.longitude,self.latitude,self.longitude,self.velocity,self.altitude,self.yaw]
        xtags=['Latitude, [$deg$]','Drone Index','Drone Index','Drone Index','Drone Index','Drone Index']
        ytags=['Longitude, [$deg$]','Latitude, [$deg$]','Longitude, [$deg$]','Velocity, [m/s]','Altitude, [$m$]','Yaw [$deg$]']
        if t_cut==False:
            for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
                ax.plot(xqtys[i],yqtys[i],'.',label='all samples')
                ax.set_xlabel(xtags[i])
                ax.set_ylabel(ytags[i])
                ax.grid()
                ax.legend()
        if t_cut==True:
            print('overplotting drone coordinates for t_cut samples: ['+str(t_bounds[0])+':'+str(t_bounds[1])+']')
            for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
                ax.plot(np.nanmin(xqtys[i][t_bounds[0]:t_bounds[1]]),np.nanmin(yqtys[i][t_bounds[0]:t_bounds[1]]))
                ax.plot(np.nanmax(xqtys[i][t_bounds[0]:t_bounds[1]]),np.nanmax(yqtys[i][t_bounds[0]:t_bounds[1]]))
                autoscalelims=ax.axis()
                ax.clear()
                ax.plot(xqtys[i],yqtys[i],'.',label='all samples')
                ax.plot(xqtys[i][t_bounds[0]:t_bounds[1]],yqtys[i][t_bounds[0]:t_bounds[1]],'.',label='selected samples')
                ax.set_xlabel(xtags[i])
                ax.set_ylabel(ytags[i])
                ax.grid()
                ax.legend()
                ax.set_xlim(autoscalelims[0],autoscalelims[1])
                ax.set_ylim(autoscalelims[2],autoscalelims[3])
        tight_layout()
        
    def Plot_Angular_Coordinates(self,t_bounds=[0,-1]):
        fig=figure(figsize=(15,4.5))
        ax1=fig.add_subplot(1, 3, 1)
        ax1.plot(self.t_index[:],(180/np.pi)*self.coords_rpt[:,1],'.')
        ax1.plot(self.t_index[t_bounds[0]:t_bounds[1]],(180/np.pi)*self.coords_rpt[t_bounds[0]:t_bounds[1],1],'.')
        ax1.set_xlabel('time index')
        ax1.set_ylabel(r'$\phi, [deg]$')
        ax2=fig.add_subplot(1, 3, 2)
        ax2.plot(self.t_index[:],(180/np.pi)*self.coords_rpt[:,2],'.')
        ax2.plot(self.t_index[t_bounds[0]:t_bounds[1]],(180/np.pi)*self.coords_rpt[t_bounds[0]:t_bounds[1],2],'.')
        ax2.set_xlabel('time index')
        ax2.set_ylabel(r'$\theta, [deg]$')
        ax3=fig.add_subplot(1, 3, 3, projection='polar')
        ax3.plot(self.coords_rpt[:,1],180/np.pi*self.coords_rpt[:,2],'.')
        ax3.plot(self.coords_rpt[t_bounds[0]:t_bounds[1],1],180/np.pi*self.coords_rpt[t_bounds[0]:t_bounds[1],2],'.')
        ax3.set_rlim(np.nanmin(180/np.pi*self.coords_rpt[t_bounds[0]:t_bounds[1],2]),1.1*np.nanmax(180/np.pi*self.coords_rpt[t_bounds[0]:t_bounds[1],2]))
        tight_layout()

    def Plot_3d(self,t_bounds=[0,-1]):
        fig=figure(figsize=(10,4.5))
        tkeys=['Geocentric Cartesian','Local Cartesian']
        for i,coordset in enumerate([self.coords_xyz_GC,self.coords_xyz_LC]):
            ax=fig.add_subplot(1, 2, i+1, projection='3d')
            ax.set_title(tkeys[i])
            ax.set_xlabel('x, [meters]')
            ax.set_ylabel('y, [meters]')
            ax.set_zlabel('z, [meters]')
            ax.plot(coordset[:,0],coordset[:,1],coordset[:,2],'.')
            ax.plot(coordset[t_bounds[0]:t_bounds[1],0],coordset[t_bounds[0]:t_bounds[1],1],coordset[t_bounds[0]:t_bounds[1],2],'.')
        tight_layout()
     
    def Plot_Transmitter_Pointing(self,t_bounds=[0,-1],t_step=1):
        fig=figure(figsize=(10,8))
        ax=fig.add_subplot(111)
        ## DRONE COORDINATE SYSTEM x,y,z=North,East,Down VARIABLES ##
        UV_nose_north=np.array([1,0,0]) #unit vector for nose pointing north, no roll/pitch, prior to rotations
        UV_trans_down=np.array([0,0,1]) #unit vector for transmitter pointing down prior to rotations
        ## drone roll, pitch, yaw angles:
        ypr=np.ndarray((len(self.t_index),3))
        ypr[:,0]=self.yaw
        ypr[:,1]=self.pitch
        ypr[:,2]=self.roll
        ## TRANSMITTER POINTING DIRECTION as fxn of time in Local Cartesian: (transform by [y,p,r]=[+90,0,+180] rot)
        trans_pointing_xyz=np.array([RotMat(np.array([90.0,0.0,180.0]))@RotMat(ypr[m,:])@UV_trans_down for m in range(len(self.t_index))])
        ## Plot Parameters:
        [Qlb,Qub,Qstep]=[t_bounds[0],t_bounds[1],t_step]
        M=np.abs(np.hypot(trans_pointing_xyz[Qlb:Qub:Qstep,0],trans_pointing_xyz[Qlb:Qub:Qstep,1]))
        CNorm=colors.Normalize()
        CNorm.autoscale(M)
        CM=cm.gnuplot2
        SM=cm.ScalarMappable(cmap=CM, norm=CNorm)
        SM.set_array([])
        q=ax.quiver(self.coords_xyz_LC[Qlb:Qub:Qstep,0],self.coords_xyz_LC[Qlb:Qub:Qstep,1],trans_pointing_xyz[Qlb:Qub:Qstep,0],trans_pointing_xyz[Qlb:Qub:Qstep,1],color=CM(CNorm(M)))
        ax.quiverkey(q,X=0.15,Y=0.05,U=1,label='Unit Vector', labelpos='E')
        fig.colorbar(SM,label='XY Projection Magnitude')
        ax.set_xlabel('Local X Position, [m]')
        ax.set_ylabel('Local Y Position, [m]')
        ax.set_title('Transmitter Deviation from Nadir [XY Projection]')
        tight_layout()

    def Plot_Polar_Lines_of_Sight(self,t_bounds=[0,-1],t_step=1,dishid=0):        
        fig1,[ax1,ax2]=subplots(nrows=1,ncols=2,figsize=(16,8),subplot_kw=dict(projection="polar"))
        ax1.set_title('Drone Position in Receiver {} Beam'.format(dishid))
        ax1.plot(self.rpt_t_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,1],180.0/np.pi*self.rpt_t_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,2],'.b',markersize=1.0)
        ax1.plot(0,0,'ro')
        ax2.set_title('Receiver {} Position in Drone Beam'.format(dishid))
        ax2.plot(self.rpt_r_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,1],180.0/np.pi*self.rpt_r_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,2],'.g',markersize=1.0)
        ax2.plot(0,0,'ro')
        tight_layout()