#     _                                    
#    | |                                   
#  __| |_ __ ___  _ __   ___   _ __  _   _ 
# / _` | '__/ _ \| '_ \ / _ \ | '_ \| | | |
#| (_| | | | (_) | | | |  __/_| |_) | |_| |
# \__,_|_|  \___/|_| |_|\___(_) .__/ \__, |
#                             | |     __/ |
#                             |_|    |___/      
                                                            
## 2021110 -- WT
## Doing the refactor, the previous DRONE MODULE has been moved here
    # moving functions in and out of the drone data class file.
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

## Import packages from our own module:
import beamcals.plotting_utils as pu
import beamcals.fitting_utils as fu
import beamcals.geometry_utils as gu
import beamcals.time_utils as tu

class Drone_Data:
    def __init__(self,Drone_Directory,FLYTAG,site_class,skip_rows=np.arange(1,500).tolist()):
        self.FLYTAG=FLYTAG
        self.Drone_Directory=Drone_Directory
        ## Import variables from site-specific site_class object:
        self.name=site_class.name
        self.n_dishes=site_class.n_dishes
        self.n_channels=site_class.n_channels
        self.chmap=site_class.chmap
        self.origin=site_class.origin
        self.prime_origin=pygeodesy.EcefCartesian(latlonh0=self.origin[0],lon0=self.origin[1],height0=self.origin[2])
        self.dish_keystrings=site_class.keystrings
        self.dish_coords=site_class.coords
        self.dish_pointings=site_class.pointings
        self.dish_polarizations=site_class.polarizations
        ## Read Drone RTK Data
        drone_data=pandas.read_csv(self.Drone_Directory+self.FLYTAG,skiprows=skip_rows,low_memory=False)
        ## Assign Drone RTK Data to class variables:
        if "_processed" in self.FLYTAG:
            print("Initializing drone data via processed_csv routine: {}".format(self.FLYTAG))
            print("  --> Skipping rows {} to {} to eliminate NAN values".format(skip_rows[0],skip_rows[-1]))
            ## Load data columns from processed files:
            self.latitude=np.array(drone_data.Lat)
            self.longitude=np.array(drone_data.Lon)
            self.pitch=np.array(drone_data.pitch)
            self.roll=np.array(drone_data.roll)
            self.yaw=np.array(drone_data.yaw)
            self.velocity=np.array(drone_data.vel)
            self.hmsl=np.array(drone_data.hmsl)
            self.altitude=np.array(drone_data.hmsl)[:]-self.origin[2]
            try:
                self.t_arr_timestamp=np.array(drone_data.timestamp)
            except AttributeError:
                self.t_arr_timestamp=np.array(drone_data.datetimestamp)
            self.t_index=np.arange(len(self.t_arr_timestamp))
            self.t_arr_datetime=np.array(drone_data.assign(UTC=pandas.to_datetime(drone_data.UTC)).UTC)
        else:
            print("Initializing drone data via datcon_csv routine: {}".format(self.FLYTAG))
            print("  --> Skipping rows {} to {} to eliminate NAN values".format(skip_rows[0],skip_rows[-1]))
            ## Load data from full datcon files:
            try:
                ## begin by trying to load RTK data if available:
                print("  --> Attempting to load position data from RTK")
                self.latitude=np.array(drone_data["RTKdata:Lat_P"])
                self.longitude=np.array(drone_data["RTKdata:Lon_P"])
                self.hmsl=np.array(drone_data["RTKdata:Hmsl_P"])
                nandtestsum=len(drone_data["RTKdata:Lat_P"][~np.isnan(drone_data["RTKdata:Lat_P"])])
                print("    --> RTK data contains {}/{} non-nan values".format(nandtestsum,len(drone_data["RTKdata:Lat_P"])))
                if nandtestsum>0:
                    pass
                if nandtestsum==0:
                    print("    --> RTK Data not usable for this data file...")
                    print("  --> Loading position data from GPS(0) instead:")
                    self.latitude=np.array(drone_data["GPS(0):Lat"])
                    self.longitude=np.array(drone_data["GPS(0):Long"])
                    self.hmsl=np.array(drone_data["GPS(0):heightMSL"])
            except KeyError:
                ## If RTK data is not present, default to GPS(0) data:
                print("    --> RTK Data not found for this data file...")
                print("  --> Loading position data from GPS(0) instead:")
                self.latitude=np.array(drone_data["GPS(0):Lat"])
                self.longitude=np.array(drone_data["GPS(0):Long"])
                self.hmsl=np.array(drone_data["GPS(0):heightMSL"])
            ## Load columns that don't depend on the RTK data...
            self.pitch=np.array(drone_data["IMU_ATTI(0):pitch"])
            self.roll=np.array(drone_data["IMU_ATTI(0):roll"])
            self.yaw=np.array(drone_data["IMU_ATTI(0):yaw360"])
            self.velocity=np.array(drone_data["IMU_ATTI(0):velComposite"])
            self.t_arr_timestamp=np.array(drone_data["GPS:dateTimeStamp"])
            self.t_index=np.arange(len(self.t_arr_timestamp))
            self.t_arr_datetime=np.array(tu.interp_time(drone_data)["UTC"],dtype='object')
            self.altitude=self.hmsl-self.origin[2]
        ## Define coordinate systems we will eventually want to use:
        print("  --> generating llh, geocentric cartesian, local cartesian, and local spherical coordinates.")
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
                print("    --> RangeError for index{}".format(i))
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
        print("  --> generating dish and receiver line of sight coordinates.")
        ## Calculate per-dish polar coordinates for drone/receiver in each other's beams as fxn of time:
        self.rpt_r_per_dish=np.zeros((len(self.dish_keystrings),len(self.t_index),3)) # drone posn wrt receiver
        self.rpt_t_per_dish=np.zeros((len(self.dish_keystrings),len(self.t_index),3)) # receiver posn wrt drone
        for i in range(len(self.dish_keystrings)):
            ## Receiver RPT after TRANS and ROT: (from receiver N to Drone in Receiver Cartesian "RC" coords)
            drone_xyz_RC=self.coords_xyz_LC-self.dish_coords[i] # translate LC to receiver i position
            ## Rotate coord system by dish pointing with rotation matrix (constant in t):
            rec_pointing_rot=gu.rot_mat(np.array([gu.xyz_to_rpt(self.dish_pointings[i])[2],0.0,gu.xyz_to_rpt(self.dish_pointings[i])[1]]))
            ## Populate receiver position wrt drone:
            self.rpt_r_per_dish[i,:,:]=np.array([gu.xyz_to_rpt(rec_pointing_rot@drone_xyz_RC[k]) for k in range(len(self.coords_xyz_LC))])
            ## Transmitter RPT after TRANS and ROT: (from Drone to receiver N in Drone coords)
            rec_xyz_LC=-1.0*(self.coords_xyz_LC)+self.dish_coords[i] # in LC relative to drone, without rotation (yet)
            ## Rotate coord system by drone pointing with rotation matrix (varies with yaw,pitch,roll as fxns of t):
            ypr=np.ndarray((len(self.t_index),3))
            ypr[:,0]=self.yaw
            ypr[:,1]=self.pitch
            ypr[:,2]=self.roll
            self.rpt_t_per_dish[i,:,:]=np.array([gu.xyz_to_rpt(gu.rot_mat(ypr[m,:])@(gu.rot_mat(np.array([90.0,0.0,180.0]))@rec_xyz_LC[m])) for m in range(len(self.coords_xyz_LC))])