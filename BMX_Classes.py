id## Import packages used in analysis script:
import numpy as np
from matplotlib.pyplot import *
import glob
import os
import datetime
import scipy.optimize as opt
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
import pandas as pd
from bmxdata import BMXFile
from bmxdata import *


## Specify relevant coordinates in llh:
VECT_Drone_Start_LOC=pygeodesy.ellipsoidalNvector.LatLon(40.87031876496191, -72.86561763277804, 23.964228339399998).to3llh()
VECT_BMX_E_LOC=pygeodesy.ellipsoidalNvector.LatLon(40.86995317295864, -72.86603925418495, 19.464228339399998).to3llh()

 ###########File ORGANIZER########  
    
meas = pd.read_csv('/home/erk26/BMX/Flight_Catalog.csv', sep=",", header=0)
file_organizer = {}
init = meas['Day'][0]
datalist = []
for i in range(len(meas['Day'])):
    if init == meas['Day'][i]:
#         print(i)
        datalist.append([meas['Drone'][i],meas['Full_Tel'][i]])
        file_organizer[meas['Day'][i]] = datalist
    if init != meas['Day'][i]:
        datalist = []
        init = meas['Day'][i+1]
        datalist.append([meas['Drone'][i], meas['Full_Tel'][i]])
        
##############

#this function just turns BMX freq index into frequency (assuming 256 equally spaced freqs bw ~1100-1650MHz, values gotten from telescope data file)

def getfreq(index):
    frequency = 1101.0742514133453+index*(1648.9257531166077-1101.0742514133453)/256
    return(frequency)

#this fn just basically interpolates BMX timestamps within each second, bc there are many repeated timestamps with 1s resolution
def Telescope_Time_Convert(tel_dat):
    test_list = tel_dat.t_arr_datetime
    test2 = []
    a = test_list[0]
    counter = 0
    counter2 = 0
    inx = 0

    for entry in test_list:
        if entry == a:
            counter += 1
            inx +=1
        if entry != a:
            for item in test_list[inx-counter:inx]:
                test2.append(test_list[inx-counter]+pd.Timedelta(seconds = counter2/counter))
                counter2+=1
            test2.append(entry)
            a = entry
            counter = 0
            counter2 = 0
            inx +=1
            
    diff = len(test2)-len(tel_dat.t_arr_datetime)
    for i in range(-1*diff):
        test2.append(test2[-1])
    timestamps = pd.to_datetime(test2)
    return(timestamps)        

def interp_position_data(xcol,ycol):
    test_list = [round(entry,3) for entry in xcol]
    test2 = []
    a = test_list[0]
    counter = 0
    counter2 = 0
    inx = 0

    x = []
    y = []
    
    for entry in test_list:
        if entry == a:
            counter += 1
            inx +=1
        if entry != a:
            x2 = list(xcol[inx-counter:inx+1])
            y2 = list(ycol[inx-counter:inx+1])
            xinterp = np.linspace(x2[0], x2[-1], len(x2))
            if x2[0]>x2[-1]:
                yinterp = np.interp(xinterp, [x2[-1],x2[0]], [y2[-1], y2[0]])
            else:
                yinterp = np.interp(xinterp, [x2[0],x2[-1]], [y2[0], y2[-1]])
            for k in range(len(xinterp)):
                x.append(xinterp[k])
                y.append(yinterp[k])
            a = entry
            counter = 0
            counter2 = 0
            inx +=1
    x2 = list(xcol[inx-counter:len(xcol)])
    y2 = list(ycol[inx-counter:len(xcol)])
    for k in range(len(x2)):
        x.append(x2[k])
        y.append(y2[k])

    return(x, y)
        

#############################
#From Will's Code

class Drone_Data:
    def __init__(self,drone_directory,FLYTAG,Origin_llh):
        print('DRONE CLASS initialized -- Loading Drone RTK Data from '+FLYTAG.split('_Drone_')[0]+':')
        self.fn=FLYTAG
        self.Origin_llh=Origin_llh
        self.prime_origin=pygeodesy.EcefCartesian(latlonh0=Origin_llh[0],lon0=Origin_llh[1],height0=Origin_llh[2],name='BMX_Tower_Center')
        ## Read Drone RTK Data
        drone_data=pd.read_csv(drone_directory+FLYTAG+'_processed.csv')
        ## Assign Drone RTK Data to class variables:
        self.latitude=np.array(drone_data.Lat)
        self.longitude=np.array(drone_data.Lon)
        self.pitch=np.array(drone_data.pitch)
        self.roll=np.array(drone_data.roll)
        self.yaw=np.array(drone_data.yaw)
#         self.velocity=np.array(drone_data.vel)
        self.hmsl=np.array(drone_data.hmsl)
        self.t_arr_timestamp=np.array(drone_data.timestamp)
        self.t_index=np.arange(len(self.t_arr_timestamp))
        ## Construct useful secondary class variables
        self.altitude=np.array(drone_data.hmsl)[:]-Origin_llh[2]
        ## Define coordinate systems we will eventually want to use:
        print( "generating llh, geocentric cartesian, local cartesian, and local spherical coordinates.")
        self.coords_llh=np.ndarray((self.t_index.shape[0],3))        ## Lat,Lon,hmsl from drone/RTK
        self.coords_xyz=np.ndarray((self.t_index.shape[0],3))        ## x,y,z in meters in geocentric cartesian
        self.coords_xyz_prime=np.ndarray((self.t_index.shape[0],3))  ## x,y,z cartesian wrt a chosen origin (x=E,y=N,z=up)
        self.coords_rpt_prime=np.ndarray((self.t_index.shape[0],3))  ## r,theta,phi wrt a chosen origin
        ## Populate and calculate these coordinate systems:
        for i in self.t_index:
            ## Create LatLon point for each recorded drone position:
            p_t=pygeodesy.ellipsoidalNvector.LatLon(self.latitude[i],lon=self.longitude[i],height=self.hmsl[i])
            ## Assign llh, xyz, xyz_prime, rpt_prime coordinates, pointwise:
            self.coords_llh[i]=p_t.to3llh()
            self.coords_xyz[i]=p_t.to3xyz()
            self.coords_xyz_prime[i]=self.prime_origin.forward(p_t).toVector()        
            r_prime=np.sqrt(self.coords_xyz_prime[i,0]**2.0+self.coords_xyz_prime[i,1]**2.0+self.coords_xyz_prime[i,2]**2.0)      
            phi_prime=np.arctan2(self.coords_xyz_prime[i,1],self.coords_xyz_prime[i,0])
            if phi_prime<0:
                phi_prime=phi_prime+(2.0*np.pi)
            theta_prime=np.arccos(self.coords_xyz_prime[i,2]/r_prime)
            self.coords_rpt_prime[i]=[r_prime,phi_prime,theta_prime]
        
        tight_layout()


        
class BMX_Data:
    def __init__(self,working_directory,idstring):
        print('BMX_Data CLASS initialized -- Loading Telescope Data from '+idstring+':')
        self.fn=idstring
        self.bmxdat=BMXFile(working_directory+idstring+'_yale_D1.data',loadD2=True,loadRFI=False)
        self.mjd=self.bmxdat.data['mjd']
        ## Assign TIME data to an array/vector:
        print('  Constructing time array from header mjd:')
        self.t_arr=Time(self.bmxdat.data['mjd'],format='mjd')
        self.t_arr.delta_ut1_utc=0.334 
        self.t_arr_datetime=np.array([np.datetime64(j.ut1.iso) for j in self.t_arr])
        self.dt=datetime.timedelta(seconds=float(self.bmxdat.deltaT))
        self.t_index=np.arange(self.bmxdat.data['chan1_0'].shape[1])
        ## FREQUENCY ARRAY: from high to low frequency (1530MHz-1280MHz)
        self.f_arr=np.array(self.bmxdat.freq[0])
        self.f_index=np.arange(self.bmxdat.data['chan1_0'].shape[0])
        ## Create raw auto-correlation array:
        print('  Constructing raw autocorrelation data array (dims=[2,4,freq,time]):')
        self.autodata_raw=np.zeros((2,4,self.bmxdat.data['chan1_0'].shape[0],self.bmxdat.data['chan1_0'].shape[1]))
        for i,chan in enumerate(['chan1_0','chan2_0','chan3_0','chan4_0']):
            self.autodata_raw[0,i,:,:]=self.bmxdat.data[chan]
        for i,chan in enumerate(['chan5_0','chan6_0','chan7_0','chan8_0']):
            self.autodata_raw[1,i,:,:]=self.bmxdat.data[chan]
        
###########################
        
def droneDataPairing(drone_dir, FLY, bmx_dir, BMXFILE, pol, time_offset_s):
        drn_dat = Drone_Data(drone_dir, FLY, Origin_llh=VECT_BMX_E_LOC)
        tel_dat = BMX_Data(bmx_dir, BMXFILE)
        
        autos = tel_dat.autodata_raw[pol,:]
        
        #interpolate position data
        xp, yp = interp_position_data(list(drn_dat.coords_xyz_prime[:,0]), list(drn_dat.coords_xyz_prime[:,1]))
                
        #interpolate drone time to telescope time        
        t_drone = pd.to_datetime(drn_dat.t_arr_timestamp)+pd.Timedelta(seconds=time_offset_s)
        t_tel = Telescope_Time_Convert(tel_dat)
        
        r = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 0])
        theta = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 1])
        phi = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 1])
        x = np.interp(t_tel, t_drone, xp)
        y = np.interp(t_tel, t_drone, yp)
        z = np.interp(t_tel, t_drone, drn_dat.coords_xyz_prime[:,2])
        rprime = np.sqrt(x**2+y**2)
        yaw = np.interp(t_tel, t_drone, drn_dat.yaw)

        
        # coordinates of dish centers (from FLY338)
        # NESW by y,x,z
        dishLoc = ([-3.7605200906121437, 3.865245048570655],\
        [-0.0641296465130804, 0.035901010584928825], \
        [-3.6868008232255924, -3.205432553991354], \
        [-7.485557261347768, -0.07592779975602093],)
        
        beamCenters = ([-4.670897458296071, 7.777616712345406],\
        [3.3385316030884016, -3.27205823116637],\
        [3.2569049644387906, 1.2983814825634061],\
        [-10.421877726648322, -0.7263793693162363])
        

        # compute angles (x=lon, y=lat) from each dish
        thetaX = []
        thetaY = []
        for chan in range(4):
	        thetaX.append(np.arctan((x - dishLoc[chan][0])/(z)))
	        thetaY.append(np.arctan((y - dishLoc[chan][1])/(z)))
        thetaX = np.array(thetaX)
        thetaY = np.array(thetaY)
        
        thetaX2 = []
        thetaY2 = []
        for chan in range(4):
	        thetaX2.append(np.arctan((x - beamCenters[chan][0])/(z)))
	        thetaY2.append(np.arctan((y - beamCenters[chan][1])/(z)))
        thetaX_beam = np.array(thetaX2)
        thetaY_beam = np.array(thetaY2)
        
        return {"FLY" : FLY, "BMXdir" :  BMXFILE, "autos" : autos, "drone_time" : t_drone, "tel_time" : t_tel, "x" : x, "y" : y, "z" : z, "thetaX_dish" : thetaX, "thetaY_dish" : thetaY, "thetaX_beam" : thetaX_beam, "thetaY_beam" : thetaY_beam, "r": rprime, "yaw": yaw, "phi":phi} 

################################
#FUNCTIONS TO CALL IN CLASS

def gauss1D(x, A, mu, sigma, offset):
    return (A*np.exp(-(x-mu)**2/(2.*sigma**2))+offset)
def format_e(n):
    a = "{:.2E}".format(n)
    return(a)
def lintodb(mags):
    mag_=10*np.log10(np.array(mags))
    return(mag_)


def plot_y_slice_db(flight_dict, dishindx, freqindx, lb, ub, flipped):
    x = list(np.degrees(flight_dict['thetaX_beam'][dishindx]))
    y = list(np.degrees(flight_dict['thetaY_beam'][dishindx]))
    z = list(flight_dict['autos'][dishindx][:,freqindx])
    
    initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)

    y_slice = [y[i] for i in range(len(x)) if lb<x[i]<ub]
    power_y_slice=[z[i]-popt[-1] for i in range(len(x)) if lb<x[i]<ub]
    
    zipped = list(zip(y_slice,power_y_slice))
    zipped.sort()
    y_slice, power_y_slice = list(zip(*zipped))
    
    if flipped == True:
        plt.plot(-1*np.array(y_slice), lintodb(power_y_slice/np.max(power_y_slice)))
    if flipped == False:
        plt.plot(y_slice, lintodb(power_y_slice/np.max(power_y_slice)))

    plt.xlabel('Degs')
    plt.ylabel('dB')
    
def plot_x_slice_db(flight_dict, dishindx, freqindx,lb, ub, flipped):
    x = list(np.degrees(flight_dict['thetaX_beam'][dishindx]))
    y = list(np.degrees(flight_dict['thetaY_beam'][dishindx]))
    z = list(flight_dict['autos'][dishindx][:,freqindx])
    
    initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)

    x_slice = [x[i] for i in range(len(x)) if lb<y[i]<ub]
    power_x_slice=[z[i]-popt[-1] for i in range(len(x)) if lb<y[i]<ub]  
    
    zipped = list(zip(x_slice,power_x_slice))
    zipped.sort()
    x_slice, power_x_slice = list(zip(*zipped))
    if flipped == True:
        plt.plot(-1*np.array(x_slice), lintodb(power_x_slice/np.max(power_x_slice)))
    if flipped == False:
        plt.plot(x_slice, lintodb(power_x_slice/np.max(power_x_slice)))

    plt.xlabel('Degs')
    plt.ylabel('dB')

    
def plot_x_slice_lin(flight_dict, dishindx, freqindx, lb, ub, params):
    x = list(np.degrees(flight_dict['thetaX_beam'][dishindx]))
    y = list(np.degrees(flight_dict['thetaY_beam'][dishindx]))
    z = list(flight_dict['autos'][dishindx][:,freqindx])
    
    x_slice = [x[i] for i in range(len(x)) if lb<y[i]<ub]
    power_x_slice=[z[i] for i in range(len(x)) if lb<y[i]<ub]  
    
    
    popt, pcov = opt.curve_fit(gauss1D, x_slice, power_x_slice, p0=[1e15, 0, 3, 1e12])
    angs_forfit = np.arange(-20,20,0.1)
    gauss_fit = gauss1D(angs_forfit, *popt)

    plt.plot(angs_forfit, gauss_fit, color='red')
    plt.plot(x_slice, power_x_slice, '.')
    if params == True:
        props = dict(boxstyle='round', alpha = 0.1)
        textstr = '\n'.join((
            r'A='+format_e(popt[0]),
            r'FWHM='+str(round(2.355*popt[2], 2)),
            r'Offset='+format_e(popt[3])))
        plt.text(5, popt[0]*0.8, textstr ,fontsize=12,
                verticalalignment='top', bbox=props)
    plt.xlabel('Degs')
    plt.ylabel('Power')
    plt.title('X Slice (linear)')

def plot_y_slice_lin(flight_dict, dishindx, freqindx, lb, ub, params):
    x = list(np.degrees(flight_dict['thetaX_beam'][dishindx]))
    y = list(np.degrees(flight_dict['thetaY_beam'][dishindx]))
    z = list(flight_dict['autos'][dishindx][:,freqindx])

    y_slice = [y[i] for i in range(len(x)) if lb<x[i]<ub]
    power_y_slice=[z[i] for i in range(len(x)) if lb<x[i]<ub]

    popt, pcov = opt.curve_fit(gauss1D, y_slice, power_y_slice, p0=[1e15, 0, 3, 1e12])
    angs_forfit = np.arange(-20,20,0.1)
    gauss_fit = gauss1D(angs_forfit, *popt)

    plt.plot(angs_forfit, gauss_fit, color='red')
    plt.plot(y_slice, power_y_slice, '.')
    if params == True:
        props = dict(boxstyle='round', alpha = 0.1)
        textstr = '\n'.join((
            r'A='+format_e(popt[0]),
            r'FWHM='+str(round(2.355*popt[2], 2)),
            r'Offset='+format_e(popt[3])))
        plt.text(5, popt[0]*0.8, textstr ,fontsize=12,
                verticalalignment='top', bbox=props)
    plt.xlabel('Degs')
    plt.ylabel('Power')
    plt.title('Y Slice (linear)')




def twoD_Gaussian(v, *params):
    [amplitude, xo, yo, sigma_x, sigma_y, theta, offset] = params
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((v[0]-xo)**2) + 2*b*(v[0]-xo)*(v[1]-yo) 
                            + c*((v[1]-yo)**2)))
    return g.ravel()


def get2Dparams(flight_dict, dishindx, freqindx):
    x = list(np.degrees(flight_dict['thetaX_beam'][dishindx]))
    y = list(np.degrees(flight_dict['thetaY_beam'][dishindx]))
    z = list(flight_dict.fulldict['autos'][dishindx][:,freqindx])
    
    initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)
    for i in range(len(pars)): print(pars[i]+' = '+"{:e}".format(popt[i]))


#################################
    
class Do_Everything_For_Data:
    def __init__(self, drone_dir, FLY, bmx_dir, BMXFILE, time_offset_s = 0):
        if FLY == 'FLY349' or FLY =='FLY351' or FLY =='FLY352': pol = 1
        else: pol = 0
        self.fulldict = droneDataPairing(drone_dir, FLY, bmx_dir, BMXFILE, pol, time_offset_s)    
    
    def plotscatter(self, dishindx, freqindx):
        plt.scatter(self.fulldict['x'],\
            self.fulldict['y'], s=50,\
            c = (self.fulldict['autos'][dishindx][:,freqindx]),\
            cmap='gnuplot2', norm=LogNorm(), marker = ',')
    
    def plot_x_lin(self, dishindx, freqindx, lb = -0.5, ub = 0.5, params = True):
        plot_x_slice_lin(self.fulldict, dishindx, freqindx, lb, ub, params)
        
    def plot_x_db(self, dishindx, freqindx, lb = -0.5, ub = 0.5, flipped = False):
        plot_x_slice_db(self.fulldict, dishindx, freqindx, lb, ub, flipped)
    
    def plot_y_lin(self, dishindx, freqindx, lb = -0.5, ub = 0.5, params = True):
        plot_y_slice_lin(self.fulldict, dishindx, freqindx, lb, ub, params)
        
    def plot_y_db(self, dishindx, freqindx, lb = -0.5, ub = 0.5, flipped = False):
        plot_y_slice_db(self.fulldict, dishindx, freqindx, lb, ub, flipped)
        
    def get2Dparams(self, dishindx, freqindx):
        x = list(np.degrees(self.fulldict['thetaX_beam'][dishindx]))
        y = list(np.degrees(self.fulldict['thetaY_beam'][dishindx]))
        z = list(self.fulldict['autos'][dishindx][:,freqindx])

        initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)
        return(popt, pcov)

    def print2Dparams(self, dishindx, freqindx):
        x = list(np.degrees(self.fulldict['thetaX_beam'][dishindx]))
        y = list(np.degrees(self.fulldict['thetaY_beam'][dishindx]))
        z = list(self.fulldict['autos'][dishindx][:,freqindx])

        initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)

        popt[3] = 2.355*popt[3]
        popt[4] = 2.355*popt[4]

        pars = ['amplitude', 'xo', 'yo', 'fwhm_x', 'fwhm_y', 'theta', 'offset']
        for i in range(len(pars)): print(pars[i]+' = '+format_e(popt[i]))       
        
###NOW: combine 2 files


class Drone_Data2:
    def __init__(self,drone_directory,FLYTAG1,FLYTAG2,Origin_llh):
        print('DRONE CLASS initialized -- Loading Drone RTK Data from '+FLYTAG1.split('_Drone_')[0]+':')
        self.fn=FLYTAG1
        self.Origin_llh=Origin_llh
        self.prime_origin=pygeodesy.EcefCartesian(latlonh0=Origin_llh[0],lon0=Origin_llh[1],height0=Origin_llh[2],name='BMX_Tower_Center')
        ## Read Drone RTK Data
        drone_data1=pd.read_csv(drone_directory+FLYTAG1+'_processed.csv')
        drone_data2=pd.read_csv(drone_directory+FLYTAG2+'_processed.csv')
        drone_data = pd.concat([drone_data1, drone_data2])
        ## Assign Drone RTK Data to class variables:
        self.latitude=np.array(drone_data.Lat)
        self.longitude=np.array(drone_data.Lon)
        self.pitch=np.array(drone_data.pitch)
        self.roll=np.array(drone_data.roll)
        self.yaw=np.array(drone_data.yaw)
#         self.velocity=np.array(drone_data.vel)
        self.hmsl=np.array(drone_data.hmsl)
        self.t_arr_timestamp=np.array(drone_data.timestamp)
        self.t_index=np.arange(len(self.t_arr_timestamp))
        ## Construct useful secondary class variables
        self.altitude=np.array(drone_data.hmsl)[:]-Origin_llh[2]
        ## Define coordinate systems we will eventually want to use:
        print( "generating llh, geocentric cartesian, local cartesian, and local spherical coordinates.")
        self.coords_llh=np.ndarray((self.t_index.shape[0],3))        ## Lat,Lon,hmsl from drone/RTK
        self.coords_xyz=np.ndarray((self.t_index.shape[0],3))        ## x,y,z in meters in geocentric cartesian
        self.coords_xyz_prime=np.ndarray((self.t_index.shape[0],3))  ## x,y,z cartesian wrt a chosen origin (x=E,y=N,z=up)
        self.coords_rpt_prime=np.ndarray((self.t_index.shape[0],3))  ## r,theta,phi wrt a chosen origin
        ## Populate and calculate these coordinate systems:
        for i in self.t_index:
            ## Create LatLon point for each recorded drone position:
            p_t=pygeodesy.ellipsoidalNvector.LatLon(self.latitude[i],lon=self.longitude[i],height=self.hmsl[i])
            ## Assign llh, xyz, xyz_prime, rpt_prime coordinates, pointwise:
            self.coords_llh[i]=p_t.to3llh()
            self.coords_xyz[i]=p_t.to3xyz()
            self.coords_xyz_prime[i]=self.prime_origin.forward(p_t).toVector()        
            r_prime=np.sqrt(self.coords_xyz_prime[i,0]**2.0+self.coords_xyz_prime[i,1]**2.0+self.coords_xyz_prime[i,2]**2.0)      
            phi_prime=np.arctan2(self.coords_xyz_prime[i,1],self.coords_xyz_prime[i,0])
            if phi_prime<0:
                phi_prime=phi_prime+(2.0*np.pi)
            theta_prime=np.arccos(self.coords_xyz_prime[i,2]/r_prime)
            self.coords_rpt_prime[i]=[r_prime,phi_prime,theta_prime]


def concat_droneDataPairing(drone_dir, FLY1, FLY2, bmx_dir, BMXFILE1, BMXFILE2, pol):
        drone_data1=pd.read_csv(drone_dir+FLY1+'_processed.csv')
        drone_data2=pd.read_csv(drone_dir+FLY2+'_processed.csv')  
        drone_timestamp1=np.array(drone_data1.timestamp)
        drone_timestamp2=np.array(drone_data2.timestamp)
        
#         drn_dat1 = Drone_Data(drone_dir, FLY1, Origin_llh=VECT_BMX_E_LOC)
#         drn_dat2 = Drone_Data(drone_dir, FLY2, Origin_llh=VECT_BMX_E_LOC)
        drn_dat = Drone_Data2(drone_dir, FLY1, FLY2, Origin_llh=VECT_BMX_E_LOC)
        
        tel_dat1 = BMX_Data(bmx_dir, BMXFILE1)
        tel_dat2 = BMX_Data(bmx_dir, BMXFILE2)
              
        autos = np.concatenate((tel_dat1.autodata_raw,tel_dat2.autodata_raw), axis = 2)[pol,:]
        
        #interpolate position data
        xp, yp = interp_position_data(list(drn_dat.coords_xyz_prime[:,0]), list(drn_dat.coords_xyz_prime[:,1]))
            
        #interpolate drone time to telescope time
        time_offset1_s = time_offset2_s = 0.0
        if FLY2 == 'FLY351': time_offset2_s = 0.5
        if FLY1 == 'FLY342': time_offset1_s = -1
            
        t_drone1 = pd.to_datetime(drone_timestamp1)+pd.Timedelta(seconds=time_offset1_s)
        t_drone2 = pd.to_datetime(drone_timestamp2)+pd.Timedelta(seconds=time_offset2_s)      
    
#         t_drone1 = pd.to_datetime(drn_dat1.t_arr_timestamp)+pd.Timedelta(seconds=time_offset1_s)
#         t_drone2 = pd.to_datetime(drn_dat2.t_arr_timestamp)+pd.Timedelta(seconds=time_offset2_s)
        t_drone = pd.to_datetime(np.concatenate((np.array(t_drone1),np.array(t_drone2))))

        
        t_tel1 = Telescope_Time_Convert(tel_dat1)
        t_tel2 = Telescope_Time_Convert(tel_dat2)
        t_tel = pd.to_datetime(np.concatenate((np.array(t_tel1),np.array(t_tel2))))
        
        r = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 0])
        theta = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 2])
        phi = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 1])
        x = np.interp(t_tel, t_drone, xp)
        y = np.interp(t_tel, t_drone, yp)
        z = np.interp(t_tel, t_drone, drn_dat.coords_xyz_prime[:,2])       
        rprime = np.sqrt(x**2+y**2)
        yaw = np.interp(t_tel, t_drone, drn_dat.yaw)

        
        # coordinates of dish centers (from FLY338)
        # NESW by y,x,z
        dishLoc = ([-3.7605200906121437, 3.865245048570655],\
        [-0.0641296465130804, 0.035901010584928825], \
        [-3.6868008232255924, -3.205432553991354], \
        [-7.485557261347768, -0.07592779975602093],)
        
        beamCenters = ([-4.670897458296071, 7.777616712345406],\
        [3.3385316030884016, -3.27205823116637],\
        [3.2569049644387906, 1.2983814825634061],\
        [-10.421877726648322, -0.7263793693162363])
        

        # compute angles (x=lon, y=lat) from each dish
        thetaX = []
        thetaY = []
        for chan in range(4):
	        thetaX.append(np.arctan((x - dishLoc[chan][0])/(z)))
	        thetaY.append(np.arctan((y - dishLoc[chan][1])/(z)))
        thetaX = np.array(thetaX)
        thetaY = np.array(thetaY)
        
        thetaX2 = []
        thetaY2 = []
        for chan in range(4):
	        thetaX2.append(np.arctan((x - beamCenters[chan][0])/(z)))
	        thetaY2.append(np.arctan((y - beamCenters[chan][1])/(z)))
        thetaX_beam = np.array(thetaX2)
        thetaY_beam = np.array(thetaY2)
        
        return {"FLY" : FLY1, "BMXdir" :  BMXFILE1, "autos" : autos, "drone_time" : t_drone, "tel_time" : t_tel, "x" : x, "y" : y, "z" : z, "thetaX_dish" : thetaX, "thetaY_dish" : thetaY, "thetaX_beam" : thetaX_beam, "thetaY_beam" : thetaY_beam,"r": rprime, "yaw": yaw, "phi":phi}

################################

class concat_files_Do_Everything_For_Data:
    def __init__(self, drone_dir, FLY1, FLY2, bmx_dir, BMXFILE1, BMXFILE2):
        if FLY1 == 'FLY349' or FLY1 =='FLY351' or FLY1 =='FLY352': pol = 1
        else: pol = 0
        self.fulldict = concat_droneDataPairing(drone_dir, FLY1, FLY2, bmx_dir, BMXFILE1, BMXFILE2, pol)
    
    def plotscatter(self, dishindx, freqindx):
        plt.scatter(self.fulldict['x'],\
            self.fulldict['y'], s=50,\
            c = (self.fulldict['autos'][dishindx][:,freqindx]),\
            cmap='gnuplot2', norm=LogNorm(), marker = ',')

    def plot_x_lin(self, dishindx, freqindx, lb = -0.5, ub = 0.5, params = True):
        plot_x_slice_lin(self.fulldict, dishindx, freqindx, lb, ub, params)
        
    def plot_x_db(self, dishindx, freqindx, lb = -0.5, ub = 0.5, flipped = False):
        plot_x_slice_db(self.fulldict, dishindx, freqindx, lb, ub, flipped)
    
    def plot_y_lin(self, dishindx, freqindx, lb = -0.5, ub = 0.5, params = True):
        plot_y_slice_lin(self.fulldict, dishindx, freqindx, lb, ub, params)
        
    def plot_y_db(self, dishindx, freqindx, lb = -0.5, ub = 0.5, flipped = False):
        plot_y_slice_db(self.fulldict, dishindx, freqindx, lb, ub, flipped)
        
    def get2Dparams(self, dishindx, freqindx):
        x = list(np.degrees(self.fulldict['thetaX_beam'][dishindx]))
        y = list(np.degrees(self.fulldict['thetaY_beam'][dishindx]))
        z = list(self.fulldict['autos'][dishindx][:,freqindx])

        initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)
        return(popt, pcov)

    def print2Dparams(self, dishindx, freqindx):
        x = list(np.degrees(self.fulldict['thetaX_beam'][dishindx]))
        y = list(np.degrees(self.fulldict['thetaY_beam'][dishindx]))
        z = list(self.fulldict['autos'][dishindx][:,freqindx])

        initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)

        popt[3] = 2.355*popt[3]
        popt[4] = 2.355*popt[4]

        pars = ['amplitude', 'xo', 'yo', 'fwhm_x', 'fwhm_y', 'theta', 'offset']
        for i in range(len(pars)): print(pars[i]+' = '+format_e(popt[i]))
        
        
###FOR PARSING DUAL-POLARIZATION FLIGHTS (LIKE ON WEDNESDAY)###############################
def droneDataPairing_dualPol(drone_dir, FLY, bmx_dir, BMXFILE):
        drn_dat = Drone_Data(drone_dir, FLY, Origin_llh=VECT_BMX_E_LOC)
        tel_dat = BMX_Data(bmx_dir, BMXFILE)
        
        #interpolate position data
        xp, yp = interp_position_data(list(drn_dat.coords_xyz_prime[:,0]), list(drn_dat.coords_xyz_prime[:,1]))
        
        #interpolate drone time to telescope time
        time_offset_s = 0.0
        if FLY == 'FLY351': time_offset_s = 0.5
        if FLY == 'FLY342': time_offset_s = -1     
        
        t_drone = pd.to_datetime(drn_dat.t_arr_timestamp)+pd.Timedelta(seconds=time_offset_s)
        t_tel = Telescope_Time_Convert(tel_dat)
        
        yaw = np.interp(t_tel, t_drone, drn_dat.yaw)

        r = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 0])
        theta = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 1])
        phi = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 2])
        x = np.interp(t_tel, t_drone, xp)
        y = np.interp(t_tel, t_drone, yp)
        z = np.interp(t_tel, t_drone, drn_dat.coords_xyz_prime[:,2])
        rprime = np.sqrt(x**2+y**2)
        
        # coordinates of dish centers (from FLY338)
        # NESW by y,x,z
        dishLoc = ([-3.7605200906121437, 3.865245048570655],\
        [-0.0641296465130804, 0.035901010584928825], \
        [-3.6868008232255924, -3.205432553991354], \
        [-7.485557261347768, -0.07592779975602093],)
        
        beamCenters = ([-4.670897458296071, 7.777616712345406],\
        [3.3385316030884016, -3.27205823116637],\
        [3.2569049644387906, 1.2983814825634061],\
        [-10.421877726648322, -0.7263793693162363])
        
        # compute angles (x=lon, y=lat) from each dish
        thetaX = []
        thetaY = []
        for chan in range(4):
	        thetaX.append(np.arctan((x - dishLoc[chan][0])/(z)))
	        thetaY.append(np.arctan((y - dishLoc[chan][1])/(z)))
        thetaX = np.array(thetaX)
        thetaY = np.array(thetaY)
        
        thetaX2 = []
        thetaY2 = []
        for chan in range(4):
	        thetaX2.append(np.arctan((x - beamCenters[chan][0])/(z)))
	        thetaY2.append(np.arctan((y - beamCenters[chan][1])/(z)))
        thetaX_beam = np.array(thetaX2)
        thetaY_beam = np.array(thetaY2)
        
        autos0 = tel_dat.autodata_raw[0,:]
        autos1 = tel_dat.autodata_raw[1,:]

        return {"FLY" : FLY, "BMXdir" :  BMXFILE, "autos0" : autos0, "autos1" : autos1, "drone_time" : t_drone, "tel_time" : t_tel, \
                "x" : x, "y" : y, "z" : z, "thetaX" : thetaX, "thetaY" : thetaY, "thetaX_beam" : thetaX_beam, \
                "thetaY_beam" : thetaY_beam, "r": rprime, "yaw":yaw} 


def getpol0(flight_dict):
    yaw = list(flight_dict['yaw'])

    newdict = {}
    autos = []
    for j in range(4): 
        autos_new = [flight_dict['autos0'][j][i,:] for i in range(len(yaw)) if -5<yaw[i]<5]
        autos.append(autos_new)
    newdict['autos'] =  np.array(autos)
    newdict['FLY'] = flight_dict['FLY']
    newdict['x'] = [flight_dict['x'][i] for i in range(len(yaw)) if -5<yaw[i]<5]
    newdict['y'] = [flight_dict['y'][i] for i in range(len(yaw)) if -5<yaw[i]<5]
    newdict['z'] = [flight_dict['z'][i] for i in range(len(yaw)) if -5<yaw[i]<5]
    
    thetax = []
    thetay = []
    for j in range(4): 
        thetax_new = [flight_dict['thetaX_beam'][j][i] for i in range(len(yaw)) if -5<yaw[i]<5]
        thetax.append(thetax_new)
        thetay_new = [flight_dict['thetaY_beam'][j][i] for i in range(len(yaw)) if -5<yaw[i]<5]
        thetay.append(thetay_new)
    newdict['thetaX_beam'] = np.array(thetax)
    newdict['thetaY_beam'] = np.array(thetay)

    return(newdict)
    
def getpol1(flight_dict):
    yaw = list(flight_dict['yaw'])

    newdict = {}
    autos = []
    for j in range(4): 
        autos_new = [flight_dict['autos1'][j][i,:] for i in range(len(yaw)) if -95<yaw[i]<-85 or 85<yaw[i]<95]
        autos.append(autos_new)
    newdict['autos'] = np.array(autos)
    newdict['x'] = [flight_dict['x'][i] for i in range(len(yaw)) if -95<yaw[i]<-85 or 85<yaw[i]<95]
    newdict['y'] = [flight_dict['y'][i] for i in range(len(yaw)) if -95<yaw[i]<-85 or 85<yaw[i]<95]
    newdict['z'] = [flight_dict['z'][i] for i in range(len(yaw)) if -95<yaw[i]<-85 or 85<yaw[i]<95]
   
    thetax = []
    thetay = []
    for j in range(4): 
        thetax_new = [flight_dict['thetaX_beam'][j][i] for i in range(len(yaw)) if -95<yaw[i]<-85 or 85<yaw[i]<95]
        thetax.append(thetax_new)
        thetay_new = [flight_dict['thetaY_beam'][j][i] for i in range(len(yaw)) if -95<yaw[i]<-85 or 85<yaw[i]<95]
        thetay.append(thetay_new)
    newdict['thetaX_beam'] = np.array(thetax)
    newdict['thetaY_beam'] = np.array(thetay)

    return(newdict)
    
    
    
class Do_Everything_For_Data_dual:
    def __init__(self, drone_dir, FLY, bmx_dir, BMXFILE):
        self.fulldict = droneDataPairing_dualPol(drone_dir, FLY, bmx_dir, BMXFILE)    
        self.pol0 = getpol0(self.fulldict)
        self.pol1 = getpol1(self.fulldict)
    
    def plotscatter(self, polarization, dishindx, freqindx):
        if polarization == 0: dictionary = self.pol0
        if polarization == 1: dictionary = self.pol1
        plt.scatter(dictionary['x'], dictionary['y'], s=50,\
            c = dictionary['autos'][dishindx][:,freqindx],\
            cmap='gnuplot2', norm=LogNorm(), marker = ',')

    def get2Dparams(self, polarization, dishindx, freqindx):
        if polarization == 0: dictionary = self.pol0
        if polarization == 1: dictionary = self.pol1
            
        x = list(np.degrees(dictionary['thetaX_beam'][dishindx]))
        y = list(np.degrees(dictionary['thetaY_beam'][dishindx]))
        z = list(dictionary['autos'][dishindx][:,freqindx])

        initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)
        return(popt, pcov)

    def print2Dparams(self, polarization, dishindx, freqindx):
        if polarization == 0: dictionary = self.pol0
        if polarization == 1: dictionary = self.pol1
            
        x = list(np.degrees(dictionary['thetaX_beam'][dishindx]))
        y = list(np.degrees(dictionary['thetaY_beam'][dishindx]))
        z = list(dictionary['autos'][dishindx][:,freqindx])

        initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)

        popt[3] = 2.355*popt[3]
        popt[4] = 2.355*popt[4]

        pars = ['amplitude', 'xo', 'yo', 'fwhm_x', 'fwhm_y', 'theta', 'offset']
        for i in range(len(pars)): print(pars[i]+' = '+format_e(popt[i]))    