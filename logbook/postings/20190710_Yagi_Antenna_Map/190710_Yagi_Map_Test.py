## Examine Data from Yale Noise Source tests:
## Look at 1/f corner frequency...
import numpy as np
from matplotlib.pyplot import *
from bmxdata import *
import os
import glob
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from random import sample
#plotting nonsense
from astropy.time import Time
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

########################################################################
## DRONE SPECTRA
########################################################################

def dBm2mW(x):
    return 10.0**(x/10.0) #mW

def mW2dBm(x):
    return 10.0*np.log10(x) #dBm
    
working_directory='/direct/astro+u/tyndall/Drone/'
csv_directory='/direct/astro+u/tyndall/Drone/190710_Yagi_Map_Test/'
os.chdir(csv_directory)
csvs=np.sort(glob.glob('*.csv'))
os.chdir(working_directory)

plt.close('all')
colorsarr=cm.gnuplot2(np.linspace(0,1,14))

figure(figsize=(14,8))
subplot(2,1,1)

for i,fn in enumerate(csvs[1:-4]):
    dat=np.genfromtxt(csv_directory+fn,delimiter=',')
    plot(dat[:,0]/1e6,dat[:,1],c=colorsarr[i+1],label=fn)

pulses=np.zeros((0,len(dat[:,0])))
bknd=np.zeros((0,len(dat[:,0])))
for i,fn in enumerate([csvs[0],csvs[-4],csvs[-3],csvs[-2]]):
    dat=np.genfromtxt(csv_directory+fn,delimiter=',')
    bknd=np.vstack((bknd,dat[:,1]))
    
plot(dat[:,0]/1e6,np.nanmean(bknd,axis=0),linewidth=2,c=colorsarr[0],label='averaged background')
legend(prop={'size': 8})
xlim(200,1000)
xlabel('Frequency [MHz]')
ylabel('Power [dBm]')
title('Spectra: 190710 Yagi Beam Mapping')

subplot(2,1,2)
for i,fn in enumerate(csvs[1:-4]):
    dat=np.genfromtxt(csv_directory+fn,delimiter=',')
    pulses=np.vstack((pulses,dat[:,1]))
    plot(dat[:,0]/1e6,mW2dBm(dBm2mW(dat[:,1])-dBm2mW(np.nanmean(bknd,axis=0))),c=colorsarr[i+1],label=fn)

legend(prop={'size': 8})
xlim(200,1000)
xlabel('Frequency [MHz]')
ylabel('Power [dBm]')
savefig('drone_spectra.png')

pulses_sub=mW2dBm(dBm2mW(pulses)-np.outer(np.ones(len(pulses)),np.nanmean(dBm2mW(bknd),axis=0)))
#figure(figsize=(11,8))
#imshow(pulses_sub)
#colorbar()
#clim(-1500,0)

########################################################################
## DRONE RTK DATA
########################################################################
flydat=np.load('FLY145_Drone_Coordinates.npz')
figure(figsize=(11,7))
plot(flydat['Lat'][:113734],flydat['Lon'][:113734])


lat_n=np.zeros(11)
lon_n=np.zeros(11)
hmsl_n=np.zeros(11)
ll_dist_n=np.zeros(11)
C=np.pi/180.0
for n,[i,j] in enumerate(np.array([[9000,14000],[17500,20500],[23500,27000],[30000,33000],[35000,40000],[41000,47000],[49500,53500],[55500,57500],[59500,61000],[62500,64500],[67500,69000]])):
    plot(flydat['Lat'][i:j],flydat['Lon'][i:j],linewidth=3,c=colorsarr[n+1],label=n)
    lat_n[n]=np.nanmean(flydat['Lat'][i:j])
    lon_n[n]=np.nanmean(flydat['Lon'][i:j])
    hmsl_n[n]=np.nanmean(flydat['hmsl'][i:j])   
    ll_dist_n[n]=np.arccos((np.sin(C*lat_n[n])*np.sin(C*lat_n[0]))+(np.cos(C*lat_n[n])*np.cos(C*lat_n[0])*np.cos((C*lon_n[n])-(C*lon_n[0]))))
    
legend()
xlabel('Lat [deg]')
ylabel('Lon [deg]')
savefig('drone_map.png')


xy_dist_n=np.array([0,2.85,2.49,2.36,2.39,4.22,4.22,4.65,3.63,3.47,5.70])
z_dist_n=(hmsl_n-flydat['hmsl'][500])
hyp_n=np.sqrt((xy_dist_n**2.0)+(z_dist_n**2.0))
theta_n=np.arctan(xy_dist_n/z_dist_n)

## plot angle as fxn of which exposure
figure()
#plot(theta_n/C,'o')

#try 528:533?
mu_theta=np.zeros(11)
sig_theta=np.zeros(11)
for i in range(len(pulses)):
    mu_theta[i]=mW2dBm(np.nanmean(dBm2mW(pulses[i,419:427])-np.nanmean(dBm2mW(bknd),axis=0)[419:427]))
    sig_theta[i]=mW2dBm(np.nanstd(dBm2mW(pulses[i,419:427])-np.nanmean(dBm2mW(bknd),axis=0)[419:427]))

plot(theta_n/C,mu_theta,'o',c=colorsarr[7])
#plot(theta_n/C,dBm2mW(vals_theta)/dBm2mW(vals_theta[0]),'o',c=colorsarr[7])
#errorbar(theta_n/C,mu_theta,sig_theta,c=colorsarr[7],ls='')
title('Azimuthally Symmetric Yagi Antenna Beam Map')
ylabel('Received Pulse Power [dBm]')
xlabel('Angular Distance from Antenna [deg]')
savefig('drone_beammap_azsym.png')


