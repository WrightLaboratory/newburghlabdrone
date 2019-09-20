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
## SPECTRA
########################################################################

def dBm2mW(x):
    return 10.0**(x/10.0) #mW

def mW2dBm(x):
    return 10.0*np.log10(x) #dBm
    
## Declare directories and glob some files:
working_directory='/direct/astro+u/tyndall/Drone/'
csv_directory='/direct/astro+u/tyndall/Drone/190711_Yagi_Map/'
os.chdir(csv_directory)
yagi_csvs=np.sort(glob.glob('beam-yagi*'))
rfi_csvs=np.sort(glob.glob('rfi-log-per*'))
os.chdir(working_directory)

## Housekeeping
plt.close('all')
colorsarr=cm.gnuplot2(np.linspace(0,1,60))

## Create Arrays with Sample Data:
dat=np.genfromtxt(csv_directory+yagi_csvs[0],delimiter=',')
f=dat[:,0]/1e6

## Loop to Concat Pulse Spectra:
pulses=np.zeros((0,len(dat[:,0])))
for i,fn in enumerate(yagi_csvs[:-5]):
    dat=np.genfromtxt(csv_directory+fn,delimiter=',')
    pulses=np.vstack((pulses,dat[:,1]))

## Loop to Concat Background Spectra:
bknd=np.zeros((0,len(dat[:,0])))
for i,fn in enumerate(yagi_csvs[-5:]):
    dat=np.genfromtxt(csv_directory+fn,delimiter=',')
    bknd=np.vstack((bknd,dat[:,1]))

## Plot Pulse and Background Spectra:
figure(figsize=(11,7))
for i in range(len(pulses)):
    plot(f,pulses[i],color=colorsarr[i+1],label=yagi_csvs[:-5][i])
plot(f,np.nanmean(bknd,axis=0),linewidth=2,c=colorsarr[0],label='averaged background')
legend(prop={'size': 6})
xlim(300,1025)
ylim(-105,-50)
xlabel('Frequency [MHz]')
ylabel('Power [dBm]')
title('Spectra: 190711 Yagi Beam Mapping')

## Plot Background Subtracted Pulse Spectra:
figure(figsize=(11,7))
for i in range(len(pulses)):
    plot(f,mW2dBm(dBm2mW(pulses[i])-dBm2mW(np.nanmean(bknd,axis=0))),color=colorsarr[i+1],label=yagi_csvs[:-5][i])
legend(prop={'size': 6})
xlim(300,1025)
ylim(-150,-50)
xlabel('Frequency [MHz]')
ylabel('Power [dBm]')
title('Spectra: 190711 Yagi Beam Mapping [Background Subtracted]')

pulses_sub=mW2dBm(dBm2mW(pulses)-np.outer(np.ones(len(pulses)),np.nanmean(dBm2mW(bknd),axis=0)))
figure(figsize=(11,7))
imshow(pulses_sub)
colorbar()
clim(-150,-50)

########################################################################
## DRONE RTK DATA
########################################################################
flydat=np.load('FLY147_FLY148_Drone_Coordinates.npz')
figure(figsize=(11,7))
plot(flydat['Lat'],flydat['Lon'])

lat_n=np.zeros(len(pulses))
lon_n=np.zeros(len(pulses))
hmsl_n=np.zeros(len(pulses))
ll_dist_n=np.zeros(len(pulses))
C=np.pi/180.0
xy_dist_n=np.array([0.0,0.0, #overhead 
                    1.07,1.07, #point 1
                    2.21,2.21, #point 2
                    4.16,4.16, #point 3
                    6.68,6.68, #point 4
                    8.45,8.45, #point 5

                    0.0,0.0, #overhead again
                    0.74,0.74, #point 6
                    1.69,1.69, #point 7
                    2.68,2.68, #point 8
                    4.14,4.14, #point 9
                    6.52,6.52, #point 10

                    0.67,0.67, #point 11
                    1.51,1.51, #point 12
                    2.48,2.48, #point 13
                    3.61,3.61, #point 14
                    5.46,5.46, #point 15

                    1.01,1.01, #point 16
                    2.16,2.16, #point 17
                    3.56,3.56, #point 18
                    4.79,4.79, #point 19
                    6.46,6.46]) #point 20


figure(5,figsize=(11,7))
plot(flydat['Lat'][100000:],flydat['Lon'][100000:],'g')

figure(6,figsize=(11,7))
subplot(2,1,1)
plot(flydat['timestamp'][100000:],flydat['Lat'][100000:],'g')
subplot(2,1,2)
plot(flydat['timestamp'][100000:],flydat['Lon'][100000:],'g')

ind_bounds_n=np.array([[114500,119000],  #overhead
                       [124000,128000],  #point 1
                       [129000,133000],  #point 2
                       [134000,136000],  #point 3
                       [137500,140000],  #point 4
                       [141500,143500],  #point 5

                       [148750,150500],  #overhead
                       [151000,153500],  #point 6
                       [154500,157000],  #point 7
                       [158000,161000],  #point 8
                       [162000,164000],  #point 9
                       [165000,167500],  #point 10

                       [173500,176250],  #point 11
                       [177000,178500],  #point 12
                       [179500,181000],  #point 13
                       [182200,184400],  #point 14
                       [185000,187000],  #point 15

                       [188500,190500],  #point 16
                       [191000,192750],  #point 17
                       [193500,195000],  #point 18
                       [196000,198000],  #point 19
                       [199000,201000]]) #point 20

for n,[i,j] in enumerate(ind_bounds_n):
    figure(5)
    plot(flydat['Lat'][i:j],flydat['Lon'][i:j],linewidth=3,c=colorsarr[2*n+1],label=n)
    figure(6)
    subplot(2,1,1)
    plot(flydat['timestamp'][i:j],flydat['Lat'][i:j],linewidth=3,c=colorsarr[2*n+1],label=n)
    subplot(2,1,2)
    plot(flydat['timestamp'][i:j],flydat['Lon'][i:j],linewidth=3,c=colorsarr[2*n+1],label=n)
    lat_n[2*n]=np.nanmean(flydat['Lat'][i:j])
    lon_n[2*n]=np.nanmean(flydat['Lon'][i:j])
    hmsl_n[2*n]=np.nanmean(flydat['hmsl'][i:j])   
    ll_dist_n[2*n]=np.arccos((np.sin(C*lat_n[n])*np.sin(C*lat_n[0]))+(np.cos(C*lat_n[n])*np.cos(C*lat_n[0])*np.cos((C*lon_n[n])-(C*lon_n[0]))))
    lat_n[2*n+1]=np.nanmean(flydat['Lat'][i:j])
    lon_n[2*n+1]=np.nanmean(flydat['Lon'][i:j])
    hmsl_n[2*n+1]=np.nanmean(flydat['hmsl'][i:j])   
    ll_dist_n[2*n+1]=np.arccos((np.sin(C*lat_n[n])*np.sin(C*lat_n[0]))+(np.cos(C*lat_n[n])*np.cos(C*lat_n[0])*np.cos((C*lon_n[n])-(C*lon_n[0]))))

figure(5)
legend()
xlabel('Lat [deg]')
ylabel('Lon [deg]')
savefig('drone_map.png')

z_dist_n=(hmsl_n-flydat['hmsl'][100000])
hyp_n=np.sqrt((xy_dist_n**2.0)+(z_dist_n**2.0))
theta_n=np.arctan(xy_dist_n/z_dist_n)


############
## BEAM MAP
############

mu_theta=np.zeros(len(pulses))
sig_theta=np.zeros(len(pulses))
good=np.where(~np.isnan(np.mean(pulses_sub,axis=0))==True)[0]


good=np.concatenate((np.arange(len(pulses[0]))[360:440],np.arange(len(pulses[0]))[450:480],np.arange(len(pulses[0]))[550:620]))

for i in range(len(pulses)):
    lb=230 #0 #375 #230
    ub=475 #-1 #400 #475
    mu_theta[i]=np.nanmean(pulses_sub[i,lb:ub][pulses_sub[i,lb:ub]>-inf])
    sig_theta[i]=np.nanstd(pulses_sub[i,lb:ub][pulses_sub[i,lb:ub]>-inf])
    mu_theta[i]=np.nanmean(pulses_sub[i,good][pulses_sub[i,good]>-inf])
    sig_theta[i]=np.nanstd(pulses_sub[i,good][pulses_sub[i,good]>-inf])

from scipy.optimize import curve_fit

def Gauss(x, A, mueff, sigma, C):
    y = A*np.exp(-(x-mueff)**2/(2.*sigma**2)) + C
    return y

#plt.close('all')
figure(figsize=(16,9))

subplot(2,1,1)
# plot axis 1 data points:
plot(-1*theta_n[0:12]/C,mu_theta[0:12],'o',c=colorsarr[10],label='axis 1')
plot(theta_n[12:24]/C,mu_theta[12:24],'o',c=colorsarr[10])
# plot axis 1 error bars:
errorbar(-1*theta_n[0:12]/C,mu_theta[0:12],sig_theta[0:12],c=colorsarr[10],ls='')
errorbar(theta_n[12:24]/C,mu_theta[12:24],sig_theta[12:24],c=colorsarr[10],ls='')
# specify params for Gaussian Fitting of axis 1 points:
x1=np.concatenate((-1*theta_n[0:12],theta_n[12:24]))/C
y1=np.concatenate((mu_theta[0:12],mu_theta[12:24]))
mueff1=np.mean(x1)
sigma1=np.std(x1)
C1=np.min(y1)
# gauss fit on axis 1 data points
popt1,pcov1 = curve_fit(Gauss,x1,y1,p0=[1,mueff1,sigma1,C1])
plot(np.linspace(-60,60,1000),Gauss(np.linspace(-60,60,1000),*popt1),label='Gauss_fit')
# append points to legend:
plot([],[],'k.',label='mean = '+str(popt1[1]))
plot([],[],'k.',label='std = '+str(popt1[2]))
plot([],[],'k.',label='coeff = '+str(popt1[0]))
plot([],[],'k.',label='y-int = '+str(popt1[3]))
title('Azimuthally Symmetric Yagi Antenna Beam Map')
ylabel('Received Pulse Power [dBm]')
xlim(-60,100)
legend()

subplot(2,1,2)
# plot axis 2 data points, with overhead points as well:
plot(-1*theta_n[24:34]/C,mu_theta[24:34],'o',c=colorsarr[30],label='axis 2')
plot(theta_n[34:]/C,mu_theta[34:],'o',c=colorsarr[30])
plot(theta_n[0:2]/C,mu_theta[0:2],'o',c=colorsarr[30])
plot(theta_n[12:14]/C,mu_theta[12:14],'o',c=colorsarr[30])
# plot axis 2 errorbars, with overhead points as well:
errorbar(-1*theta_n[24:34]/C,mu_theta[24:34],sig_theta[24:34],c=colorsarr[30],ls='')
errorbar(theta_n[34:]/C,mu_theta[34:],sig_theta[34:],c=colorsarr[30],ls='')
errorbar(-1*theta_n[0:2]/C,mu_theta[0:2],sig_theta[0:2],c=colorsarr[30],ls='')
errorbar(theta_n[12:14]/C,mu_theta[12:14],sig_theta[12:14],c=colorsarr[30],ls='')
# specify params for Gaussian Fitting of axis 2 points:
x2=np.concatenate((theta_n[0:2],theta_n[12:14],-1*theta_n[24:34],theta_n[34:]))/C
y2=np.concatenate((mu_theta[0:2],mu_theta[12:14],mu_theta[24:34],mu_theta[34:]))
mueff2=np.mean(x2)
sigma2=np.std(x2)
C2=np.min(y2)
# gauss fit on axis 2 data points
popt2,pcov2 = curve_fit(Gauss,x2,y2,p0=[1,mueff2,sigma2,C2])
plot(np.linspace(-60,60,1000),Gauss(np.linspace(-60,60,1000),*popt2),label='Gauss_fit')
plot([],[],'k.',label='mean = '+str(popt2[1]))
plot([],[],'k.',label='std = '+str(popt2[2]))
plot([],[],'k.',label='coeff = '+str(popt2[0]))
plot([],[],'k.',label='y-int = '+str(popt2[3]))
ylabel('Received Pulse Power [dBm]')
xlabel('Angular Distance from Antenna [deg]')
xlim(-60,100)
legend()

#stop

#############
# ALT AZ
#############
#plt.close('all')

pointsdat=np.load('FLY147_FLY148_points.npz') #['mean_alt', 'std_alt', 'mean_az', 'std_az']
ax=plt.subplot(111,projection='polar')
ax.plot(pointsdat['mean_az']*C,pointsdat['mean_alt'],'.')

###################
# BEAMMAP IMSHOW  #
###################
plt.close('all')



nx=10
ny=10
lat_grid=np.linspace(41.32045,41.32056,nx+1)
lon_grid=np.linspace(-72.92247,-72.92233,ny+1)

#stop

beammap=np.zeros((nx,ny))
for i,m in enumerate(lat_grid[:-1]):
    for j,n in enumerate(lon_grid[:-1]):
        passed_pts=np.intersect1d(np.intersect1d(np.where(lat_grid[i]<=lat_n)[0],np.where(lat_n<=lat_grid[i+1])[0]),np.intersect1d(np.where(lon_grid[j]<=lon_n)[0],np.where(lon_n<=lon_grid[j+1])[0]))
        beammap[i,j]=np.nanmean(dBm2mW(mu_theta[passed_pts])/np.max(dBm2mW(mu_theta)))
        if len(passed_pts)>=1:
            print i,j,passed_pts

#figure(figsize=(12,10))
#imshow(beammap.T,origin='lower',cmap='copper',extent=[lat_grid[0],lat_grid[-1],lon_grid[0],lon_grid[-1]])
#plot(lat_grid[:-1],lon_grid[:-1],'.')
#plot(lat_n,lon_n,'wx')
#colorbar()
#xlabel('Latitude')
#ylabel('Longitude')
#title('190711 Yagi Beammap')
#savefig('yagi_beammap.png')

fig,ax1=plt.subplots(figsize=(10,9),nrows=1,ncols=1)
im1=ax1.imshow(beammap.T,origin='lower',cmap='copper',extent=[lat_grid[0],lat_grid[-1],lon_grid[0],lon_grid[-1]])
ax1.plot(lat_n,lon_n,'wo')
ax1.set_xlabel('Latitude [deg]')
ax1.set_ylabel('Longitude [deg]')
divider1=make_axes_locatable(ax1)
cax1=divider1.append_axes("right", size="5%", pad=0.05)
cbar1=fig.colorbar(im1,cax=cax1)
cbar1.set_label('Normalized Power')
plt.tight_layout(pad=0.2, w_pad=1.0, h_pad=0.0)
#plt.savefig('Composite_WF_Plots.png')

#stop



## Replot in thousands of degrees for Emily Kuhn:
fig,ax1=plt.subplots(figsize=(10,9),nrows=1,ncols=1)
im1=ax1.imshow(beammap.T,origin='lower',cmap='copper',extent=[1000.0*lat_grid[0],1000.0*lat_grid[-1],1000.0*lon_grid[0],1000.0*lon_grid[-1]])
ax1.plot(1000.0*lat_n,1000.0*lon_n,'wx')
ax1.set_xlabel('Latitude [$10^{-3}$ deg]')
ax1.set_ylabel('Longitude [$10^{-3}$ deg]')
divider1=make_axes_locatable(ax1)
cax1=divider1.append_axes("right", size="5%", pad=0.05)
cbar1=fig.colorbar(im1,cax=cax1)
cbar1.set_label('Normalized Power')
plt.tight_layout(pad=0.2, w_pad=1.0, h_pad=0.0)
plt.savefig('yagi_beammap.png')
stop

