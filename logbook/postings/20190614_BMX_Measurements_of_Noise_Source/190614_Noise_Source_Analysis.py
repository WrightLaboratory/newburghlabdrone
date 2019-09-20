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

raw_directory='/gpfs/mnt/gpfs02/astro/workarea/bmxdata/raw/1906/'
working_directory='/direct/astro+u/tyndall/Drone/'

os.chdir(raw_directory)
D1_files=np.sort(glob.glob('*pulse__D1.data'))
D2_files=np.sort(glob.glob('*pulse__D2.data'))
os.chdir(working_directory)
#print D1_files
#print D2_files
fileslist=[]
for i,fn in enumerate(D1_files):
    if fn.split('__D1')[0]+'__D2.data' in D2_files:
        fileslist.append(fn)

#['190614_1807_pulse__D1.data',      ## Data from direct injection test
# '190614_1951_pulse__D1.data',      ## Data from cross-pol test
# '190614_2100_pulse__D1.data']      ## Data from antenna on dish test

CM=cm.gnuplot2
plt.close('all')

########################################################################
## SAMPLE DRONE DATA: NOT FROM THIS TEST...
########################################################################
flydat=np.load('FLY130_Drone_Coordinates.npz')
figure(figsize=(11,7))
plot(flydat['Lat'][83000:],flydat['Lon'][83000:])
stop


########################################################################
## Broadcast Test: Antenna on Parabola: Dish 3
########################################################################
dat=BMXFile(raw_directory+fileslist[2],loadD2=True)#'190614_2100_pulse__D1.data'


t_arr=Time(dat.data['mjd'],format='mjd')
t_arr.delta_ut1_utc = 0.334 
datetime_arr=np.array([np.datetime64(j.ut1.iso) for j in t_arr])

accepts_3=[]
accepts_7=[]
for i,j in [[0,400],[600,1150],[1350,1800]]:
    accepts_3.append(np.where(dat.data['chan3_0'][i:j,185]>=3.9e13)[0]+i)
    accepts_7.append(np.where(dat.data['chan7_0'][i:j,185]>=8.5e13)[0]+i)
accepts_3=np.concatenate(array(accepts_3)).ravel()           #hot inds 3
accepts_7=np.concatenate(array(accepts_7)).ravel()           #hot inds 7

quiet_3=np.where(dat.data['chan3_0'][0:5200,185]<=2.5e13)[0] #quiet inds
quiet_7=np.where(dat.data['chan7_0'][0:5200,185]<=2.5e13)[0] #quiet inds

diode_on=np.where(dat.data['lj_diode']>=9)[0]               #diode inds

diode_quiet_3=np.intersect1d(quiet_3,diode_on)               #diode quiet 3
diode_quiet_7=np.intersect1d(quiet_7,diode_on)               #diode quiet 3

spec_quiet_3=setdiff1d(quiet_3,diode_on)                     #quiet clean inds 3
spec_quiet_7=setdiff1d(quiet_7,diode_on)                     #quiet clean inds 7

diode_hot_3=np.intersect1d(accepts_3,diode_on)
diode_hot_7=np.intersect1d(accepts_7,diode_on)

spec_hot_3=setdiff1d(accepts_3,diode_on)
spec_hot_7=setdiff1d(accepts_7,diode_on)


man_g3_q=(np.nanmean(dat.data['chan3_0'][diode_quiet_3],axis=0)-np.nanmean(dat.data['chan3_0'][spec_quiet_3],axis=0))/2.3652
man_g3_h=(np.nanmean(dat.data['chan3_0'][diode_hot_3],axis=0)-np.nanmean(dat.data['chan3_0'][spec_hot_3],axis=0))/2.36515
man_g7_q=(np.nanmean(dat.data['chan7_0'][diode_quiet_7],axis=0)-np.nanmean(dat.data['chan7_0'][spec_quiet_7],axis=0))/2.3652
man_g7_h=(np.nanmean(dat.data['chan7_0'][diode_hot_7],axis=0)-np.nanmean(dat.data['chan7_0'][spec_hot_7],axis=0))/2.36515

# Gain Arrays for both Main Channels
figure(figsize=(11,7))
subplot(2,1,1)
title('Gain: 190614 Antenna Broadcast Test')
semilogy(man_g3_q,'.',label='chan3 low')
semilogy(man_g3_h,'.',label='chan3 hot')
ylabel('gain [(ADU**2)/K]')
legend()
subplot(2,1,2)
semilogy(man_g7_q,'.',label='chan7 low')
semilogy(man_g7_h,'.',label='chan7 hot')
ylabel('gain [(ADU**2)/K]')
xlabel('frequency [MHz]')
legend()
#savefig('antenna_broadcast_gain.png')

# Time Series of Both Main Channels
figure(figsize=(11,7))
subplot(2,1,1)
title('Time Series: 190614 Antenna Broadcast Test')
plot(dat.data['chan3_0'][:,185],'.',label='all samples')
plot(spec_hot_3,dat.data['chan3_0'][:,185][spec_hot_3],'.',label='chan3_source_pulses')
plot(spec_quiet_3,dat.data['chan3_0'][:,185][spec_quiet_3],'.',label='chan3_spectra_quiet')
plot(diode_quiet_3,dat.data['chan3_0'][:,185][diode_quiet_3],'.',label='chan3_cal_diode_pulses_low')
plot(diode_hot_3,dat.data['chan3_0'][:,185][diode_hot_3],'.',label='chan3_cal_diode_pulses_hot')
ylabel('Power [ADU**2]')
xlim(0,3000)
legend()
subplot(2,1,2)
plot(dat.data['chan7_0'][:,185],'.',label='all samples')
plot(spec_hot_7,dat.data['chan7_0'][:,185][spec_hot_7],'.',label='chan7_source_pulses')
plot(spec_quiet_7,dat.data['chan7_0'][:,185][spec_quiet_7],'.',label='chan7_spectra_quiet')
plot(diode_quiet_7,dat.data['chan7_0'][:,185][diode_quiet_7],'.',label='chan7_cal_diode_pulses_low')
plot(diode_hot_7,dat.data['chan7_0'][:,185][diode_hot_7],'.',label='chan7_cal_diode_pulses_hot')
ylabel('Power [ADU**2]')
xlabel('samples [time~18m]')
xlim(0,3000)
legend()
#savefig('antenna_broadcast_time_series.png')

# Spectra Plots from Data Collection

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(14,8), nrows=2, ncols=2,sharex=True)
for axis in [ax1,ax2,ax3,ax4]:
    axis.set_xlabel('frequency [MHz]')
    axis.set_yscale('log')

ax1.set_title('Spectra: 190614 Antenna Broadcast Test [Channel 3]')
ax1.plot(dat.freq[0],np.nanmean(dat.data['chan3_0'][spec_hot_3],axis=0),'.',label='chan3_source_pulse_spectra')
ax1.plot(dat.freq[0],np.nanmean(dat.data['chan3_0'][spec_quiet_3],axis=0),'.',label='chan3_spectra_quiet')
ax1.set_ylabel('Power [ADU**2]')
ax1.legend()
ax2.set_title('Spectra: 190614 Antenna Broadcast Test [Channel 7]')
ax2.plot(dat.freq[0],np.nanmean(dat.data['chan7_0'][spec_hot_7],axis=0),'.',label='chan7_source_pulse_spectra')
ax2.plot(dat.freq[0],np.nanmean(dat.data['chan7_0'][spec_quiet_7],axis=0),'.',label='chan7_spectra_quiet')
ax2.set_ylabel('Power [ADU**2]')
ax2.legend()
ax3.plot(dat.freq[0],(np.nanmean(dat.data['chan3_0'][spec_hot_3],axis=0)-np.nanmean(dat.data['chan3_0'][spec_quiet_3],axis=0))/man_g3_q,'.',label='chan3_source_pulse_spectra_q')
ax3.plot(dat.freq[0],(np.nanmean(dat.data['chan3_0'][spec_hot_3],axis=0)-np.nanmean(dat.data['chan3_0'][spec_quiet_3],axis=0))/man_g3_h,'.',label='chan3_source_pulse_spectra_h')
ax3.set_ylabel('Antenna Temperature [K]')
ax3.legend()
ax4.plot(dat.freq[0],(np.nanmean(dat.data['chan7_0'][spec_hot_7],axis=0)-np.nanmean(dat.data['chan7_0'][spec_quiet_7],axis=0))/man_g7_q,'.',label='chan7_source_pulse_spectra_q')
ax4.plot(dat.freq[0],(np.nanmean(dat.data['chan7_0'][spec_hot_7],axis=0)-np.nanmean(dat.data['chan7_0'][spec_quiet_7],axis=0))/man_g7_h,'.',label='chan7_source_pulse_spectra_h')
ax4.set_ylabel('Antenna Temperature [K]')
ax4.legend()
#savefig('antenna_broadcast_spectra.png')

#plot other channels to see if pulses are visible: answer is yeeees?
plt.close('all')
for i,j in zip(['chan1_0','chan2_0','chan3_0','chan4_0'],['chan5_0','chan6_0','chan7_0','chan8_0']):
    #k=750
    g_i=(np.nanmean(dat.data[i][diode_quiet_3],axis=0)-np.nanmean(dat.data[i][spec_quiet_3],axis=0))/2.36515
    g_j=(np.nanmean(dat.data[j][diode_quiet_3],axis=0)-np.nanmean(dat.data[j][spec_quiet_3],axis=0))/2.36515
    figure(figsize=(11,7))
    subplot(2,1,1)
    title(i+' Spectra from Antenna Broadcast Test')
    plot(dat.freq[0],(np.nanmean(dat.data[i][spec_hot_3],axis=0)-np.nanmean(dat.data[i][spec_quiet_3],axis=0))/g_i,'.',label=i+' source signal')
    #plot(dat.freq[0],np.nanmean(dat.data[i][spec_hot_3],axis=0),'.',label=i+' source signal')
    #plot(dat.freq[0],np.nanmean(dat.data[i][spec_quiet_3],axis=0),'.',label=i+' no signal')
    ylabel('Antenna Temperature [K]')  
    ylim(0,200)
    #plot(quiet_3,dat.data[i][:,k][quiet_3],'.',label=i+' all samples')
    #plot(diode_on,dat.data[i][:,k][diode_on],'.',label=i+' diode')
    #ylabel('Power [ADU**2]')
    subplot(2,1,2)
    title(j+' Spectra from Antenna Broadcast Test')
    plot(dat.freq[0],(np.nanmean(dat.data[j][spec_hot_3],axis=0)-np.nanmean(dat.data[j][spec_quiet_3],axis=0))/g_j,'.',label=j+' source signal')
    ylim(0,200)
    #plot(dat.freq[0],np.nanmean(dat.data[j][spec_hot_3],axis=0),'.',label=j+' source signal')
    #plot(dat.freq[0],np.nanmean(dat.data[j][spec_quiet_3],axis=0),'.',label=j+' no signal')
    ylabel('Antenna Temperature [K]')
    xlabel('Frequency [MHz]')
    #plot(quiet_3,dat.data[j][:,k][quiet_3],'.',label=j+' all samples')
    #plot(diode_on,dat.data[j][:,k][diode_on],'.',label=j+' diode')
    #ylabel('Power [ADU**2]')
    #xlabel('samples [time~18m]')
stop


########################################################################
## DIRECT INJECTION CODE:
########################################################################
dat=BMXFile(raw_directory+fileslist[0],loadD2=True)#'190614_1807_pulse__D1.data'

accepts=[]
for i,j in [[0,290],[450,1020],[1150,1730],[1900,2490],[2640,3168],[3365,3900],[4070,4600],[4800,5200]]:
    accepts.append(np.where(dat.data['chan8_0'][i:j,185]>=6.5e13)[0]+i)
accepts=np.concatenate(array(accepts)).ravel()           #hot inds
quiet=np.where(dat.data['chan8_0'][0:5200,185]<=2e13)[0] #quiet inds
diode_on=np.where(dat.data['lj_diode']>=9)               #diode inds

diode_quiet=np.intersect1d(quiet,diode_on)               #diode quiet
spec_quiet=setdiff1d(quiet,diode_on)                     #quiet clean inds
diode_hot=np.intersect1d(accepts,diode_on)
spec_hot=setdiff1d(accepts,diode_on)


man_g1_q=(np.nanmean(dat.data['chan8_0'][diode_quiet],axis=0)-np.nanmean(dat.data['chan8_0'][spec_quiet],axis=0))/2.3652
man_g1_h=(np.nanmean(dat.data['chan8_0'][diode_hot],axis=0)-np.nanmean(dat.data['chan8_0'][spec_hot],axis=0))/2.36515

figure(figsize=(11,7))
title('Gain: 190614 Direct Injection of Source on one BMX Channel')
semilogy(man_g1_q,'.')
#semilogy(man_g1_h,'.')
ylabel('gain [(ADU**2)/K]')
xlabel('frequency [MHz]')
#savefig('direct_injection_gain.png')

figure(figsize=(11,7))
subplot(2,1,1)
title('Time Series: 190614 Direct Injection of Source on one BMX Channel')
imshow(dat.data['chan8_0'],cmap=CM,clim=(1e11,1e14),extent=(dat.freq[0][0],dat.freq[0][-1],dat.nSamples*dat.deltaT/60.0,0.0))
ylim(0.5,0)
ylabel('time [min]')
xlabel('frequency [MHz]')
subplot(2,1,2)
plot(dat.data['chan8_0'][:,185],'.')
plot(accepts,dat.data['chan8_0'][:,185][accepts],'.',label='source_pulses')
plot(spec_quiet,dat.data['chan8_0'][:,185][spec_quiet],'.',label='spectra_quiet')
plot(diode_quiet,dat.data['chan8_0'][:,185][diode_quiet],'.',label='cal_diode_pulses')
ylabel('Power [ADU**2]')
xlabel('samples [time~100m]')
legend()
#savefig('direct_injection_time_series.png')

figure(figsize=(11,7))
subplot(2,1,1)
title('Spectra: 190614 Direct Injection of Source on one BMX Channel')
semilogy(dat.freq[0],np.nanmean(dat.data['chan8_0'][accepts],axis=0),'.',label='source_pulse_spectra')
semilogy(dat.freq[0],np.nanmean(dat.data['chan8_0'][spec_quiet],axis=0),'.',label='spectra_quiet')
semilogy(dat.freq[0],np.nanmean(dat.data['chan8_0'][diode_quiet],axis=0),'.',label='diode_quiet')
xlabel('frequency [MHz]')
ylabel('Power [ADU**2]')
legend()
subplot(2,1,2)
semilogy(dat.freq[0],(np.nanmean(dat.data['chan8_0'][accepts],axis=0)-np.nanmean(dat.data['chan8_0'][spec_quiet],axis=0))/man_g1_q,'.',label='source_pulse_spectra')
ylabel('Antenna Temperature [K]')
xlabel('frequency [MHz]')
legend()
#savefig('direct_injection_spectra.png')

stop

########################################################################
## Examine all Channels: Produce 2x2x2 WF Plots
########################################################################

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(11,8), nrows=2, ncols=2,sharex=True,sharey=True)
for axis,chan in zip([ax1,ax2,ax3,ax4],['chan1_0','chan2_0','chan3_0','chan4_0']):
    im1=axis.imshow(dat.data[chan],cmap=CM,clim=(1e11,1e14),extent=(dat.freq[0][0],dat.freq[0][-1],dat.nSamples*dat.deltaT/60.0,0.0))
    #axis.set_xlim(1100,1510)
    #axis.set_ylim(25.0,0.0)
    axis.set_xlabel('frequency [MHz]')
    axis.set_ylabel('time [min]')
    divider1=make_axes_locatable(axis)
    cax1=divider1.append_axes("right", size="5%", pad=0.05)
    cbar1=fig.colorbar(im1,cax=cax1)
    cbar1.set_label('ADU$**2$')

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(11,8), nrows=2, ncols=2,sharex=True,sharey=True)
for axis,chan in zip([ax1,ax2,ax3,ax4],['chan5_0','chan6_0','chan7_0','chan8_0']):
    im1=axis.imshow(dat.data[chan],cmap=CM,clim=(1e11,1e14),extent=(dat.freq[0][0],dat.freq[0][-1],dat.nSamples*dat.deltaT/60.0,0.0))
    #axis.set_xlim(1100,1510)
    #axis.set_ylim(25.0,0.0)
    axis.set_xlabel('frequency [MHz]')
    axis.set_ylabel('time [min]')
    divider1=make_axes_locatable(axis)
    cax1=divider1.append_axes("right", size="5%", pad=0.05)
    cbar1=fig.colorbar(im1,cax=cax1)
    cbar1.set_label('ADU$**2$')






########################################################################
## GARBAGE:
########################################################################


#dat=BMXFile(raw_directory+fileslist[0],loadD2=True)
#t_arr=Time(dat.data['mjd'],format='mjd')
#t_arr.delta_ut1_utc = 0.334 #???
#t_arr.ut1.iso

## For this file, the proper tag is: dat.data['chan8_0']
#np.where(dat.data['chan8_0'][1150:1760,185]>=6.5e13)[0]

##Trying to isolate high drone pulses without rise:
comb=np.zeros(len(dat.data['chan8_0']))
calseed=4
calul=calseed+0.5
calll=calseed-0.5

for ind in range(len(dat.data['chan8_0'])):
    for n in np.arange(-5,10000):
        #if (n*5.0)+calll <= ind <= (n*5.0)+calul:
        if (n*4.975409836065574)+calll <= ind <= (n*4.975409836065574)+calul:
            comb[ind]=1
pulse_index=list(np.where(comb==1)[0])

diode=np.where(dat.data['lj_diode']>=5)

print len(pulse_index)

figure(figsize=(11,7))
plot(np.linspace(0,dat.nSamples*dat.deltaT/60.0,len(dat.data['chan8_0'][:,185])),dat.data['chan8_0'][:,185],'.')
plot(np.linspace(0,dat.nSamples*dat.deltaT/60.0,len(dat.data['chan8_0'][:,185]))[pulse_index],dat.data['chan8_0'][:,185][pulse_index],'.')
plot(np.linspace(0,dat.nSamples*dat.deltaT/60.0,len(dat.data['chan8_0'][:,185]))[diode],dat.data['chan8_0'][:,185][diode],'.')
