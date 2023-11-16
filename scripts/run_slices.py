## First import general packages for running python analysis:
import os, h5py, datetime,pytz
import numpy as np
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pickle, glob, yaml, string

from scipy.interpolate import griddata
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import least_squares

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

## Then import the beamcals module packages and initialize 'gbosite' class:
from beamcals import corr
from beamcals import concat
from beamcals import drone
from beamcals import bicolog
import beamcals.plotting_utils as pu
import beamcals.fitting_utils as fu
import beamcals.geometry_utils as gu
import beamcals.time_utils as tu
from beamcals import beammap as bp
from beamcals.sites import site

gbosite=site.site('../beamcals/beamcals/sites/GBO_config.npz')

# various gridding attempts
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging


import argparse

# Get which flight in the list to process
parser = argparse.ArgumentParser()

parser.add_argument('EorN', type=str, help='E or N pol')
parser.add_argument('freqit', type=int, help='which frequency')
args = parser.parse_args()

pol_input = args.EorN
freq_input = args.freqit

freqs = 800.0*np.ones(1024) + (-400/1024.)*np.arange(1024)

fitdir='/hirax/GBO_Analysis_Outputs/main_beam_fits/'
ymldir = '/hirax/GBO_Analysis_Outputs/concat_config_files/'
pckldir = '/hirax/GBO_Analysis_Outputs/flight_pickles/'
ampdir = '/hirax/GBO_Analysis_Outputs/amplitude_corrections/'

def get_flightinfo(fly):
    for fi in range(0,len(documents["flight_info"]["flights"])):
        if fly == flights[fi]: f = fi
    
    if polarn[f]=='N':
        pols = N_pols
        cpols = E_pols
    else:
        pols = E_pols
        cpols = N_pols
    return polarn[f], pols, cpols, attns[f], f


def get_slice(beam,Z,val, sliceOrientation='h'):
    # this gradually increases the tolerance until it finds something
    tol = abs(beam.x_centers_grid[1,0,0] - beam.x_centers_grid[0,0,0])/1.5
    N = len(beam.x_centers_grid[:,0,0]) #figure out the importance of this 
    ok = True
    while(ok):
        if sliceOrientation=='h': #keeping the y value constant and changing the x value 
            sliceIndex = np.where((beam.y_centers_grid[0,:,0] < (val + tol)) & (beam.y_centers_grid[0,:,0] > (val-tol)))[0]
            n = np.count_nonzero(np.isfinite(Z)) #count number of 'good' data
            if n > 10: ok = False
            else: ok = True
                #still need to do this one 
        if sliceOrientation=='v':#keeping the x value constant and changing the y value 
            sliceIndex = np.where((beam.x_centers_grid[:,0,0] < (val+tol)) & (beam.x_centers_grid[:,0,0] > (val-tol)))[0]
            n = np.count_nonzero(np.isfinite(Z)) #count number of 'good' data
            if n > 10: ok = False
            else: ok = True
        tol+=1
        if tol > 30: ok = False
    return sliceIndex[0]


with open('/hirax/GBO_Analysis_Outputs/GBO_flights_forscripts.yaml', 'r') as fff:
    documents = yaml.safe_load(fff)
flights = documents["flight_info"]["flights"]
N_pols = documents["flag_info"]["N_pols"]
E_pols = documents["flag_info"]["E_pols"]
polarn = documents["flight_info"]["pols"]
attns = documents["flight_info"]["attns"]

pcklarr=np.sort(os.listdir(pckldir))
gfitarr=np.sort(os.listdir(fitdir))
print(pcklarr)


# FREQUENCY DEFAULT:
find=896#640#896#992#900#570#800#900
f_intern = int((find-512)/16)
print(freqs[find])

# SLICE DEFAULTS # 
sliw = 10 # This defines slices for (some) plots
sz = 80 # use this to set the size of the Xargs and Yargs for beammapping, usually 80 or 50

Npolflights = ['618','619','623','625','646','647','533','536']
Epolflights = ['620','648','649','535']

#good_uniq = [518, 519, 522, 523, 535, 538, 553, 554, 555, 556, 557, 558, 560, 561, 562, 563, 564, 565,
# 566, 567, 568, 569, 571, 572, 573, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 599,
# 630, 631, 632, 633, 636, 640, 645, 676, 691, 692, 693, 695, 696, 697, 698, 699, 700, 702,
# 703, 705, 706, 707, 720, 768, 799, 801, 807, 811, 814, 845, 846, 848, 849, 850, 851, 854,
# 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 871, 873, 874, 875, 876,
# 877, 878, 879, 880, 881, 882, 883, 886, 887, 888, 889, 890, 891, 893, 894, 895, 896, 897,
# 898, 900, 901, 902, 903, 904, 905, 906, 907, 908, 910, 911, 915, 916, 917, 918, 919, 920,
# 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 935, 938] # files from summer 2023

#good_uniq = [513, 514, 515, 518, 520, 522, 523, 525, 526, 531, 532, 533, 536, 538, 553, 554, 555, 556,
# 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 568, 570, 571, 572, 573, 575, 576, 577,
# 578, 579, 581, 582, 583, 584, 599, 630, 631, 632, 633, 639, 640, 645, 676, 691, 692, 695,
# 696, 697, 698, 699, 700, 702, 703, 704, 705, 706, 707, 719, 720, 768, 787, 795, 799, 801,
# 802, 803, 805, 808, 810, 811, 814, 845, 846, 847, 849, 850, 851, 852, 853, 854, 855, 856,
# 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 871, 872, 873, 874, 875,
# 876, 877, 878, 879, 880, 881, 882, 883, 884, 887, 889, 890, 891, 893, 894, 895, 897, 898,
# 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 915, 916, 917,
# 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 932, 933, 934, 935, 936, 937,
# 938]

good_uniq = [517, 518, 519, 520, 522, 523, 525, 532, 533, 536, 538, 553, 554,
       556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 568, 569,
       571, 572, 575, 576, 577, 578, 579, 580, 581, 584, 630, 631, 632,
       633, 636, 639, 645, 676, 691, 692, 695, 696, 697, 698, 699, 700,
       702, 703, 705, 706, 707, 719, 720, 768, 788, 799, 801, 802, 803,
       805, 807, 808, 810, 811, 814, 845, 846, 847, 848, 849, 851, 852,
       853, 854, 855, 856, 857, 858, 860, 861, 862, 863, 864, 865, 866,
       867, 869, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881,
       883, 884, 885, 887, 888, 890, 891, 892, 895, 896, 899, 900, 902,
       903, 904, 905, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916,
       917, 918, 919, 921, 922, 923, 924, 925, 926, 928, 929, 930, 931,
       932, 933, 935, 936, 937, 938, 939]


if pol_input == 'N': pflights = Npolflights
else: pflights = Epolflights

find = good_uniq[freq_input]


flightarr = np.zeros([64,64,16,len(pflights)]) # pixel, pixel, freq, dish, Nflights
normarr = np.zeros([64,64,16,len(pflights)])
   
for i, fly in enumerate(pflights):
    

    pol, pols, cpols, attn, fi = get_flightinfo(fly)
    print(fly,pol,pols,cpols, attn)
    concytest=[glob.glob(pckldir+'*'+fly+'*')[0] for x in flights]
    print(concytest[fi])

    with open(glob.glob(pckldir+'*'+fly+'*')[0], 'rb') as pfile:
        print(pfile)
        concattest1=pickle.load(pfile)
    t_cut=concattest1.inds_on

    beam=bp.Beammap(concatlist=pcklarr[[fi]],gfitlist=gfitarr[[fi]],
                 normalization='Gauss',operation='coadd',Xargs=[-1*sz,sz,2.5],
                 Yargs=[-1*sz,sz,2.5],Fargs=[find,find+1,1],f_index=find,vplot=False)

    if fly == '618':
        normarr[:,:,:,i] = 1.0
    elif fly == '620':
        normarr[:,:,:,i] = 1.0
    elif fly in Npolflights[1::]:
        pklfile = ampdir+'FLY'+str(fly)+'_Corrected_amplitudes.pkl'
        with open(pklfile, 'rb') as inp:
            amps = pickle.load(inp)
        normarr[:,:,:,i] = amps[0,find,:]
    elif fly in Epolflights[1::]:
        pklfile = ampdir+'FLY'+str(fly)+'_Corrected_amplitudes.pkl'
        with open(pklfile, 'rb') as inp:
            amps = pickle.load(inp)
        normarr[:,:,:,i] = amps[0,find,:]
    flightarr[:,:,:,i] = beam.V_LC_mean[:,:,0,:,0]*normarr[:,:,:,i]


xx = 0
chind = pols[0]
colx = get_slice(beam,beam.V_LC_mean[:,:,0,chind,0],xx,'v')
coly = get_slice(beam,beam.V_LC_mean[:,:,0,chind,0],xx,'h')
fig = plt.figure(figsize=(15,20))
for j, chind in enumerate(pols):
    plt.subplot(5,2,2*j+1)
    for i,p in enumerate(pflights):
        plt.semilogy(beam.y_centers_grid[colx,:,0],flightarr[colx,:,chind,i],'o',
             label='Flight '+str(p)+'  x :'+str(beam.x_centers_grid[colx,0,0]))
        plt.title('Input: '+str(chind))
        plt.ylim(0.0001,1)
        plt.xlim(-1*sz,sz)
        plt.legend(loc='lower center')
    plt.subplot(5,2,2*j+2)
    for i,p in enumerate(pflights):
        plt.semilogy(beam.x_centers_grid[:,coly,0],flightarr[:,coly,chind,i],'o',
             label='Flight '+str(p)+'  y :'+str(beam.y_centers_grid[0,coly,0]))
        plt.ylim(0.0001,1)
        plt.xlim(-1*sz,sz)
        plt.title('Input: '+str(chind))
        plt.legend(loc='lower center')
plt.suptitle('Freq ind'+str(find)+' Freq '+str(freqs[find]))
fig = plt.gcf()
fig.savefig('Polin_'+pol_input+'Freqi_'+str(find)+'_slices.png')















