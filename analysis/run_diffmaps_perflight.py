## First import general packages for running python analysis:
import os, h5py, datetime,pytz
import numpy as np
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pickle, glob, yaml

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
import argparse

# Get which flight in the list to process
parser = argparse.ArgumentParser()

parser.add_argument('flight_iterator', type=str, help='which flight')
parser.add_argument('flight_control', type=str, help='which control flight')
parser.add_argument('inner_mask', type=int, help='mask inside')
parser.add_argument('outer_mask', type=int, help='mask outside')
parser.add_argument('minfreq', type=int, help='start freq index')
args = parser.parse_args()

fly_input = args.flight_iterator
fly_control = args.flight_control

high = args.inner_mask #20 # inner mask, 20 for large flights
low = -1*high #-20 # inner mask, 20 for large flights
maskin = high # annulus mask, set equal to high
maskout = args.outer_mask #40 # annulus mask, 40 for large flights

fmin = args.minfreq
fmax = fmin+32
fstep = 1
find = fmin+fstep

print('Input flight: '+fly_input+'Freq range '+str(fmin)+' '+str(fmax))

gbosite=site.site('../beamcals/beamcals/sites/GBO_config.npz')
freqs = 800.0*np.ones(1024) + (-400/1024.)*np.arange(1024)

fitdir='/hirax/GBO_Analysis_Outputs/main_beam_fits/'
ymldir = '/hirax/GBO_Analysis_Outputs/concat_config_files/'
pckldir = '/hirax/GBO_Analysis_Outputs/flight_pickles/'

## Read in Yaml file for info per flight:
with open('/home/ln267/newburghlabdrone/analysis/GBO_flights_forscripts.yaml', 'r') as fff:
    documents = yaml.safe_load(fff)
flights = documents["flight_info"]["flights"]
N_pols = documents["flag_info"]["N_pols"]
E_pols = documents["flag_info"]["E_pols"]
polarn = documents["flight_info"]["pols"]
attns = documents["flight_info"]["attns"]

pcklarr=np.sort(os.listdir(pckldir))
gfitarr=np.sort(os.listdir(fitdir))
print(pcklarr)

## Define anything I want to keep common
dotsize=1
res = 500
LX,LY = np.meshgrid(np.linspace(-100,100,res), np.linspace(-100,100,res))
X = np.arange(-100,100,2.0,dtype='float64')
Y = np.arange(-100,100,2.0,dtype='float64')


# Define colormap for this nb:
cmap = matplotlib.cm.get_cmap('gnuplot2')
norm = matplotlib.colors.Normalize(vmin=-25, vmax=25)

sliw = 10 # This defines slices for (some) plots
sz = 80 # use this to set the size of the Xargs and Yargs for beammapping, usually 80 or 50

nss = np.arange(0.1,3.0,0.001) # this defines the range of amplitudes for variance checking

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


def get_beam_diff(beam2, beam1,n):
    return beam2[:,:] - n*beam1[:,:]

def get_stat(new_d,whstat='stddev'): # default is stddev
    diff_flat = np.ndarray.flatten(new_d)
    if whstat=='stddev':
        stat = np.nanstd(diff_flat)
    elif whstat=='median':
        # compute median
        stat = np.abs(np.nanmedian(diff_flat))
    elif whstat=='sum':
        stat = np.nanmean(np.abs(diff_flat))
    return stat


def run_one_pair(fly1,fly2):
   pol, pols, cpols, attn, fi = get_flightinfo(fly1)
   print(fly1,pol,pols,cpols, attn)
   chind = pols[0]

   concytest=[glob.glob(pckldir+'*'+fly1+'*')[0] for x in flights]
   print(concytest[fi])

   with open(glob.glob(pckldir+'*'+fly1+'*')[0], 'rb') as pfile:
      print(pfile)
      concattest1=pickle.load(pfile)
   t_cut=concattest1.inds_on    

   beam1=bp.Beammap(concatlist=pcklarr[[fi]],gfitlist=gfitarr[[fi]],
                 normalization='Gauss',operation='coadd',Xargs=[-1*sz,sz,2.5],
                 Yargs=[-1*sz,sz,2.5],Fargs=[fmin,fmax,fstep],f_index=find,vplot=False)

   pol, pols, cpols, attn, fi = get_flightinfo(fly2)
   print(fly2,pol,pols,cpols, attn)

   concytest=[glob.glob(pckldir+'*'+fly2+'*')[0] for x in flights]
   print(concytest[fi])

   with open(glob.glob(pckldir+'*'+fly2+'*')[0], 'rb') as pfile:
      print(pfile)
      concattest2=pickle.load(pfile)
   t_cut=concattest2.inds_on    

   beam2=bp.Beammap(concatlist=pcklarr[[fi]],gfitlist=gfitarr[[fi]],
                 normalization='Gauss',operation='coadd',Xargs=[-1*sz,sz,2.5],
                 Yargs=[-1*sz,sz,2.5],Fargs=[fmin,fmax,fstep],f_index=find,vplot=False)

   ## Create masks:

   # Mask inner region
   thingyx = np.ma.masked_inside(beam1.x_centers_grid[:,:,chind], low, high, copy=True)
   thingyy = np.ma.masked_inside(beam1.y_centers_grid[:,:,chind], low, high, copy=True)
   inner_mask = np.logical_and(thingyx.mask,thingyy.mask)

   # Mask outer region
   outer_mask = np.logical_not(inner_mask)

   # Mask annulus
   thingyx = np.ma.masked_inside(beam1.x_centers_grid[:,:,chind], -1*maskin, maskin, copy=True)
   thingyy = np.ma.masked_inside(beam1.y_centers_grid[:,:,chind], -1*maskin, maskin, copy=True)
   inner_mask = np.logical_and(thingyx.mask,thingyy.mask)

   thingyx = np.ma.masked_inside(beam1.x_centers_grid[:,:,chind], -1*maskout, maskout, copy=True)
   thingyy = np.ma.masked_inside(beam1.y_centers_grid[:,:,chind], -1*maskout, maskout, copy=True)
   thingy = np.logical_and(thingyx.mask,thingyy.mask)
   outer_mask = np.logical_not(thingy)

   full_mask = np.logical_or(inner_mask,outer_mask)

   tots = np.zeros([4,1024,beam1.n_channels])

   for freq in np.arange(0,len(beam1.faxis)):
      for chind in pols:
         statarr = np.zeros([len(nss),3])
         for i in np.arange(0,len(nss)):
            ns = nss[i]
            diffn = get_beam_diff(beam2.V_LC_mean[:,:,freq,chind,0], beam1.V_LC_mean[:,:,freq,chind,0],ns)
            new_d = np.ma.masked_where(full_mask, diffn).filled(np.nan)
            statarr[i,0] = get_stat(new_d,'median')
            statarr[i,1] = get_stat(new_d,'stddev')
            statarr[i,2] = get_stat(new_d,'sum')

         mm = np.argmin(statarr[:,0])
         tots[0,beam1.faxis[freq],chind] = nss[mm]  # mimimum median
         tots[1,beam1.faxis[freq],chind] = statarr[mm,0] # median value at min
         mm = np.argmin(statarr[:,1])
         tots[2,beam1.faxis[freq],chind] = nss[mm] # stddev
         mm = np.argmin(statarr[:,2])
         tots[3,beam1.faxis[freq],chind] = nss[mm] # sum
         

   pklfile = 'Flight_'+str(fly2)+'-Flight'+str(fly1)+'_freqind_'+str(fmin)+'_norms.pkl'
   with open(pklfile, 'wb') as outp:
      pickle.dump(tots, outp, pickle.HIGHEST_PROTOCOL)
   print('DONE')


####################################################################
################ EXECUTE THE ENORMOUS FUNCTION #####################
####################################################################

print(fly_input)
print(high,low,maskin,maskout)
run_one_pair(str(fly_input),str(fly_control))

