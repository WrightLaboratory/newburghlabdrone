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
from scipy.stats import binned_statistic_2d

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
import beamcals.beam_autoprocessing as ba
from beamcals.sites import site
import beamcals.reduce_ccclass as rc

gbosite=site.site('../beamcals/beamcals/sites/GBO_config.npz')

# various gridding attempts
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import emcee

freqs = 800.0*np.ones(1024) + (-400/1024.)*np.arange(1024)

fitdir='/hirax/GBO_Analysis_Outputs/main_beam_fits/'
ymldir = '/hirax/GBO_Analysis_Outputs/concat_config_files/'
pckldir = '/hirax/GBO_Analysis_Outputs/flight_pickles/'
ampdir = '/hirax/GBO_Analysis_Outputs/amplitude_corrections/'
beamdir = '/hirax/GBO_Analysis_Outputs/beam_pickles/'
polbeamdir = '/hirax/GBO_Analysis_Outputs/beam_pickles_polar/'

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

def model(P,x,y):
    amp,x0,xsig,y0,ysig,c=P
    #xsig,ysig = P
    xx = ((x-x0)**2)/(2*(xsig**2))
    yy = ((y-y0)**2)/(2*(ysig**2))
    return amp*np.exp(-1.0*(xx + yy))+c

def lnlike(P,x,y,V,Verr):
    LnLike = -0.5*np.nansum(((model(P,x,y)-V)/Verr)**2)
    return LnLike

def lnprior(P):
    amp,x0,xsig,y0,ysig,c=P
    if x0>-1 and x0<1 and y0>-1 and y0<1 and c<0.1 and xsig>5.0 and xsig<12.0 and ysig>5.0 and ysig<12.0 and amp>-1.5 and amp<1.5:
        return 0.0
    else:
        return -np.inf

def lnprob(P,x,y,V,Verr):
    lp = lnprior(P)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(P,x,y,V,Verr) #recall if lp not -inf, its 0, so this just returns likelihood



def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state


def get_ml_solns(mbx,mby,mbV,mbVerr,gi):
    pG=np.array([1.0,0.0,8.0,0.0,8.0,1E-8])
    nwalkers = 500
    data = (mbx[gi],mby[gi],mbV[gi],mbVerr[gi])
    niter = 3000
    ndim = len(pG)
    p0 = [np.array(pG) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)
    samples = sampler.flatchain
    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except: 'TAU DIDNT WORK'

    fits = np.zeros([ndim,4])
    for i in np.arange(ndim):
        results = np.percentile(samples[:, i], [16, 50, 84])
        fits[i,0] = results[1] # 50% error bars
        fits[i,1] = np.std(samples[:,i]) # use the stdd of the dist.
        fits[i,2] = np.diff(results)[0]
        fits[i,3] = np.diff(results)[1]
    return fits

with open('/hirax/GBO_Analysis_Outputs/GBO_flights_forscripts.yaml', 'r') as fff:
    documents = yaml.safe_load(fff)
flights = documents["flight_info"]["flights"]
N_pols = documents["flag_info"]["N_pols"]
E_pols = documents["flag_info"]["E_pols"]
polarn = documents["flight_info"]["pols"]
attns = documents["flight_info"]["attns"]
masks = documents["flight_info"]["masks"]
good_freqs = documents["freq_info"]["good_freqs"]

pcklarr=np.sort(os.listdir(pckldir))
gfitarr=np.sort(os.listdir(fitdir))
amparr=np.sort(os.listdir(ampdir))
print(gfitarr,amparr)


# after looking through the slices and being very picky
good_freqs =  [538, 553, 554,
       556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 568, 569,
       571, 572, 575, 576, 577, 578, 579, 580, 581, 584, 630, 631, 632,
       633, 636, 639, 645, 676, 691, 692, 695, 696, 697, 698, 699, 700,
       702, 703, 705, 706, 707, 719, 720, 768, 799, 801, 802, 803,
       805, 807, 808, 810, 811, 814, 845, 846, 847, 848, 849, 851, 852,
       853, 854, 855, 856, 857, 858, 860, 861, 862, 863, 864, 865, 866,
       867, 869, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881,
       883, 884, 885, 887, 888, 890, 891, 892, 895, 896, 899, 900, 902,
       903, 904, 905, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916,
       917, 918, 919, 921, 922, 923, 924, 925, 926, 928, 929, 930, 931,
       932, 933, 935, 936, 937, 938, 939]


pG=np.array([1.0,0.0,8.0,0.0,8.0,1E-6])

emcfits = np.zeros([1024,16,6,4]) # freq,channel,fits, value/error

# Npols first
copoldir = 'N'
pols = N_pols
for f,find in enumerate(good_freqs):

   picklefile = beamdir+'Beamcoadd_pol_'+copoldir+'_freq_'+str(find)+'.pkl'
   with open(picklefile, 'rb') as inp:
       ptest = pickle.load(inp)

   for c, chind in enumerate(pols):
       mbx = ptest.x_centers_grid[:,:,chind].flatten()
       mby = ptest.y_centers_grid[:,:,chind].flatten()
       mbV = ptest.V_LC_operation[:,:,0,chind].flatten()
       mbVerr = ptest.V_LC_operation_err[:,:,0,chind].flatten()
       gi = np.where((np.isfinite(mbV)) & (np.isfinite(mbVerr)) & 
                  (mbVerr!=0) & (np.abs(mbx)<20) & (np.abs(mby)<20))[0]
    
       emcfits[find,chind,:,:] = get_ml_solns(mbx,mby,mbV,mbVerr,gi)

## Epols next
copoldir = 'E'
pols = E_pols
for f,find in enumerate(good_freqs):

   picklefile = beamdir+'Beamcoadd_pol_'+copoldir+'_freq_'+str(find)+'.pkl'
   with open(picklefile, 'rb') as inp:
       ptest = pickle.load(inp)

   for c, chind in enumerate(pols):
       mbx = ptest.x_centers_grid[:,:,chind].flatten()
       mby = ptest.y_centers_grid[:,:,chind].flatten()
       mbV = ptest.V_LC_operation[:,:,0,chind].flatten()
       mbVerr = ptest.V_LC_operation_err[:,:,0,chind].flatten()
       gi = np.where((np.isfinite(mbV)) & (np.isfinite(mbVerr)) & 
                  (mbVerr!=0) & (np.abs(mbx)<20) & (np.abs(mby)<20))[0]
    
       emcfits[find,chind,:,:] = get_ml_solns(mbx,mby,mbV,mbVerr,gi)


outpkl = '/hirax/GBO_Analysis_Outputs/emcee_fits.pkl'
with open(outpkl, 'wb') as outp:
    pickle.dump(emcfits, outp, pickle.HIGHEST_PROTOCOL)
print('DONE')
