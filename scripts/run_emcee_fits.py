## First import general packages for running python analysis:
import os, h5py, datetime,pytz, yaml
import numpy as np
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import pickle, glob

from scipy.interpolate import griddata
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from matplotlib import cm

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

import csv
from math import *

## Then import the beamcals module packages and initialize 'gbosite' class:
from beamcals import corr
from beamcals import concat
from beamcals import drone
from beamcals import bicolog
import beamcals.plotting_utils as pu
import beamcals.fitting_utils as fu
import beamcals.geometry_utils as gu
import beamcals.time_utils as tu
from beamcals.sites import site
gbosite=site.site('../beamcals/beamcals/sites/GBO_config.npz')
import emcee

# Defining anything I want to keep the same 

freqs = 800.0*np.ones(1024) + (-400/1024.)*np.arange(1024)


#basedir = '/hirax/GBO_Analysis_Outputs/2023_SpringSummer_products/'
basedir = '/hirax/GBO_Analysis_Outputs/'
fitdir=basedir+'main_beam_fits/'
ymldir = basedir+'concat_config_files/'
pckldir = basedir+'flight_pickles/'
ampdir = basedir+'amplitude_corrections/'
beamdir = basedir+'beam_pickles/'

dronedir='/hirax/all_drone_data/datcon_csv/'
fltyaml = '/hirax/GBO_Analysis_Outputs/GBO_flights_forscripts.yaml'

## Useful functions
def get_flightinfo(fly):
    with open('/hirax/GBO_Analysis_Outputs/GBO_flights_forscripts.yaml', 'r') as fff:
        documents = yaml.safe_load(fff)
    flights = documents["flight_info"]["flights"]
    
    for fi in range(0,len(documents["flight_info"]["flights"])):
        if fly == flights[fi]: f = fi
    
    if polarn[f]=='N':
        pols = N_pols
        cpols = E_pols
    else:
        pols = E_pols
        cpols = N_pols
    return polarn[f], pols, cpols, attns[f], f


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

Npolflights = ['618','619','625','646','647','533','536']
Epolflights = ['620','648','649','535']
NF = len(Npolflights)
NE = len(Epolflights)
F = NF+NE
fi_i =650

colormap='viridis'


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
    niter = 500
    ndim = len(pG)
    p0 = [np.array(pG) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)
    samples = sampler.flatchain
    fits = np.zeros([ndim,2])
    for i in np.arange(ndim):
        fits[i,0] = samples[np.argmax(sampler.flatlnprobability),i]
        fits[i,1] = np.std(samples[:,i])
    return fits


mlfits = np.zeros([16,1024,6,2])

for f,find in enumerate(good_freqs[0:1]):
    print('Start time, freq', find, time.time())
    copoldir = 'N'
    pols=N_pols
    picklefile = beamdir+'Beamcoadd_pol_'+copoldir+'_freq_'+str(find)+'.pkl'
    with open(picklefile, 'rb') as inp:
        ptest = pickle.load(inp)

    for j, chind in enumerate(pols): 
        mbx = ptest.x_centers_grid[:,:,chind].flatten()
        mby = ptest.y_centers_grid[:,:,chind].flatten()
        mbV = ptest.V_LC_operation[:,:,0,chind].flatten()
        mbVerr = ptest.V_LC_operation_err[:,:,0,chind].flatten()
        gi = np.where((np.isfinite(mbV) & (np.abs(mbx) < 25) & (np.abs(mby) < 25) & (np.isfinite(mbV))))[0]
        mlfits[chind,find,:,:] = get_ml_solns(mbx,mby,mbV,mbVerr,gi)
        
    copoldir = 'E'
    pols=E_pols
    picklefile = beamdir+'Beamcoadd_pol_'+copoldir+'_freq_'+str(find)+'.pkl'
    with open(picklefile, 'rb') as inp:
        ptest = pickle.load(inp)

    for j, chind in enumerate(pols): 
        mbx = ptest.x_centers_grid[:,:,chind].flatten()
        mby = ptest.y_centers_grid[:,:,chind].flatten()
        mbV = ptest.V_LC_operation[:,:,0,chind].flatten()
        mbVerr = ptest.V_LC_operation_err[:,:,0,chind].flatten()
        gi = np.where((np.isfinite(mbV) & (np.abs(mbx) < 25) & (np.abs(mby) < 25) & (np.isfinite(mbV))))[0]
        mlfits[chind,find,:,:] = get_ml_solns(mbx,mby,mbV,mbVerr,gi)
    print('End time', time.time())        

tmpfitpath='/hirax/GBO_Analysis_Outputs/Coadded_2dGauss_With_Errors.npz'
np.savez(tmpfitpath,coadded=mlfits)
