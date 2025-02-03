# practice the script

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

freqs = 800.0*np.ones(1024) + (-400/1024.)*np.arange(1024)

fitdir='/hirax/GBO_Analysis_Outputs/main_beam_fits/'
ymldir = '/hirax/GBO_Analysis_Outputs/concat_config_files/'
pckldir = '/hirax/GBO_Analysis_Outputs/flight_pickles/'
ampdir = '/hirax/GBO_Analysis_Outputs/amplitude_corrections/'
beamdir = '/hirax/GBO_Analysis_Outputs/beam_pickles/'

sz = 80


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

Npolflights = ['618','619','625','646','647','533']
Epolflights = ['620','648','649','535']

delt_the = np.radians(6)
delt_phi = np.radians(1)

for find in good_freqs:
    print('Working on Frequency: ',find,freqs[find])

    flights = Npolflights
    concytest=[glob.glob(pckldir+'*'+fly+'*')[0] for fly in flights]    

    #beam=ba.Beammap_forautoprocessing(concatlist=concytest,
    #             coordsys='cartesian',normalization='Gauss_wcorr',operation='std',mask=True,d0args=[-1*sz,sz,2.0],
    #             d1args=[-1*sz,sz,2.0],Fargs=[find,find+1,1])
    beam=ba.Beammap_polar(concatlist=concytest,
                coordsys='polar',
                d0args=[-delt_the/2.0,(60*delt_the)-(delt_the/2.0),delt_the],
                d1args=[0,np.radians(36),delt_phi],
                normalization='Gauss_wcorr',operation='std',Fargs=[find,find+1,1],
                f_index=find,vplot=False,mask=True)

    thingy = rc.Smallify_comap(beam)
    for j,fstr in enumerate(documents['flight_info']['flights']):
        if beam.FLYNUM in fstr:
            copoldir=documents['flight_info']['pols'][j]
    #write_pickle = beamdir+'Beamcoadd_pol_'+copoldir+'_freq_'+str(find)+'.pkl'
    write_pickle = beamdir+'Beamcoadd_pol_'+copoldir+'_freq_'+str(find)+'_polar.pkl'

    print(write_pickle)
    with open(write_pickle, 'wb') as outp:
        pickle.dump(thingy, outp, pickle.HIGHEST_PROTOCOL)
    print('DONE')

    
    flights = Epolflights
    concytest=[glob.glob(pckldir+'*'+fly+'*')[0] for fly in flights]    

    #beam=ba.Beammap_forautoprocessing(concatlist=concytest,
    #             coordsys='cartesian',normalization='Gauss_wcorr',operation='std',mask=True,d0args=[-1*sz,sz,2.0],
    #             d1args=[-1*sz,sz,2.0],Fargs=[find,find+1,1])
    beam=ba.Beammap_polar(concatlist=concytest,
                 coordsys='polar',
                 d0args=[-delt_the/2.0,(60*delt_the)-(delt_the/2.0),delt_the],
                 d1args=[0,np.radians(36),delt_phi],
                 normalization='Gauss_wcorr',operation='std',Fargs=[find,find+1,1],
                 f_index=find,vplot=False,mask=True)

    thingy = rc.Smallify_comap(beam)
    for j,fstr in enumerate(documents['flight_info']['flights']):
        if beam.FLYNUM in fstr:
            copoldir=documents['flight_info']['pols'][j]
    print(copoldir)
    #write_pickle = beamdir+'Beamcoadd_pol_'+copoldir+'_freq_'+str(find)+'.pkl'
    write_pickle = beamdir+'Beamcoadd_pol_'+copoldir+'_freq_'+str(find)+'_polar.pkl'

    print(write_pickle)
    with open(write_pickle, 'wb') as outp:
        pickle.dump(thingy, outp, pickle.HIGHEST_PROTOCOL)
    print('DONE')
print('For loop ended successfully')
