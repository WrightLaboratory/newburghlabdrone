#  _                                                             
# | |                                                            
# | |__   ___  __ _ _ __ ___  _ __ ___   __ _ _ __   _ __  _   _ 
# | '_ \ / _ \/ _` | '_ ` _ \| '_ ` _ \ / _` | '_ \ | '_ \| | | |
# | |_) |  __/ (_| | | | | | | | | | | | (_| | |_) || |_) | |_| |
# |_.__/ \___|\__,_|_| |_| |_|_| |_| |_|\__,_| .__(_) .__/ \__, |
#                                            | |    | |     __/ |
#                                            |_|    |_|    |___/ 
        
## 20230411_WT: introducing class and adding it to the package....

## Class for position-space sorting/averaging of beammap flight data, from time domain data:

from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import h5py
import hdf5plugin
import os
import glob
from matplotlib import colors
import pandas
import csv
import datetime
import pytz
import bisect
import pygeodesy
import yaml
from scipy.signal import square
from scipy.stats import pearsonr
import glob
import pickle

#defines coordinate vector from xmax to xmin spaced roughly by xres:
def cedges(args):
    cmin,cmax,cres=args
    return np.linspace(cmin,cmax,int((cmax-cmin)/cres)+1)

class Beammap:
    def __init__(self,concatlist=[],operation='coadd',inputstyle='pickle',Xargs=[-100,100,5],Yargs=[-100,100,5]):
        ## enable format of input to be load from 'pickle' filestring or bin/map using concat 'class':        
        nchanslist=np.zeros(len(concatlist))
        if inputstyle=='pickle':
            for h,cstring in enumerate(concatlist):
                ## using the with loop structure, the pickle file is closed after ccc is loaded:
                with open(cstring, "rb") as f:
                    CONCATCLASS=pickle.load(f)
                nchanslist[h]=CONCATCLASS.n_channels
        elif inputstyle=='class':
            CONCATCLASS=concatlist[0]
            for h,cstring in enumerate(concatlist):
                nchanslist[h]=cstring.n_channels
        
        #get variables that should be kept along from the first concat class:         
        self.name=CONCATCLASS.name
        self.Data_Directory=CONCATCLASS.Data_Directory
        self.Gain_Directory=CONCATCLASS.Gain_Directory
        self.filenames=CONCATCLASS.filenames
        self.gainfile=CONCATCLASS.gainfile
        self.Drone_Directory=CONCATCLASS.Drone_Directory
        self.FLYTAG=CONCATCLASS.FLYTAG
        self.n_dishes=CONCATCLASS.n_dishes
        self.n_channels=int(np.nanmin(nchanslist))
        self.n_concats=len(concatlist)
        self.chmap=CONCATCLASS.chmap
        self.automap=CONCATCLASS.automap
        self.crossmap=CONCATCLASS.crossmap
        self.origin=CONCATCLASS.origin
        #self.prime_origin=CONCATCLASS.prime_origin ## omit, it breaks the pickle
        self.dish_keystrings=CONCATCLASS.dish_keystrings
        self.dish_coords=CONCATCLASS.dish_coords
        self.dish_pointings=CONCATCLASS.dish_pointings
        self.dish_polarizations=CONCATCLASS.dish_polarizations
        self.freq=CONCATCLASS.freq
        
        #create x,y cartesian vectors (edges and centers) and grids for the beammap:
        self.operation=operation
        xedges,yedges=(cedges(Xargs),cedges(Yargs))
        xedgesgrid,yedgesgrid=np.meshgrid(xedges,yedges)
        xcenters,ycenters=(cedges(Xargs)[:-1]+Xargs[2]/2.0,cedges(Yargs)[:-1]+Yargs[2]/2.0)
        xcentersgrid,ycentersgrid=np.meshgrid(xcenters,ycenters)
        
        #need to extend this to dimensionality of channels in concatclass.V
        self.x_edges=np.zeros(xedges.shape+(self.n_channels,))
        self.y_edges=np.zeros(yedges.shape+(self.n_channels,))
        self.x_edges_grid=np.zeros((xedgesgrid.shape+(self.n_channels,)))
        self.y_edges_grid=np.zeros((yedgesgrid.shape+(self.n_channels,)))
        self.x_centers=np.zeros(xcenters.shape+(self.n_channels,))
        self.y_centers=np.zeros(ycenters.shape+(self.n_channels,))
        self.x_centers_grid=np.zeros((xcentersgrid.shape+(self.n_channels,)))
        self.y_centers_grid=np.zeros((ycentersgrid.shape+(self.n_channels,)))
        for i in range(self.n_channels):
            if 'D3A' in CONCATCLASS.name:
                xoff=CONCATCLASS.dish_coords[i][0]
                yoff=CONCATCLASS.dish_coords[i][1]
            elif 'GBO' in CONCATCLASS.name:
                xoff=CONCATCLASS.dish_coords[i][0]
                yoff=CONCATCLASS.dish_coords[i][1]
            self.x_edges[:,i]=xedges+xoff
            self.x_centers[:,i]=xcenters+xoff
            self.y_edges[:,i]=yedges+yoff
            self.y_centers[:,i]=ycenters+yoff
            self.x_centers_grid[:,:,i]=xcentersgrid+xoff
            self.y_centers_grid[:,:,i]=ycentersgrid+yoff
            self.x_edges_grid[:,:,i]=xedgesgrid+xoff
            self.y_edges_grid[:,:,i]=yedgesgrid+yoff          
        
        ## create arrays for V mean, V std, and histo: shape is (gridx, gridy, freq, chans, concatlist)
        self.V_LC_mean=np.NAN*np.ones((len(self.x_centers[:,0]),len(self.y_centers[:,0]),len(self.freq),self.n_channels,self.n_concats))
        self.V_LC_std=np.NAN*np.ones((len(self.x_centers[:,0]),len(self.y_centers[:,0]),len(self.freq),self.n_channels,self.n_concats))
        self.histogram_LC=np.NAN*np.ones((len(self.x_centers[:,0]),len(self.y_centers[:,0]),self.n_channels,self.n_concats))
        ## loop through the concat classes (h,ccc=concatclass) and extract hist/V parameters:
        for h,cstring in enumerate(concatlist):
            ## using the with loop structure, the pickle file is closed after ccc is loaded:
            if inputstyle=='pickle':
                with open(cstring, "rb") as f:
                    ccc=pickle.load(f)   
            elif inputstyle=='class':
                ccc=cstring
            ## loop through channels (i,chan) to find indices of nonzero cells in histogram
            for i,chan in enumerate(range(self.n_channels)):
                x,y=[ccc.drone_xyz_per_dish_interp[i,:,0],ccc.drone_xyz_per_dish_interp[i,:,1]]
                histo2d,xbins,ybins=np.histogram2d(x,y,bins=[self.x_edges[:,i],self.y_edges[:,i]])
                self.histogram_LC[:,:,i,h]=histo2d
                xnz,ynz=np.where(histo2d!=0)
                ## loop over the bins [m] where the histogram is nonzero to populate the bins of the V_arrays:
                for m in range(len(xnz)):
                    j,k=[xnz[m],ynz[m]]
                    ## find times [tacc] where the drones coordinates x,y are with in the bin m x,y values:
                    xacc=np.intersect1d(np.where(self.x_edges[j,i]<=x),np.where(self.x_edges[j+1,i]>x))
                    yacc=np.intersect1d(np.where(self.y_edges[k,i]<=y),np.where(self.y_edges[k+1,i]>y))
                    tacc0=np.intersect1d(xacc,yacc)
                    tacc=np.intersect1d(tacc0,ccc.inds_on)
                    ## Populate the V arrays with the appropriate values:
                    self.V_LC_mean[j,k,:,i,h]=np.nanmean(ccc.V_bgsub[tacc,:,i],axis=0)
                    self.V_LC_std[j,k,:,i,h]=np.nanstd(ccc.V_bgsub[tacc,:,i],axis=0)
        if operation=='coadd':
            self.V_LC_operation=np.NAN*np.ones(self.V_LC_mean[:,:,:,:,0].shape)
            self.V_LC_operation=np.nanmean(self.V_LC_mean,axis=4)
        elif operation=='difference':
            if len(concatlist)==2:
                self.V_LC_operation=np.NAN*np.ones(self.V_LC_mean[:,:,:,:,0].shape)
                self.V_LC_operation=np.nansum(np.array([self.V_LC_mean[:,:,:,:,0],-1*self.V_LC_mean[:,:,:,:,1]]),axis=0)
            else:
                print("--> V_LC_operation can only be instantiated if the length of concatlist is 2")