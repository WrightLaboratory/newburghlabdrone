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
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

## Class for position-space sorting/averaging of beammap flight data, from time domain data:
#defines coordinate vector from xmax to xmin spaced roughly by xres:
def cedges(args):
    cmin,cmax,cres=args
    return np.linspace(cmin,cmax,int((cmax-cmin)/cres)+1)


class Beammap_forautoprocessing:
    
    ## Initialize:
    def __init__(self,concatlist=[],gfit=True,ampcorr=True,Xargs=[-100,100,5],Yargs=[-100,100,5],\
                 Fargs=[0,1024,512],\
                 operation='coadd',normalization='none',mask=False,\
                 pickle_directory='/hirax/GBO_Analysis_Outputs/flight_pickles/',\
                 gfit_directory='/hirax/GBO_Analysis_Outputs/main_beam_fits/',\
                 flightmasterpath='/hirax/GBO_Analysis_Outputs/GBO_flights_forscripts.yaml',\
                 ampcorr_directory='/hirax/GBO_Analysis_Outputs/amplitude_corrections/'):
        ## enable format of input to be load from 'pickle' filestring or bin/map using concat 'class':        
        self.concat_list=concatlist
        nchanslist=np.zeros(len(concatlist))
        for h,cstring in enumerate(concatlist):
            ## using the with loop structure, the pickle file is closed after ccc is loaded:
            with open(cstring, "rb") as f:
                CONCATCLASS=pickle.load(f)
            nchanslist[h]=CONCATCLASS.n_channels
        
        ## determine which channel is co-pol, for frequency dependent centroid corrections:
        with open(flightmasterpath, 'r') as flightmaster:
            doccs = yaml.safe_load(flightmaster)

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
        self.fmin,self.fmax,self.fstep=Fargs
        self.faxis=np.arange(self.fmin,self.fmax,self.fstep)
        self.freq=CONCATCLASS.freq[self.faxis]
        find=0
        self.n_freqs = len(self.faxis)
        self.normalization = normalization
        self.gfit_directory = gfit_directory
        self.ampcorr_directory = ampcorr_directory
        
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
            xoff=CONCATCLASS.dish_coords[i][0]
            yoff=CONCATCLASS.dish_coords[i][1]
            self.x_edges[:,i]=xedges
            self.x_centers[:,i]=xcenters
            self.y_edges[:,i]=yedges
            self.y_centers[:,i]=ycenters
            self.x_centers_grid[:,:,i]=xcentersgrid
            self.y_centers_grid[:,:,i]=ycentersgrid
            self.x_edges_grid[:,:,i]=xedgesgrid
            self.y_edges_grid[:,:,i]=yedgesgrid
            
        ## now need frequency dependent offset terms in shape (freq, channel, concat) to mimic V
        self.x_offsets=np.NAN*np.ones((self.n_freqs,self.n_channels,self.n_concats))
        self.y_offsets=np.NAN*np.ones((self.n_freqs,self.n_channels,self.n_concats))
        
        ## create arrays for V mean, V std, and histo: shape is (gridx, gridy, freq, chans, concatlist)
        self.V_LC_sum=np.NAN*np.ones((len(self.x_centers[:,0]),len(self.y_centers[:,0]),
                                      self.n_freqs,self.n_channels,self.n_concats))
        self.V_LC_count=np.NAN*np.ones((len(self.x_centers[:,0]),len(self.y_centers[:,0]),
                                        self.n_freqs,self.n_channels,self.n_concats))

        ## define inner and outer masks:
        self.maskin = 18
        self.maskout = 40           
        
        print("start of big ass loop is: {}".format(datetime.datetime.now()))

        for h,cstring in enumerate(concatlist):
            self.FLYNUM=cstring.split('FLY')[1].split('_')[0]
            
            Vvals, fccoords = self.get_VVals_tempcoords(cstring,h,doccs)

            if mask==True: # get mask type for this flight
                arrmask = get_mask(self,doccs,self.FLYNUM)
                
            ## loop through channels (i,chan) to find indices of nonzero cells in histogram
            for i,chan in enumerate(range(self.n_channels)):
                for j,fr in enumerate(self.faxis):
                    print('Concat:{}/{}, Channel:{}/{}, Frequency:{}/{},           '.format(h+1,self.n_concats,i+1,self.n_channels,j+1,len(self.freq)),end='\r')
                    xf,yf=fccoords[i,:,0,j],fccoords[i,:,1,j]
                    valsf=Vvals[:,j,i]                        
                    #histo2d,xbins,ybins=np.histogram2d(x,y,bins=[self.x_edges[:,i],self.y_edges[:,i]])
                    self.V_LC_count[:,:,j,i,h]=binned_statistic_2d(x=xf,y=yf,values=valsf,
                                                                statistic='count',
                                                                bins=[self.x_edges[:,i],self.y_edges[:,i]]).statistic
                    self.V_LC_sum[:,:,j,i,h]=binned_statistic_2d(x=xf,y=yf,values=valsf,
                                                                 statistic='sum',
                                                                 bins=[self.x_edges[:,i],self.y_edges[:,i]]).statistic       

                    if mask==True:
                        self.V_LC_count[:,:,j,i,h] = np.ma.masked_where(arrmask, 
                                                                self.V_LC_count[:,:,j,i,h]).filled(np.nan) 
                        self.V_LC_sum[:,:,j,i,h] = np.ma.masked_where(arrmask, 
                                                                self.V_LC_sum[:,:,j,i,h]).filled(np.nan) 
                                  
                
        print("end of bigass loop is: {}".format(datetime.datetime.now()))

        # set 0 values to nans for all
        self.V_LC_count[self.V_LC_count == 0] = 'nan'
        self.V_LC_sum[self.V_LC_sum == 0] = 'nan'
        
        
        if operation=='coadd' or operation=='std': 
            self.V_LC_operation=np.NAN*np.ones(self.V_LC_sum[:,:,:,:,0].shape)
            self.V_LC_operation_err=np.NAN*np.ones(self.V_LC_sum[:,:,:,:,0].shape)
            self.V_LC_operation_count=np.NAN*np.ones(self.V_LC_sum[:,:,:,:,0].shape)
            
            # average all bins together
            self.V_LC_operation_count = np.nansum(self.V_LC_count,axis=4)
            self.V_LC_operation = np.nansum(self.V_LC_sum,axis=4)/self.V_LC_operation_count
                        
        if operation=='std':
            # unfortunately, repeat the above.... assume you're only doing this with pickle files, normalized
            self.V_LC_err=np.NAN*np.ones((len(self.x_centers[:,0]),len(self.y_centers[:,0]),
                                        self.n_freqs,self.n_channels,self.n_concats))
            
            for h,cstring in enumerate(concatlist):
                self.FLYNUM=cstring.split('FLY')[1].split('_')[0]

                ## using the with loop structure, the pickle file is closed after ccc is loaded:
                Vvals, fccoords = self.get_VVals_tempcoords(cstring,h,doccs)
            
                if mask==True: # get mask type for this flight
                    arrmask = get_mask(self,doccs,self.FLYNUM)
            
                ## loop through channels (i,chan) to find indices of nonzero cells in histogram
                for i,chan in enumerate(range(self.n_channels)):
                    for j,fr in enumerate(self.faxis):
                        try:
                        
                            ## Unravel the mean value into the right bins, per file, and subtract:
                            xf,yf=fccoords[i,:,0,j],fccoords[i,:,1,j]
                            valsf=Vvals[:,j,i]                        
                            #histo2d,xbins,ybins=np.histogram2d(x,y,bins=[self.x_edges[:,i],self.y_edges[:,i]])
                            binning=binned_statistic_2d(x=xf,y=yf,values=valsf,statistic='count',
                                                    bins=[self.x_edges[:,i],self.y_edges[:,i]],
                                                    expand_binnumbers=True).binnumber

                            badbin = len(self.x_edges[:,i])
                            gbins = np.where((binning[0,:]<badbin) & (binning[1,:]<badbin))[0]
                            subtracteddata = (valsf[gbins]-self.V_LC_operation[binning[0,gbins]-1,binning[1,gbins]-1,j,i])**2
                            self.V_LC_err[:,:,j,i,h] = binned_statistic_2d(x=xf[gbins],y=yf[gbins],
                                                                    values=subtracteddata,
                                                                    statistic='sum',
                                                                    bins=[self.x_edges[:,i],
                                                                          self.y_edges[:,i]]).statistic
  
                            if mask==True:
                                self.V_LC_err[:,:,j,i,h] = np.ma.masked_where(arrmask, self.V_LC_err[:,:,j,i,h]).filled(np.nan) 
        
                        except:''
            
            self.V_LC_operation_err = (np.nansum(self.V_LC_err,axis=4)/self.V_LC_operation_count)**0.5 # stddev

    def get_mask(self,doccs,flynum):

        for j,fstr in enumerate(doccs['flight_info']['flights']): # what mask
            if flynum in fstr: maskit = doccs['flight_info']['masks'][j]
        if maskit=='inner':
            thingyx = np.ma.masked_inside(self.x_centers_grid[:,:,0], -1*self.maskin, self.maskin, copy=True)
            thingyy = np.ma.masked_inside(self.y_centers_grid[:,:,0], -1*self.maskin, self.maskin, copy=True)
            arrmask = np.logical_and(thingyx.mask,thingyy.mask)
        elif maskit=='outer':
            thingyx = np.ma.masked_inside(self.x_centers_grid[:,:,0], -1*self.maskout, self.maskout, copy=True)
            thingyy = np.ma.masked_inside(self.y_centers_grid[:,:,0], -1*self.maskout, self.maskout, copy=True)
            thingy = np.logical_and(thingyx.mask,thingyy.mask)
            arrmask = np.logical_not(thingy)
        return arrmask
            
    def get_VVals_tempcoords(self,cstring,h,doccs):
            
    def get_VVals_tempcoords(self,cstring,h,doccs):
                
            for j,fstr in enumerate(doccs['flight_info']['flights']):
                if self.FLYNUM in fstr:
                    self.copoldir=doccs['flight_info']['pols'][j]
                    
            ## using the with loop structure, the pickle file is closed after ccc is loaded:
            with open(cstring, "rb") as f:
                ccc=pickle.load(f)   
            t_cut=ccc.inds_on
            if self.normalization=='none':
                Vvals=ccc.V_bgsub[ccc.inds_on,:,:]
            if self.normalization=='Gauss' or self.normalization=='Gauss_wcorr':
                # get the normalization:
                gfit=glob.glob(self.gfit_directory+'*'+self.FLYNUM+'*.npz')[0]
                print(self.FLYNUM,gfit)
                gff = np.load(gfit)
                g_norm=gff['G_popt'][:,:,0]
                Vvals=(np.repeat(
                    np.swapaxes(g_norm[:,self.fmin:self.fmax:self.fstep],0,1)[np.newaxis,:,:],len(ccc.inds_on),axis=0)**-1)*ccc.V_bgsub[ccc.inds_on,self.fmin:self.fmax:self.fstep,:]       
                
                for i in range(self.n_channels):
                    if self.copoldir in 'E':
                        COPOLIND=np.arange(self.n_channels).reshape(int(self.n_channels/2),2)[int(i/2)][0]
                        self.x_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,1]-ccc.dish_coords[i][0] # seems like gauss params were found with original xyz, so remove dish offset
                        self.y_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,3]-ccc.dish_coords[i][1] # seems like gauss params were found with original xyz, so remove dish offset
                    elif self.copoldir in 'N':
                        COPOLIND=np.arange(self.n_channels).reshape(int(self.n_channels/2),2)[int(i/2)][1]
                        self.x_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,1]-ccc.dish_coords[i][0] # seems like gauss params were found with original xyz, so remove dish offset
                        self.y_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,3]-ccc.dish_coords[i][1] # seems like gauss params were found with original xyz, so remove dish offset
                                
                                                
            if self.normalization=='Gauss_wcorr':
                # get normalization
                gcorrfile=glob.glob(self.ampcorr_directory+'*'+flynum+'*.pkl')[0]
                with open(gcorrfile,'rb') as acf:
                    gcorr_norm=pickle.load(acf)
                Vvals *= gcorr_norm[self.faxis,:][np.newaxis,:,:]
                            
            ## create centroid-corrected per channel and frequency drone coordinate maps on a per-concat basis:
            tmpcoords=np.repeat(ccc.drone_xyz_per_dish_interp[:,:,:,np.newaxis],self.n_freqs,axis=3)
            shiftvec=np.array((self.x_offsets[:,:,h],self.y_offsets[:,:,h],np.zeros(self.x_offsets[:,:,h].shape)))
            tmpshifts=np.repeat(np.swapaxes(np.swapaxes(shiftvec,0,2),1,2)[:,np.newaxis,:,:],len(ccc.t),axis=1)
            fccoords=(tmpcoords-tmpshifts)[:,ccc.inds_on]
                
            return Vvals,fccoords 
                
            for j,fstr in enumerate(doccs['flight_info']['flights']):
                if self.FLYNUM in fstr:
                    self.copoldir=doccs['flight_info']['pols'][j]
                    
            ## using the with loop structure, the pickle file is closed after ccc is loaded:
            with open(cstring, "rb") as f:
                ccc=pickle.load(f)   
            t_cut=ccc.inds_on
            if self.normalization=='none':
                Vvals=ccc.V_bgsub[ccc.inds_on,:,:]
            if self.normalization=='Gauss' or self.normalization=='Gauss_wcorr':
                # get the normalization:
                gfit=glob.glob(self.gfit_directory+'*'+self.FLYNUM+'*.npz')[0]
                print(self.FLYNUM,gfit)
                gff = np.load(gfit)
                g_norm=gff['G_popt'][:,:,0]
                Vvals=(np.repeat(
                    np.swapaxes(g_norm[:,self.fmin:self.fmax:self.fstep],0,1)[np.newaxis,:,:],len(ccc.inds_on),axis=0)**-1)*ccc.V_bgsub[ccc.inds_on,self.fmin:self.fmax:self.fstep,:]       
                
                for i in range(self.n_channels):
                    if self.copoldir in 'E':
                        COPOLIND=np.arange(self.n_channels).reshape(int(self.n_channels/2),2)[int(i/2)][0]
                        self.x_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,1]-ccc.dish_coords[i][0] # seems like gauss params were found with original xyz, so remove dish offset
                        self.y_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,3]-ccc.dish_coords[i][1] # seems like gauss params were found with original xyz, so remove dish offset
                    elif self.copoldir in 'N':
                        COPOLIND=np.arange(self.n_channels).reshape(int(self.n_channels/2),2)[int(i/2)][1]
                        self.x_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,1]-ccc.dish_coords[i][0] # seems like gauss params were found with original xyz, so remove dish offset
                        self.y_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,3]-ccc.dish_coords[i][1] # seems like gauss params were found with original xyz, so remove dish offset
                                
                                                
            if self.normalization=='Gauss_wcorr':
                # get normalization
                gcorrfile=glob.glob(self.ampcorr_directory+'*'+flynum+'*.pkl')[0]
                with open(gcorrfile,'rb') as acf:
                    gcorr_norm=pickle.load(acf)
                Vvals *= gcorr_norm[self.faxis,:][np.newaxis,:,:]
                            
            ## create centroid-corrected per channel and frequency drone coordinate maps on a per-concat basis:
            tmpcoords=np.repeat(ccc.drone_xyz_per_dish_interp[:,:,:,np.newaxis],self.n_freqs,axis=3)
            shiftvec=np.array((self.x_offsets[:,:,h],self.y_offsets[:,:,h],np.zeros(self.x_offsets[:,:,h].shape)))
            tmpshifts=np.repeat(np.swapaxes(np.swapaxes(shiftvec,0,2),1,2)[:,np.newaxis,:,:],len(ccc.t),axis=1)
            fccoords=(tmpcoords-tmpshifts)[:,ccc.inds_on]
                
            return Vvals,fccoords 
