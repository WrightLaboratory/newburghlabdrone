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
from scipy.stats import binned_statistic_2d

## Class for position-space sorting/averaging of beammap flight data, from time domain data:
#defines coordinate vector from xmax to xmin spaced roughly by xres:
def cedges(args):
    cmin,cmax,cres=args
    return np.linspace(cmin,cmax,int((cmax-cmin)/cres)+1)

class Beammap:
    def __init__(self,concatlist=[],gfitlist=[],Xargs=[-100,100,5],Yargs=[-100,100,5],\
                 Fargs=[0,1024,1],f_index=900,\
                 operation='coadd',inputstyle='pickle',normalization='none',vplot=True,\
                 pickle_directory='/hirax/GBO_Analysis_Outputs/flight_pickles/',\
                 gfit_directory='/hirax/GBO_Analysis_Outputs/main_beam_fits/',\
                 flightmasterpath='../analysis/GBO_flights_forscripts.yaml'):
        ## enable format of input to be load from 'pickle' filestring or bin/map using concat 'class':        
        self.concat_list=concatlist
        self.gfit_list=gfitlist
        nchanslist=np.zeros(len(concatlist))
        if inputstyle=='pickle':
            for h,cstring in enumerate(concatlist):
                ## using the with loop structure, the pickle file is closed after ccc is loaded:
                with open(pickle_directory+cstring, "rb") as f:
                    CONCATCLASS=pickle.load(f)
                nchanslist[h]=CONCATCLASS.n_channels
                self.FLYNUM=cstring.split('FLY')[1].split('_')[0]
        elif inputstyle=='class':
            CONCATCLASS=concatlist[0]
            for h,cstring in enumerate(concatlist):
                nchanslist[h]=cstring.n_channels
                self.FLYNUM=cstring.FLYTAG.split('FLY')[1].split('.')[0]
        
        ## determine which channel is co-pol, for frequency dependent centroid corrections:
        with open(flightmasterpath, 'r') as flightmaster:
            doccs = yaml.safe_load(flightmaster)
            for j,fstr in enumerate(doccs['flight_info']['flights']):
                if self.FLYNUM in fstr:
                    self.copoldir=doccs['flight_info']['pols'][j]

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
        fmin,fmax,fstep=Fargs
        self.faxis=np.arange(fmin,fmax,fstep)
        self.freq=CONCATCLASS.freq[self.faxis]
        find=np.where(self.faxis==f_index)[0][0]
        
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
            self.x_edges[:,i]=xedges
            self.x_centers[:,i]=xcenters
            self.y_edges[:,i]=yedges
            self.y_centers[:,i]=ycenters
            self.x_centers_grid[:,:,i]=xcentersgrid
            self.y_centers_grid[:,:,i]=ycentersgrid
            self.x_edges_grid[:,:,i]=xedgesgrid
            self.y_edges_grid[:,:,i]=yedgesgrid
            
        ## now need frequency dependent offset terms in shape (freq, channel, concat) to mimic V
        self.x_offsets=np.NAN*np.ones((len(self.freq),self.n_channels,self.n_concats))
        self.y_offsets=np.NAN*np.ones((len(self.freq),self.n_channels,self.n_concats))
        
        ## create arrays for V mean, V std, and histo: shape is (gridx, gridy, freq, chans, concatlist)
        self.V_LC_mean=np.NAN*np.ones((len(self.x_centers[:,0]),len(self.y_centers[:,0]),len(self.freq),self.n_channels,self.n_concats))
        self.V_LC_std=np.NAN*np.ones((len(self.x_centers[:,0]),len(self.y_centers[:,0]),len(self.freq),self.n_channels,self.n_concats))
        self.histogram_LC=np.NAN*np.ones((len(self.x_centers[:,0]),len(self.y_centers[:,0]),len(self.freq),self.n_channels,self.n_concats))
        
        ## loop through the concat classes (h,ccc=concatclass) and extract hist/V parameters:
        if vplot==True:
            fig0,axes0=subplots(nrows=2,ncols=4,figsize=(40,20))
            
        print("start of big ass loop is: {}".format(datetime.datetime.now()))

        for h,cstring in enumerate(concatlist):
            ## create automatic verification plot axes:
            if vplot==True:
                fig,axes=subplots(nrows=2,ncols=4,figsize=(40,20))
            ## using the with loop structure, the pickle file is closed after ccc is loaded:
            if inputstyle=='pickle':
                with open(pickle_directory+cstring, "rb") as f:
                    ccc=pickle.load(f)   
            elif inputstyle=='class':
                ccc=cstring
            t_cut=ccc.inds_on
            if normalization=='none':          
                if vplot==True:
                    im1x=axes[0][0].scatter(ccc.drone_xyz_per_dish_interp[0,t_cut,0],ccc.drone_xyz_per_dish_interp[0,t_cut,1],c=ccc.V_bgsub[t_cut,f_index,0],cmap=cm.gnuplot2,norm=LogNorm())
                    im1y=axes[1][0].scatter(ccc.drone_xyz_per_dish_interp[1,t_cut,0],ccc.drone_xyz_per_dish_interp[1,t_cut,1],c=ccc.V_bgsub[t_cut,f_index,1],cmap=cm.gnuplot2,norm=LogNorm())    
                    im1x0=axes0[0][0].scatter(ccc.drone_xyz_per_dish_interp[0,t_cut,0],ccc.drone_xyz_per_dish_interp[0,t_cut,1],c=ccc.V_bgsub[t_cut,f_index,0],cmap=cm.gnuplot2,norm=LogNorm())
                    im1y0=axes0[1][0].scatter(ccc.drone_xyz_per_dish_interp[1,t_cut,0],ccc.drone_xyz_per_dish_interp[1,t_cut,1],c=ccc.V_bgsub[t_cut,f_index,1],cmap=cm.gnuplot2,norm=LogNorm())    
                else:
                    pass
            elif normalization=='Gauss':
                if len(gfitlist)==0:
                    print('ERROR: --> The normalization cannot be applied because the gfitlist is empty')
                elif len(gfitlist)!=len(concatlist):
                    print('ERROR: --> The length of the gfitlist is not equal to the length of the concatlist')
                elif len(gfitlist)==len(concatlist):
                    ## make sure the normalization is being appropriately applied to the correct file:
                    if concatlist[h].split('_')[0]==gfitlist[h].split('_')[0]:
                        with np.load(gfit_directory+gfitlist[h]) as gff:
                            g_norm=gff['G_popt'][:,:,0]
                            for i in range(self.n_channels):
                                if self.copoldir in 'E':
                                    COPOLIND=np.arange(self.n_channels).reshape(int(self.n_channels/2),2)[int(i/2)][0]
                                    self.x_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,1]-ccc.dish_coords[i][0] # seems like gauss params were found with original xyz, so remove dish offset
                                    self.y_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,3]-ccc.dish_coords[i][1] # seems like gauss params were found with original xyz, so remove dish offset
                                elif self.copoldir in 'N':
                                    COPOLIND=np.arange(self.n_channels).reshape(int(self.n_channels/2),2)[int(i/2)][1]
                                    self.x_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,1]-ccc.dish_coords[i][0] # seems like gauss params were found with original xyz, so remove dish offset
                                    self.y_offsets[:,i,h]=gff['G_popt'][COPOLIND,self.faxis,3]-ccc.dish_coords[i][1] # seems like gauss params were found with original xyz, so remove dish offset
                                else:
                                    print('your loop sucks idiot')                                 
                            if vplot==True:
                                im1x=axes[0][0].scatter(ccc.drone_xyz_per_dish_interp[0,t_cut,0]+self.x_offsets[find,i,h],ccc.drone_xyz_per_dish_interp[0,t_cut,1]+self.y_offsets[find,i,h],c=(1.0/g_norm[0,f_index])*ccc.V_bgsub[t_cut,f_index,0],cmap=cm.gnuplot2,norm=LogNorm())
                                im1y=axes[1][0].scatter(ccc.drone_xyz_per_dish_interp[1,t_cut,0]+self.x_offsets[find,i,h],ccc.drone_xyz_per_dish_interp[1,t_cut,1]+self.y_offsets[find,i,h],c=(1.0/g_norm[1,f_index])*ccc.V_bgsub[t_cut,f_index,1],cmap=cm.gnuplot2,norm=LogNorm())
                                im1x0=axes0[0][0].scatter(ccc.drone_xyz_per_dish_interp[0,t_cut,0]+self.x_offsets[find,i,h],ccc.drone_xyz_per_dish_interp[0,t_cut,1]+self.y_offsets[find,i,h],c=(1.0/g_norm[0,f_index])*ccc.V_bgsub[t_cut,f_index,0],cmap=cm.gnuplot2,norm=LogNorm())
                                im1y0=axes0[1][0].scatter(ccc.drone_xyz_per_dish_interp[1,t_cut,0]+self.x_offsets[find,i,h],ccc.drone_xyz_per_dish_interp[1,t_cut,1]+self.y_offsets[find,i,h],c=(1.0/g_norm[1,f_index])*ccc.V_bgsub[t_cut,f_index,1],cmap=cm.gnuplot2,norm=LogNorm())
                            else:
                                pass
                            
                            
            ## ATTEMPT TO BYPASS chan,freq LOOPS:
            ## create centroid-corrected per channel and frequency drone coordinate maps on a per-concat basis:
            tmpcoords=np.repeat(ccc.drone_xyz_per_dish_interp[:,:,:,np.newaxis],len(self.freq),axis=3)
            shiftvec=np.array((self.x_offsets[:,:,h],self.y_offsets[:,:,h],np.zeros(self.x_offsets[:,:,h].shape)))
            tmpshifts=np.repeat(np.swapaxes(np.swapaxes(shiftvec,0,2),1,2)[:,np.newaxis,:,:],len(ccc.t),axis=1)
            fccoords=(tmpcoords-tmpshifts)[:,ccc.inds_on]
            if normalization=='none':
                Vvals=ccc.V_bgsub[ccc.inds_on,:,:]
            elif normalization=='Gauss':
                Vvals=(np.repeat(np.swapaxes(g_norm[:,fmin:fmax],0,1)[np.newaxis,:,:],len(ccc.inds_on),axis=0)**-1)*ccc.V_bgsub[ccc.inds_on,fmin:fmax,:]           
            ## loop through channels (i,chan) to find indices of nonzero cells in histogram
            for i,chan in enumerate(range(self.n_channels)):
                for j,fr in enumerate(self.faxis):
                    print('Concat:{}/{}, Channel:{}/{}, Frequency:{}/{},           '.format(h+1,self.n_concats,i+1,self.n_channels,j+1,len(self.freq)),end='\r')
                    xf,yf=fccoords[i,:,0,j],fccoords[i,:,1,j]
                    valsf=Vvals[:,j,i]                        
                    #histo2d,xbins,ybins=np.histogram2d(x,y,bins=[self.x_edges[:,i],self.y_edges[:,i]])
                    self.histogram_LC[:,:,j,i,h]=binned_statistic_2d(x=xf,y=yf,values=valsf,statistic='count',bins=[self.x_edges[:,i],self.y_edges[:,i]]).statistic
                    self.V_LC_mean[:,:,j,i,h]=binned_statistic_2d(x=xf,y=yf,values=valsf,statistic='mean',bins=[self.x_edges[:,i],self.y_edges[:,i]]).statistic
                    self.V_LC_std[:,:,j,i,h]=binned_statistic_2d(x=xf,y=yf,values=valsf,statistic='std',bins=[self.x_edges[:,i],self.y_edges[:,i]]).statistic
                              
#             ## THIS LOOP IS SLOW, MABE WE FIND A WAY TO DO IT IN ARRAY SPACE: SAVE THIS DONT DELETE YET
#             ## loop through channels (i,chan) to find indices of nonzero cells in histogram
#             for i,chan in enumerate(range(self.n_channels)):
#                 ## now must loop through frequency space to accomodate freq-dependent centroid removal:
#                 for j,fr in enumerate(self.faxis):
#                     print('Concat:{}/{}, Channel:{}/{}, Frequency:{}/{},           '.format(h+1,self.n_concats,i+1,self.n_channels,j+1,len(self.freq)),end='\r')
#                     x,y=[ccc.drone_xyz_per_dish_interp[i,:,0]+self.x_offsets[j,i,h],ccc.drone_xyz_per_dish_interp[i,:,1]+self.y_offsets[j,i,h]]
#                     histo2d,xbins,ybins=np.histogram2d(x,y,bins=[self.x_edges[:,i],self.y_edges[:,i]])
#                     self.histogram_LC[:,:,j,i,h]=histo2d
#                     xnz,ynz=np.where(histo2d!=0)
#                     ## loop over the bins [m] where the histogram is nonzero to populate the bins of the V_arrays:
#                     for m in range(len(xnz)):
#                         k,l=[xnz[m],ynz[m]]
#                         ## find times [tacc] where the drones coordinates x,y are with in the bin m x,y values:
#                         xacc=np.intersect1d(np.where(self.x_edges[k,i]<=x),np.where(self.x_edges[k+1,i]>x))
#                         yacc=np.intersect1d(np.where(self.y_edges[l,i]<=y),np.where(self.y_edges[l+1,i]>y))
#                         tacc0=np.intersect1d(xacc,yacc)
#                         tacc=np.intersect1d(tacc0,ccc.inds_on)
#                         if normalization=='none':
#                             ## Populate the V arrays with the appropriate values:
#                             self.V_LC_mean[k,l,j,i,h]=np.nanmean(ccc.V_bgsub[tacc,fr,i],axis=0)
#                             self.V_LC_std[k,l,j,i,h]=np.nanstd(ccc.V_bgsub[tacc,fr,i],axis=0)       
#                         elif normalization=='Gauss':
#                             ## Apply normalization, if possible from gauss fit params:
#                             self.V_LC_mean[k,l,j,i,h]=np.nanmean((1.0/g_norm[i,j])*ccc.V_bgsub[tacc,fr,i],axis=0)
#                             self.V_LC_std[k,l,j,i,h]=np.nanstd((1.0/g_norm[i,j])*ccc.V_bgsub[tacc,fr,i],axis=0)
            if vplot==True:
                for i in range(2):
                    ax1,ax2,ax3,ax4=axes[i]
                    im2=ax2.pcolormesh(self.x_edges_grid[:,:,i],self.y_edges_grid[:,:,i],self.histogram_LC[:,:,find,i,h].T,cmap=cm.gnuplot2)
                    im2.set_clim(0,30)
                    im3=ax3.pcolormesh(self.x_edges_grid[:,:,i],self.y_edges_grid[:,:,i],self.V_LC_mean[:,:,find,i,h].T,cmap=cm.gnuplot2,norm=LogNorm())
                    im4=ax4.pcolormesh(self.x_edges_grid[:,:,i],self.y_edges_grid[:,:,i],self.V_LC_std[:,:,find,i,h].T,cmap=cm.gnuplot2,norm=LogNorm())
                    images=[[im1x,im1y][i],im2,im3,im4]
                    titles=['Unbinned Beammaps ({} CH{} {:.2f} MHz)'.format(self.concat_list[h].split('_')[0],i,self.freq[find]),\
                            'Histogram ({} CH{} {:.2f} MHz)'.format(self.concat_list[h].split('_')[0],i,self.freq[find]),\
                            r'Coadded $\bar{x}$'+' Map ({} CH{} {:.2f} MHz)'.format(self.concat_list[h].split('_')[0],i,self.freq[find]),\
                            'Coadded $\sigma$ Map ({} CH{} {:.2f} MHz)'.format(self.concat_list[h].split('_')[0],i,self.freq[find])]
                    cbarlabels=['Power [$ADU^2$]','Counts','Power [$ADU^2$]','Power [$ADU^2$]']
                    for j,ax in enumerate([ax1,ax2,ax3,ax4]):
                        ax.set_title(titles[j])
                        ax.set_facecolor('k')
                        ax.set_xlabel('$x,[m]$')
                        ax.set_ylabel('$y, [m]$')
                        divider=make_axes_locatable(ax)
                        cax=divider.append_axes("right", size="3%", pad=0.05)
                        cbar=fig.colorbar(images[j],cax=cax)
                        cbar.set_label(cbarlabels[j])
                tight_layout()
                
        print("end of bigass loop is: {}".format(datetime.datetime.now()))

        if operation=='coadd':
            self.V_LC_operation=np.NAN*np.ones(self.V_LC_mean[:,:,:,:,0].shape)
            self.V_LC_operation=np.nanmean(self.V_LC_mean,axis=4)
        elif operation=='difference':
            if len(concatlist)==2:
                self.V_LC_operation=np.NAN*np.ones(self.V_LC_mean[:,:,:,:,0].shape)
                self.V_LC_operation=np.nansum(np.array([self.V_LC_mean[:,:,:,:,0],-1*self.V_LC_mean[:,:,:,:,1]]),axis=0)
            else:
                print("--> V_LC_operation can only be instantiated if the length of concatlist is 2")
        if vplot==True:
            for i in range(2):
                ax1,ax2,ax3,ax4=axes0[i]
                im2=ax2.pcolormesh(self.x_edges_grid[:,:,i],self.y_edges_grid[:,:,i],np.nansum(self.histogram_LC[:,:,find,i,:],axis=2).T,cmap=cm.gnuplot2)
                im2.set_clim(0,300)
                im3=ax3.pcolormesh(self.x_edges_grid[:,:,i],self.y_edges_grid[:,:,i],self.V_LC_operation[:,:,find,i].T,cmap=cm.gnuplot2,norm=LogNorm())
                im4=ax4.pcolormesh(self.x_edges_grid[:,:,i],self.y_edges_grid[:,:,i],np.nanmean(self.V_LC_std[:,:,find,i,:],axis=2).T,cmap=cm.gnuplot2,norm=LogNorm())
                images=[[im1x0,im1y0][i],im2,im3,im4]
                titles=['Unbinned Beammaps ({} CH{} {:.2f} MHz)'.format(self.name,i,self.freq[find]),\
                        'Histogram ({} CH{} {:.2f} MHz)'.format(self.name,i,self.freq[find]),\
                        r'Coadded $\bar{x}$'+' Map ({} CH{} {:.2f} MHz)'.format(self.name,i,self.freq[find]),\
                        'Coadded $\sigma$ Map ({} CH{} {:.2f} MHz)'.format(self.name,i,self.freq[find])]
                cbarlabels=['Power [$ADU^2$]','Counts','Power [$ADU^2$]','Power [$ADU^2$]']
                for j,ax in enumerate([ax1,ax2,ax3,ax4]):
                    ax.set_title(titles[j])
                    ax.set_facecolor('k')
                    ax.set_xlabel('$x,[m]$')
                    ax.set_ylabel('$y, [m]$')
                    divider=make_axes_locatable(ax)
                    cax=divider.append_axes("right", size="3%", pad=0.05)
                    cbar=fig.colorbar(images[j],cax=cax)
                    cbar.set_label(cbarlabels[j])
            tight_layout()
                
                


## Ongoing issues/targets: n_dishes degeneracy in siteclass --> move to channel based approach...
## Laura's Objectives:
## X - standard deviation 
## gaussian normalization, centroid matching, & radial binning: laura's gaussian fitting ipynb
## Polar transformation
## Interpolation
## Slices in coordinates: x,y,theta,phi
## Add/Subtract/
## Danny Jacobs: Sometimes you dont want to localize to a single point, some gridding is nearest-neighbors?
            
