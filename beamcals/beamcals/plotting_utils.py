#       _       _   _   _                      _   _ _                   
#      | |     | | | | (_)                    | | (_) |                  
# _ __ | | ___ | |_| |_ _ _ __   __ _    _   _| |_ _| |___   _ __  _   _ 
#| '_ \| |/ _ \| __| __| | '_ \ / _` |  | | | | __| | / __| | '_ \| | | |
#| |_) | | (_) | |_| |_| | | | | (_| |  | |_| | |_| | \__ \_| |_) | |_| |
#| .__/|_|\___/ \__|\__|_|_| |_|\__, |   \__,_|\__|_|_|___(_) .__/ \__, |
#| |                             __/ |_____                 | |     __/ |
#|_|                            |___/______|                |_|    |___/ 

## 20211110 WT - The module structure is being completely refactored...
## This is the brand new plotting_utils.py file
## Plotting functions will be written here in a more general format... hopefully! ;)

import os
import glob
import pandas
import csv
import datetime
import pytz
from matplotlib.pyplot import *
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from scipy.optimize import least_squares
import numpy as np
import h5py

import beamcals.fitting_utils as fu

################################################
##                  Corr_Data                 ##
################################################
      
def Plot_Waterfalls(corr_class):
    ## Express bounds for the plot axes
    wfbounds=[corr_class.freq[-1],corr_class.freq[0],corr_class.t[-1]-corr_class.t[0],0.0]
    ## This should plot waterfalls for the imported gain calibrated data:
    fig1=figure(figsize=(16,int(4*corr_class.n_channels/2)))
    ## Plotting the individual waterfall plots (note freq ind is reversed!)
    for i in range(corr_class.n_channels):
        ax=fig1.add_subplot(int(corr_class.n_channels/2),2,i+1)
        im=ax.imshow(corr_class.V[:,::-1,corr_class.chmap[i]].real,extent=wfbounds,cmap='gnuplot2',aspect='auto',norm=LogNorm())
        ax.set_title('Auto-Corr: Channel {}x{} - Ind {}'.format(corr_class.chmap[i],corr_class.chmap[i],corr_class.automap[i]))
        ax.set_xlabel('Frequency, [$MHz$]')
        ax.set_ylabel('$\Delta$Time [$s$]')
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig1.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
    tight_layout()

def Plot_Saturation_Maps(corr_class):
    ## Express bounds for the plot axes
    wfbounds=[corr_class.freq[-1],corr_class.freq[0],corr_class.t[-1]-corr_class.t[0],0.0]
    ## This should plot waterfalls for the imported gain calibrated data:
    fig1=figure(figsize=(16,int(4*corr_class.n_channels/2)))
    ## Plotting the individual waterfall plots (note freq ind is reversed!)
    for i in range(corr_class.n_channels):
        ax=fig1.add_subplot(int(corr_class.n_channels/2),2,i+1)
        im=ax.imshow(corr_class.sat[:,::-1,corr_class.chmap[i]].real,extent=wfbounds,cmap='gnuplot2',aspect='auto')
        ax.set_title('Auto-Corr: Channel {}x{} - Ind {}'.format(corr_class.chmap[i],corr_class.chmap[i],corr_class.automap[i]))
        ax.set_xlabel('Frequency, [$MHz$]')
        ax.set_ylabel('$\Delta$Time [$s$]')
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig1.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
    tight_layout()

def Plot_Time_Series(corr_class,tbounds=[0,-1],freqlist=[100,700,900]):
    fig1=figure(figsize=(16,int(4*corr_class.n_channels/2)))
    for i in range(corr_class.n_channels):
        ax=fig1.add_subplot(int(corr_class.n_channels/2),2,i+1)
        for k in freqlist:
            ax.plot(corr_class.t_index[tbounds[0]:tbounds[1]],corr_class.V[tbounds[0]:tbounds[1],k,corr_class.chmap[i]].real,'.',label="F={}".format(corr_class.freq[k]))
        ax.set_title('Time Series: Channel {}x{} - Ind {}'.format(corr_class.chmap[i],corr_class.chmap[i],corr_class.automap[i]))
        ax.set_ylabel('Power [$ADU^2$]')
        ax.set_xlabel('Time Index')
        ax.legend()
    tight_layout()

def Plot_Spectra(corr_class,tbounds=[5,-5],tstep=2000):
    fig1=figure(figsize=(16,4*int(corr_class.n_channels/2)))
    CNorm=colors.Normalize()
    CNorm.autoscale(np.arange(len(corr_class.t))[tbounds[0]:tbounds[1]:tstep])
    CM=cm.gnuplot2
    CM=cm.magma
    for i in range(corr_class.n_channels):
        ax=fig1.add_subplot(int(corr_class.n_channels/2),2,i+1)
        for k,t_ind in enumerate(np.arange(len(corr_class.t))[tbounds[0]:tbounds[1]:tstep]):
            ax.semilogy(corr_class.freq,corr_class.V[t_ind,:,corr_class.chmap[i]],'.',c=CM(CNorm(t_ind)),label='t = {:.2f}'.format(float(corr_class.t[t_ind]-corr_class.t[0])))
        ax.set_title('Spectra: Channel {}x{} - Ind {}'.format(corr_class.chmap[i],corr_class.chmap[i],corr_class.automap[i]))
        ax.set_ylabel('Log Power [$ADU^2$]')
        ax.set_xlabel('Frequency [MHz]')
        ax.legend(fontsize='small')
    tight_layout()

def Plot_Gains_vs_Data(corr_class,tind=1):
    ## Let's plot the calculated gain solution for the data:
    fig1=figure(figsize=(16,int(4*corr_class.n_channels/2)))
    for i in range(corr_class.n_channels):
        ax=fig1.add_subplot(int(corr_class.n_channels/2),4,int(2*i)+1)
        ax.plot(corr_class.freq,corr_class.gain[:,corr_class.chmap[i]],'.',label="Gain Exp = {}".format(corr_class.gain_exp[corr_class.chmap[i]]))
        ax.set_title("Gain: Channel {}x{} - Ind {}".format(corr_class.chmap[i],corr_class.chmap[i],corr_class.automap[i]))
        ax.set_ylabel('Gain')            
        ax.set_xlabel('Frequency [MHz]')
        ax.legend()
        ax=fig1.add_subplot(int(corr_class.n_channels/2),4,int(2*i)+2)
        ax.plot(corr_class.freq,corr_class.V[tind,:,corr_class.chmap[i]],'.')
        ax.set_title("Spectra: Channel {}x{} - Ind {}".format(corr_class.chmap[i],corr_class.chmap[i],corr_class.automap[i]))
        ax.set_ylabel('Power [$ADU^2$]')            
        ax.set_xlabel('Frequency [MHz]')
    tight_layout()
    
################################################
##                 Drone_Data                 ##
################################################

def Plot_Drone_Coordinates(drone_class,coo='lat',t_bounds=[0,-1]):
    print('plotting drone coordinates for all time samples:')
    fig1,[[ax1,ax2,ax3],[ax4,ax5,ax6]]=subplots(nrows=2,ncols=3,figsize=(15,9))
    ## Plot p0 coordinate origin:
    ax1.plot(drone_class.origin[0],drone_class.origin[1],'ro')
    ax2.axhline(drone_class.origin[0],c='b')
    ax3.axhline(drone_class.origin[1],c='b')

    if coo=='lat':
        ## Title each coordinate subplot:        
        ax1.set_title('Lat vs Lon')
        ax2.set_title('Lat vs Time')
        ax3.set_title('Lon vs Time')
        ax4.set_title('Velocity vs Time')
        ax5.set_title('Altitude vs Time')
        ax6.set_title('Yaw vs Time')
        ## Specify arrays/vectors to plot in 1,3,4 coordinate subplot
        xqtys=[drone_class.latitude,drone_class.t_index,drone_class.t_index,drone_class.t_index,drone_class.t_index,drone_class.t_index]
        yqtys=[drone_class.longitude,drone_class.latitude,drone_class.longitude,drone_class.velocity,drone_class.altitude,drone_class.yaw]
        xtags=['Latitude, [$deg$]','Drone Index','Drone Index','Drone Index','Drone Index','Drone Index']
        ytags=['Longitude, [$deg$]','Latitude, [$deg$]','Longitude, [$deg$]','Velocity, [m/s]','Altitude, [$m$]','Yaw [$deg$]']
    if coo=='xy': 
        ax1.set_title('X vs Y')
        ax2.set_title('X vs Time')
        ax3.set_title('Y vs Time')
        ax4.set_title('Velocity vs Time')
        ax5.set_title('Altitude vs Time')
        ax6.set_title('Yaw vs Time')
        ## Specify arrays/vectors to plot in 1,3,4 coordinate subplot
        xqtys=[drone_class.coords_xyz_LC[:,0],drone_class.t_index,drone_class.t_index,drone_class.t_index,drone_class.t_index,drone_class.t_index]
        yqtys=[drone_class.coords_xyz_LC[:,1],drone_class.latitude,drone_class.longitude,drone_class.velocity,drone_class.altitude,drone_class.yaw]
        xtags=['X, [$m$]','Drone Index','Drone Index','Drone Index','Drone Index','Drone Index']
        ytags=['Y, [$m$]','X, [$m$]','Y, [$m$]','Velocity, [m/s]','Altitude, [$m$]','Yaw [$deg$]']
    print('overplotting drone coordinates for t_cut samples: ['+str(t_bounds[0])+':'+str(t_bounds[1])+']')
    for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
        ax.plot(np.nanmin(xqtys[i][t_bounds[0]:t_bounds[1]]),np.nanmin(yqtys[i][t_bounds[0]:t_bounds[1]]))
        ax.plot(np.nanmax(xqtys[i][t_bounds[0]:t_bounds[1]]),np.nanmax(yqtys[i][t_bounds[0]:t_bounds[1]]))
        autoscalelims=ax.axis()
        ax.clear()
        ax.plot(xqtys[i],yqtys[i],'.',label='all samples')
        ax.plot(xqtys[i][t_bounds[0]:t_bounds[1]],yqtys[i][t_bounds[0]:t_bounds[1]],'.',label='selected samples')
        ax.set_xlabel(xtags[i])
        ax.set_ylabel(ytags[i])
        ax.grid()
        ax.legend()
        ax.set_xlim(autoscalelims[0],autoscalelims[1])
        ax.set_ylim(autoscalelims[2],autoscalelims[3])
    tight_layout()

def Plot_Angular_Coordinates(drone_class,t_bounds=[0,-1]):
    fig=figure(figsize=(15,4.5))
    ax1=fig.add_subplot(1, 3, 1)
    ax1.plot(drone_class.t_index[:],(180/np.pi)*drone_class.coords_rpt[:,1],'.')
    ax1.plot(drone_class.t_index[t_bounds[0]:t_bounds[1]],(180/np.pi)*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],1],'.')
    ax1.set_xlabel('time index')
    ax1.set_ylabel(r'$\phi, [deg]$')
    ax2=fig.add_subplot(1, 3, 2)
    ax2.plot(drone_class.t_index[:],(180/np.pi)*drone_class.coords_rpt[:,2],'.')
    ax2.plot(drone_class.t_index[t_bounds[0]:t_bounds[1]],(180/np.pi)*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],2],'.')
    ax2.set_xlabel('time index')
    ax2.set_ylabel(r'$\theta, [deg]$')
    ax3=fig.add_subplot(1, 3, 3, projection='polar')
    ax3.plot(drone_class.coords_rpt[:,1],180/np.pi*drone_class.coords_rpt[:,2],'.')
    ax3.plot(drone_class.coords_rpt[t_bounds[0]:t_bounds[1],1],180/np.pi*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],2],'.')
    ax3.set_rlim(np.nanmin(180/np.pi*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],2]),1.1*np.nanmax(180/np.pi*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],2]))
    tight_layout()

def Plot_3d(drone_class,t_bounds=[0,-1]):
    fig=figure(figsize=(10,4.5))
    tkeys=['Geocentric Cartesian','Local Cartesian']
    for i,coordset in enumerate([drone_class.coords_xyz_GC,drone_class.coords_xyz_LC]):
        ax=fig.add_subplot(1, 2, i+1, projection='3d')
        ax.set_title(tkeys[i])
        ax.set_xlabel('x, [meters]')
        ax.set_ylabel('y, [meters]')
        ax.set_zlabel('z, [meters]')
        ax.plot(coordset[:,0],coordset[:,1],coordset[:,2],'.')
        ax.plot(coordset[t_bounds[0]:t_bounds[1],0],coordset[t_bounds[0]:t_bounds[1],1],coordset[t_bounds[0]:t_bounds[1],2],'.')
    tight_layout()

def Plot_Transmitter_Pointing(drone_class,t_bounds=[0,-1],t_step=1):
    fig=figure(figsize=(10,8))
    ax=fig.add_subplot(111)
    ## DRONE COORDINATE SYSTEM x,y,z=North,East,Down VARIABLES ##
    UV_nose_north=np.array([1,0,0]) #unit vector for nose pointing north, no roll/pitch, prior to rotations
    UV_trans_down=np.array([0,0,1]) #unit vector for transmitter pointing down prior to rotations
    ## drone roll, pitch, yaw angles:
    ypr=np.ndarray((len(drone_class.t_index),3))
    ypr[:,0]=drone_class.yaw
    ypr[:,1]=drone_class.pitch
    ypr[:,2]=drone_class.roll
    ## TRANSMITTER POINTING DIRECTION as fxn of time in Local Cartesian: (transform by [y,p,r]=[+90,0,+180] rot)
    trans_pointing_xyz=np.array([gu.rot_mat(np.array([90.0,0.0,180.0]))@gu.rot_mat(ypr[m,:])@UV_trans_down for m in range(len(drone_class.t_index))])
    ## Plot Parameters:
    [Qlb,Qub,Qstep]=[t_bounds[0],t_bounds[1],t_step]
    M=np.abs(np.hypot(trans_pointing_xyz[Qlb:Qub:Qstep,0],trans_pointing_xyz[Qlb:Qub:Qstep,1]))
    CNorm=colors.Normalize()
    CNorm.autoscale(M)
    CM=cm.gnuplot2
    SM=cm.ScalarMappable(cmap=CM, norm=CNorm)
    SM.set_array([])
    q=ax.quiver(drone_class.coords_xyz_LC[Qlb:Qub:Qstep,0],drone_class.coords_xyz_LC[Qlb:Qub:Qstep,1],trans_pointing_xyz[Qlb:Qub:Qstep,0],trans_pointing_xyz[Qlb:Qub:Qstep,1],color=CM(CNorm(M)))
    ax.quiverkey(q,X=0.15,Y=0.05,U=1,label='Unit Vector', labelpos='E')
    fig.colorbar(SM,label='XY Projection Magnitude')
    ax.set_xlabel('Local X Position, [m]')
    ax.set_ylabel('Local Y Position, [m]')
    ax.set_title('Transmitter Deviation from Nadir [XY Projection]')
    tight_layout()

def Plot_Polar_Lines_of_Sight(drone_class,t_bounds=[0,-1],t_step=1,dishid=0):        
    fig1,[ax1,ax2]=subplots(nrows=1,ncols=2,figsize=(16,8),subplot_kw=dict(projection="polar"))
    ax1.set_title('Drone Position in Receiver {} Beam'.format(dishid))
    ax1.plot(drone_class.rpt_t_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,1],180.0/np.pi*drone_class.rpt_t_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,2],'.b',markersize=1.0)
    ax1.plot(0,0,'ro')
    ax2.set_title('Receiver {} Position in Drone Beam'.format(dishid))
    ax2.plot(drone_class.rpt_r_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,1],180.0/np.pi*drone_class.rpt_r_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,2],'.g',markersize=1.0)
    ax2.plot(0,0,'ro')
    tight_layout()


################################################
##                   CONCAT                   ##
################################################

def Plot_Beammap(concat_class,t_bounds=[0,-1],coord_args="LC",pulse_args=None,f_bounds=[300,340],cbounds=[],dotsize=40):
    fig1=figure(figsize=(16,int(7*concat_class.n_channels/2)))
    for i in range(int(concat_class.n_channels/2)):
        ## No pulse_args: all data
        if pulse_args==None:
            t_cut=np.arange(concat_class.t_index[t_bounds[0]],concat_class.t_index[t_bounds[1]])
            pt_colors_1=np.nanmean(concat_class.V[t_cut,f_bounds[0]:f_bounds[1],int(2*i)],axis=1)
            pt_colors_2=np.nanmean(concat_class.V[t_cut,f_bounds[0]:f_bounds[1],int(2*i)+1],axis=1)
        ## pulse_args="on" only source on
        elif pulse_args=="on":
            t_cut=np.intersect1d(np.arange(concat_class.t_index[t_bounds[0]],concat_class.t_index[t_bounds[1]]),concat_class.inds_on).tolist()
            pt_colors_1=np.nanmean(concat_class.V[t_cut,f_bounds[0]:f_bounds[1],int(2*i)],axis=1)
            pt_colors_2=np.nanmean(concat_class.V[t_cut,f_bounds[0]:f_bounds[1],int(2*i)+1],axis=1)
        ## pulse_args="off" only source off
        elif pulse_args=="off":
            t_cut=np.intersect1d(np.arange(concat_class.t_index[t_bounds[0]],concat_class.t_index[t_bounds[1]]),concat_class.inds_off).tolist()
            pt_colors_1=np.nanmean(concat_class.V[t_cut,f_bounds[0]:f_bounds[1],int(2*i)],axis=1)
            pt_colors_2=np.nanmean(concat_class.V[t_cut,f_bounds[0]:f_bounds[1],int(2*i)+1],axis=1)
        ## pulse_args="bg" only show background 
        elif pulse_args=="bg":
            t_cut=np.arange(concat_class.t_index[t_bounds[0]],concat_class.t_index[t_bounds[1]]).tolist()
            pt_colors_1=np.nanmean(concat_class.V_bg[t_cut,f_bounds[0]:f_bounds[1],int(2*i)],axis=1)
            pt_colors_2=np.nanmean(concat_class.V_bg[t_cut,f_bounds[0]:f_bounds[1],int(2*i)+1],axis=1)
        ## pulse_args="bgsub" only show background subtracted on points
        elif pulse_args=="bgsub":
            t_cut=np.intersect1d(np.arange(concat_class.t_index[t_bounds[0]],concat_class.t_index[t_bounds[1]]),concat_class.inds_on).tolist()
            pt_colors_1=np.nanmean(concat_class.V_bgsub[t_cut,f_bounds[0]:f_bounds[1],int(2*i)],axis=1)
            pt_colors_2=np.nanmean(concat_class.V_bgsub[t_cut,f_bounds[0]:f_bounds[1],int(2*i)+1],axis=1)
        ## Create axes and assign assign x,y points from drone using coords of choice:
        if coord_args=="LC":
            ax1=fig1.add_subplot(int(concat_class.n_channels/2),2,int(2*i)+1)
            ax2=fig1.add_subplot(int(concat_class.n_channels/2),2,int(2*i)+2)
            x=concat_class.drone_xyz_LC_interp[t_cut,0]
            y=concat_class.drone_xyz_LC_interp[t_cut,1]
            im1=ax1.scatter(x,y,s=dotsize,c=pt_colors_1,cmap='gnuplot2',norm=LogNorm())
            im2=ax2.scatter(x,y,s=dotsize,c=pt_colors_2,cmap='gnuplot2',norm=LogNorm())
        elif coord_args=="Pol":
            ax1=fig1.add_subplot(int(concat_class.n_channels/2),2,int(2*i)+1,projection="polar")
            ax2=fig1.add_subplot(int(concat_class.n_channels/2),2,int(2*i)+2,projection="polar")
            x=concat_class.drone_rpt_r_per_dish_interp[i,t_cut,1]
            y=180.0/np.pi*concat_class.drone_rpt_r_per_dish_interp[i,t_cut,2]
            im1=ax1.scatter(x,y,s=dotsize,c=pt_colors_1,cmap='gnuplot2',norm=LogNorm())
            im2=ax2.scatter(x,y,s=dotsize,c=pt_colors_2,cmap='gnuplot2',norm=LogNorm())
        ## set color limits to fix the L,R plots to same colorscale:
        images=[im1,im2]
        for im in images:
            mincl=np.nanmin([im1.get_clim()[0],im2.get_clim()[0]])
            maxcl=np.nanmax([im1.get_clim()[1],im2.get_clim()[1]])
            if len(cbounds)==2:
                im.set_clim(cbounds[0],cbounds[1])
            else:
                im.set_clim(mincl,maxcl)
        for j,ax in enumerate([ax1,ax2]):
            ax.set_facecolor('k')
            ax.set_title(concat_class.name+' Channel {} Beammap'.format(concat_class.chmap[int(2*i)+j]))
            if coord_args=="LC":
                ax.set_xlabel('X Position $[m]$')
                ax.set_ylabel('Y Position $[m]$')
                divider=make_axes_locatable(ax)
                cax=divider.append_axes("right", size="3%", pad=0.05)
                cbar=fig1.colorbar(images[j],cax=cax)
                cbar.set_label('Power [$ADU^2$]')
            if coord_args=="Pol":
                cbar=fig1.colorbar(images[j],ax=ax,aspect=40)
                cbar.set_label('Power [$ADU^2$]')
    tight_layout()
    
def Synchronization_Verification_Plots(inputconcat,chans=np.array([2,3]),find=900,coordbounds=[50.0,50.0,150.0],ampbound=0.999):
    ## Produce additional verification plots:
    ## Make figures for the best-fit time offset at a particular frequency:
    figure,[[ax1,ax2],[ax3,ax4],[ax5,ax6],[ax7,ax8],[ax9,ax10],[ax11,ax12]]=subplots(nrows=6,ncols=2,figsize=(16,40))
    ## Make temp concat file with tfinex (best fit time offset)  
    ## Loop through fits and make the plot:
    for j,chan in enumerate(chans):
        ## define timecuts for amplitude:
        tacut=inputconcat.t_index[inputconcat.V[:,find,chan]<ampbound*(np.nanmax(inputconcat.V[:,find,chan]))]
        ## define timecuts for cartesian coordinates:
        txcut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,0])<coordbounds[0]]
        tycut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,1])<coordbounds[1]]
        tzcut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,2])>coordbounds[2]]
        coordcut=np.intersect1d(np.intersect1d(txcut,tycut),tzcut)
        try:
            ttcut=np.intersect1d(np.intersect1d(coordcut,tacut),inputconcat.inds_on)
        except AttributeError:
            ttcut=np.intersect1d(coordcut,tacut)
        ## data points for fit:
        mbx=inputconcat.drone_xyz_LC_interp[ttcut,0]
        mby=inputconcat.drone_xyz_LC_interp[ttcut,1]
        mbz=inputconcat.drone_xyz_LC_interp[ttcut,2]
        mbV=inputconcat.V[ttcut,find,j]
        mb_input_data=np.array([mbx,mby,mbV])
        ## shared params:
        amp0=np.nanmax(mbV)
        bg0=np.nanmin(mbV)
        x00=inputconcat.dish_coords[int(j/2),0]
        y00=inputconcat.dish_coords[int(j/2),1]
        ## airy params:
        rad0=25.0
        ## 2dgauss params:
        xsig0=6.0
        ysig0=6.0
        theta0=0.0
        ## initial guess and bounds:
        pA=np.array([amp0,x00,y00,rad0,bg0])
        pG=np.array([amp0,x00,xsig0,y00,ysig0,theta0,bg0])
        ## run the fits:
        Apopt=least_squares(fu.Airy_2d_LC_opt,x0=pA,args=mb_input_data).x
        Gpopt=least_squares(fu.Gauss_2d_LC_opt,x0=pG,args=mb_input_data).x
        ## simulate space of these coords:
        simx=np.outer(np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),np.ones(100)).flatten()
        simy=np.outer(np.ones(100),np.linspace(np.nanmin(mby),np.nanmax(mby),100)).flatten()
        simV=fu.Gauss_2d_LC_func(Gpopt,simx,simy)
        ## PLOT 1
        ax=[ax1,ax2][j]
        ax.set_title('Channel {} Beammap - Full Time'.format(j))
        ax.set_facecolor('k')
        try:
            tbig=np.intersect1d(tzcut,inputconcat.inds_on)
        except AttributeError:
            tbig=tzcut
        im=ax.scatter(inputconcat.drone_xyz_LC_interp[tbig,0],inputconcat.drone_xyz_LC_interp[tbig,1],c=inputconcat.V[tbig,find,j],s=20,norm=LogNorm())
        #im.set_clim(np.nanmin(mbV),np.nanmax(mbV))
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="3%", pad=0.05)
        cbar=figure.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
        ## PLOT 1
        ax=[ax3,ax4][j]
        ax.set_title('Channel {} Beammap - Points Fit w/ 2DGauss'.format(j))
        ax.set_facecolor('k')
        im=ax.scatter(mbx,mby,c=mbV,norm=LogNorm())
        im.set_clim(np.nanmax((np.nanmin(simV),np.nanmin(mbV))),np.nanmax((np.nanmax(simV),np.nanmax(mbV))))
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="3%", pad=0.05)
        cbar=figure.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
        ## PLOT 2
        ax=[ax5,ax6][j]
        ax.set_title('Channel {} Beammap - Best Fit 2DGauss'.format(j))
        ax.set_facecolor('k')
        im=ax.scatter(simx,simy,c=simV,norm=LogNorm())
        im.set_clim(np.nanmax((np.nanmin(simV),np.nanmin(mbV))),np.nanmax((np.nanmax(simV),np.nanmax(mbV))))
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="3%", pad=0.05)
        cbar=figure.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
        ax.plot(Gpopt[1],Gpopt[3],'wx',label='[{:.2f},{:.2f}]'.format(Gpopt[1],Gpopt[3]))
        ax.legend()
        ## PLOT 3
        ax=[ax7,ax8][j]
        ax.set_title('Channel {} Beammap - Overlay'.format(j))
        ax.set_facecolor('k')
        im=ax.scatter(simx,simy,c=simV,norm=LogNorm())
        im.set_clim(np.nanmax((np.nanmin(simV),np.nanmin(mbV))),np.nanmax((np.nanmax(simV),np.nanmax(mbV))))
        im=ax.scatter(mbx,mby,c=mbV,norm=LogNorm())
        im.set_clim(np.nanmax((np.nanmin(simV),np.nanmin(mbV))),np.nanmax((np.nanmax(simV),np.nanmax(mbV))))
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="3%", pad=0.05)
        cbar=figure.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
        ax.plot(Gpopt[1],Gpopt[3],'wx',label='[{:.2f},{:.2f}]'.format(Gpopt[1],Gpopt[3]))
        ax.legend()
        ## PLOT 4
        ax=[ax9,ax10][j]
        ax.set_title('Channel {} Data vs Best-Fit 2DGauss'.format(j))
        ax.set_facecolor('k')
        ax.scatter(mbx,mbV,c=mbV,norm=LogNorm())
        ax.plot(np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),fu.Gauss_2d_LC_func(Gpopt,np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),Gpopt[3]*np.ones(100)),'w.--',label='X center pass 2DGauss')
        ax.plot(np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),fu.Airy_2d_LC_func(Apopt,np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),Apopt[2]*np.ones(100)),'c.--',label='X center pass Airy')
        ax.axvline(Gpopt[1],c='w',label='X center = {:.2f}'.format(Gpopt[1]))
        ax.semilogy
        ax.legend()
        ## PLOT 4
        ax=[ax11,ax12][j]
        ax.set_title('Channel {} Data vs Best-Fit 2DGauss'.format(j))
        ax.set_facecolor('k')
        ax.scatter(mby,mbV,c=mbV,norm=LogNorm())
        ax.plot(np.linspace(np.nanmin(mby),np.nanmax(mby),100),fu.Gauss_2d_LC_func(Gpopt,Gpopt[1]*np.ones(100),np.linspace(np.nanmin(mby),np.nanmax(mby),100)),'w.--',label='Y center pass 2DGauss')
        ax.plot(np.linspace(np.nanmin(mby),np.nanmax(mby),100),fu.Airy_2d_LC_func(Apopt,Apopt[1]*np.ones(100),np.linspace(np.nanmin(mby),np.nanmax(mby),100)),'c.--',label='Y center pass Airy')
        ax.axvline(Gpopt[3],c='w',label='Y center = {:.2f}'.format(Gpopt[3]))
        ax.semilogy
        ax.legend()
    for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]):
        ax.set_xlabel('X $[m]$')
        ax.set_ylabel('Y $[m]$')
    for i,ax in enumerate([ax9,ax10]):
        ax.set_xlabel('X $[m]$')
        ax.set_ylabel('Power [$ADU^2$]')
    for i,ax in enumerate([ax11,ax12]):
        ax.set_xlabel('Y $[m]$')
        ax.set_ylabel('Power [$ADU^2$]')
    tight_layout()
