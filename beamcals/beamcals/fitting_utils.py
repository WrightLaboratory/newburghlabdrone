#  __ _ _   _   _                      _   _ _                   
# / _(_) | | | (_)                    | | (_) |                  
#| |_ _| |_| |_ _ _ __   __ _    _   _| |_ _| |___   _ __  _   _ 
#|  _| | __| __| | '_ \ / _` |  | | | | __| | / __| | '_ \| | | |
#| | | | |_| |_| | | | | (_| |  | |_| | |_| | \__ \_| |_) | |_| |
#|_| |_|\__|\__|_|_| |_|\__, |   \__,_|\__|_|_|___(_) .__/ \__, |
#                        __/ |_____                 | |     __/ |
#                       |___/______|                |_|    |___/ 

## 20211110 - WT - creating a new file to contain fit functions...
## relocating functions from drone, concat, etc scripts:

import numpy as np
from matplotlib.pyplot import *
from scipy.optimize import least_squares
from astropy.modeling.models import AiryDisk2D
from scipy.stats import pearsonr

## DEFN the Gauss Fit function:
def Gauss(x,a,x0,sigma,k):
    return a*np.exp(-(x-x0)**2.0/(2.0*sigma**2.0))+k

def Airy_2d_LC_opt(P,x,y,V):
    AD=AiryDisk2D(1,5,5,radius=1)
    amp,x0,y0,rad,c=P
    return AD.evaluate(x,y,amp,x0,y0,rad)+c-V

def Airy_2d_LC_func(P,x,y):
    AD=AiryDisk2D(1,5,5,radius=1)
    amp,x0,y0,rad,c=P
    return AD.evaluate(x,y,amp,x0,y0,rad)+c

def Gauss_2d_LC_func(P,x,y):
    amp,x0,xsig,y0,ysig,theta,c=P
    A=(0.5*(((np.cos(theta)/xsig)**2.0)+((np.sin(theta)/ysig)**2.0)))
    B=(0.25*((np.sin(2.0*theta)/(ysig**2.0))-(np.sin(2.0*theta)/(xsig**2.0))))
    C=(0.5*(((np.sin(theta)/xsig)**2.0)+((np.cos(theta)/ysig)**2.0)))
    return amp*np.exp(-1.0*((A*((x-x0)**2.0))+(2.0*B*(x-x0)*(y-y0))+(C*((y-y0)**2.0))))+c

def Gauss_2d_LC_opt(P,x,y,V):
    amp,x0,xsig,y0,ysig,theta,c=P
    A=(0.5*(((np.cos(theta)/xsig)**2.0)+((np.sin(theta)/ysig)**2.0)))
    B=(0.25*((np.sin(2.0*theta)/(ysig**2.0))-(np.sin(2.0*theta)/(xsig**2.0))))
    C=(0.5*(((np.sin(theta)/xsig)**2.0)+((np.cos(theta)/ysig)**2.0)))
    return amp*np.exp(-1.0*((A*((x-x0)**2.0))+(2.0*B*(x-x0)*(y-y0))+(C*((y-y0)**2.0))))+c-V

def Fit_Main_Beam(inputconcat,chans,freqs,coordbounds=[50.0,50.0,150.0],ampbound=0.999):
    A_popt=np.zeros((len(chans),len(freqs),5))
    A_PR=np.zeros((len(chans),len(freqs)))
    G_popt=np.zeros((len(chans),len(freqs),7))
    G_PR=np.zeros((len(chans),len(freqs)))
    ## define timecuts for cartesian coordinates:
    txcut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,0])<coordbounds[0]]
    tycut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,1])<coordbounds[1]]
    tzcut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,2])>coordbounds[2]]
    coordcut=np.intersect1d(np.intersect1d(txcut,tycut),tzcut)
    for i,chan in enumerate(chans):
        for j,find in enumerate(freqs):
            try:
                ## apply amplitude cut:
                tacut=inputconcat.t_index[inputconcat.V[:,find,chan]<ampbound*(np.nanmax(inputconcat.V[:,find,chan]))]
                try:
                    ttcut=np.intersect1d(np.intersect1d(coordcut,tacut),inputconcat.inds_on)
                except AttributeError:
                    ttcut=np.intersect1d(coordcut,tacut)                    
                ## pull the proper coords for fitting:
                mbx=inputconcat.drone_xyz_LC_interp[ttcut,0]
                mby=inputconcat.drone_xyz_LC_interp[ttcut,1]
                mbz=inputconcat.drone_xyz_LC_interp[ttcut,2]
                mbV=inputconcat.V[ttcut,find,chan]
                mb_input_data=np.array([mbx,mby,mbV])
                ## shared params:
                amp0=np.nanmax(mbV)
                bg0=np.nanmin(mbV)
                x00=inputconcat.dish_coords[int(chan/2),0]
                y00=inputconcat.dish_coords[int(chan/2),1]
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
                A_popt[i,j]=least_squares(Airy_2d_LC_opt,x0=pA,args=mb_input_data).x
                A_PR[i,j]=pearsonr(mbV,Airy_2d_LC_func(A_popt[i,j,:],mbx,mby))[0]
                G_popt[i,j]=least_squares(Gauss_2d_LC_opt,x0=pG,args=mb_input_data).x
                G_PR[i,j]=pearsonr(mbV,Gauss_2d_LC_func(G_popt[i,j,:],mbx,mby))[0]
            except ValueError:
                A_popt[i,j,:]=np.NAN*np.zeros(5)
                A_PR[i,j]=np.NAN
                G_popt[i,j,:]=np.NAN*np.zeros(7)
                G_PR[i,j]=np.NAN
    return A_popt,A_PR,G_popt,G_PR