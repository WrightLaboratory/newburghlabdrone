from BMX_Classes import *
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import scipy.optimize as opt
from scipy.optimize import curve_fit


def droneDataPairing_offset(drone_dir, FLY, bmx_dir, BMXFILE, time_offset_s, pol):
        drn_dat = Drone_Data(drone_dir, FLY, Origin_llh=VECT_BMX_E_LOC)
        tel_dat = BMX_Data(bmx_dir, BMXFILE)
        
        autos = tel_dat.autodata_raw[pol,:]
        
        #interpolate position data
        xp, yp = interp_position_data(list(drn_dat.coords_xyz_prime[:,0]), list(drn_dat.coords_xyz_prime[:,1]))
                   
        
        t_drone = pd.to_datetime(drn_dat.t_arr_timestamp)+pd.Timedelta(seconds=time_offset_s)
        t_tel = Telescope_Time_Convert(tel_dat)
        
        r = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 0])
        theta = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 1])
        phi = np.interp(t_tel, t_drone, drn_dat.coords_rpt_prime[:, 1])
        x = np.interp(t_tel, t_drone, xp)
        y = np.interp(t_tel, t_drone, yp)
        z = np.interp(t_tel, t_drone, drn_dat.coords_xyz_prime[:,2])
        rprime = np.sqrt(x**2+y**2)
        yaw = np.interp(t_tel, t_drone, drn_dat.yaw)

        
        # coordinates of dish centers (from FLY338)
        # NESW by y,x,z
        dishLoc = ([-3.7605200906121437, 3.865245048570655],\
        [-0.0641296465130804, 0.035901010584928825], \
        [-3.6868008232255924, -3.205432553991354], \
        [-7.485557261347768, -0.07592779975602093],)
        
        beamCenters = ([-4.670897458296071, 7.777616712345406],\
        [3.3385316030884016, -3.27205823116637],\
        [3.2569049644387906, 1.2983814825634061],\
        [-10.421877726648322, -0.7263793693162363])
        

        # compute angles (x=lon, y=lat) from each dish
        thetaX = []
        thetaY = []
        for chan in range(4):
	        thetaX.append(np.arctan((x - dishLoc[chan][0])/(z)))
	        thetaY.append(np.arctan((y - dishLoc[chan][1])/(z)))
        thetaX = np.array(thetaX)
        thetaY = np.array(thetaY)
        
        thetaX2 = []
        thetaY2 = []
        for chan in range(4):
	        thetaX2.append(np.arctan((x - beamCenters[chan][0])/(z)))
	        thetaY2.append(np.arctan((y - beamCenters[chan][1])/(z)))
        thetaX_beam = np.array(thetaX2)
        thetaY_beam = np.array(thetaY2)
        
        return {"FLY" : FLY, "BMXdir" :  BMXFILE, "autos" : autos, "drone_time" : t_drone, "tel_time" : t_tel, "x" : x, "y" : y, "z" : z, "thetaX" : thetaX, "thetaY" : thetaY, "thetaX_beam" : thetaX_beam, "thetaY_beam" : thetaY_beam, "r": rprime, "yaw": yaw, "phi":phi} 


    
class Do_Everything_For_Data_offset:
    def __init__(self, drone_dir, FLY, bmx_dir, BMXFILE, offset):
        if FLY == 'FLY349' or FLY =='FLY351' or FLY =='FLY352': pol = 1
        else: pol = 0
        self.fulldict = droneDataPairing_offset(drone_dir, FLY, bmx_dir, BMXFILE, offset, pol)    
    
    def plotscatter(self, dishindx, freqindx):
        plt.scatter(self.fulldict['x'],\
            self.fulldict['y'], s=50,\
            c = (self.fulldict['autos'][dishindx][:,freqindx]),\
            cmap='gnuplot2', norm=LogNorm(), marker = ',')
    
    def plot_x_lin(self, dishindx, freqindx, lb = -0.5, ub = 0.5, params = True):
        plot_x_slice_lin(self.fulldict, dishindx, freqindx, lb, ub, params)
        
    def plot_x_db(self, dishindx, freqindx, lb = -0.5, ub = 0.5, flipped = False):
        plot_x_slice_db(self.fulldict, dishindx, freqindx, lb, ub, flipped)
    
    def plot_y_lin(self, dishindx, freqindx, lb = -0.5, ub = 0.5, params = True):
        plot_y_slice_lin(self.fulldict, dishindx, freqindx, lb, ub, params)
        
    def plot_y_db(self, dishindx, freqindx, lb = -0.5, ub = 0.5, flipped = False):
        plot_y_slice_db(self.fulldict, dishindx, freqindx, lb, ub, flipped)
        
    def get2Dparams(self, dishindx, freqindx):
        x = list(np.degrees(self.fulldict['thetaX_beam'][dishindx]))
        y = list(np.degrees(self.fulldict['thetaY_beam'][dishindx]))
        z = list(self.fulldict['autos'][dishindx][:,freqindx])

        initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)
        return(popt, pcov)

    def print2Dparams(self, dishindx, freqindx):
        x = list(np.degrees(self.fulldict['thetaX_beam'][dishindx]))
        y = list(np.degrees(self.fulldict['thetaY_beam'][dishindx]))
        z = list(self.fulldict['autos'][dishindx][:,freqindx])

        initial_guess = (1e15, 0, 0, 5, 5, 0, 1e12)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), z, p0=initial_guess)

        popt[3] = 2.355*popt[3]
        popt[4] = 2.355*popt[4]

        pars = ['amplitude', 'xo', 'yo', 'fwhm_x', 'fwhm_y', 'theta', 'offset']
        for i in range(len(pars)): print(pars[i]+' = '+format_e(popt[i]))       
            
def get_y_centroid_loc(flight_dict, dishindx, freqindx, bounds):
    x = list(np.degrees(flight_dict['thetaX_beam'][dishindx]))
    y = list(np.degrees(flight_dict['thetaY_beam'][dishindx]))
    z = list(flight_dict['autos'][dishindx][:,freqindx])

    y_slice = [y[i] for i in range(len(x)) if bounds[0]<x[i]<bounds[1]]
    power_y_slice=[z[i] for i in range(len(x)) if bounds[0]<x[i]<bounds[1]]

    popt, pcov = opt.curve_fit(gauss1D, y_slice, power_y_slice, p0=[1e15, 0, 3, 1e12])
    return(popt[1])

def get_x_centroid_loc(flight_dict, dishindx, freqindx, bounds):
    x = list(np.degrees(flight_dict['thetaX_beam'][dishindx]))
    y = list(np.degrees(flight_dict['thetaY_beam'][dishindx]))
    z = list(flight_dict['autos'][dishindx][:,freqindx])

    x_slice = [x[i] for i in range(len(x)) if bounds[0]<y[i]<bounds[1]]
    power_x_slice=[z[i] for i in range(len(x)) if bounds[0]<y[i]<bounds[1]]

    popt, pcov = opt.curve_fit(gauss1D, x_slice, power_x_slice, p0=[1e15, 0, 3, 1e12])
    return(popt[1])


def get_fits_by_time(offets_dict):
    pars = ['amp', 'x0', 'y0', 'fwhmx', 'fwhmy', 'theta', 'offset']
    params = {}
    cov = {}
    for i in range(len(pars)):
        sub_par = []
        sub_par_std = []
        for key in offets_dict.keys():
            sub_par.append(offets_dict[key].get2Dparams(0,100)[0][i])
            sub_par_std.append(np.sqrt(offets_dict[key].get2Dparams(0,100)[1][i][i]))
        params[pars[i]] = sub_par
        cov[pars[i]] = sub_par_std
    return(params, cov)

def ploterrors(offets_dict,cov):
    pars = ['amp', 'x0', 'y0', 'fwhmx', 'fwhmy', 'theta', 'offset']

    fig = plt.figure(figsize = (9,5))
    counter = 1
    for param in pars:
        ax = fig.add_subplot(2,4,counter)
        ax.plot(list(offets_dict.keys()),np.sqrt(cov[param]),'.')
        ax.set_yscale("log")
        counter+=1
        
def getcetroidbounds(dictname, flightname, dish_num):
    x = [round(dictname[flightname][0].fulldict['x'][i]) \
         for i in range(len(dictname[flightname][0].fulldict['x']))]
    y = [round(dictname[flightname][0].fulldict['y'][i]) \
         for i in range(len(dictname[flightname][0].fulldict['x']))]

    thetax = [round(np.degrees(dictname[flightname][0].fulldict['thetaX_beam'][dish_num][i]),1) \
         for i in range(len(dictname[flightname][0].fulldict['x']))]
    thetay = [round(np.degrees(dictname[flightname][0].fulldict['thetaY_beam'][dish_num][i]),1) \
         for i in range(len(dictname[flightname][0].fulldict['x']))]
    
    xdataatzero = [thetax[i] for i in range(len(thetax)) if -1<thetay[i]<1 and -8<thetax[i]<6]
    test = xdataatzero
    testdict = {}
    for key in test: testdict[key] = 0
    test = list(testdict.keys())

    passes = []
    for i in range(len(test)-1):
        if -0.2<(test[i]-test[i+1])<0.2:
            pass
        else: 
            passes.append(test[i])

    bounds = [(_pass-0.5,_pass+0.5) for _pass in passes]
    return(bounds)

    
def getcentroidmedians(dictname, flightname,dish_num):
    bounds = getcetroidbounds(dictname, flightname,dish_num)
    medians = []
    for bound in bounds:
        stuff = np.degrees(dictname[flight_dict][0].fulldict['thetaX_beam'][dish_num])
        y = [stuff[i] for i in range(len(stuff)) if bound[0]<stuff[i]<bound[1]]
        medians.append(np.median(y))
    return(medians)

def plotcentroids(dictname, flightname, dictelements):
    fig = plt.figure(figsize = (6,7))
    bounds = getcetroidbounds(dictname, flightname,0)
    medians = getcentroidmedians(dictname, flightname,0)
    size = 6

    colors = ['blue','orange','pink', 'green', 'red']
    colors = colors[0:len(dictelements)]
    
    for i,element in enumerate(medians):
        for j in range(4):
            ax = fig.add_subplot(2,2,j+1)
            ax.axvline(element)
            for k, key in enumerate(dictelements):
                try: ax.plot(element, get_y_centroid_loc(dictname[flightname][key].fulldict,j,100,bounds[i]),'o', \
                         color = colors[k], markersize = size)
                except: pass
            ax.set_title('Dish '+str(j+1))
            ax.set_xlim(-5,5)
            ax.set_ylim(-2,2)
            if j == 0 or j ==2:
                ax.set_ylabel('Degrees')
            if j == 2 or j ==3:
                ax.set_xlabel('Degrees')

    patches = [mpatches.Patch(color=colors[i], label=dictelements[i]) for i in range(len(colors))]

    plt.legend(handles=patches, ncol = 2)
    plt.suptitle(flightname+'Centroid Location from 1D Fit Along Passes Thru Beam', y = 0.95, fontsize = 12)

    plt.show()
    
def plotcentroids_x2(dictname, flightname, dictelements, freq):
    fig = plt.figure(figsize = (6,7))
    size = 6

    colors = ['blue','orange','pink', 'green', 'red']
    colors = colors[0:len(dictelements)]
    
    for j in range(4):
        medians = getcentroidmedians_x(dictname, flightname,j)
        for i,element in enumerate(medians):
            ax = fig.add_subplot(2,2,j+1)
            ax.axhline(element)
            bounds = getcetroidbounds_x(dictname, flightname,j)
            for k, key in enumerate(dictelements):
                try: ax.plot(get_x_centroid_loc(dictname[flightname][key].fulldict, j,freq,bounds[i]),element,'o', \
                         color = colors[k], markersize = size)
                except: pass
            ax.axvspan(-0.25,0.25, color = 'pink', alpha = 0.05)
            ax.set_title('Dish '+str(j+1))
            ax.set_xlim(-2,2)
            ax.set_ylim(-5,5)
            if j == 0 or j ==2:
                ax.set_ylabel('Degrees')
            if j == 2 or j ==3:
                ax.set_xlabel('Degrees')

    patches = [mpatches.Patch(color=colors[i], label=dictelements[i]) for i in range(len(colors))]

    plt.legend(handles=patches, ncol = 2)
    plt.suptitle(flightname+'Centroid Location from 1D Fit Along Passes Thru Beam @indx '+str(freq), y = 0.95, fontsize = 12)

    plt.show()
    
def getcetroidbounds_x(dictname, flight_dict, dish_num):
    x = [round(dictname[flight_dict][0].fulldict['x'][i]) \
         for i in range(len(dictname[flight_dict][0].fulldict['x']))]
    y = [round(dictname[flight_dict][0].fulldict['y'][i]) \
         for i in range(len(dictname[flight_dict][0].fulldict['x']))]

    thetax = [round(np.degrees(dictname[flight_dict][0].fulldict['thetaX_beam'][dish_num][i]),1) \
         for i in range(len(dictname[flight_dict][0].fulldict['x']))]
    thetay = [round(np.degrees(dictname[flight_dict][0].fulldict['thetaY_beam'][dish_num][i]),1) \
         for i in range(len(dictname[flight_dict][0].fulldict['x']))]
    
    ydataatzero = [thetay[i] for i in range(len(thetax)) if -1<thetax[i]<1 and -6<thetay[i]<6]
    test = ydataatzero
    testdict = {}
    for key in test: testdict[key] = 0
    test = list(testdict.keys())

    passes = []
    for i in range(len(test)-1):
        if -0.2<(test[i]-test[i+1])<0.2:
            pass
        else: 
            passes.append(test[i])

    bounds = [(_pass-0.5,_pass+0.5) for _pass in passes]
    return(bounds)

    
def getcentroidmedians_x(dictname, flightname, dish_num):
    bounds = getcetroidbounds_x(dictname, flightname,dish_num)
    medians = []
    for bound in bounds:
        stuff = np.degrees(dictname[flightname][0].fulldict['thetaY_beam'][dish_num])
        stuff_x = np.degrees(dictname[flightname][0].fulldict['thetaX_beam'][dish_num])
        y = [stuff[i] for i in range(len(stuff)) if bound[0]<stuff[i]<bound[1] and -2<stuff_x[i]<2]
        medians.append(np.median(y))
    return(medians)

def plotcentroids_x(dictname, flightname, dictelements):
    fig = plt.figure(figsize = (6,7))
    bounds = getcetroidbounds_x(dictname, flightname,0)
    medians = getcentroidmedians_x(dictname, flightname,0)
    size = 6

    colors = ['blue','orange','pink', 'green', 'red']
    colors = colors[0:len(dictelements)]
    
    for i,element in enumerate(medians):
        for j in range(4):
            ax = fig.add_subplot(2,2,j+1)
            ax.axhline(element)
            for k, key in enumerate(dictelements):
                try: ax.plot(get_x_centroid_loc(dictname[flightname][key].fulldict, j,100,bounds[i]),element,'o', \
                         color = colors[k], markersize = size)
                except: pass
            ax.set_title('Dish '+str(j+1))
            ax.set_xlim(-2,2)
            ax.set_ylim(-5,5)
            if j == 0 or j ==2:
                ax.set_ylabel('Degrees')
            if j == 2 or j ==3:
                ax.set_xlabel('Degrees')

    patches = [mpatches.Patch(color=colors[i], label=dictelements[i]) for i in range(len(colors))]

    plt.legend(handles=patches, ncol = 2)
    plt.suptitle(flightname+'Centroid Location from 1D Fit Along Passes Thru Beam', y = 0.95, fontsize = 12)

    plt.show()