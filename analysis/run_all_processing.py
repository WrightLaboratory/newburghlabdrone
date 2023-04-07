import os, h5py, datetime,pytz,pickle, yaml
import numpy as np
from matplotlib.pyplot import *
from matplotlib import pyplot as plt

## Then import the beamcals module packages and initialize 'gbosite' class:
from beamcals import corr, concat, drone, bicolog
import beamcals.plotting_utils as pu
import beamcals.fitting_utils as fu
import beamcals.geometry_utils as gu
import beamcals.time_utils as tu
from beamcals.sites import site
import beamcals.reduce_ccclass as rc
gbosite=site.site('../beamcals/beamcals/sites/GBO_config.npz')

# set up directories and files, defaults
yamlfile = '/home/ln267/newburghlabdrone/analysis/GBO_flights_forscripts.yaml'
config_directory="/hirax/GBO_Analysis_Outputs/concat_config_files/"

tclb,tcub=[8500,20000]

# open yamlfile of flights and read in relevant info
with open(yamlfile, 'r') as fff:
    documents = yaml.safe_load(fff)
flights = documents["flight_info"]["flights"]
cdats = documents["flight_info"]["cdats"]
tlbs = documents["flight_info"]["tlbs"]
tubs = documents["flight_info"]["tubs"]
fmaxes = documents["flight_info"]["fmaxes"]

# run through Will's tutorial
for f in np.arange(3,4):
    fly = flights[f]
    tub = tubs[f]
    tlb = tlbs[f]
    cdat = cdats[f]
    
    if int(fly) < 537:
        mdir='/hirax/GBO_Aug_2021/TONE_ACQ/'+cdat+'_yale_drone/corr/' # August 2021
        gaindir='/hirax/GBO_Aug_2021/TONE_ACQ/digital_gains/'+cdat+'_yale_drone_yale_digitalgain/' # August 2021
    else:
        mdir='/hirax/GBO_Oct_2021/TONE_ACQ/'+cdat+'_yale_drone/corr/' # Oct 2021
        gaindir='/hirax/GBO_Oct_2021/TONE_ACQ/digital_gains/'+cdat+'_yale_drone_yale_digitalgain/' # Oct 2021

    sdir = os.listdir(mdir)[0]
    datadir=mdir+sdir+'/'
    print(sdir, datadir,gaindir)
    
    dronedir='/hirax/all_drone_data/datcon_csv/'
    dronetest0825=drone.Drone_Data(Drone_Directory=dronedir,FLYTAG='FLY'+fly+'.csv',site_class=gbosite,tlb=tlb,tub=tub)

    print('DONE reading in drone data')
    print(dronetest0825.t_arr_datetime[0], dronetest0825.t_arr_datetime[-1])

    print('start time: ', datetime.datetime.now())
    if str(fmaxes[f]) != 'None':# 
        corrtest0825=corr.Corr_Data(Data_Directory=datadir,
                            Gain_Directory=gaindir,site_class=gbosite,
                            crossmap=[],Data_File_Index=np.arange(0,int(fmaxes[f])))
    else: 
        corrtest0825=corr.Corr_Data(Data_Directory=datadir,
                            Gain_Directory=gaindir,site_class=gbosite,
                            crossmap=[])
    print('end time: ', datetime.datetime.now())
    concattest0825=concat.CONCAT(CORRDATCLASS=corrtest0825,DRONEDATCLASS=dronetest0825,\
                             load_yaml=False,traceback=True,save_traceback=True)
    concattest0825.Extract_Source_Pulses(Period=0.4e6,Dutycycle=0.2e6,t_bounds=[tclb,tcub])
    concattest0825.Perform_Background_Subtraction(window_size=5)
    concattest0825.Synchronization_Function(inputcorr=corrtest0825,inputdrone=dronetest0825,
                                        coarse_params=[-5.0,5.0,0.2],
                                        FMB_coordbounds=[30.0,30.0,150.0],
                                        FMB_ampbound=0.999)
    concattest0825.Export_yaml()   
    concattest0825.Main_Beam_Fitting(fit_param_directory='/hirax/GBO_Analysis_Outputs/main_beam_fits/',
                                 FMB_ampbound=0.999,
                                 Vargs='bgsub')
    thingy = rc.Smallify(concattest0825)
    with open(thingy.tmppath, 'wb') as outp:
        pickle.dump(thingy, outp, pickle.HIGHEST_PROTOCOL)
    print('Done with this flight, and pickled')
    
