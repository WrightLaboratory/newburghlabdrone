import os, h5py, datetime,pytz, yaml
import numpy as np
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import pickle, glob

basedir='/hirax/GBO_Analysis_Outputs/'
ampdir = basedir+'amplitude_corrections/'

with open('/hirax/GBO_Analysis_Outputs/GBO_flights_forscripts.yaml', 'r') as fff:
    documents = yaml.safe_load(fff)
flights = documents["flight_info"]["flights"]

for fly in flights:
    ampls = np.zeros([4,1024,16])
    if fly!='618' and fly!='620':
        ampfiles = np.sort(glob.glob(ampdir+'ampsubfiles/'+'*'+fly+'*'))
        for ffile in ampfiles:
            print('Loading: ',ffile)
            with open(ffile, 'rb') as inp:
                amps = pickle.load(inp)
                noz = np.where(np.nanmean(amps[0,:,:],axis=-1)!=0)[0]
                ampls[:,noz,:] = amps[:,noz,:]
    
        pklfile = ampdir+'FLY'+str(fly)+'_Corrected_amplitudes.pkl'
        print(pklfile)
        with open(pklfile, 'wb') as outp:
            pickle.dump(ampls, outp, pickle.HIGHEST_PROTOCOL)
print('DONE')
