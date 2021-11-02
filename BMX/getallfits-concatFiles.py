from BMX_Classes import *
import pandas as pd
import numpy as np
import pickle 

working_directory=u"/home/tyndall/20200309_Yale_BMX_Data/"
drone_directory=u'/hirax/rf_data/bnl_202003/drone_data/drone_20200312/'

## Pull list of BMX data files:
fileslist=np.sort(glob.glob('*.data'))
idstringlist=np.sort(list(set([fn.split('_yale_')[0] for i,fn in enumerate(fileslist)])))
os.chdir(working_directory)
    
## Pull list of processed drone csv files:
os.chdir(drone_directory)
drone_list=np.sort(glob.glob('*processed.csv'))

os.chdir(working_directory)

file= '/home/erk26/BMX_for_github/concat_flights.txt'
with open(file, 'rb') as handle:
    flights = pickle.loads(handle.read())

fitsDict = {}
for key in flights.keys():
    fitsDict[key] = {}
    for dish in range(4):
        fitsDict[key][dish] = {}
        for freq in range(256):
            try: fitsDict[key][dish][freq] = flights[key].get2Dparams(dish,freq)
            except: pass
            if freq < 10: print(freq)
        print(key+' dish '+str(dish)+' done!')

file = '/home/erk26/BMX_for_github/gaussianfits_concat.txt'    
with open(file, 'wb') as handle:
    pickle.dump(fitsDict, handle)

print('wrote ' + file)