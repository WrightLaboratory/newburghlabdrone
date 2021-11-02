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


file = '/home/erk26/BMX_for_github/concat_flights.txt'

keys = list(np.array(file_organizer['Thursday'])[2:-1][:,0])

###### MAKE Raw Data DICT #########
# flightDict = {}
# for j in np.arange(2,len(file_organizer['Thursday'])-1):
#     print('starting write for '+file_organizer['Thursday'][j][0])
#     flightDict[file_organizer['Thursday'][j][0]] = Do_Everything_For_Data(drone_directory, file_organizer['Thursday'][j][0], working_directory, file_organizer['Thursday'][j][1])


####### MAKE Offset Data Dict ##########    
# offsets = {}
# off_elements = [-0.47, 0.63, 0.34, 0.39, 0.39, 0.59, 0.38]
# keys = list(np.array(file_organizer['Thursday'])[2:-1][:,0])
# for i, key in enumerate(keys):
#     offsets[key] = off_elements[i]
# flightDict = {}
# for j in np.arange(2,len(file_organizer['Thursday'])-1):
#     print('starting write for '+file_organizer['Thursday'][j][0])
#     flightDict[file_organizer['Thursday'][j][0]] = Do_Everything_For_Data(drone_directory, file_organizer['Thursday'][j][0], working_directory, file_organizer['Thursday'][j][1], offsets[file_organizer['Thursday'][j][0]])

##### MAKE Wed Dict ####
# flightDict = {}
# for i in range(len(file_organizer['Wednesday'])):
#     print(file_organizer['Wednesday'][i][0])
#     flightDict[i] = Do_Everything_For_Data_dual('/hirax/rf_data/bnl_202003/drone_data/drone_20200311/', file_organizer['Wednesday'][i][0], working_directory, file_organizer['Wednesday'][i][1])
    
#### MAKE Concat Flight Dict ####
flightDict = {}
offsets = {}
off_elements = [-0.47, 0.63, 0.34, 0.39, 0.39, 0.59, 0.38]
keys = list(np.array(file_organizer['Thursday'])[2:-1][:,0])
for i, key in enumerate(keys):
    offsets[key] = off_elements[i]

for i in np.arange(2, len(file_organizer['Thursday'])-2, 2):
    print(file_organizer['Thursday'][i][0]+file_organizer['Thursday'][i+1][0])
    flightDict[file_organizer['Thursday'][i][0]+file_organizer['Thursday'][i+1][0]] = concat_files_Do_Everything_For_Data(drone_directory, file_organizer['Thursday'][i][0], file_organizer['Thursday'][i+1][0], working_directory, file_organizer['Thursday'][i][1], file_organizer['Thursday'][i+1][1])

with open(file, 'wb') as handle:
    pickle.dump(flightDict, handle)

print('wrote ' + file)