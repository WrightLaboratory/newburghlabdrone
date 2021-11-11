# _   _                       _   _ _                   
#| | (_)                     | | (_) |                  
#| |_ _ _ __ ___   ___  _   _| |_ _| |___   _ __  _   _ 
#| __| | '_ ` _ \ / _ \| | | | __| | / __| | '_ \| | | |
#| |_| | | | | | |  __/| |_| | |_| | \__ \_| |_) | |_| |
# \__|_|_| |_| |_|\___| \__,_|\__|_|_|___(_) .__/ \__, |
#                   ______                 | |     __/ |
#                  |______|                |_|    |___/ 

## 20211110 - WT - creating a new file to contain time fitting functions:

import datetime
from scipy.signal import square
from scipy.stats import pearsonr
import numpy as np
import pandas

## Annie's function for fixing the time axis:
    # (9/28/2021) function for adding sub-second accuracy to DJI timestamps
    # now detects and eliminates >1s errors
def interp_time(df_in):
    # find where the GPS turns on
    gps_idx = df_in[df_in.gpsUsed == True].index[0]
    # interpolate the time and see if it works out!
    while (gps_idx < len(df_in)):
        # look for where the datetimestamp ticks
        first_dts = df_in["GPS:dateTimeStamp"][gps_idx]
        start_sec = int(first_dts[-3:-1])
        while(int(df_in["GPS:dateTimeStamp"][gps_idx][-3:-1]) == start_sec):
            gps_idx = gps_idx + 1
        # use this reference timestamp to convert the offsetTime column into proper datetimes
        start_dt = pandas.to_datetime(df_in["GPS:dateTimeStamp"][gps_idx])
        offsets = np.array(df_in["offsetTime"]-df_in["offsetTime"][gps_idx])
        offsets = pandas.to_timedelta(offsets, unit='s')
        timestamps = start_dt + offsets
        # put them in the dataframe
        df_in = df_in.assign(timestamp = timestamps)
        df_in = df_in.assign(UTC = timestamps)
        # check for excessive error by comparing the interpolated and uninterpolated timestamp columns
        gps_dts = pandas.to_datetime(df_in["GPS:dateTimeStamp"][gps_idx:-20]).values
        interp_dts = pandas.to_datetime(df_in["timestamp"][gps_idx:-20]).values
        if (np.mean(np.abs(gps_dts - interp_dts)/np.timedelta64(1,'ms')) < 1000):
            print("Timestamp interpolation succeeded")
            break
        else:
            print("Detected >1s error, retrying")
            gps_idx += 10 # increment the start timestamp index by an arbitrary amount and retry
    return df_in

## source switching signal functions:

def Pulsed_Data_Waveform(total_duration,period,duty_cycle_on):
    ## Outputs should be an array of timedeltas and an array of switch voltages (1s and 0s)
    ## Let's make the time resolution of these arrays milliseconds (10^-3 sec):
    t_steps_ms=int(datetime.timedelta(seconds=total_duration).total_seconds()*1e3)+1 #n_steps
    t_arr_s=np.linspace(0,total_duration,t_steps_ms)
    ## Use the square function from scipy.signal to produce the 1s and 0s:
    switch_signal_arr=0.5*square((2*np.pi/(period*1e-6))*t_arr_s,duty_cycle_on/period)+0.5
    ## Create a timedelta array for interpolation purposes so we can interpolate the square wave later:
    t_arr_datetime=np.array([datetime.timedelta(seconds=timeval) for timeval in t_arr_s])
    return t_arr_s,t_arr_datetime,switch_signal_arr