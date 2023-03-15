import datetime

class Smallify:
    def __init__(self,concatclass):

        # basics #
        self.chmap = concatclass.chmap
        self.crossmap = concatclass.crossmap
        self.Data_Directory = concatclass.Data_Directory
        self.FLYTAG = concatclass.FLYTAG
        self.freq = concatclass.freq
        self.t_arr_datetime = concatclass.t_arr_datetime
        self.t = concatclass.t
        self.t_delta_dji = concatclass.t_delta_dji
        self.t_delta_pulse = concatclass.t_delta_pulse
       
        # pulsing info #
        self.inds_off = concatclass.inds_off
        self.inds_on = concatclass.inds_on
        self.inds_span = concatclass.inds_span
        self.pulse_dutycycle = concatclass.pulse_dutycycle
        self.pulse_period = concatclass.pulse_period
        self.switch_signal = concatclass.switch_signal
        self.switch_signal_interp = concatclass.switch_signal_interp
        
        # drone #
        self.drone_xyz_LC_interp = concatclass.drone_xyz_LC_interp
        self.drone_yaw_interp = concatclass.drone_yaw_interp

        # RF data # 
        #self.V = concatclass.V
        self.V_bg = concatclass.V_bg
        self.V_bgsub = concatclass.V_bgsub
        #self.V_cross = concatclass.V_cross
        self.V_cross_bg = concatclass.V_cross_bg
        self.V_cross_bgsub = concatclass.V_cross_bgsub

        # fitting
        self.G_popt = concatclass.G_popt

        # text for pickle file name
        tmppickdir='/hirax/GBO_Analysis_Outputs/flight_pickles/'
        tmpcorrdir=self.Data_Directory.split("_yale")[0].split("TONE_ACQ/")[1]
        tmpdronedir=self.FLYTAG.split('.')[0]
        suff=datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        Output_Prefix='{}{}_{}_ver_{}'.format(tmppickdir,tmpdronedir,tmpcorrdir,suff)
        tmppath=Output_Prefix+'_concat.pkl'
        self.tmppath = tmppath
