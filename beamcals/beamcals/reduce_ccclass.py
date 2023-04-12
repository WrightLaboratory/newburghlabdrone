import datetime

class Smallify:
    def __init__(self,concatclass):

        # basics # grabbed more vars (WT)
        self.name = concatclass.name
        self.chmap = concatclass.chmap
        self.automap = concatclass.automap
        self.crossmap = concatclass.crossmap
        self.Data_Directory = concatclass.Data_Directory
        self.Gain_Directory = concatclass.Gain_Directory
        self.filenames = concatclass.filenames
        self.gainfile = concatclass.gainfile
        self.Drone_Directory = concatclass.Drone_Directory
        self.FLYTAG = concatclass.FLYTAG
        self.freq = concatclass.freq
        self.t_arr_datetime = concatclass.t_arr_datetime
        self.t = concatclass.t
        self.t_delta_dji = concatclass.t_delta_dji
        self.t_delta_pulse = concatclass.t_delta_pulse
        
        # site info #
        self.n_dishes = concatclass.n_dishes
        self.n_channels = concatclass.n_channels
        self.origin = concatclass.origin
        #self.prime_origin = concatclass.prime_origin
        self.dish_keystrings = concatclass.dish_keystrings
        self.dish_coords = concatclass.dish_coords
        self.dish_pointings = concatclass.dish_pointings
        self.dish_polarizations = concatclass.dish_polarizations
       
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
        self.drone_xyz_per_dish_interp = concatclass.drone_xyz_per_dish_interp
        self.drone_yaw_interp = concatclass.drone_yaw_interp

        # RF data # 
        #self.V = concatclass.V
        #self.V_bg = concatclass.V_bg
        self.V_bgsub = concatclass.V_bgsub
        #self.V_cross = concatclass.V_cross
        #self.V_cross_bg = concatclass.V_cross_bg
        #self.V_cross_bgsub = concatclass.V_cross_bgsub

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

