import datetime

class Smallify:
    def __init__(self,concatclass,pickle_directory='/hirax/GBO_Analysis_Outputs/flight_pickles/',include_cross_data=False):

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
        
        # WT 20230613 - Include binary flag/parse options for cross-correlations:
        if include_cross_data==True:
            self.V_cross_bgsub = concatclass.V_cross_bgsub
        elif include_cross_data==False:
            pass          

        # text for pickle file name
        tmppickdir=pickle_directory
        if 'TONE_ACQ' in self.Data_Directory:
            tmpcorrdir=self.Data_Directory.split("_yale")[0].split("TONE_ACQ/")[1]
        elif 'NFandFF' in self.Data_Directory:
            tmpcorrdir=self.Data_Directory.split("_Suit")[0].split("NFandFF/")[1]
        #tmpcorrdir=self.Data_Directory.split("_yale")[0].split("TONE_ACQ/")[1]
        tmpdronedir=self.FLYTAG.split('.')[0]
        suff=datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        Output_Prefix='{}{}_{}_ver_{}'.format(tmppickdir,tmpdronedir,tmpcorrdir,suff)
        tmppath=Output_Prefix+'_concat.pkl'
        self.tmppath = tmppath



class Smallify_comap():
    def __init__(self,beamclass):
        self.concat_list=beamclass.concat_list        
        self.name=beamclass.name
        self.Data_Directory=beamclass.Data_Directory
        self.Gain_Directory=beamclass.Gain_Directory
        self.filenames=beamclass.filenames
        self.gainfile=beamclass.gainfile
        self.Drone_Directory=beamclass.Drone_Directory
        self.FLYTAG=beamclass.FLYTAG
        self.n_dishes=beamclass.n_dishes
        self.n_channels=beamclass.n_channels
        self.n_concats=beamclass.n_concats
        self.chmap=beamclass.chmap
        self.automap=beamclass.automap
        self.crossmap=beamclass.crossmap
        self.origin=beamclass.origin
        self.dish_keystrings=beamclass.dish_keystrings
        self.dish_coords=beamclass.dish_coords
        self.dish_pointings=beamclass.dish_pointings
        self.dish_polarizations=beamclass.dish_polarizations
        self.fmin,self.fmax,self.fstep=beamclass.fmin,beamclass.fmax,beamclass.fstep
        self.faxis=beamclass.faxis
        self.freq=beamclass.freq
        self.n_freqs = beamclass.n_freqs
        self.normalization = beamclass.normalization
        self.gfit_directory = beamclass.gfit_directory
        self.ampcorr_directory = beamclass.ampcorr_directory
        
        #create x,y cartesian vectors (edges and centers) and grids for the beammap:
        self.operation=beamclass.operation
        
        #need to extend this to dimensionality of channels in concatclass.V
        self.x_edges=beamclass.x_edges
        self.y_edges=beamclass.y_edges
        self.x_edges_grid=beamclass.x_edges_grid
        self.y_edges_grid=beamclass.y_edges_grid
        self.x_centers=beamclass.x_centers
        self.y_centers=beamclass.y_centers
        self.x_centers_grid=beamclass.x_centers_grid
        self.y_centers_grid=beamclass.y_centers_grid
            
        ## now need frequency dependent offset terms in shape (freq, channel, concat) to mimic V
        self.x_offsets=beamclass.x_offsets
        self.y_offsets=beamclass.y_offsets
        
        ## create arrays for V mean, V std, and histo: shape is (gridx, gridy, freq, chans, concatlist)
        self.V_LC_sum=beamclass.V_LC_sum
        self.V_LC_count=beamclass.V_LC_count

        ## define inner and outer masks:
        self.maskin = beamclass.maskin
        self.maskout = beamclass.maskout    
        
        self.V_LC_operation_count = beamclass.V_LC_operation_count
        self.V_LC_operation = beamclass.V_LC_operation
                        
        if self.operation=='std':
            # unfortunately, repeat the above.... assume you're only doing this with pickle files, normalized
            self.V_LC_err= beamclass.V_LC_err
            
            self.V_LC_operation_err = beamclass.V_LC_operation_err
    
