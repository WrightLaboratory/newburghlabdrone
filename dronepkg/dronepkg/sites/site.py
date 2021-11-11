#  _____ _____ _____ _____ _____ ________   __
# /  ___|_   _|_   _|  ___/  ___|| ___ \ \ / /
# \ `--.  | |   | | | |__ \ `--. | |_/ /\ V / 
#  `--. \ | |   | | |  __| `--. \|  __/  \ /  
# /\__/ /_| |_  | | | |___/\__/ /| |     | |  
# \____/ \___/  \_/ \____/\____(_)_|     \_/  
                                                                                
## This is a module that reads in site-specific information and returns it in a convenient class structure
## The formatting is not yet set in stone but the future plan is to read something from a config type file
## Currently, we are doing this:

## Import packages:
import numpy as np
import pygeodesy

## Define a class structure that will contain all of the necessary data to initialize geometry: Load from an NPZ file
class site:
    def __init__(self,NPZ_Filename):
        NPZ_config=np.load(NPZ_Filename)
        self.name=str(NPZ_config['name'])
        self.n_dishes=NPZ_config['n_dishes']
        self.n_channels=NPZ_config['n_channels']
        self.chmap=NPZ_config['chmap']
        self.origin=NPZ_config['origin']
        self.keystrings=NPZ_config['keystrings']
        self.coords=NPZ_config['coords']
        self.pointings=NPZ_config['pointings']
        self.polarizations=NPZ_config['polarizations']