#  __ _ _   _   _                      _   _ _                   
# / _(_) | | | (_)                    | | (_) |                  
#| |_ _| |_| |_ _ _ __   __ _    _   _| |_ _| |___   _ __  _   _ 
#|  _| | __| __| | '_ \ / _` |  | | | | __| | / __| | '_ \| | | |
#| | | | |_| |_| | | | | (_| |  | |_| | |_| | \__ \_| |_) | |_| |
#|_| |_|\__|\__|_|_| |_|\__, |   \__,_|\__|_|_|___(_) .__/ \__, |
#                        __/ |_____                 | |     __/ |
#                       |___/______|                |_|    |___/ 

## 20211110 - WT - creating a new file to contain fit functions...
## relocating functions from drone, concat, etc scripts:

import numpy as np

## DEFN the Gauss Fit function:
def Gauss(x,a,x0,sigma,k):
    return a*np.exp(-(x-x0)**2.0/(2.0*sigma**2.0))+k