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

## Define a class structure that will contain all of the necessary data to initialize geometry
class site:
    def __init__(self,name,n_dishes,n_channels,chmap,origin,keystrings,coords,pointings,polarizations):
        self.name=name
        self.n_dishes=n_dishes
        self.n_channels=n_channels
        self.chmap=chmap
        self.origin=origin
        self.keystrings=keystrings
        self.coords=coords
        self.pointings=pointings
        self.polarizations=polarizations

## Key, Position, Channel, Polarization, Pointing Variable Assignment:
## RECEIVER INFORMATION is needed to initialize the geometric environments: ##
## Must include some 'key' strings for labelling/identifying receivers: [Pranav's Conventions]
Array_Keys=["Dish_0","Dish_7","Dish_2","Dish_3","Dish_4","Dish_5","Dish_6","Dish_8"]
## Specify Coordinate Origin: #810m about MSL
GBO_HIRAX_DISH_0=pygeodesy.ellipsoidalNvector.LatLon(38.429280, -79.844990, 810).to3llh()
## Must Specify Coordinates and Pointings of receivers in Local Cartesian in an array of shape: n_receivers x 3 ##
Array_Coords=np.array([[0.0, 0.0, 0.0], # "Dish_0" \
                       [-24.215395745765452, -12.605737141316427, 0.0], # "Dish_7" \
                       [-8.403824760877617, 16.143597163843634, 0.0], # "Dish_2" \
                       [-8.071798581921817, -4.2019123804388085, 0.0], # "Dish_3" \
                       [-12.273710962360624, 3.869886201483008, 0.0], # "Dish_4" \
                       [-16.475623342799434, 11.941684783404824, 0.0], # "Dish_5" \
                       [-20.345509544282443, -0.33202617895580033, 0.0], # "Dish_6" \
                       [-28.41730812620426, -4.53393855939461, 0.0]]) # "Dish_8" \
## If pointings are defined by vectors in Local Coordinates (E,N,U) use:
Array_Pointings=np.array([[0,0,1], # +Z (up) in LC \
                          [0,0,1],\
                          [0,0,1],\
                          [0,0,1],\
                          [0,0,1],\
                          [0,0,1],\
                          [0,0,1],\
                          [0,0,1]])
## If pointings are defined by angles from zenith, use:
#Array_Pointings=np.array([RotMat(np.array([0.0,0.0,0.0]))@np.array([0.0,0.0,1.0]),  # e.g. 1 deg of yaw and roll \
#                          RotMat(np.array([0.0,0.0,0.0]))@np.array([0.0,0.0,1.0]),\
#                          RotMat(np.array([0.0,0.0,0.0]))@np.array([0.0,0.0,1.0]),\
#                          RotMat(np.array([0.0,0.0,0.0]))@np.array([0.0,0.0,1.0]),\
#                          RotMat(np.array([0.0,0.0,0.0]))@np.array([0.0,0.0,1.0]),\
#                          RotMat(np.array([0.0,0.0,0.0]))@np.array([0.0,0.0,1.0]),\
#                          RotMat(np.array([0.0,0.0,0.0]))@np.array([0.0,0.0,1.0]),\
#                          RotMat(np.array([0.0,0.0,0.0]))@np.array([0.0,0.0,1.0])])
## Define Array Polarizations (2 per dish) in Local Cartesian E,N,U:
Array_Pols=np.array([[[1,0,0],[0,1,0]], # E,N \
                     [[1,0,0],[0,1,0]], \
                     [[1,0,0],[0,1,0]], \
                     [[1,0,0],[0,1,0]], \
                     [[1,0,0],[0,1,0]], \
                     [[1,0,0],[0,1,0]], \
                     [[1,0,0],[0,1,0]], \
                     [[1,0,0],[0,1,0]]])

## 8 DISH PLOTTING:
GBO8_chmap=np.array([0,1,3,2,4,5,7,6,9,8,10,11,12,13,14,15])
GBO8_automap=np.array([0,  16,  45,  31,  58,  70,  91,  81, 108, 100, 115, 121, 126, 130, 133, 135])