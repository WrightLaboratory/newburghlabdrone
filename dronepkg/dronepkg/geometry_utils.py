#                                 _                       _   _ _                   
#                                | |                     | | (_) |                  
#  __ _  ___  ___  _ __ ___   ___| |_ _ __ _   _    _   _| |_ _| |___   _ __  _   _ 
# / _` |/ _ \/ _ \| '_ ` _ \ / _ \ __| '__| | | |  | | | | __| | / __| | '_ \| | | |
#| (_| |  __/ (_) | | | | | |  __/ |_| |  | |_| |  | |_| | |_| | \__ \_| |_) | |_| |
# \__, |\___|\___/|_| |_| |_|\___|\__|_|   \__, |   \__,_|\__|_|_|___(_) .__/ \__, |
#  __/ |                                    __/ |_____                 | |     __/ |
# |___/                                    |___/______|                |_|    |___/ 
    
## 20211110 - WT - creating this file during the refactor, going to shuffle several functions around
## This is a .py file for geometric tools like conversion functions:

import numpy as np

## Rotation Matrix for Yaw,Pitch,Roll rotations about z,y,x axes:
def rot_mat(ypr_arr):
    [a,b,c]=(np.pi/180.0)*ypr_arr
    RM=np.ndarray((3,3))
    RM[0,:]=[np.cos(a)*np.cos(b),np.cos(a)*np.sin(b)*np.sin(c)-np.sin(a)*np.cos(c),np.cos(a)*np.sin(b)*np.cos(c)+np.sin(a)*np.sin(c)]
    RM[1,:]=[np.sin(a)*np.cos(b),np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c),np.sin(a)*np.sin(b)*np.cos(c)-np.cos(a)*np.sin(c)]
    RM[2,:]=[-1*np.sin(b),np.cos(b)*np.sin(c),np.cos(b)*np.cos(c)]
    return RM

## Convert from cartesian to polar (r,phi,theta):
def xyz_to_rpt(xyz):
    r_prime=np.sqrt(xyz[0]**2.0+xyz[1]**2.0+xyz[2]**2.0)
    phi_prime=np.arctan2(xyz[1],xyz[0])
    if phi_prime<0:
        phi_prime=phi_prime+(2.0*np.pi)
    theta_prime=np.arccos(xyz[2]/r_prime)
    rpt=[r_prime,phi_prime,theta_prime]
    return rpt