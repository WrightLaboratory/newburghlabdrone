import pandas as pd
import numpy as np

def cstreader(file):
    data = open(file).readlines()[2::]
    theta = []
    phi = []
    gainTheta = []
    gainPhi = []
    for row in data:
        theta.append(float(row.rsplit()[0]))
        phi.append(float(row.rsplit()[1]))
        gainTheta.append(float(row.rsplit()[3]))
        gainPhi.append(float(row.rsplit()[5]))
    df = pd.DataFrame({'theta':theta, 'phi':phi, 'gain(theta)':gainTheta, 'gain(phi)':gainPhi})
    return(df)

def Theta_0(df):
    df1 = df.truncate(before=find_indx(df['phi'], 0), after=find_indx(df['phi'], 1)-1)
    return(df1)
def Theta_90(df):
    df1 = df.truncate(before=find_indx(df['phi'], -90), after=find_indx(df['phi'], -89)-1)
    return(df1)

def lintodb(mags):
    mag_=10*np.log10(np.array(mags))
    return(mag_)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def find_indx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return(idx)
 
class variables:
    def __init__(self, df):
        #seems mixed up but also seems to work?
        
        self.e_co = [Theta_90(df)['theta'], Theta_90(df)['gain(phi)']]
        self.e_crp = [Theta_90(df)['theta'], Theta_90(df)['gain(theta)']]
        self.h_co = [Theta_0(df)['theta'], Theta_0(df)['gain(phi)']]
        self.h_crp = [Theta_0(df)['theta'], Theta_0(df)['gain(theta)']]

        self.e_co_angle, self.e_co_gain = self.e_co[0], self.e_co[1]
        self.e_crp_angle, self.e_crp_gain = self.e_crp[0], self.e_crp[1]
        self.h_co_angle, self.h_co_gain = self.h_co[0], self.h_co[1]
        self.h_crp_angle, self.h_crp_gain = self.h_crp[0], self.h_crp[1]

class cst_beam_sims:
    def __init__(self, file):
        df = cstreader(file)
        self.vars = variables(df)
    def plot_CoPol(self):
        plot(self.vars.e_co[0], lintodb(self.vars.e_co[1]), label = 'E co')
        plot(self.vars.h_co[0], lintodb(self.vars.h_co[1]), label = 'H co')
    def plot_Crp(self):
        plot(self.vars.e_crp[0], lintodb(self.vars.e_crp[1]), label = 'E crp')
        plot(self.vars.h_crp[0], lintodb(self.vars.h_crp[1]), label = 'H crp')
    def plot_Hco(self):
        plot(self.vars.h_co[0], lintodb(self.vars.h_co[1]), label = 'H co')
    def plot_Eco(self):
        plot(self.vars.e_co[0], lintodb(self.vars.e_co[1]), label = 'E co')
        