
#!/usr/bin/python
"""
Python module for the calculation of the Atomic Limit
functions created by Dominik Robert Fus
"""
__author__='Dominik Robert Fus'
import numpy as np
import matplotlib as mpl
import scipy as sp

def N(U, beta, mu):
    n = np.where(
        #if
        mu>9*U/2,
            #then
            (np.exp(-mu*beta)+np.exp(-U*beta))/(np.exp(-2*mu*beta)+2*np.exp(-mu*beta)+np.exp(-U*beta))
            #elif
            ,np.where(mu<-9*U/2,
                #then
                (np.exp(mu*beta)+np.exp((2*mu-U)*beta))/(1+2*np.exp(mu*beta)+np.exp((2*mu-U)*beta))
                #else
                ,(1+np.exp((mu-U)*beta))/(np.exp(-mu*beta)+2+np.exp((mu-U)*beta)) # times exp(-mu*beta)
    ))
    return n

def g(U, beta, mu, Niwf):
    size = 2*Niwf
    n=N(U, beta, mu)
    #z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta))
    #n=(1+np.exp((mu-U)*beta))/z  #time np.exp(-mu*beta)

    iw =1j*np.pi/beta*(2*np.arange(-Niwf,Niwf)+1)
    return n/(iw+mu-U) + (1-n)/(iw+mu)

def chi_c(U, beta, mu, Niwf):
    size = 2*Niwf
    z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta)) # times exp(-mu*beta)
    n=N(U, beta, mu)
    #n=(1+np.exp((mu-U)*beta))/z

    iw =1j*np.pi/beta*(2*np.arange(-Niwf,Niwf)+1)
    nu = iw[:,None]
    nup =  iw[None,:]

    nu_m  = nu+mu
    nu_mU = nu+mu-U

    np_m  = nup+mu
    np_mU = nup+mu-U
    delta = np.eye(size)

    if mu == U/2:
        quot = delta[:,::-1]*beta/2*1/(1+np.exp(beta*U/2))
    else:
        quot = (2*n-1)/(nu+nup+2*mu-U)


    return (-beta*((1-n)/nu_m + n/nu_mU)**2*delta \
        + beta * U**2 * n*(1-n) * (1-delta)/(nu_m*nu_mU*np_m*np_mU) \
        + quot *(1/nu_mU+1/np_mU)**2 \
        - beta * 1/z * delta * (1/nu_m - 1/nu_mU)**2 \
        + beta * U**2 * (np.exp(-U*beta)-1)/z**2 * 1/(nu_m*nu_mU*np_m*np_mU) \
        + (n-1)/(nu_m**2 * np_mU) + (1-n)/(nu_m*np_mU**2) \
        + 2*((1-n)/(nu_mU * np_m * np_mU) + (n-1)/(nu_m * np_m * np_mU)) \
        + (1-n)/(nu_m**2*np_m) + (1-n)/(nu_m*np_m**2) \
        + (1-n)/(nu_mU**2*np_m) + (n-1)/(nu_mU*np_m**2) \
        - n/(nu_mU**2*np_mU) - n/(nu_mU*np_mU**2))/beta

# plotting helper:
# -------------------------------------
# colormap inspired by Patrick Chalupa
# -------------------------------------
cdict_white = {'blue':  [[0.0, 0.6, 0.6],
                   [0.499, 1.0, 1.0],
                   #[0.5, 0.0, 0.0],
                   [0.5, 1.0, 1.0],
                   [0.501, 0.0, 0.0],
                   [1.0, 0., 0.]],
         'green': [[0.0, 0.0, 0.0],
                   [0.02631578947368421, 7.673360394717657e-06, 7.673360394717657e-06],
                   [0.05263157894736842, 0.00012277376631548252, 0.00012277376631548252],
                   [0.07894736842105263, 0.0006215421919721302, 0.0006215421919721302],
                   [0.10526315789473684, 0.0019643802610477203, 0.0019643802610477203],
                   [0.13157894736842105, 0.004795850246698536, 0.004795850246698536],
                   [0.15789473684210525, 0.009944675071554084, 0.009944675071554084],
                   [0.18421052631578946, 0.018423738307717093, 0.018423738307717093],
                   [0.21052631578947367, 0.031430084176763524, 0.031430084176763524],
                   [0.23684210526315788, 0.050344917549742546, 0.050344917549742546],
                   [0.2631578947368421, 0.07673360394717657, 0.07673360394717657],
                   [0.2894736842105263, 0.11234566953906126, 0.11234566953906126],
                   [0.3157894736842105, 0.15911480114486534, 0.15911480114486534],
                   [0.3421052631578947, 0.21915884623353094, 0.21915884623353094],
                   [0.3684210526315789, 0.2947798129234735, 0.2947798129234735],
                   [0.39473684210526316, 0.3884638699825815, 0.3884638699825815],
                   [0.42105263157894735, 0.5028813468282164, 0.5028813468282164],
                   [0.4473684210526315, 0.6408867335272133, 0.6408867335272133],
                   [0.47368421052631576, 0.8055186807958807, 0.8055186807958807],
                   [0.499, 1.0, 1.0],
                   #[0.5, 0.0, 0.0],
                   [0.5, 1.0, 1.0],
                   [0.501, 1.0, 1.0],
                   [0.5263157894736843, 0.8055186807958807, 0.8055186807958807],
                   [0.5526315789473685, 0.6408867335272133, 0.6408867335272133],
                   [0.5789473684210527, 0.5028813468282164, 0.5028813468282164],
                   [0.6052631578947368, 0.3884638699825815, 0.3884638699825815],
                   [0.631578947368421, 0.2947798129234735, 0.2947798129234735],
                   [0.6578947368421053, 0.21915884623353094, 0.21915884623353094],
                   [0.6842105263157895, 0.15911480114486534, 0.15911480114486534],
                   [0.7105263157894737, 0.11234566953906126, 0.11234566953906126],
                   [0.736842105263158, 0.07673360394717657, 0.07673360394717657],
                   [0.7631578947368421, 0.050344917549742546, 0.050344917549742546],
                   [0.7894736842105263, 0.031430084176763524, 0.031430084176763524],
                   [0.8157894736842105, 0.018423738307717093, 0.018423738307717093],
                   [0.8421052631578947, 0.009944675071554084, 0.009944675071554084],
                   [0.868421052631579, 0.004795850246698536, 0.004795850246698536],
                   [0.8947368421052632, 0.0019643802610477203, 0.0019643802610477203],
                   [0.9210526315789473, 0.0006215421919721302, 0.0006215421919721302],
                   [0.9473684210526316, 0.00012277376631548252, 0.00012277376631548252],
                   [0.9736842105263158, 7.673360394717657e-06, 7.673360394717657e-06],
                   [1.0, 0.0, 0.0]],
         'red':   [[0.0, 0., 0.],
                   [0.499, 0.0, 0.0],
                   [0.5, 1.0, 1.0],
                   #[0.5, 0.0, 0.0],
                   [0.501, 1.0, 1.0],
                   [1.0, 0.6, 0.6]]}

cmap_w = mpl.colors.LinearSegmentedColormap('chalupa_white',segmentdata = cdict_white,N=10000)

# ---------------------------------------
# normalized colormap from stackoverflow
# ---------------------------------------
class norm(mpl.colors.Normalize):
    def __init__(self, matrix, midpoint=0, clip=False):
        vmin = np.amin(matrix)
        vmax = np.amax(matrix)
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if self.vmax == 0:
            normalized_min = 0
        else:
            normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        if self.vmin == 0:
            normalized_max = 1
        else:
            normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))
