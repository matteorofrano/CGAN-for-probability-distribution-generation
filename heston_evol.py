#!/usr/bin/env python3

from math import *
import numpy as np
from time import time
try:
    from cir_obj import cir_obj
    from cir_evol import cir_euler, QT_cir_evol, fe_cir_evol
except ModuleNotFoundError:
    from CFLib.cir_obj import cir_obj
    from CFLib.cir_evol import cir_euler, QT_cir_evol, fe_cir_evol
# -----------------------------------------------------

def mc_heston( rand, vol, intVol, cir, rho, tf  ):

    '''
    @parms intVol: volatility integral trajectory
    @parms cir   : CIR object
    @parms rho   : correlation between vol and underlying innovations
    @parms tf    : schedule of the underlying trajectory
    @parms N     : number of underlying trajectories
    '''

    # length of the volatility trajectory
    # (including initial point)
    Nt  = intVol.shape[0]

    # Number of trajectories ...
    N   = intVol.shape[1]

    th  = cir.theta
    k   = cir.kappa
    eta = cir.sigma
    nu  = vol
    I   = intVol

    # underlying trajectorie
    S  = np.zeros((Nt, N), dtype=np.float64 ) # S[N, L] in fortran matrix notation

    xi = rand.normal( loc = 0.0, scale = 1.0, size=(Nt-1, N))

    # prime with So the starting value of each trajectory
    S[0] = 1.0

    for n in range(1,Nt):
        DI   = I[n] - I[n-1]
        Dt   = tf[n]-tf[n-1]
        X    = -.5 * DI + (rho/eta)*( nu[n] - nu[n-1] - k*( th*Dt - DI) ) + np.sqrt((1. - rho*rho)*DI)*xi[n-1]
        try:
            S[n] = S[n-1]*np.exp(X)
        except ValueError as e:
           print(f"S[n]: {S[n].shape}, X: {X.shape}") 
           raise e

    return S

# ----------------------------------------------------

def heston_trj    ( rand
                  , heston
                  , tf     # SChedule for the output result
                  , dt     # step per vol inegration
                  , NV     # number of vol trajectories
                  ):

    cir = heston.cir
    rho = heston.rho
    #
    # Computes NV Cir trajectories
    # vol and Ivol have the geometry ( Nt+1, NV)
    # vol[n] = r(t_n)
    # Ivol[n] = \int_0^{t_n} r(s) ds
    #
    vol, Ivol = QT_cir_evol( rand, cir, tf, dt, NV)
    #vol = vol.T
    #Ivol = Ivol.T

    #S = np.zeros( shape=(len(tf),NV) )
    #for n in range(NV):
    #    s = mc_heston( rand, vol[n], Ivol[n], cir, rho, tf )
    #    S[:,n,:] = s
    S = mc_heston( rand, vol, Ivol, cir, rho, tf )
    
    return S
