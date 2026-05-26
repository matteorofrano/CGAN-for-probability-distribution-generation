from math import *
import numpy as np

'''
A selection of CIR evolution algorithm
    cir_euler  ( rand, cir, tf, dt, N):
    QT_cir_evol( rand, cir, tf, dt, N):
    fe_cir_evol( rand, cir, tf, dt, N):
'''

PSI_c = 1.5
EPS   = 1.e-08

def euler_step(rand, Ro, k, th, vol, dt, N):
        eta  = rand.normal(size=N)
        Rn = Ro + k*(th - Ro)*dt + vol*np.sqrt( Ro*dt)*eta
        return np.maximum(Rn,0.0)
# ---------------------------------------------------------------------

def cir_euler( rand, cir, tf, dt, N):

    '''
    @params rand:  random number generator
    @params cir :  the CIR object
    @parmas tf  :  time schedule of the output
    @params dt  :  step size of the cir trajectory simulation
    @params N   :  number of trajectories
    '''

    Nt = len(tf)
    X  = np.zeros(shape = ( Nt, N ), dtype=np.float64 )
    I  = np.zeros(shape = ( Nt, N ), dtype=np.float64 )

    vol   = cir.sigma
    th  = cir.theta
    k   = cir.kappa
    ro  = cir.ro
    X[0] = ro;

    Rn     = np.full( N, ro, dtype=np.float64)
    In     = np.zeros(N, dtype=np.float64)
    tn     = tf[0]
    p      = 1
    count  = 0

    for tm in tf[1:]:
        t   = tn
        while True:
            if t + dt > tm: break
            Ro = Rn
            Rn = euler_step( rand, Ro, k, th, vol, dt, N)
            In = In + .5*(Rn+Ro)*dt
            t += dt
            count += 1

        if tm - t >= EPS:
            Ro = Rn
            eps = tm-t
            Rn = euler_step( rand, Ro, k, th, vol, eps, N)
            In = In + .5*(Rn+Ro)*eps
            t += eps
            count += 1
        tn = t

        X[p] = Rn
        I[p] = In
        p += 1

    return (X, I) 
# ----------------------------------------------------------

def QT_step(rand, Ro, k, th, vol, dt, h, N):

    xi   = rand.normal( loc = 0.0, scale = 1.0, size=N)
    Zero = ( Ro == 0.0 )
    Rn   = np.where(Zero,k*th*dt, 0.0)

    #h   = 1. - exp(-k*dt)
    m   = th + ( Ro - th)*(1. - h)
    s2  = (vol*vol*h/k)*( Ro * (1. - h ) + .5*th*h )
    PSI = s2/(m*m)

    #
    #
    #
    Mask   = np.logical_and( PSI > PSI_c, ~Zero )
    u      = rand.uniform(low=0.0, high=1.0, size = N)
    p      = (PSI-1)/(PSI+1)
    opMask = np.logical_and( u > p, Mask == 1 )
    beta   = (1. - p)/m
    x      = np.where(opMask, np.log( (1-p)/(1-u))/beta, 0.0)

    Mask   = np.logical_and( PSI <= PSI_c, ~Zero )
    o      = np.where( Mask, 2/PSI - 1., 0.0)
    b2     = np.where(Mask, o + np.sqrt(o*(o+1)), 0.0) 
    a      = m/(1. + b2)
    c      = np.power( (np.sqrt(b2)+ xi), 2, where=Mask)
    y      = np.where(Mask, a*c, 0)
    return Rn + ( x + y )



##
## From: "Efficient Simulation of the Heston Stochastic Volatility Model" by Leif Andersen
## the algorithm here implementd is denoted as 'QE' in the paper
##
def QT_cir_evol( rand, cir, tf, dt, N):

    Nt = len(tf)
    X  = np.ndarray(shape = ( Nt, N ), dtype=np.float64 )
    I  = np.ndarray(shape = ( Nt, N ), dtype=np.float64 )


    vol = cir.sigma
    th  = cir.theta
    k   = cir.kappa
    ro  = cir.ro

    X[0]   = ro
    I[0]   = 0.0
    Rn     = np.full( N, ro, dtype=np.float64)
    In     = np.zeros(N, dtype=np.float64)
    tn     = tf[0]
    p      = 1
    count  = 0

    for tm in tf[1:]:
        t   = tn
        h   = 1. - exp(-k*dt)
        while True:
            if t + dt > tm: break
            Ro = Rn
            Rn = QT_step( rand, Ro, k, th, vol, dt, h, N)
            In = In + .5*(Rn+Ro)*dt
            t += dt
            count += 1

        if tm - t >= EPS:
            Ro = Rn
            eps = tm-t
            h   = 1. - exp(-k*eps)
            Rn = QT_step( rand, Ro, k, th, vol, eps, h, N)
            In = In + .5*(Rn+Ro)*eps
            t += eps
            count += 1
        tn = t

        X[p] = Rn
        I[p] = In
        p += 1

    return (X, I) 
# =======================================================================================


def fe_step( rand, df, M2, N):
    '''
    df: degrees of freedom
    M2: non central parameter
    '''
    Rn   = rand.chisquare( df, N)
    U    = rand.uniform(size=N)
    V    = M2 + 2*np.log(U)
    mask = ( V > 0 )
    V    = np.where(mask, V, 0.)
    eta  = rand.normal(size=2*N)
    Rn   = Rn + np.where( mask, (eta[0:N] + np.sqrt( V ) )**2 + eta[N:]**2, 0)

    #Rn = rho*Rn
    return Rn

def fe_cir_evol( rand, cir, tf, dt, N):

    '''
    fast and exact CIR evolution

    @params rand: random number generator
    @params cir :  the CIR object
    @params dt  :  step size of the cir trajectory simulation
    @params N   :  number of trajectories

    @return X, I where X_n = r(t_n) is the array of short rate, 
                       I_n = \int_0^{t_n} r(s) ds is the short rate integral
    '''
    EPS = 1.e-08
    Nt  = len(tf)

    
    X  = np.zeros(shape = ( Nt, N ), dtype=np.float64 )
    I  = np.zeros(shape = ( Nt, N ), dtype=np.float64 )

    s    = cir.sigma
    th   = cir.theta
    k    = cir.kappa
    ro   = cir.ro
    X[0] = ro

    # degrees of freedom
    df     = (4*k*th)/(s*s)

    Rn     = np.full( N, ro, dtype=np.float64)
    In     = np.zeros(N, dtype=np.float64)
    tn     = tf[0]
    p      = 1

    for tm in tf[1:]:

        rho = ((s*s)/(4*k))*(1 - exp(-k*dt))
        M2  = exp(-k*dt)/rho
        t   = tn
        while True:
            if t + dt > tm: break
            Ro = Rn
            Rn = rho*fe_step( rand, df, Ro*M2, N)
            In = In + .5*(Rn+Ro)*dt
            t += dt

        if tm - t >= EPS:
            Ro = Rn
            eps = tm-t
            rho = ((s*s)/(4*k))*(1 - exp(-k*eps))
            M2  = exp(-k*eps)/rho
            Rn = rho*fe_step( rand, df, Ro*M2, N)
            In = In + .5*(Rn+Ro)*eps
            t += eps
        tn = t

        X[p] = Rn
        I[p] = In
        p += 1

    return (X, I) 
# ----------------------------------------------------------

'''
Rappers do provide a different interface
where we assume e constant separated schedule
'''
def fe_wrapper( rand, cir, L, dt, Nt, DT, N):
    tf = np.arange( 0.0, Nt*DT+DT/2, DT)
    return fe_cir_evol( rand, cir, tf, dt, N)
#
def euler_wrapper( rand, cir, L, dt, Nt, DT, N):
    tf = np.arange( 0.0, Nt*DT+DT/2, DT)
    return cir_euler( rand, cir, tf, dt, N)

def QT_wrapper( rand, cir, L, dt, Nt, DT, N):
    tf = np.arange( 0.0, Nt*DT+DT/2, DT)
    return QT_cir_evol( rand, cir, tf, dt, N)
# ----------------------------------------------------------
# ----------------------------------------------------------
