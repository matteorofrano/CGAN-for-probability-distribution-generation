from sys import stdout as cout
from math import *
import numpy as np



class cir_obj:

    def __init__(self, **kwargs):
        self.kappa = kwargs["kappa"] 
        self.sigma = kwargs["sigma"] 
        self.theta = kwargs["theta"] 
        self.ro    = kwargs["ro"] 
        self.gamma = sqrt( self.kappa * self.kappa + 2 * self.sigma*self.sigma)
    # --------------

    def B(self, t):
        g = self.gamma
        k = self.kappa

        #
        # when g >> 1 we do neglect terms of the 
        # type g*exp(-gt)
        # the situation g >> 1 occurs only when we try to test 
        # very large violation from the Feller condition
        #
        if g > 30: return 2 /(g+k)
        h = np.exp(g*t) - 1
        return 2 * h/( (g+k)*h + 2*g)
    # ------------------------

    def A(self, t):
        g  = self.gamma
        k  = self.kappa
        th = self.theta
        s  = self.sigma
        #
        # when g >> 1 we do neglect terms of the 
        # type g*exp(-gt)
        # the situation g >> 1 occurs only when we try to test 
        # very large violation from the Feller condition
        #
        if g > 30:
            return ( 2*k*th/(s*s) ) * ( log( 2 * g ) + .5 * (k+g)*t - g*t + log(g+k))

        h = np.exp(g*t) - 1
        return ( 2*k*th/(s*s) ) * ( log( 2 * g ) + .5 * (k+g)*t - np.log( (g+k)*h + 2*g) )
    # --------------------------------------

    def P_tT( self, t, r=None):
        '''
        The price process of the zero coupon bond P(t,T)
        The calling interface will be cir.P_tT( T-t_n, r = r_n)
        '''
        if r == None: r = self.ro
        return np.exp( -self.B(t)*r + self.A(t) )

    def feller(self):
        return  self.sigma*self.sigma/(2*self.kappa*self.theta)

    def show(self, fp=cout):
        fp.write("@ %-12s: Feller = %8.4f\n" %("Info", self.feller() ))
        fp.write("@ %-12s: kappa  = %8.4f\n" %("Info", self.kappa ))
        fp.write("@ %-12s: theta  = %8.4f\n" %("Info", self.theta ))
        fp.write("@ %-12s: sigma  = %8.4f\n" %("Info", self.sigma ))
        fp.write("@ %-12s: ro     = %8.4f\n" %("Info", self.ro ))
        fp.write('\n')
        
