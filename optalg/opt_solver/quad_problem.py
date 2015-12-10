#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from scipy.sparse import tril
from opt_problem import OptProblem

class QuadProblem(OptProblem):
    """
    Quadratic problem class.
    It represents problem of the form
    
    minimize    (1/2)x^THx + g^Tx
    subject to  Ax = b
                l <= x <= u
    """

    def __init__(self,H,g,A,b,l,u,x=None,lam=None,mu=None,pi=None):
        """
        Class constructor.
        
        Parameters
        ----------
        H : symmetric sparse matrix
        g : vector
        A : sparse matrix
        l : vector
        u : vector
        x : vector
        """

        OptProblem.__init__(self)

        self.Hphi = tril(H) # lower triangular
        self.H = H
        self.g = g
        self.A = A
        self.b = b
        self.u = u
        self.l = l

        self.x = x
        
        self.lam = lam
        self.mu = mu
        self.pi = pi
        
    def eval(self,x):

        self.phi = 0.5*np.dot(x,self.H*x) + np.dot(self.g,x)
        self.gphi = self.H*x + self.g
        
        
