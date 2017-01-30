#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from .problem import OptProblem
from scipy.sparse import tril,coo_matrix

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
        Quadratic program class.
        
        Parameters
        ----------
        H : symmetric matrix
        g : vector
        A : matrix
        l : vector
        u : vector
        x : vector
        """

        OptProblem.__init__(self)

        self.H = coo_matrix(H)
        self.Hphi = tril(self.H) # lower triangular
        self.g = g
        self.A = coo_matrix(A)
        self.b = b
        self.u = u
        self.l = l

        self.x = x
        
        self.lam = lam
        self.mu = mu
        self.pi = pi

        # Check data
        # TODO
        
    def eval(self,x):

        self.phi = 0.5*np.dot(x,self.H*x) + np.dot(self.g,x)
        self.gphi = self.H*x + self.g
        
    def show(self):
        
        print('\nQP Problem')
        print('----------')
        print('H shape : (%d,%d)' %(self.H.shape[0],self.H.shape[1]))
        print('H nnz   : %.2f %%' %(100.*self.H.nnz/(self.H.shape[0]*self.H.shape[1])))
        print('A shape : (%d,%d)' %(self.A.shape[0],self.A.shape[1]))
        print('A nnz   : %.2f %%' %(100.*self.A.nnz/(self.A.shape[0]*self.A.shape[1])))
        
