#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from .problem import OptProblem
from scipy.sparse import tril,triu,coo_matrix

class QuadProblem(OptProblem):
    """
    Quadratic problem class.
    It represents problem of the form
    
    minimize    (1/2)x^THx + g^Tx
    subject to  Ax = b
                l <= x <= u
    """

    def __init__(self,H,g,A,b,l,u,x=None,lam=None,mu=None,pi=None,problem=None):
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
        problem : OptProblem
        """

        OptProblem.__init__(self)

        if problem is not None:
            problem.eval(problem.x)
            H = problem.Hphi + problem.Hphi.T - triu(problem.Hphi)
            g = problem.gphi - H*problem.x
            A = problem.A
            b = problem.b
            l = problem.l
            u = problem.u
            
        self.H = coo_matrix(H)
        self.Hphi = tril(self.H) # lower triangular
        self.g = g
        self.A = coo_matrix(A)
        self.b = b
        self.u = u
        self.l = l

        self.f = np.zeros(0)
        self.J = coo_matrix((0,H.shape[0]))
        self.H_combined = coo_matrix(H.shape)

        self.x = x
        
        self.lam = lam
        self.mu = mu
        self.pi = pi
        
        # Check data
        assert(H.shape[0] == H.shape[1])
        assert(H.shape[0] == A.shape[1])
        assert(b.size == A.shape[0])
        assert(u.size == l.size)
        if x is not None:
            assert(x.size == H.shape[0])
            assert(x.size == A.shape[1])
            assert(x.size == l.size)
        if lam is not None:
            assert(lam.size == A.shape[0])
        if mu is not None:
            assert(mu.size == u.size)
        if pi is not None:
            assert(pi.size == u.size)
 
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
