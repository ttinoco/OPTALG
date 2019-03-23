#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2019, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from .problem import OptProblem
from scipy.sparse import coo_matrix

class LinProblem(OptProblem):
    """
    Linear problem class.
    It represents problem of the form
    
    minimize    c^Tx
    subject to  Ax = b
                l <= x <= u
    """

    def __init__(self, c, A, b, l, u, x=None, lam=None, mu=None, pi=None):
        """
        Linear program class.
        
        Parameters
        ----------
        c : vector
        A : matrix
        l : vector
        u : vector
        x : vector
        """

        OptProblem.__init__(self)

        self.c = c
        self.A = coo_matrix(A)
        self.b = b
        self.u = u
        self.l = l

        self.n = self.get_num_primal_variables()

        self.P = np.zeros(self.n, dtype=bool)

        self.f = np.zeros(0)
        self.J = coo_matrix((0,self.n))
        self.H_combined = coo_matrix((self.n,self.n))
        self.Hphi = coo_matrix((self.n,self.n))
        self.gphi = self.c

        self.x = x if x is not None else np.zeros(self.n)
        
        self.lam = lam
        self.mu = mu
        self.pi = pi

        # Check data
        assert(c.size == self.n)
        assert(c.size == A.shape[1])
        assert(b.size == A.shape[0])
        assert(u.size == l.size)
        assert(u.size == c.size)
        if x is not None:
            assert(x.size == A.shape[1])
        if lam is not None:
            assert(lam.size == A.shape[0])
        if mu is not None:
            assert(mu.size == u.size)
        if pi is not None:
            assert(pi.size == u.size)
 
    def eval(self,x):

        self.phi = np.dot(self.c,x)
        
    def show(self):
        
        print('\nLP Problem')
        print('----------')
        print('A shape : (%d,%d)' %(self.A.shape[0],self.A.shape[1]))
        print('A nnz   : %.2f %%' %(100.*self.A.nnz/(self.A.shape[0]*self.A.shape[1])))

    def write_to_lp_file(self, filename):

        p = self.to_mixintlin()
        p.write_to_lp_file(filename)
