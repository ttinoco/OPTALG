#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from .problem import OptProblem
from scipy.sparse import coo_matrix

class MixIntLinProblem(OptProblem):
    """
    Mixed integer linear problem class.
    It represents problem of the form
    
    minimize    c^Tx
    subject to  Ax = b
                l <= x <= u
                Px integer
    """

    def __init__(self,c,A,b,l,u,P,x=None):
        """
        Mixed integer linear program class.
        
        Parameters
        ----------
        c : vector
        A : matrix
        l : vector
        u : vector
        P : boolean array
        """

        OptProblem.__init__(self)

        self.c = c
        self.A = coo_matrix(A)
        self.b = b
        self.u = u
        self.l = l
        self.P = P

        self.n = self.get_num_primal_variables()

        self.f = np.zeros(0)
        self.J = coo_matrix((0,self.n))
        self.H_combined = coo_matrix((self.n,self.n))
        self.Hphi = coo_matrix((self.n,self.n))
        self.gphi = self.c

        self.x = x
        
        # Check data
        assert(c.size == self.n)
        assert(c.size == A.shape[1])
        assert(b.size == A.shape[0])
        assert(u.size == l.size)
        assert(u.size == c.size)
        assert(P.size == c.size)
        assert(P.dtype == 'bool')
        if x is not None:
            assert(x.size == A.shape[1])
 
    def eval(self,x):

        self.phi = np.dot(self.c,x)
        
    def show(self):
        
        print('\nMILP Problem')
        print('------------')
        print('A shape : (%d,%d)' %(self.A.shape[0],self.A.shape[1]))
        print('A nnz   : %.2f %%' %(100.*self.A.nnz/(self.A.shape[0]*self.A.shape[1])))
        print('integer : %d' %(np.sum(self.P)))
