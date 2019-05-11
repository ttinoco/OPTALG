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

class MixIntLinProblem(OptProblem):
    """
    Mixed integer linear problem class.
    It represents problem of the form
    
    minimize    c^Tx
    subject to  Ax = b
                l <= x <= u
                Px integer
    """

    def __init__(self, c, A, b, l, u, P, x=None):
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

        self.x = x if x is not None else np.zeros(self.n)
        
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
 
    def eval(self, x):

        self.phi = np.dot(self.c,x)
        
    def show(self):
        
        print('\nMILP Problem')
        print('------------')
        print('A shape : (%d,%d)' %(self.A.shape[0],self.A.shape[1]))
        print('A nnz   : %.2f %%' %(100.*self.A.nnz/(self.A.shape[0]*self.A.shape[1])))
        print('integer : %d' %(np.sum(self.P)))

    def write_to_lp_file(self, filename):

        f = open(filename, 'w')

        # Objective
        f.write('Minimize\n')
        f.write(' obj:\n')
        for i in np.where(self.c != 0.)[0]:
            ci = self.c[i]
            if ci > 0:
                pre = '+'
            else:
                pre = '-'
            if np.abs(ci) == 1.:
                f.write('     %s x_%d\n' %(pre, i))
            else:
                f.write('     %s %.10e x_%d\n' %(pre, np.abs(ci), i))

        # Constraints
        f.write('Subject to\n')
        A = self.A.tocsr()
        for i in range(A.shape[0]):
            f.write(' c_%d:\n' %i)
            for k in range(A.indptr[i], A.indptr[i+1]):
                j = A.indices[k]
                d = A.data[k]
                b = self.b[i]
                if d == 0.:
                    continue
                if d > 0:
                    pre = '+'
                else:
                    pre = '-'
                if np.abs(d) == 1.:
                    f.write('     %s x_%d\n' %(pre, j))
                else:
                    f.write('     %s %.10e x_%d\n' %(pre, np.abs(d), j))
            f.write('     = %.10e\n' %b)

        # Bounds
        f.write('Bounds\n')
        for i in range(self.c.size):
            f.write(' %.10e <= x_%d <= %.10e\n' %(self.l[i], i, self.u[i]))

        # General
        f.write('General\n')
        #row = ' '
        for i in np.where(self.P)[0]:
            f.write('x_%d\n' %i)
        f.write('End\n')
        
        f.close()
