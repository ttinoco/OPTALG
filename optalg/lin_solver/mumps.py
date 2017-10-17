#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from .lin_solver import LinSolver
from scipy.sparse import coo_matrix

class LinSolverMUMPS(LinSolver):
    """
    Linear solver based on MUMPS.
    """

    def __init__(self,prop='unsymmetric'):
        """
        Linear solver based on MUMPS.
        """

        # Import mumps
        from ._mumps import DMumpsContext
        
        # Parent
        LinSolver.__init__(self,prop)

        # Name
        self.name ='mumps'
        
        # Load mumps
        if prop == self.UNSYMMETRIC:
            self.mumps = DMumpsContext(par=1,sym=0)
        elif prop == self.SYMMETRIC:
            self.mumps = DMumpsContext(par=1,sym=2)
        else:
            raise ValueError('invalid property')
            
        # Configure
        self.mumps.set_silent()
        self.mumps.set_icntl(14,200) # % increase of estimated working space

    def analyze(self,A):
        """
        Analyzes structure of A.

        Parameters
        ----------
        
        A : matrix 
           For symmetric systems, should contain only lower diagonal part.
        """

        A = coo_matrix(A)
        
        self.mumps.set_shape(A.shape[0])
        self.mumps.set_centralized_assembled_rows_cols(A.row+1,A.col+1)
        self.mumps.run(job=1)

        self.analyzed = True
        
    def factorize(self,A):
        """
        Factorizes A.

        Parameters
        ----------
        A : matrix
           For symmetric systems, should contain only lower diagonal part.
        """

        A = coo_matrix(A)

        self.mumps.set_centralized_assembled_values(A.data)
        self.mumps.run(job=2)

    def solve(self,b):
        """
        Solves system Ax=b.
        
        Parameters
        ----------
        b : ndarray
        
        Returns
        -------
        x : ndarray
        """

        x = b.copy()
        self.mumps.set_rhs(x)
        self.mumps.run(job=3)

        return x

    def factorize_and_solve(self,A,b):
        """
        Factorizes A and sovles Ax=b.

        Parameters
        ----------
        A : matrix
        b : ndarray

        Returns
        -------
        x : ndarray
        """

        A = coo_matrix(A)

        x = b.copy()
        self.mumps.set_centralized_assembled_values(A.data)
        self.mumps.set_rhs(x)
        self.mumps.run(job=5)

        return x
