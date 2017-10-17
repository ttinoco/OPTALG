#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        # 
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from .lin_solver import LinSolver
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix,triu

class LinSolverSUPERLU(LinSolver):
    """
    Linear solver based on SuperLU.
    """


    def __init__(self,prop='unsymmetric'):
        """    
        Linear solver based on SuperLU.
        """

        # Parent
        LinSolver.__init__(self,prop)

        # Name
        self.name = 'superlu'
        
        # Factorization
        self.lu = None
                
    def factorize(self,A):
        """
        Factorizes A.

        Parameters
        ----------
        A : matrix
           For symmetric systems, should contain only lower diagonal part.
        """
        
        A = csc_matrix(A)
        if self.prop == self.SYMMETRIC:
            A = (A + A.T) - triu(A)

        self.lu = splu(A)
        
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

        return self.lu.solve(b)
