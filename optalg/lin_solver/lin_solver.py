#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class LinSolver:

    # Class constants
    SYMMETRIC = 'symmetric'
    UNSYMMETRIC = 'unsymmetric'
    
    def __init__(self, prop='unsymmetric'):
        """
        Linear solver class.
        
        Parameters
        ----------
        prop : {``symmetric``, ``unsymmetric``}
        """

        # Check
        if prop not in [self.SYMMETRIC,self.UNSYMMETRIC]:
            raise ValueError('invalid property')

        #: Name (string)
        self.name = ''
            
        #: Linear system property {``'symmetric'``, ``'unsymmetric'``}.
        self.prop = prop

        #: Flag that specifies whether the matrix has been analyzed.
        self.analyzed = False
        
    def is_analyzed(self):
        """
        Determine whether the matrix has been analyzed.

        Returns
        -------
        flags : {``True``, ``False``}
        """

        return self.analyzed

    def analyze(self, A):
        """
        Analyzes structure of A.

        Parameters
        ----------
        A : matrix
        """

        self.analyzed = True
        
    def factorize(self, A):
        """
        Factorizes A.

        Parameters
        ----------
        A : matrix
        """

        pass

    def solve(self, b):
        """
        Solves system Ax=b.
        
        Parameters
        ----------
        b: vector
        
        Returns
        -------
        x : vector
        """

        return None

    def factorize_and_solve(self, A, b):
        """
        Factorizes A and solves Ax=b.

        Returns
        -------
        x : vector
        """

        self.factorize(A)

        return self.solve(b)

        

        
