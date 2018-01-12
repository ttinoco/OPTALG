#*****************************************************#
# This file is part of OPTALG.                        #
#                                                     #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.    #
#                                                     #
# OPTALG is released under the BSD 2-clause license.  #
#*****************************************************#

import unittest
import numpy as np
import optalg as opt
from numpy.linalg import norm
from scipy.sparse import coo_matrix

class TestOptSolvers(unittest.TestCase):
   
    def setUp(self):

        np.random.seed(2)

    def test_umfpack(self):

        A = np.random.randn(100,100)
        b = np.random.randn(100)
        
        try:
            umf = opt.lin_solver.new_linsolver('umfpack','unsymmetric')
        except ImportError:
            raise unittest.SkipTest('no umfpack')

        self.assertTrue(isinstance(umf, opt.lin_solver.LinSolverUMFPACK))
        umf.analyze(A)
        umf.factorize(A)
        x = umf.solve(b)

        self.assertLess(norm(np.dot(A,x)-b),1e-10)

    def test_superlu(self):

        A = np.random.randn(100,100)
        b = np.random.randn(100)

        superlu = opt.lin_solver.new_linsolver('superlu','unsymmetric')
        self.assertTrue(isinstance(superlu, opt.lin_solver.LinSolverSUPERLU))

        superlu.analyze(A)
        superlu.factorize(A)
        x = superlu.solve(b)

        self.assertLess(norm(np.dot(A,x)-b),1e-10)

    def test_mumps(self):

        A = np.random.randn(100,100)
        b = np.random.randn(100)
        
        try:
            mumps = opt.lin_solver.new_linsolver('mumps','unsymmetric')
        except ImportError:
            raise unittest.SkipTest('no mumps')

        self.assertTrue(isinstance(mumps, opt.lin_solver.LinSolverMUMPS))
        mumps.analyze(A)
        mumps.factorize(A)
        x = mumps.solve(b)

        self.assertLess(norm(np.dot(A,x)-b),1e-10)
