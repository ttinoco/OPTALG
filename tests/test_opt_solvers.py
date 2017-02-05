#*****************************************************#
# This file is part of GRIDOPT.                       #
#                                                     #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.    #
#                                                     #
# GRIDOPT is released under the BSD 2-clause license. #
#*****************************************************#

import unittest
import numpy as np
import optalg as opt
from numpy.linalg import norm
from scipy.sparse import coo_matrix

class TestOptSolvers(unittest.TestCase):
   
    def setUp(self):

        np.random.seed(2)
         
    def test_iqp_random(self):
        
        solver = opt.opt_solver.OptSolverIQP()
        solver.set_parameters({'tol': 1e-8,
                               'quiet': True})

        for i in range(10):

            n = 50
            m = 10
            p = 20
            A = coo_matrix(np.random.randn(m,n))
            b = np.random.randn(m)
            g = np.random.randn(n)
            B = np.matrix(np.random.randn(p,n))
            H = coo_matrix(B.T*B)
            l = np.random.randn(n)
            u = l + 10*np.random.rand()
            
            prob = opt.opt_solver.QuadProblem(H,g,A,b,l,u)

            solver.solve(prob)

            x = solver.get_primal_variables()
            lam,nu,mu,pi = solver.get_dual_variables()

            eps = 1e-10
            self.assertLess(norm(g + H*x - A.T*lam + mu - pi),eps)
            self.assertLess(norm(A*x-b),eps)
            self.assertTrue(np.all(x <= u))
            self.assertTrue(np.all(x >= l))
            self.assertTrue(norm(mu*(u-x),np.inf),eps)
            self.assertTrue(norm(pi*(x-l),np.inf),eps)
            
    def test_solvers_in_QPs(self):

        IQP = opt.opt_solver.OptSolverIQP()
        IQP.set_parameters({'quiet': True})

        AugL = opt.opt_solver.OptSolverAugL()
        AugL.set_parameters({'quiet': True})

        IPOPT = opt.opt_solver.OptSolverIPOPT()
        IPOPT.set_parameters({'quiet': True})
            
        for i in range(20):
            
            n = 50
            m = 10 if i%2 == 0 else 0
            p = 20
            A = coo_matrix(np.random.randn(m,n))
            b = np.random.randn(m)
            g = np.random.randn(n)
            B = np.matrix(np.random.randn(p,n))
            H = coo_matrix(B.T*B+1e-5*np.eye(n))
            l = -1e8*np.ones(n)
            u = 1e8*np.ones(n)
            
            prob = opt.opt_solver.QuadProblem(H,g,A,b,l,u)
            
            IQP.solve(prob)
            self.assertEqual(IQP.get_status(),'solved')
            xIQP = IQP.get_primal_variables()
            lamIQP,nuIQP,muIQP,piIQP = IQP.get_dual_variables()
            
            AugL.solve(prob)
            self.assertEqual(AugL.get_status(),'solved')
            xAugL = AugL.get_primal_variables()
            lamAugL,nuAugL,muAugL,piAugL = AugL.get_dual_variables()

            IPOPT.solve(prob)
            self.assertEqual(IPOPT.get_status(),'solved')
            xIPOPT = IPOPT.get_primal_variables()
            lamIPOPT,nuIPOPT,muIPOPT,piIPOPT = IPOPT.get_dual_variables()

            self.assertTrue(np.all(xIQP == xIQP))
            self.assertFalse(np.all(xIQP == xAugL))
            self.assertFalse(np.all(xIQP == xIPOPT))
            self.assertLess(100*norm(xAugL-xIQP)/norm(xIQP),1e-5)
            self.assertLess(100*norm(xIPOPT-xIQP)/norm(xIQP),1e-5)

            if m > 0:
                self.assertTrue(np.all(lamIQP == lamIQP))
                self.assertFalse(np.all(lamIQP == lamAugL))
                self.assertFalse(np.all(lamIQP == lamIPOPT))
                self.assertLess(100*norm(lamAugL-lamIQP)/norm(lamIQP),1e-5)
                self.assertLess(100*norm(lamIPOPT-lamIQP)/norm(lamIQP),1e-5)

            prob.eval(xIQP)
            objIQP = prob.phi

            prob.eval(xAugL)
            objAugL = prob.phi

            prob.eval(xIPOPT)
            objIPOPT = prob.phi

            self.assertNotEqual(objIQP,objAugL)
            self.assertNotEqual(objIQP,objIPOPT)
            self.assertLess(100*np.abs(objIQP-objAugL)/np.abs(objIQP),1e-5)
            self.assertLess(100*np.abs(objIQP-objIPOPT)/np.abs(objIQP),1e-5)
            
