#*****************************************************#
# This file is part of OPTALG.                        #
#                                                     #
# Copyright (c) 2019, Tomas Tinoco De Rubira.         #
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

    def test_augl_properties_support(self):

        augl = opt.opt_solver.OptSolverAugL()

        # augl
        self.assertTrue(augl.supports_properties(['linear',
                                                  'quadratic',
                                                  'nonlinear',
                                                  'continuous',
                                                  'feasibility',
                                                  'optimization']))
        self.assertFalse(augl.supports_properties(['integer']))

    def test_cbc_properties_support(self):
        
        try:
            cbc = opt.opt_solver.OptSolverCbc()
        except ImportError:
            raise unittest.SkipTest('no cbc')

        # cbc
        self.assertTrue(cbc.supports_properties(['linear',
                                                 'continuous',
                                                 'integer',
                                                 'feasibility',
                                                 'optimization']))
        self.assertFalse(cbc.supports_properties(['quadratic']))
        self.assertFalse(cbc.supports_properties(['nonlinear']))
            
    def test_cbc_cmd_properties_support(self):

        try:
            cbc_cmd = opt.opt_solver.OptSolverCbcCMD()
        except ImportError:
            raise unittest.SkipTest('no cbc cmd')

        # cbc
        self.assertTrue(cbc_cmd.supports_properties(['linear',
                                                     'continuous',
                                                     'integer',
                                                     'feasibility',
                                                     'optimization']))
        self.assertFalse(cbc_cmd.supports_properties(['quadratic']))
        self.assertFalse(cbc_cmd.supports_properties(['nonlinear']))

    def test_cplex_cmd_properties_support(self):

        try:
            cplex_cmd = opt.opt_solver.OptSolverCplexCMD()
        except ImportError:
            raise unittest.SkipTest('no cplex cmd')

        # cplex
        self.assertTrue(cplex_cmd.supports_properties(['linear',
                                                     'continuous',
                                                     'integer',
                                                     'feasibility',
                                                     'optimization']))
        self.assertFalse(cplex_cmd.supports_properties(['quadratic']))
        self.assertFalse(cplex_cmd.supports_properties(['nonlinear']))

    def test_clp_properties_support(self):

        try:
            clp = opt.opt_solver.OptSolverClp()
        except ImportError:
            raise unittest.SkipTest('no clp')

        # clp
        self.assertTrue(clp.supports_properties(['linear',
                                                 'continuous',
                                                 'feasibility',
                                                 'optimization']))
        self.assertFalse(clp.supports_properties(['quadratic']))
        self.assertFalse(clp.supports_properties(['nonlinear']))
        self.assertFalse(clp.supports_properties(['integer']))

    def test_clp_cmd_properties_support(self):

        try:
            clp_cmd = opt.opt_solver.OptSolverClpCMD()
        except ImportError:
            raise unittest.SkipTest('no clp cmd')

        self.assertTrue(clp_cmd.supports_properties(['linear',
                                                     'continuous',
                                                     'feasibility',
                                                     'optimization']))
        self.assertFalse(clp_cmd.supports_properties(['quadratic']))
        self.assertFalse(clp_cmd.supports_properties(['nonlinear']))
        self.assertFalse(clp_cmd.supports_properties(['integer']))

    def test_inlp_properties_support(self):

        inlp = opt.opt_solver.OptSolverINLP()

        # inlp
        self.assertTrue(inlp.supports_properties(['linear',
                                                  'quadratic',
                                                  'nonlinear',
                                                  'continuous',
                                                  'feasibility',
                                                  'optimization']))
        self.assertFalse(inlp.supports_properties(['integer']))

    def test_ipopt_properties_support(self):
        
        try:
            ipopt = opt.opt_solver.OptSolverIpopt()
        except ImportError:
            raise unittest.SkipTest('no ipopt')

        # ipopt
        self.assertTrue(ipopt.supports_properties(['linear',
                                                   'quadratic',
                                                   'nonlinear',
                                                   'continuous',
                                                   'feasibility',
                                                   'optimization']))
        self.assertFalse(ipopt.supports_properties(['integer']))

    def test_iqp_properties_support(self):
        
        iqp = opt.opt_solver.OptSolverIQP()

        # iqp
        self.assertTrue(iqp.supports_properties(['linear',
                                                 'quadratic',
                                                 'continuous',
                                                 'feasibility',
                                                 'optimization']))
        self.assertFalse(iqp.supports_properties(['integer']))
        self.assertFalse(iqp.supports_properties(['nonlinear']))

    def test_nr_properties_support(self):
        
        nr = opt.opt_solver.OptSolverNR()

        # nr
        self.assertTrue(nr.supports_properties(['linear',
                                                'quadratic',
                                                'nonlinear',
                                                'continuous',
                                                'feasibility']))
        self.assertFalse(nr.supports_properties(['integer']))
        self.assertFalse(nr.supports_properties(['optimization']))

    def test_ipopt(self):

        try:
            Ipopt = opt.opt_solver.OptSolverIpopt()
        except ImportError:
            raise unittest.SkipTest('no ipopt')
        
        Ipopt.set_parameters({'quiet': True, 'sb': 'yes'})

        n = 50
        m = 10
        p = 20
        A = coo_matrix(np.random.randn(m,n))
        b = np.random.randn(m)
        g = np.random.randn(n)
        B = np.matrix(np.random.randn(p,n))
        H = coo_matrix(B.T*B+1e-3*np.eye(n))
        l = np.random.randn(n)
        u = l+20*np.random.rand(n)
        
        prob = opt.opt_solver.QuadProblem(H,g,A,b,l,u)
            
        # Default parameters
        Ipopt.solve(prob)
        self.assertEqual(Ipopt.get_status(),'solved')
        
        # Modify exposed parameters
        new_parameters = {'tol': 1e-7,
                          'inf': 1e8,
                          'derivative_test': 'first-order',
                          'hessian_approximation': 'exact',
                          'linear_solver': 'mumps',
                          'print_level': 1,
                          'max_iter': 100,
                          'mu_init': 1e-2,
                          'expect_infeasible_problem' : 'yes',
                          'check_derivatives_for_naninf' : 'yes',
                          'diverging_iterates_tol' : 1e6,
                          'max_cpu_time' : 10}
        
        Ipopt.set_parameters(new_parameters)
        Ipopt.solve(prob)
                    
        # Test with inf and nan
        x = np.random.randn(n)
        
        for x_bad in [np.inf, np.nan]:
            x[int(n/2)] = x_bad
            bad_prob = opt.opt_solver.QuadProblem(H,g,A,b,l,u,x=x)
            self.assertRaises(opt.opt_solver.OptSolverError_Ipopt, Ipopt.solve, bad_prob)
            self.assertEqual(Ipopt.get_status(), 'error')

    def test_cplex_cmd_lp_duals(self):
            
        A = np.array([[6.,1.,1.,0.,0.],
                      [3.,1.,0.,1.,0.],
                      [4.,6.,0.,0.,1.]])
        b = np.array([12.,8.,24.])
        
        l = np.array([0.,0.,-1e0,-1e8,-1e8])
        u = np.array([5.,5.,0.,0.,0.])
        
        c = np.array([180.,160.,0.,0.,0.])

        problem = opt.opt_solver.LinProblem(c,A,b,l,u)

        try:
            solver = opt.opt_solver.OptSolverCplexCMD()
        except ImportError:
            raise unittest.SkipTest('no cplex cmd')

        solver.set_parameters({'debug': False, 'quiet': True})
        solver.solve(problem)
        
        x = solver.get_primal_variables()
        lam, nu, mu, pi = solver.get_dual_variables()
        
        problem.eval(x)
        self.assertLess(np.linalg.norm(np.dot(A,x)-b),1e-6)
        self.assertTrue(np.all(l <= x))
        self.assertTrue(np.all(x <= u))
        self.assertEqual(nu.size, 0.)
        self.assertEqual(lam.size, 3)
        self.assertEqual(mu.size, 5)
        self.assertEqual(pi.size, 5)
        self.assertLess(norm(problem.gphi - problem.A.T*lam + mu - pi, np.inf), 1e-8)
        self.assertGreaterEqual(np.min(mu), 0.)
        self.assertGreaterEqual(np.min(pi), 0.)
        self.assertLess(np.abs(np.dot(problem.u-x, mu)), 1e-8)
        self.assertLess(np.abs(np.dot(x-problem.l, pi)), 1e-8)

    def test_cbc_cmd_lp_duals(self):
            
        A = np.array([[6.,1.,1.,0.,0.],
                      [3.,1.,0.,1.,0.],
                      [4.,6.,0.,0.,1.]])
        b = np.array([12.,8.,24.])
        
        l = np.array([0.,0.,-1e0,-1e8,-1e8])
        u = np.array([5.,5.,0.,0.,0.])
        
        c = np.array([180.,160.,0.,0.,0.])

        problem = opt.opt_solver.LinProblem(c,A,b,l,u)

        try:
            solver = opt.opt_solver.OptSolverCbcCMD()
        except ImportError:
            raise unittest.SkipTest('no cbc cmd')

        solver.set_parameters({'debug': False, 'quiet': True})
        solver.solve(problem)
        
        x = solver.get_primal_variables()
        lam, nu, mu, pi = solver.get_dual_variables()
        
        problem.eval(x)
        self.assertLess(np.linalg.norm(np.dot(A,x)-b),1e-6)
        self.assertTrue(np.all(l <= x))
        self.assertTrue(np.all(x <= u))
        self.assertEqual(nu.size, 0.)
        self.assertEqual(lam.size, 3)
        self.assertEqual(mu.size, 5)
        self.assertEqual(pi.size, 5)
        self.assertLess(norm(problem.gphi - problem.A.T*lam + mu - pi, np.inf), 1e-8)
        self.assertGreaterEqual(np.min(mu), 0.)
        self.assertGreaterEqual(np.min(pi), 0.)
        self.assertLess(np.abs(np.dot(problem.u-x, mu)), 1e-8)
        self.assertLess(np.abs(np.dot(x-problem.l, pi)), 1e-8)

    def test_ipopt_lp_duals(self):
            
        A = np.array([[6.,1.,1.,0.,0.],
                      [3.,1.,0.,1.,0.],
                      [4.,6.,0.,0.,1.]])
        b = np.array([12.,8.,24.])
        
        l = np.array([0.,0.,-1e0,-1e8,-1e8])
        u = np.array([5.,5.,0.,0.,0.])
        
        c = np.array([180.,160.,0.,0.,0.])

        problem = opt.opt_solver.LinProblem(c,A,b,l,u)

        try:
            solver = opt.opt_solver.OptSolverIpopt()
        except ImportError:
            raise unittest.SkipTest('no ipopt')

        solver.set_parameters({'debug': False, 'quiet': True})
        solver.solve(problem)
        
        x = solver.get_primal_variables()
        lam, nu, mu, pi = solver.get_dual_variables()
        
        problem.eval(x)
        self.assertLess(np.linalg.norm(np.dot(A,x)-b), 1e-6)
        self.assertTrue(np.all(l <= x))
        self.assertTrue(np.all(x <= u))
        self.assertEqual(nu.size, 0.)
        self.assertEqual(lam.size, 3)
        self.assertEqual(mu.size, 5)
        self.assertEqual(pi.size, 5)
        self.assertLess(norm(problem.gphi - problem.A.T*lam + mu - pi, np.inf), 1e-8)
        self.assertGreaterEqual(np.min(mu), 0.)
        self.assertGreaterEqual(np.min(pi), 0.)
        self.assertLess(np.abs(np.dot(problem.u-x, mu)), 1e-7)
        self.assertLess(np.abs(np.dot(x-problem.l, pi)), 1e-7)

    def test_clp_cmd_lp_duals(self):

        A = np.array([[6.,1.,1.,0.,0.],
                      [3.,1.,0.,1.,0.],
                      [4.,6.,0.,0.,1.]])
        b = np.array([12.,8.,24.])
        
        l = np.array([0.,0.,-1e0,-1e8,-1e8])
        u = np.array([5.,5.,0.,0.,0.])
        
        c = np.array([180.,160.,0.,0.,0.])

        problem = opt.opt_solver.LinProblem(c,A,b,l,u)

        try:
            solver = opt.opt_solver.OptSolverClpCMD()
        except ImportError:
            raise unittest.SkipTest('no clp cmd')

        solver.set_parameters({'debug': False, 'quiet': True})
        solver.solve(problem)
        
        x = solver.get_primal_variables()
        lam, nu, mu, pi = solver.get_dual_variables()
        
        problem.eval(x)
        self.assertLess(np.linalg.norm(np.dot(A,x)-b),1e-6)
        self.assertTrue(np.all(l <= x))
        self.assertTrue(np.all(x <= u))
        self.assertEqual(nu.size, 0.)
        self.assertEqual(lam.size, 3)
        self.assertEqual(mu.size, 5)
        self.assertEqual(pi.size, 5)
        self.assertLess(norm(problem.gphi - problem.A.T*lam + mu - pi, np.inf), 1e-8)
        self.assertGreaterEqual(np.min(mu), 0.)
        self.assertGreaterEqual(np.min(pi), 0.)
        self.assertLess(np.abs(np.dot(problem.u-x, mu)), 1e-8)
        self.assertLess(np.abs(np.dot(x-problem.l, pi)), 1e-8)

    def test_clp(self):
        
        A = np.array([[6.,1.,1.,0.,0.],
                      [3.,1.,0.,1.,0.],
                      [4.,6.,0.,0.,1.]])
        b = np.array([12.,8.,24.])
        
        l = np.array([0.,0.,-1e8,-1e8,-1e8])
        u = np.array([5.,5.,0.,0.,0.])
        
        c = np.array([180.,160.,0.,0.,0.])

        problem = opt.opt_solver.LinProblem(c,A,b,l,u)

        try:
            solver = opt.opt_solver.OptSolverClp()
        except ImportError:
            raise unittest.SkipTest('no clp')
        
        solver.set_parameters({'quiet':True})
        
        solver.solve(problem)
            
        x = solver.get_primal_variables()
        lam,nu,mu,pi = solver.get_dual_variables()

        problem.eval(x)
        self.assertLess(np.linalg.norm(np.dot(A,x)-b),1e-8)
        self.assertTrue(np.all(l <= x))
        self.assertTrue(np.all(x <= u))
        self.assertLess(np.abs(x[0]-1.71428571),1e-8)
        self.assertLess(np.abs(x[1]-2.85714286),1e-8)
        self.assertLess(np.abs(problem.phi-765.714285218),1e-6)
        
        qp = opt.opt_solver.OptSolverIQP()
        qp.set_parameters({'quiet':True})
        problem1 = opt.opt_solver.QuadProblem(coo_matrix((5,5)),c,A,b,l,u) 
        qp.solve(problem1)
        x1 = qp.get_primal_variables()
        lam1,nu1,mu1,pi1 = qp.get_dual_variables()

        problem1.eval(x1)
        self.assertLess(np.linalg.norm(np.dot(A,x1)-b),1e-8)
        self.assertTrue(np.all(l-1e-5 <= x1))
        self.assertTrue(np.all(x1 <= u+1e-5))
        
        self.assertLess(100.*norm(x-x1,np.inf)/norm(x,np.inf),0.1)
        self.assertLess(100.*norm(lam-lam1,np.inf)/norm(lam,np.inf),0.1)
        self.assertLess(100.*norm(mu-mu1,np.inf)/max([norm(mu,np.inf),norm(mu,np.inf),1e-8]),0.1)
        self.assertLess(100.*norm(pi-pi1,np.inf)/max([norm(mu,np.inf),norm(mu,np.inf),1e-8]),0.1)
        
        self.assertRaises(opt.opt_solver.OptSolverError,solver.solve,4)

    def test_clp_cmd(self):

        A = np.array([[6.,1.,1.,0.,0.],
                      [3.,1.,0.,1.,0.],
                      [4.,6.,0.,0.,1.]])
        b = np.array([12.,8.,24.])
        
        l = np.array([0.,0.,-1e8,-1e8,-1e8])
        u = np.array([5.,5.,0.,0.,0.])
        
        c = np.array([180.,160.,0.,0.,0.])

        problem = opt.opt_solver.LinProblem(c,A,b,l,u)

        try:
            solver = opt.opt_solver.OptSolverClpCMD()
        except ImportError:
            raise unittest.SkipTest('no clp command-line solver')
            
        solver.set_parameters({'quiet':True})

        solver.solve(problem)
            
        x = solver.get_primal_variables()
        #lam,nu,mu,pi = solver.get_dual_variables()

        problem.eval(x)
        self.assertLess(np.linalg.norm(np.dot(A,x)-b),1e-6)
        self.assertTrue(np.all(l <= x))
        self.assertTrue(np.all(x <= u))
        self.assertLess(np.abs(x[0]-1.71428571),1e-7)
        self.assertLess(np.abs(x[1]-2.85714286),1e-7)
        self.assertLess(np.abs(problem.phi-765.714285218),1e-5)
        
    def test_cbc(self):

        A = np.array([[-2.,2. ,1.,0.],
                      [-8.,10.,0.,1.]])
        b = np.array([1.,13.])
        
        l = np.array([-1e8,-1e8,-1e8,0.])
        u = np.array([1e8,1e8,0.,1e8])
        
        c = np.array([-1.,-1.,0.,0.])
        
        P = np.array([True,True,False,False])
        
        problem = opt.opt_solver.MixIntLinProblem(c,A,b,l,u,P)

        try:
            solver = opt.opt_solver.OptSolverCbc()
        except ImportError:
            raise unittest.SkipTest('no cbc')
        
        solver.set_parameters({'quiet':True})

        solver.solve(problem)

        x = solver.get_primal_variables()

        self.assertAlmostEqual(x[0],1.)
        self.assertAlmostEqual(x[1],2.)

        problem.P[:] = False

        solver.solve(problem)

        x = solver.get_primal_variables()

        self.assertAlmostEqual(x[0],4.)
        self.assertAlmostEqual(x[1],4.5)

    def test_cbc_cmd(self):

        A = np.array([[-2.,2. ,1.,0.],
                      [-8.,10.,0.,1.]])
        b = np.array([1.,13.])
        
        l = np.array([-1e8,-1e8,-1e8,0.])
        u = np.array([1e8,1e8,0.,1e8])
        
        c = np.array([-1.,-1.,0.,0.])
        
        P = np.array([True,True,False,False])
        
        problem = opt.opt_solver.MixIntLinProblem(c,A,b,l,u,P)

        try:
            solver = opt.opt_solver.OptSolverCbcCMD()
        except ImportError:
            raise unittest.SkipTest('no cbc command-line solver')
        
        solver.set_parameters({'quiet':True})

        solver.solve(problem)

        self.assertEqual(solver.get_status(), 'solved')
        x = solver.get_primal_variables()

        self.assertAlmostEqual(x[0],1.)
        self.assertAlmostEqual(x[1],2.)

        problem.P[:] = False

        solver.solve(problem)
        
        self.assertEqual(solver.get_status(), 'solved')
        x = solver.get_primal_variables()

        self.assertAlmostEqual(x[0],4.)
        self.assertAlmostEqual(x[1],4.5)

    def test_cplex_cmd(self):

        A = np.array([[-2.,2. ,1.,0.],
                      [-8.,10.,0.,1.]])
        b = np.array([1.,13.])
        
        l = np.array([-1e8,-1e8,-1e8,0.])
        u = np.array([1e8,1e8,0.,1e8])
        
        c = np.array([-1.,-1.,0.,0.])
        
        P = np.array([True,True,False,False])
        
        problem = opt.opt_solver.MixIntLinProblem(c,A,b,l,u,P)

        try:
            solver = opt.opt_solver.OptSolverCplexCMD()
        except ImportError:
            raise unittest.SkipTest('no cplex command-line solver')
        
        solver.set_parameters({'quiet': True, 'feasibility': 1e-4})

        solver.solve(problem)

        self.assertEqual(solver.get_status(), 'solved')
        x = solver.get_primal_variables()

        self.assertAlmostEqual(x[0],1.)
        self.assertAlmostEqual(x[1],2.)

        problem.P[:] = False

        solver.solve(problem)
        
        self.assertEqual(solver.get_status(), 'solved')
        x = solver.get_primal_variables()

        self.assertAlmostEqual(x[0],4.)
        self.assertAlmostEqual(x[1],4.5)
        
    def test_iqp_random(self):
        
        solver = opt.opt_solver.OptSolverIQP()
        solver.set_parameters({'tol': 1e-8,
                               'quiet': True})

        self.assertRaises(Exception,solver.solve,4)

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
            self.assertTrue(np.all(x <= u+1e-5))
            self.assertTrue(np.all(x >= l-1e-5))
            self.assertTrue(norm(mu*(u-x),np.inf),eps)
            self.assertTrue(norm(pi*(x-l),np.inf),eps)

    def test_solvers_on_LPs(self):

        A = np.array([[6.,1.,1.,0.,0.],
                      [3.,1.,0.,1.,0.],
                      [4.,6.,0.,0.,1.]])
        b = np.array([12.,8.,24.])
        
        l = np.array([0.,0.,-1e8,-1e8,-1e8])
        u = np.array([5.,5.,0.,0.,0.])
        
        c = np.array([180.,160.,0.,0.,0.])

        problem = opt.opt_solver.LinProblem(c,A,b,l,u)

        for solver in [opt.opt_solver.OptSolverIpopt,
                       opt.opt_solver.OptSolverINLP,
                       opt.opt_solver.OptSolverAugL]:

            try:
                solver = solver()
            except ImportError:
                continue
            
            solver.set_parameters({'quiet': True})
            
            try:
                solver.solve(problem)
            except opt.opt_solver.OptSolverError_NotAvailable:
                continue
            
            x = solver.get_primal_variables()
            lam,nu,mu,pi = solver.get_dual_variables()

            problem.eval(x)
            self.assertLess(np.linalg.norm(np.dot(A,x)-b),1e-5*np.linalg.norm(b))
            self.assertTrue(np.all(l-1e-5 <= x))
            self.assertTrue(np.all(x <= u+1e-5))
            self.assertLess(np.abs(x[0]-1.71428571),1e-5)
            self.assertLess(np.abs(x[1]-2.85714286),1e-5)
            self.assertLess(np.abs(problem.phi-765.714285218),1e-5*765)
            
    def test_solvers_on_QPs(self):

        eps = 1.5 # %
        num_trials = 10
        
        IQP = opt.opt_solver.OptSolverIQP()
        IQP.set_parameters({'quiet': True, 'tol':1e-5})

        INLP = opt.opt_solver.OptSolverINLP()
        INLP.set_parameters({'quiet': True,'feastol':1e-5, 'optol': 1e-5})
        
        AugL = opt.opt_solver.OptSolverAugL()
        AugL.set_parameters({'quiet': True, 'feastol':1e-5, 'optol': 1e-5})

        try:
            Ipopt = opt.opt_solver.OptSolverIpopt()
            Ipopt.set_parameters({'quiet': True})
        except ImportError:
            Ipopt = None
            
        for i in range(num_trials):
            
            n = 50
            m = 10 if i%2 == 0 else 0
            p = 20
            A = coo_matrix(np.random.randn(m,n))
            b = np.random.randn(m)
            g = np.random.randn(n)
            B = np.matrix(np.random.randn(p,n))
            H = coo_matrix(B.T*B+1e-5*np.eye(n))
            l = np.random.randn(n)
            u = l+20*np.random.rand(n)
            
            prob = opt.opt_solver.QuadProblem(H,g,A,b,l,u)

            IQP.solve(prob)
            self.assertEqual(IQP.get_status(),'solved')
            xIQP = IQP.get_primal_variables()
            lamIQP,nuIQP,muIQP,piIQP = IQP.get_dual_variables()

            INLP.solve(prob)
            self.assertEqual(INLP.get_status(),'solved')
            xINLP = INLP.get_primal_variables()
            lamINLP,nuINLP,muINLP,piINLP = INLP.get_dual_variables()
            
            AugL.solve(prob)
            self.assertEqual(AugL.get_status(),'solved')
            xAugL = AugL.get_primal_variables()
            lamAugL,nuAugL,muAugL,piAugL = AugL.get_dual_variables()

            if Ipopt is not None:
                Ipopt.solve(prob)
                self.assertEqual(Ipopt.get_status(),'solved')
                xIpopt = Ipopt.get_primal_variables()
                lamIpopt,nuIpopt,muIpopt,piIpopt = Ipopt.get_dual_variables()

            self.assertFalse(xIQP is xAugL)
            self.assertFalse(xIQP is xINLP)
            self.assertLess(100*norm(xAugL-xIQP)/(norm(xIQP)+eps),eps)
            self.assertLess(100*norm(xINLP-xIQP)/(norm(xIQP)+eps),eps)
            if Ipopt is not None:
                self.assertFalse(xIQP is xIpopt)
                self.assertLess(100*norm(xIpopt-xIQP)/(norm(xIQP)+eps),eps)

            if m > 0:
                self.assertFalse(lamIQP is lamAugL)
                self.assertFalse(lamIQP is lamINLP)
                self.assertLess(100*norm(lamAugL-lamIQP)/(norm(lamIQP)+eps),eps)
                self.assertLess(100*norm(lamINLP-lamIQP)/(norm(lamIQP)+eps),eps)
                if Ipopt is not None:
                    self.assertFalse(lamIQP is lamIpopt)
                    self.assertLess(100*norm(lamIpopt-lamIQP)/(norm(lamIQP)+eps),eps)

            self.assertFalse(muIQP is muAugL)
            self.assertFalse(muIQP is muINLP)
            #self.assertLess(100*norm(muAugL-muIQP)/(norm(muIQP)+eps),eps)
            self.assertLess(100*norm(muINLP-muIQP)/(norm(muIQP)+eps),eps)
            if Ipopt is not None:
                self.assertFalse(muIQP is muIpopt)
                self.assertLess(100*norm(muIpopt-muIQP)/(norm(muIQP)+eps),eps)

            self.assertFalse(piIQP is piAugL)
            self.assertFalse(piIQP is piINLP)
            #self.assertLess(100*norm(piAugL-piIQP)/(norm(piIQP)+eps),eps)
            self.assertLess(100*norm(piINLP-piIQP)/(norm(piIQP)+eps),eps)
            if Ipopt is not None:
                self.assertFalse(piIQP is piIpopt)
                self.assertLess(100*norm(piIpopt-piIQP)/(norm(piIQP)+eps),eps)

            prob.eval(xIQP)
            objIQP = prob.phi

            prob.eval(xINLP)
            objINLP = prob.phi

            prob.eval(xAugL)
            objAugL = prob.phi

            if Ipopt is not None:
                prob.eval(xIpopt)
                objIpopt = prob.phi

            self.assertLess(100*np.abs(objIQP-objAugL)/(np.abs(objIQP)+eps),eps)
            self.assertLess(100*np.abs(objIQP-objINLP)/(np.abs(objIQP)+eps),eps)
            if Ipopt is not None:
                self.assertLess(100*np.abs(objIQP-objIpopt)/(np.abs(objIQP)+eps),eps)

    def test_augl_barrier(self):

        from optalg.opt_solver.augl import AugLBarrier

        h = 1e-9
        tol = 1.

        bounds = AugLBarrier(5)
        self.assertTrue(np.all(bounds.umin <= -bounds.inf*np.ones(5)))
        self.assertTrue(np.all(bounds.umax >= bounds.inf*np.ones(5)))
    
        bounds = AugLBarrier(0,np.zeros(0),np.zeros(0))
        bounds.eval(np.ones(0))

        for i in range(10):
            
            n = 10
            umin = 10*np.random.randn(n)
            umax = umin + 10*np.random.rand(n)
            bounds = AugLBarrier(n,umin,umax)
            self.assertEqual(bounds.inf,1e8)

            self.assertEqual(bounds.phi,0.)
            self.assertTrue(np.all(bounds.gphi == 0.))
            self.assertTupleEqual(bounds.gphi.shape,(n,))
            self.assertTupleEqual(bounds.Hphi.shape,(n,n))
            self.assertTrue(np.all(bounds.Hphi.data == 0.))
            self.assertTrue(np.all(bounds.Hphi.row == bounds.Hphi.col))
            self.assertTrue(np.all(bounds.Hphi.row == range(n)))

            du = umax-umin
            points = [(umin+umax)/2.,
                      umax-1e-2*du,
                      umin+1e-2*du]

            for x0 in points:

                bounds.eval(x0)
                phi0 = bounds.phi
                gphi0 = bounds.gphi.copy()
                Hphi0 = bounds.Hphi.copy()

                for j in range(10):
                    
                    d = np.random.randn(n)
                    x = x0 + h*d
                    bounds.eval(x)
                    phi1 = bounds.phi
                    gphi1 = bounds.gphi.copy()
                    Hphi1 = bounds.Hphi.copy()
                    
                    gTd = np.dot(gphi0,d)
                    gTd_approx = (phi1-phi0)/h

                    self.assertLess(100*abs(gTd-gTd_approx)/np.maximum(abs(gTd),1e-3),tol)

                    Hd = Hphi0*d
                    Hd_approx = (gphi1-gphi0)/h

                    self.assertLess(100*norm(Hd-Hd_approx)/np.maximum(norm(Hd),1e-3),tol)
