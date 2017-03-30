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

    def test_clp(self):

        A = np.array([[6.,1.,1.,0.,0.],
                      [3.,1.,0.,1.,0.],
                      [4.,6.,0.,0.,1.]])
        b = np.array([12.,8.,24.])
        
        l = np.array([0.,0.,-1e8,-1e8,-1e8])
        u = np.array([5.,5.,0.,0.,0.])
        
        c = np.array([180.,160.,0.,0.,0.])
        
        problem = opt.opt_solver.LinProblem(c,A,b,l,u)
        
        solver = opt.opt_solver.OptSolverClp()
        solver.set_parameters({'quiet':True})

        try:
            solver.solve(problem)
        except ImportError:
            return

        x = solver.get_primal_variables()
        lam,nu,mu,pi = solver.get_dual_variables()

        self.assertLess(np.abs(x[0]-1.71428571),1e-8)
        self.assertLess(np.abs(x[1]-2.85714286),1e-8)
        
        qp = opt.opt_solver.OptSolverIQP()
        qp.set_parameters({'quiet':True})
        qp.solve(opt.opt_solver.QuadProblem(coo_matrix((5,5)),c,A,b,l,u))
        x1 = qp.get_primal_variables()
        lam1,nu1,mu1,pi1 = qp.get_dual_variables()
        
        self.assertLess(100.*norm(x-x1,np.inf)/norm(x,np.inf),0.1)
        self.assertLess(100.*norm(lam-lam1,np.inf)/norm(lam,np.inf),0.1)
        self.assertLess(100.*norm(mu-mu1,np.inf)/max([norm(mu,np.inf),norm(mu,np.inf),1e-8]),0.1)
        self.assertLess(100.*norm(pi-pi1,np.inf)/max([norm(mu,np.inf),norm(mu,np.inf),1e-8]),0.1)
        
        self.assertRaises(opt.opt_solver.OptSolverError,solver.solve,4)
        
    def test_cbc(self):

        A = np.array([[-2.,2. ,1.,0.],
                      [-8.,10.,0.,1.]])
        b = np.array([1.,13.])
        
        l = np.array([-1e8,-1e8,-1e8,0.])
        u = np.array([1e8,1e8,0.,1e8])
        
        c = np.array([-1.,-1.,0.,0.])
        
        P = np.array([True,True,False,False])
        
        problem = opt.opt_solver.MixIntLinProblem(c,A,b,l,u,P)
        
        solver = opt.opt_solver.OptSolverCbc()
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

    def test_iqp_random(self):
        
        solver = opt.opt_solver.OptSolverIQP()
        solver.set_parameters({'tol': 1e-8,
                               'quiet': True})

        self.assertRaises(opt.opt_solver.OptSolverError,solver.solve,4)

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
            
    def test_solvers_on_QPs(self):

        eps = 1.5 # %
        
        IQP = opt.opt_solver.OptSolverIQP()
        IQP.set_parameters({'quiet': True})
        
        AugL = opt.opt_solver.OptSolverAugL()
        AugL.set_parameters({'quiet': True})
        
        Ipopt = opt.opt_solver.OptSolverIpopt()
        Ipopt.set_parameters({'quiet': True})
            
        for i in range(30):
            
            n = 50
            m = 10 if i%2 == 0 else 0
            p = 20
            A = coo_matrix(np.random.randn(m,n))
            b = np.random.randn(m)
            g = np.random.randn(n)
            B = np.matrix(np.random.randn(p,n))
            H = coo_matrix(B.T*B+1e-3*np.eye(n))
            l = np.random.randn(n)
            u = l+20*np.random.rand(n)

            #l = -1e8*np.ones(n)
            #u = 1e8*np.ones(n)
            
            prob = opt.opt_solver.QuadProblem(H,g,A,b,l,u)
            
            IQP.solve(prob)
            self.assertEqual(IQP.get_status(),'solved')
            xIQP = IQP.get_primal_variables()
            lamIQP,nuIQP,muIQP,piIQP = IQP.get_dual_variables()
            
            AugL.solve(prob)
            self.assertEqual(AugL.get_status(),'solved')
            xAugL = AugL.get_primal_variables()
            lamAugL,nuAugL,muAugL,piAugL = AugL.get_dual_variables()

            try:
                Ipopt.solve(prob)
                self.assertEqual(Ipopt.get_status(),'solved')
                xIpopt = Ipopt.get_primal_variables()
                lamIpopt,nuIpopt,muIpopt,piIpopt = Ipopt.get_dual_variables()
                has_ipopt = True
            except ImportError:
                has_ipopt = False

            self.assertTrue(np.all(xIQP == xIQP))
            self.assertFalse(np.all(xIQP == xAugL))
            self.assertLess(100*norm(xAugL-xIQP)/(norm(xIQP)+eps),eps)
            if has_ipopt:
                self.assertFalse(np.all(xIQP == xIpopt))
                self.assertLess(100*norm(xIpopt-xIQP)/(norm(xIQP)+eps),eps)

            if m > 0:
                self.assertTrue(np.all(lamIQP == lamIQP))
                self.assertFalse(np.all(lamIQP == lamAugL))
                self.assertLess(100*norm(lamAugL-lamIQP)/(norm(lamIQP)+eps),eps)
                if has_ipopt:
                    self.assertFalse(np.all(lamIQP == lamIpopt))
                    self.assertLess(100*norm(lamIpopt-lamIQP)/(norm(lamIQP)+eps),eps)

            self.assertTrue(np.all(muIQP == muIQP))
            self.assertLess(100*norm(muAugL-muIQP)/(norm(muAugL)+eps),eps)
            if has_ipopt:
                self.assertFalse(np.all(muIQP == muIpopt))
                self.assertLess(100*norm(muIpopt-muIQP)/(norm(muIQP)+eps),eps)

            self.assertTrue(np.all(piIQP == piIQP))
            self.assertLess(100*norm(piAugL-piIQP)/(norm(piIQP)+eps),eps)
            if has_ipopt:
                self.assertFalse(np.all(piIQP == piIpopt))
                self.assertLess(100*norm(piIpopt-piIQP)/(norm(piIQP)+eps),eps)

            prob.eval(xIQP)
            objIQP = prob.phi

            prob.eval(xAugL)
            objAugL = prob.phi

            if has_ipopt:
                prob.eval(xIpopt)
                objIpopt = prob.phi

            self.assertNotEqual(objIQP,objAugL)
            self.assertLess(100*np.abs(objIQP-objAugL)/(np.abs(objIQP)+eps),eps)
            if has_ipopt:
                self.assertNotEqual(objIQP,objIpopt)
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

    def test_augl_bounds(self):

        from optalg.opt_solver.augl import AugLBounds

        h = 1e-7
        tol = 1.

        bounds = AugLBounds(5)
        self.assertTrue(np.all(bounds.umin == -bounds.inf*np.ones(5)))
        self.assertTrue(np.all(bounds.umax == bounds.inf*np.ones(5)))

        bounds = AugLBounds(0,np.zeros(0),np.zeros(0))
        bounds.eval(np.ones(0))
        bounds.combine_H(np.ones(0))

        for i in range(10):
            
            n = 10
            umin = 10*np.random.randn(n)
            umax = umin + 10*np.random.rand(n)
            bounds = AugLBounds(n,umin,umax)
            self.assertEqual(bounds.eps,1e-4)
            self.assertEqual(bounds.inf,1e8)

            self.assertEqual(bounds.f.size,2*n)
            self.assertTupleEqual(bounds.J.shape,(2*n,n))
            self.assertTupleEqual(bounds.H_combined.shape,(n,n))
            self.assertTrue(np.all(bounds.f == 0.))
            self.assertTrue(np.all(bounds.J.data == 0.))
            self.assertTrue(np.all(bounds.H_combined.data == 0.))
            self.assertTrue(np.all(bounds.H_combined.row == bounds.H_combined.col))
            
            points = [(umin+umax)/2.,
                      umax+np.random.randn(n)*0.1,
                      umin+np.random.randn(n)*0.1]

            for x0 in points:
                                
                lam = 10.*np.random.randn(2*x0.size)
                bounds.eval(x0)
                f0 = bounds.f.copy()
                J0 = bounds.J.copy()
                bounds.combine_H(lam)
                H0 = bounds.H_combined.copy()
 
                for j in range(10):
                    
                    d = np.random.randn(n)
                    x = x0 + h*d
                    bounds.eval(x)

                    Jd1 = (bounds.f-f0)/h
                    Jd2 = J0*d
                    
                    self.assertLess(100*norm(Jd1-Jd2)/np.maximum(norm(Jd2),1e-5),tol)

                    bounds.combine_H(lam)
                    
                    Hd1 = (bounds.J.T*lam-J0.T*lam)/h
                    Hd2 = H0*d
                    
                    self.assertLess(100*norm(Hd1-Hd2)/np.maximum(norm(Hd2),1e-5),tol)
