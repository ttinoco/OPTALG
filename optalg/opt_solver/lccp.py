#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from opt_solver_error import *
from opt_solver import OptSolver
from optalg.lin_solver import new_linsolver
from scipy.sparse import bmat,triu,eye,spdiags,coo_matrix,tril

class OptSolverLCCP(OptSolver):
    """
    Interior-point linearly-constrained convex program solver.
    """
    
    # Solver parameters
    parameters = {'tol': 1e-4,      # optimality tolerance
                  'maxiter': 100,   # max iterations
                  'sigma': 0.1,     # factor for increasing subproblem solution accuracy
                  'eps': 1e-3,      # boundary proximity factor 
                  'eps_cold': 1e-2, # boundary proximity factor (cold start)
                  'quiet': False}   # quiet flag

    def __init__(self):
        """
        This algorithm solves problems of the form
        
        minimize    varphi(x)
        subject to  Ax = b
                    l <= x <= u
        
        using an interior point method, where
        varphi is convex.
        """
        
        # Init
        OptSolver.__init__(self)
        self.parameters = OptSolverLCCP.parameters.copy()                
        self.linsolver = None
        self.problem = None

    def extract_components(self,y):

        n = self.n
        m = self.m
        
        x = y[:n]
        lam = y[n:n+m]
        mu = y[n+m:2*n+m]
        pi = y[2*n+m:]

        return x,lam,mu,pi
        
    def func(self,y):

        fdata = self.fdata
        sigma = self.parameters['sigma']
        prob = self.problem

        x,lam,mu,pi = self.extract_components(y)
        ux = self.u-x
        xl = x-self.l

        prob.eval(x)

        H = prob.Hphi + prob.Hphi.T - triu(prob.Hphi) # symmetric
        
        rd = prob.gphi-self.AT*lam+mu-pi     # dual residual
        rp = self.A*x-self.b                 # primal residual
        ru = mu*ux-sigma*self.eta_mu*self.e  # residual of perturbed complementarity
        rl = pi*xl-sigma*self.eta_pi*self.e  # residual of perturbed complementarity
        
        Dmu = spdiags(self.mu,0,self.n,self.n) 
        Dux = spdiags(ux,0,self.n,self.n)
        Dpi = spdiags(self.pi,0,self.n,self.n)
        Dxl = spdiags(xl,0,self.n,self.n)

        f = np.hstack((rd,rp,ru,rl))         # residuals
        J = bmat([[H,-self.AT,self.I,-self.I],
                  [self.A,None,None,None],
                  [-Dmu,None,Dux,None],
                  [Dpi,self.Onm,None,Dxl]],
                 format='csr')

        fdata.rp = rp
        fdata.rd = rd
        fdata.ru = ru
        fdata.rl = rl
        
        fdata.f = f
        fdata.J = J
        fdata.F = 0.5*np.dot(f,f)                          # merit function
        fdata.GradF = J.T*f                                # gradient of merit function

        return fdata

    def solve(self,problem):
        """
        Solves optimization problem.

        Parameters
        ----------
        problem : QuadProblem
        """
        
        # Local vars
        norm2 = self.norm2
        norminf = self.norminf
        parameters = self.parameters
        
        # Parameters
        tol = parameters['tol']
        maxiter = parameters['maxiter']
        quiet = parameters['quiet']
        sigma = parameters['sigma']
        eps = parameters['eps']
        eps_cold = parameters['eps_cold']
        
        # Linsolver
        self.linsolver = new_linsolver('mumps','symmetric')
        
        # Problem
        self.problem = problem

        # Reset
        self.reset()

        # Data
        self.A = problem.A
        self.AT = problem.A.T
        self.b = problem.b
        self.l = problem.l
        self.u = problem.u
        self.n = self.A.shape[1]
        self.m = self.A.shape[0]
        self.e = np.ones(self.n)
        self.I = eye(self.n,format='coo')
        self.Onm = coo_matrix((self.n,self.m))
        self.Omm = coo_matrix((self.m,self.m))
    
        # Checks
        assert(np.all(self.l < self.u))

        # Initial point
        if problem.x is None:
            self.x = (self.u + self.l)/2.
        else:
            dul = eps*(self.u-self.l)
            self.x = np.maximum(np.minimum(problem.x,self.u-dul),self.l+dul)
        if problem.lam is None:
            self.lam = np.zeros(self.m)
        else:
            self.lam = problem.lam.copy()
        if problem.mu is None:
            self.mu = np.ones(self.x.size)*eps_cold
        else:
            self.mu = np.maximum(problem.mu,eps)
        if problem.pi is None:
            self.pi = np.ones(self.x.size)*eps_cold
        else:
            self.pi = np.maximum(problem.pi,eps)

        # Check interior
        assert(np.all(self.l < self.x)) 
        assert(np.all(self.x < self.u))
        assert(np.all(self.mu > 0))
        assert(np.all(self.pi > 0))

        # Init vector
        self.y = np.hstack((self.x,self.lam,self.mu,self.pi))

        # Header
        if not quiet:
            print '\nSolver: LCCP'
            print '------------'
                                   
        # Outer
        s = 0.
        self.k = 0
        while True:

            # Complementarity measures
            self.eta_mu = np.dot(self.mu,self.u-self.x)/self.x.size
            self.eta_pi = np.dot(self.pi,self.x-self.l)/self.x.size
            
            # Init eval
            fdata = self.func(self.y)
            fmax = norminf(fdata.f)
            gmax = norminf(fdata.GradF)
            
            # Done
            if fmax < tol and sigma*np.maximum(self.eta_mu,self.eta_pi) < tol:
                self.set_status(self.STATUS_SOLVED)
                self.set_error_msg('')
                return                

            # Target
            tau = sigma*norminf(fdata.GradF)
           
            # Header
            if not quiet:
                if self.k > 0:
                    print ''
                print '{0:^3s}'.format('iter'),
                print '{0:^9s}'.format('phi'),
                print '{0:^9s}'.format('fmax'),
                print '{0:^9s}'.format('gmax'),
                print '{0:^8s}'.format('cu'),
                print '{0:^8s}'.format('cl'),
                print '{0:^8s}'.format('s')
 
            # Inner
            while True:
                
                # Eval
                fdata = self.func(self.y)
                fmax = norminf(fdata.f)
                gmax = norminf(fdata.GradF)
                compu = norminf(self.mu*(self.u-self.x))
                compl = norminf(self.pi*(self.x-self.l))
                
                # Show progress
                if not quiet:
                    print '{0:^3d}'.format(self.k),
                    print '{0:^9.2e}'.format(problem.phi),
                    print '{0:^9.2e}'.format(fmax),
                    print '{0:^9.2e}'.format(gmax),
                    print '{0:^8.1e}'.format(compu),
                    print '{0:^8.1e}'.format(compl),
                    print '{0:^8.1e}'.format(s)
                
                # Done
                if gmax < tau:
                    break

                # Maxiters
                if self.k >= maxiter:
                    raise OptSolverError_MaxIters(self)
                    
                # Search direction
                ux = self.u-self.x
                xl = self.x-self.l
                D1 = spdiags(self.mu/ux,0,self.n,self.n,format='coo')
                D2 = spdiags(self.pi/xl,0,self.n,self.n,format='coo')
                fbar = np.hstack((-fdata.rd+fdata.ru/ux-fdata.rl/xl,fdata.rp))
                Jbar = bmat([[problem.Hphi+D1+D2,None],
                             [-self.A,self.Omm]],format='coo')
                if not self.linsolver.is_analyzed():
                    self.linsolver.analyze(Jbar)
                pbar = self.linsolver.factorize_and_solve(Jbar,fbar)
                px = pbar[:self.n]
                plam = pbar[self.n:self.n+self.m]                
                pmu = (-fdata.ru + self.mu*px)/ux
                ppi = (-fdata.rl - self.pi*px)/xl
                p = np.hstack((pbar,pmu,ppi))

                # Steplength bounds
                indices = px > 0
                s1 = np.min(np.hstack(((1.-eps)*(self.u-self.x)[indices]/px[indices],np.inf)))
                indices = px < 0
                s2 = np.min(np.hstack(((eps-1.)*(self.x-self.l)[indices]/px[indices],np.inf)))
                indices = pmu < 0
                s3 = np.min(np.hstack(((eps-1.)*self.mu[indices]/pmu[indices],np.inf)))
                indices = ppi < 0
                s4 = np.min(np.hstack(((eps-1.)*self.pi[indices]/ppi[indices],np.inf)))
                smax = np.min([s1,s2,s3,s4])
                
                # Line search
                s,fdata = self.line_search(self.y,p,fdata.F,fdata.GradF,self.func,smax)

                # Update x
                self.y += s*p
                self.k += 1
                self.x,self.lam,self.mu,self.pi = self.extract_components(self.y)

                # Check
                assert(np.all(self.x < self.u))
                assert(np.all(self.x > self.l))
                assert(np.all(self.mu > 0))
                assert(np.all(self.pi > 0))

        
