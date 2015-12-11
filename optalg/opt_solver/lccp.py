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
    parameters = {'tol': 1e-4,        # optimality tolerance
                  'maxiter': 1000,    # max iterations
                  'sigma': 0.1,       # factor for increasing subproblem solution accuracy
                  'eps': 1e-3,        # boundary proximity factor 
                  'eps_cold': 1e-2,   # boundary proximity factor (cold start)
                  'quiet': False}     # quiet flag

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
        
    def func(self,y):

        fdata = self.fdata
        sigma = self.parameters['sigma']
        prob = self.problem

        x = y[:self.n]
        lam = y[self.n:]
        
        ux = self.u-x
        xl = x-self.l

        rp = self.A*x-self.b

        prob.eval(x)
        
        fdata.F = (prob.phi - 
                   self.rho*np.sum(np.log(ux)) -
                   self.rho**np.sum(np.log(xl)) +
                   np.dot(lam,rp) + 
                   0.5*self.beta*np.dot(rp,rp))

        fdata.GradF = np.hstack((prob.gphi +
                                 self.rho*1./ux -
                                 self.rho*1./xl +
                                 self.AT*lam + 
                                 self.beta*self.AT*rp,
                                 rp))

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
        self.linsolver = new_linsolver('superlu','symmetric')
        
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

        # Check interior
        assert(np.all(self.l < self.x)) 
        assert(np.all(self.x < self.u))

        # Header
        if not quiet:
            print '\nSolver: LCCP'
            print '------------'
                                   
        # Outer
        s = 0.
        pmax = 0
        self.k = 0
        self.beta = 0
        self.rho = 1e0
        self.mu = self.rho/(self.u-self.x)
        self.pi = self.rho/(self.x-self.l)
        while True:
            
            # Init eval
            fdata = self.func(np.hstack((self.x,self.lam)))
            gmax = norminf(fdata.GradF)
            
            # Done
            if gmax < tol and self.rho < tol:
                self.set_status(self.STATUS_SOLVED)
                self.set_error_msg('')
                return                

            # Target
            tau = sigma*gmax
           
            # Header
            if not quiet:
                if self.k > 0:
                    print ''
                print '{0:^3s}'.format('iter'),
                print '{0:^9s}'.format('phi'),
                print '{0:^9s}'.format('gmax'),
                print '{0:^8s}'.format('rho'),
                print '{0:^8s}'.format('beta'),
                print '{0:^8s}'.format('s'),
                print '{0:^8s}'.format('pmax')
 
            # Inner
            while True:
                
                # Eval
                fdata = self.func(np.hstack((self.x,self.lam)))
                gmax = norminf(fdata.GradF)
                
                # Show progress
                if not quiet:
                    print '{0:^3d}'.format(self.k),
                    print '{0:^9.2e}'.format(problem.phi),
                    print '{0:^9.2e}'.format(gmax),
                    print '{0:^8.1e}'.format(self.rho),
                    print '{0:^8.1e}'.format(self.beta),
                    print '{0:^8.1e}'.format(s),
                    print '{0:^8.1e}'.format(pmax)
                
                # Done
                if gmax < tau:
                    break

                # Maxiters
                if self.k >= maxiter:
                    raise OptSolverError_MaxIters(self)

                # Merit function
                if self.beta >= 1e10:
                    raise OptSolverError('what?')
                    
                # Search direction
                ux = self.u-self.x
                xl = self.x-self.l
                self.mu = self.rho/ux
                self.pi = self.rho/xl
                rd = problem.gphi-self.AT*self.lam+self.mu-self.pi     # dual residual
                rp = self.A*self.x-self.b                 # primal residual
                ru = self.mu*ux-self.rho*self.e  # residual of perturbed complementarity
                rl = self.pi*xl-self.rho*self.e  # residual of perturbed complementarity
                D1 = spdiags(self.mu/ux,0,self.n,self.n,format='coo')
                D2 = spdiags(self.pi/xl,0,self.n,self.n,format='coo')
                fbar = np.hstack((-rd+ru/ux-rl/xl,rp))
                Jbar = bmat([[problem.Hphi+D1+D2,None],
                             [-self.A,self.Omm]],format='coo')
                if not self.linsolver.is_analyzed():
                    self.linsolver.analyze(Jbar)
                p = self.linsolver.factorize_and_solve(Jbar,fbar)
                px = p[:self.n]
                plam = p[self.n:]
                pmax = norminf(p)

                # Steplength bounds
                indices = px > 0
                s1 = np.min(np.hstack(((1.-eps)*(self.u-self.x)[indices]/px[indices],np.inf)))
                indices = px < 0
                s2 = np.min(np.hstack(((eps-1.)*(self.x-self.l)[indices]/px[indices],np.inf)))
                smax = np.min([s1,s2])
                
                # Line search
                try:
                    s,fdata = self.line_search(np.hstack((self.x,self.lam)),pbar,fdata.F,fdata.GradF,self.func,smax)
                except OptSolverError_LineSearch,e:
                    print e
                    s = 0
                    if self.beta == 0:
                        self.beta = 1.
                    else:
                        self.beta*=10.

                # Update x
                self.x += s*px
                self.lam += s*plam
                self.mu = self.rho/(self.u-self.x)
                self.pi = self.rho/(self.x-self.l)
                self.k += 1

                # Check
                assert(np.all(self.x < self.u))
                assert(np.all(self.x > self.l))

        
