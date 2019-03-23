#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2019, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from __future__ import print_function
import numpy as np
from .opt_solver_error import *
from .opt_solver import OptSolver
from .problem import cast_problem, OptProblem
from optalg.lin_solver import new_linsolver
from scipy.sparse import bmat, triu, eye, spdiags, coo_matrix, tril

class OptSolverIQP(OptSolver):
    """
    Interior-point quadratic program solver.
    """
    
    # Solver parameters
    parameters = {'tol': 1e-4,            # optimality tolerance
                  'maxiter': 1000,        # max iterations
                  'sigma': 0.1,           # factor for increasing subproblem solution accuracy
                  'eps': 1e-3,            # boundary proximity factor 
                  'eps_cold': 1e-2,       # boundary proximity factor (cold start)
                  'linsolver': 'default', # linear solver
                  'quiet': False}         # quiet flag

    def __init__(self):
        """
        Interior-point quadratic program solver.
        """
        
        # Init
        OptSolver.__init__(self)
        self.parameters = OptSolverIQP.parameters.copy()
        self.linsolver = None

    def supports_properties(self, properties):

        for p in properties:
            if p not in [OptProblem.PROP_CURV_LINEAR,
                         OptProblem.PROP_CURV_QUADRATIC,
                         OptProblem.PROP_VAR_CONTINUOUS,
                         OptProblem.PROP_TYPE_FEASIBILITY,
                         OptProblem.PROP_TYPE_OPTIMIZATION]:
                return False
        return True
        
    def solve(self, problem):
        """
        Solves optimization problem.

        Parameters
        ----------
        problem : Object
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

        # Problem
        try:
            problem = cast_problem(problem)
            quad_problem = problem.to_quad()
            self.problem = problem
            self.quad_problem = quad_problem
        except:
            raise OptSolverError_BadProblemType(self)

        # Linsolver
        self.linsolver = new_linsolver(parameters['linsolver'],'symmetric')

        # Reset
        self.reset()

        # Checks
        if not np.all(problem.l <= problem.u):
            raise OptSolverError_NoInterior(self)
    
        # Data
        self.H = quad_problem.H
        self.g = quad_problem.g
        self.A = quad_problem.A
        self.AT = quad_problem.A.T
        self.b = quad_problem.b
        self.l = quad_problem.l-tol/10.
        self.u = quad_problem.u+tol/10.
        self.n = quad_problem.H.shape[0]
        self.m = quad_problem.A.shape[0]
        self.e = np.ones(self.n)
        self.I = eye(self.n,format='coo')
        self.Onm = coo_matrix((self.n,self.m))
        self.Omm = coo_matrix((self.m,self.m))

        # Initial primal
        if quad_problem.x is None:
            self.x = (self.u + self.l)/2.
        else:
            self.x = np.maximum(np.minimum(quad_problem.x,problem.u),problem.l)

        # Initial duals
        if quad_problem.lam is None:
            self.lam = np.zeros(self.m)
        else:
            self.lam = quad_problem.lam.copy()
        if quad_problem.mu is None:
            self.mu = np.ones(self.x.size)*eps_cold
        else:
            self.mu = np.maximum(quad_problem.mu,eps)
        if quad_problem.pi is None:
            self.pi = np.ones(self.x.size)*eps_cold
        else:
            self.pi = np.maximum(quad_problem.pi,eps)

        # Check interior
        try:
            assert(np.all(self.l < self.x)) 
            assert(np.all(self.x < self.u))
            assert(np.all(self.mu > 0))
            assert(np.all(self.pi > 0))
        except AssertionError:
            raise OptSolverError_Infeasibility(self)

        # Init vector
        self.y = np.hstack((self.x,self.lam,self.mu,self.pi))

        # Complementarity measures
        self.eta_mu = np.dot(self.mu,self.u-self.x)/self.x.size
        self.eta_pi = np.dot(self.pi,self.x-self.l)/self.x.size

        # Objective scaling
        fdata = self.func(self.y)
        self.obj_sca = np.maximum(norminf(self.g+self.H*self.x)/10.,1.)
        self.H = self.H/self.obj_sca
        self.g = self.g/self.obj_sca
        fdata = self.func(self.y)

        # Header
        if not quiet:
            print('\nSolver: IQP')
            print('-----------')
                                   
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
                    print('')
                print('{0:^3s}'.format('iter'), end=' ')
                print('{0:^9s}'.format('phi'), end=' ')
                print('{0:^9s}'.format('fmax'), end=' ')
                print('{0:^9s}'.format('gmax'), end=' ')
                print('{0:^8s}'.format('cu'), end=' ')
                print('{0:^8s}'.format('cl'), end=' ')
                print('{0:^8s}'.format('s'))
 
            # Inner
            while True:
                
                # Eval
                fdata = self.func(self.y)
                fmax = norminf(fdata.f)
                gmax = norminf(fdata.GradF)
                compu = norminf(self.mu*(self.u-self.x))
                compl = norminf(self.pi*(self.x-self.l))
                phi = (0.5*np.dot(self.x,self.H*self.x)+np.dot(self.g,self.x))*self.obj_sca
                
                # Show progress
                if not quiet:
                    print('{0:^3d}'.format(self.k), end=' ')
                    print('{0:^9.2e}'.format(phi), end=' ')
                    print('{0:^9.2e}'.format(fmax), end=' ')
                    print('{0:^9.2e}'.format(gmax), end=' ')
                    print('{0:^8.1e}'.format(compu), end=' ')
                    print('{0:^8.1e}'.format(compl), end=' ')
                    print('{0:^8.1e}'.format(s))
                
                # Done
                if gmax < tau:
                    break

                # Done
                if fmax < tol and np.maximum(compu,compl) < tol:
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
                if self.A.shape[0] > 0:
                    Jbar = bmat([[tril(self.H)+D1+D2,None],
                                 [-self.A,self.Omm]],format='coo')
                else:
                    Jbar = bmat([[tril(self.H)+D1+D2]],
                                format='coo')
                try:
                    if not self.linsolver.is_analyzed():
                        self.linsolver.analyze(Jbar)
                    pbar = self.linsolver.factorize_and_solve(Jbar,fbar)
                except RuntimeError:
                    raise OptSolverError_BadLinSystem(self)
                px = pbar[:self.n]
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
                try:
                    assert(np.all(self.x < self.u))
                    assert(np.all(self.x > self.l))
                    assert(np.all(self.mu > 0))
                    assert(np.all(self.pi > 0))
                except AssertionError:
                    raise OptSolverError_Infeasibility(self)


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

        x,lam,mu,pi = self.extract_components(y)
        ux = self.u-x
        xl = x-self.l
        
        rd = self.H*x+self.g-self.AT*lam+mu-pi       # dual residual
        rp = self.A*x-self.b                         # primal residual
        ru = mu*ux-sigma*self.eta_mu*self.e  # residual of perturbed complementarity
        rl = pi*xl-sigma*self.eta_pi*self.e  # residual of perturbed complementarity
        
        Dmu = spdiags(self.mu,0,self.n,self.n) 
        Dux = spdiags(ux,0,self.n,self.n)
        Dpi = spdiags(self.pi,0,self.n,self.n)
        Dxl = spdiags(xl,0,self.n,self.n)

        f = np.hstack((rd,rp,ru,rl))
        
        J = bmat([[self.H,-self.AT,self.I,-self.I],
                  [self.A,None,None,None],
                  [-Dmu,None,Dux,None],
                  [Dpi,self.Onm,None,Dxl]])

        fdata.rd = rd
        fdata.rp = rp
        fdata.ru = ru
        fdata.rl = rl
        fdata.f = f
        fdata.J = J
        
        fdata.F = 0.5*np.dot(f,f) # merit function
        fdata.GradF = J.T*f       # gradient of merit function

        return fdata
