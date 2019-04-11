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
from .problem import cast_problem, OptProblem
from .opt_solver import OptSolver
from scipy.sparse import bmat
from optalg.lin_solver import new_linsolver

class OptSolverNR(OptSolver):
    
    parameters = {'feastol':1e-4,
                  'acc_factor': 1.,
                  'maxiter':100,
                  'linsolver':'default',
                  'quiet':False}

    def __init__(self):
        """
        Newton-Raphson algorithm.
        """
        
        # Init
        OptSolver.__init__(self)
        self.parameters = OptSolverNR.parameters.copy()     
        self.linsolver = None
        self.problem = None

    def supports_properties(self, properties):

        for p in properties:
            if p not in [OptProblem.PROP_CURV_LINEAR,
                         OptProblem.PROP_CURV_QUADRATIC,
                         OptProblem.PROP_CURV_NONLINEAR,
                         OptProblem.PROP_VAR_CONTINUOUS,
                         OptProblem.PROP_TYPE_FEASIBILITY]:
                return False
        return True
        
    def func(self, x):

        fdata = self.fdata
        p = self.problem

        p.eval(x)
        
        J = p.J
        f = p.f
        fTf = np.dot(f,f)
        JTf = J.T*f

        A = p.A
        r = A*x-p.b
        rTr = np.dot(r,r)
        ATr = A.T*r

        fdata.f = f
        fdata.r = r
        
        fdata.F = 0.5*(fTf+rTr)
        fdata.GradF = JTf+ATr
        return fdata

    def solve(self, problem):
    
        # Local vars
        norm2 = self.norm2
        norminf = self.norminf
        params = self.parameters

        # Parameters
        feastol = params['feastol']
        maxiter = params['maxiter']
        quiet = params['quiet']
        acc_factor = params['acc_factor']

        # Linear solver
        self.linsolver = new_linsolver(params['linsolver'],'unsymmetric')

        # Problem
        problem = cast_problem(problem)
        self.problem = problem

        # Reset
        self.reset()
                
        # Init point
        if problem.x is not None:
            self.x = problem.x.copy()
        else:
            raise OptSolverError_BadInitPoint(self)
            
        # Init eval
        fdata = self.func(self.x)
            
        # Print header
        if not quiet:
            print('\nSolver: NR')
            print('----------')
            print('{0:^3}'.format('k'), end=' ')
            print('{0:^9}'.format('fmax'), end=' ')
            print('{0:^9}'.format('gmax'), end=' ')
            print('{0:^8}'.format('pmax'), end=' ')
            print('{0:^8}'.format('alpha'), end=' ')
            if self.info_printer:
                self.info_printer(self,True)
            else:
                print('')

        # Main loop
        s = 0.         
        pmax = 0.      
        self.k = 0
        analyzed = False
        while True:
            
            # Callbacks
            for c in self.callbacks:
                c(self)
            fdata = self.func(self.x)
                        
            # Compute info quantities
            fmax = np.maximum(norminf(fdata.f),norminf(fdata.r))
            gmax = norminf(fdata.GradF)

            # Show progress
            if not quiet:
                print('{0:^3d}'.format(self.k), end=' ')
                print('{0:^9.2e}'.format(fmax), end=' ')
                print('{0:^9.2e}'.format(gmax), end=' ')
                print('{0:^8.1e}'.format(pmax), end=' ')
                print('{0:^8.1e}'.format(s), end=' ')
                if self.info_printer:
                    self.info_printer(self,False)
                else:
                    print('')
                
            # Check solved
            if fmax < feastol:
                self.set_status(self.STATUS_SOLVED)
                self.set_error_msg('')
                return

            # Check maxiters
            if self.k >= maxiter:
                raise OptSolverError_MaxIters(self)
            
            # Check custom terminations
            for t in self.terminations:
                t(self)
            
            # Search direction
            try:
                if not analyzed:
                    self.linsolver.analyze(bmat([[problem.J],[problem.A]]))
                    analyzed = True
                p = self.linsolver.factorize_and_solve(bmat([[problem.J],[problem.A]]),
                                                       np.hstack([-fdata.f,-fdata.r]))
            except Exception:
                raise OptSolverError_BadLinSystem(self)
            pmax = norminf(p)

            # Line search
            s,fdata = self.line_search(self.x,p,fdata.F,fdata.GradF,self.func)

            # Update x
            self.x += acc_factor*s*p
            self.k += 1
