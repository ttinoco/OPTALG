#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from __future__ import print_function
import time
import numpy as np
from numpy.linalg import norm
from .stoch_solver import StochSolver

class StochHybrid(StochSolver):

    parameters = {'maxiters': 1000,
                  'period': 50,
                  'quiet' : True,
                  'theta': 1.,
                  'num_samples': 500,
                  'warm_start': False,
                  'k0': 0,
                  'tol': 1e-4}

    def __init__(self):
        """
        Stochastic Hybrid Approximation Algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochHybrid.parameters.copy()

    def solve(self,problem):

        # Local vars
        params = self.parameters

        # Parameters
        maxiters = params['maxiters']
        period = params['period']
        quiet = params['quiet']
        theta = params['theta']
        num_samples = params['num_samples']
        warm_start = params['warm_start']
        k0 = params['k0']
        tol = params['tol']

        # Header
        if not quiet:
            print('\nStochastic Hybrid')
            print('-----------------')
            print('{0:^8s}'.format('iter'), end=' ')
            print('{0:^10s}'.format('time(s)'), end=' ')
            print('{0:^12s}'.format('prop'), end=' ')
            print('{0:^12s}'.format('EF'))

        # Init
        sol_data = None
        t0 = time.time()
        g = np.zeros(problem.get_size_x())
        
        # Loop
        for k in range(maxiters+1):

            # Solve approx
            if warm_start:
                self.x,sol_data = problem.solve_approx(g_corr=g,quiet=True,init_data=sol_data,tol=tol)
            else:
                self.x,sol_data = problem.solve_approx(g_corr=g,quiet=True,tol=tol)                
            
            # Sample
            w = problem.sample_w()
            
            # Eval
            F,gF = problem.eval_F(self.x,w,tol=tol)

            # Eval approx (should be extracted from solve_approx)
            F_approx,gF_approx = problem.eval_F_approx(self.x,tol=tol)
            
            # Output
            if k % period == 0:
                t1 = time.time()
                if not quiet:
                    print('{0:^8d}'.format(k), end=' ')
                    print('{0:^10.2f}'.format(t1-t0), end=' ')
                    print('{0:^12.5e}'.format(problem.get_prop_x(self.x)), end=' ')
                    EF,EgF = problem.eval_EF(x,samples=num_samples,tol=tol)
                    print('{0:^12.5e}'.format(EF))
                t0 += time.time()-t1

            # Update
            alpha = theta/(k0+k+1.)
            g += alpha*(gF-gF_approx-g)        
    
            
