#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import time
import numpy as np
from stoch_solver import StochSolver

class StochGradient(StochSolver):

    parameters = {'maxiters': 1000,
                  'period': 50,
                  'quiet' : True,
                  'theta': 1.,
                  'num_samples': 500,
                  'k0': 0
                  'tol': 1e-4}
    
    def __init__(self):
        """
        Stochastic Gradient Algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochGradient.parameters.copy()

    def solve(self,problem):
        
        # Local vars
        params = self.parameters

        # Parameters
        maxiters = params['maxiters']
        period = params['period']
        quiet = params['quiet']
        theta = params['theta']
        num_samples = params['num_samples']
        k0 = params['k0']
        tol = params['tol']

        # Header
        if not quiet:
            print '\nStochastic Gradient'
            print '-------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^12s}'.format('prop'),
            print '{0:^12s}'.format('EF')

        # Init
        t0 = time.time()
        self.x = problem.x
        
        # Loop
        for k in range(maxiters+1):
            
            # Sample
            w = problem.sample_w()
            
            # Eval
            F,gF = problem.eval_F(self.x,w,tol=tol)
            
            # Show progress
            if k % period == 0:
                t1 = time.time()
                if not quiet:
                    print '{0:^8d}'.format(k),
                    print '{0:^10.2f}'.format(t1-t0),
                    print '{0:^12.5e}'.format(problem.get_prop_x(self.x)),
                    EF,EgF = problem.eval_EF(x,samples=num_samples,tol=tol)
                    print '{0:^12.5e}'.format(EF)
                t0 += time.time()-t1
            
            # Update
            alpha = theta/(k0+k+1.)
            self.x = problem.project_x(self.x - alpha*gF)

        
    
            
