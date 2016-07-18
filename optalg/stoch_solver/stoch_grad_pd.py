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

class StochGradientPD(StochSolver):

    parameters = {'maxiters': 1000,
                  'period': 50,
                  'quiet' : True,
                  'theta': 1.,
                  'num_samples': 500,
                  'k0': 0,
                  'tol': 1e-4,
                  'no_G': False,
                  'callback': None}

    def __init__(self):
        """
        Primal-Dual Stochastic Gradient Algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochGradientPD.parameters.copy()

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
        no_G = params['no_G']
        callback = params['callback']

        # Header
        if not quiet:
            print '\nPrimal-Dual Stochastic Gradient'
            print '-------------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^12s}'.format('prop'),
            print '{0:^12s}'.format('lmax'),
            print '{0:^12s}'.format('EF_run'),
            print '{0:^12s}'.format('EGmax_run'),
            print '{0:^12s}'.format('EF'),
            print '{0:^12s}'.format('EGmax'),
            print '{0:^12s}'.format('info')

        # Init
        t0 = time.time()
        self.x = problem.x
        lam = np.zeros(problem.get_size_lam())
        
        # Loop
        for k in range(maxiters+1):
            
            # Sample
            w = problem.sample_w()
            
            # Eval
            F,gF,G,JG = problem.eval_FG(self.x,w,tol=tol)
            
            # Lagrangian subgradient
            gL = gF + JG.T*lam

            # Running
            if k == 0:
                EF_run = F
                EG_run = G.copy()
            else:
                EF_run += 0.05*(F-EF_run)
                EG_run += 0.05*(G-EG_run)
            
            # Show progress
            if k % period == 0:
                t1 = time.time()
                if callback:
                    callback(self.x)
                if not quiet:
                    print '{0:^8d}'.format(k),
                    print '{0:^10.2f}'.format(t1-t0),
                    print '{0:^12.5e}'.format(problem.get_prop_x(self.x)),
                    print '{0:^12.5e}'.format(np.max(lam)),
                    print '{0:^12.5e}'.format(EF_run),
                    print '{0:^12.5e}'.format(np.max(EG_run)),
                    EF,EgF,EG,EJG,info = problem.eval_EFG(self.x,samples=num_samples,tol=tol,info=True)
                    print '{0:^12.5e}'.format(EF),
                    print '{0:^12.5e}'.format(np.max(EG)),
                    print '{0:^12.5e}'.format(info)
                t0 += time.time()-t1
            
            # Update
            alpha_x = theta/(k0+k+1.)
            alpha_lam = theta/(k0+k+1.)
            self.x = problem.project_x(self.x - alpha_x*gL)
            if not no_G:
                lam = problem.project_lam(lam + alpha_lam*G)

        
    
            
