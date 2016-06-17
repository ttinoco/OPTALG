#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import time
import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix
from stoch_solver import StochSolver

class StochHybridPD(StochSolver):

    parameters = {'maxiters': 1000,
                  'period': 50,
                  'quiet' : True,
                  'theta': 1.,
                  'num_samples': 500,
                  'warm_start': False,
                  'callback': None,
                  'k0': 0,
                  'tol': 1e-4,
                  'no_G': False}

    def __init__(self):
        """
        Primal-Dual Stochastic Hybrid Approximation Algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochHybridPD.parameters.copy()

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
        callback = params['callback']
        k0 = params['k0']
        tol = params['tol']
        no_G = params['no_G']
        
        # Header
        if not quiet:
            print '\nPrimal-Dual Stochastic Hybrid'
            print '-----------------------------'
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
        sol_data = None
        t0 = time.time()
        g = np.zeros(problem.get_size_x())
        lam = np.zeros(problem.get_size_lam())
        J = coo_matrix((problem.get_size_lam(),problem.get_size_x()))
        
        # Loop
        for k in range(maxiters+1):

            # Solve approx
            if warm_start:
                self.x,sol_data = problem.solve_Lrelaxed_approx(lam,g_corr=g,J_corr=J,quiet=True,init_data=sol_data,tol=tol)
            else:
                self.x,sol_data = problem.solve_Lrelaxed_approx(lam,g_corr=g,J_corr=J,quiet=True,tol=tol)
            
            # Sample
            w = problem.sample_w()
            
            # Eval
            F,gF,G,JG = problem.eval_FG(self.x,w,tol=tol)
            
            # Running
            if k == 0:
                EF_run = F
                EG_run = G.copy()
            else:
                EF_run += 0.05*(F-EF_run)
                EG_run += 0.05*(G-EG_run)

            # Eval approx (should be able to extract this from solve_Lrelaxed_approx)
            F_approx,gF_approx,G_approx,JG_approx = problem.eval_FG_approx(self.x,tol=tol)
            
            # Output
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
            alpha_slope = theta/(k0+k+1.)
            alpha_lam = theta/(k0+k+1.)
            if not no_G:
                lam = problem.project_lam(lam + alpha_lam*G)
            g += alpha_slope*(gF-gF_approx-g)
            J = J + alpha_slope*(JG-JG_approx-J)        
    
            
