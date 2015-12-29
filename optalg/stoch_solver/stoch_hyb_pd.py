#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import time
import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix
from solver import StochasticSolver

class PrimalDual_StochasticHybrid(StochasticSolver):

    def solve(self,maxiters=1001,period=50,quiet=True,samples=500,k0=0,theta=1.,warm_start=False,tol=1e-4):
        
        # Local vars
        prob = self.problem

        # Header
        if not quiet:
            print '\nPrimal-Dual AdaCE'
            print '-----------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^12s}'.format('prop'),
            print '{0:^17s}'.format('lmax'),
            print '{0:^12s}'.format('EF_run'),
            print '{0:^12s}'.format('EGmax_run'),
            print '{0:^12s}'.format('EF'),
            print '{0:^12s}'.format('EGmax')

        # Init
        sol_data = None
        t0 = time.time()
        g = np.zeros(prob.get_size_x())
        lam = np.zeros(prob.get_size_lam())
        J = coo_matrix((prob.get_size_lam(),prob.get_size_x()))
        
        # Loop
        for k in range(maxiters):

            # Solve approx
            if warm_start:
                x,sol_data = prob.solve_Lrelaxed_approx(lam,g_corr=g,J_corr=J,quiet=True,init_data=sol_data,tol=tol)
            else:
                x,sol_data = prob.solve_Lrelaxed_approx(lam,g_corr=g,J_corr=J,quiet=True,tol=tol)
            
            # Sample
            w = prob.sample_w()
            
            # Eval
            F,gF,G,JG = prob.eval_FG(x,w,tol=tol)
            
            # Running
            if k == 0:
                EF_run = F
                EG_run = G.copy()
            else:
                EF_run += 0.05*(F-EF_run)
                EG_run += 0.05*(G-EG_run)

            # Eval approx
            F_approx,gF_approx,G_approx,JG_approx = prob.eval_FG_approx(x,tol=tol)
            
            # Output
            if not quiet:
                t1 = time.time()
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(t1-t0),
                print '{0:^12.2e}'.format(prob.get_prop_x(x)),
                print '{0:^17.7e}'.format(np.max(lam)),
                print '{0:^12.5e}'.format(EF_run),
                print '{0:^12.5e}'.format(np.max(EG_run)),
                if k % period == 0:
                    EF,EgF,EG,EJG = prob.eval_EFG(x,samples=samples,tol=tol)
                    print '{0:^12.5e}'.format(EF),
                    print '{0:^12.5e}'.format(np.max(EG))
                    t0 += time.time()-t1
                else:
                    print ''
                    
            # Update
            alpha_slope = theta/(k0+k+1.)
            alpha_lam = theta/(k0+k+1.)
            lam = prob.project_lam(lam + alpha_lam*G)
            g += alpha_slope*(gF-gF_approx-g)
            J = J + alpha_slope*(JG-JG_approx-J)

        return x
        
    
            
