#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import time
import numpy as np
from solver import StochasticSolver

class PrimalDual_StochasticGradient(StochasticSolver):

    def solve(self,x=None,maxiters=1001,period=50,quiet=True,theta=1.,samples=500,k0=0,tol=1e-4):

        # Local vars
        prob = self.problem

        # Header
        if not quiet:
            print '\nPrimal-Dual Stochastic Gradient'
            print '-------------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^12s}'.format('prop'),
            print '{0:^17s}'.format('lmax'),
            print '{0:^12s}'.format('EF_run'),
            print '{0:^12s}'.format('EGmax_run'),
            print '{0:^12s}'.format('EF'),
            print '{0:^12s}'.format('EGmax'),
            print '{0:^12s}'.format('info')

        # Init
        t0 = time.time()
        lam = np.zeros(prob.get_size_lam())
        
        # Loop
        for k in range(maxiters):
            
            # Sample
            w = prob.sample_w()
            
            # Eval
            F,gF,G,JG = prob.eval_FG(x,w,tol=tol)
            
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
            if not quiet and k % period == 0:
                t1 = time.time()
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(t1-t0),
                print '{0:^12.2e}'.format(prob.get_prop_x(x)),
                print '{0:^17.7e}'.format(np.max(lam)),
                print '{0:^12.5e}'.format(EF_run),
                print '{0:^12.5e}'.format(np.max(EG_run)),
                EF,EgF,EG,EJG,info = prob.eval_EFG(x,samples=samples,tol=tol,info=True)
                print '{0:^12.5e}'.format(EF),
                print '{0:^12.5e}'.format(np.max(EG)),
                print '{0:^12.5e}'.format(info)
                t0 += time.time()-t1
            
            # Update
            alpha_x = theta/(k0+k+1.)
            alpha_lam = theta/(k0+k+1.)
            x = prob.project_x(x - alpha_x*gL)
            lam = prob.project_lam(lam + alpha_lam*G)

        return x


        
    
            
