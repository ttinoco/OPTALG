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

    def solve(self,x=None,maxiters=1001,period=50,quiet=True,theta=1.,samples=500,k0=0):

        # Local vars
        prob = self.problem

        # Header
        if not quiet:
            print '\nPrimal-Dual Stochastic Gradient'
            print '-------------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^10s}'.format('prop'),
            print '{0:^10s}'.format('lmax'),
            print '{0:^12s}'.format('EF_run'),
            print '{0:^12s}'.format('EGmax_run'),
            print '{0:^12s}'.format('EF'),
            print '{0:^12s}'.format('EGmax')

        # Init
        t0 = time.time()
        lam = np.zeros(prob.get_size_lam())
        EF_run = 0.
        EG_run = np.zeros(prob.get_size_lam())
        
        # Loop
        for k in range(maxiters):
            
            # Sample
            w = prob.sample_w()
            
            # Eval
            F,gF,G,JG = prob.eval_FG(x,w,debug=False)
            
            # Lagrangian subgradient
            gL = gF + JG.T*lam

            # Running
            EF_run += 0.05*(F-EF_run)
            EG_run += 0.05*(G-EG_run)
            
            # Show progress
            if not quiet:
                t1 = time.time()
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(t1-t0),
                print '{0:^10.2f}'.format(prob.get_prop_x(x)),
                print '{0:^10.2f}'.format(np.max(lam)),
                print '{0:^12.5e}'.format(EF_run),
                print '{0:^12.5e}'.format(np.max(EG_run)),
                if k % period == 0 and False:
                    EF,EgF,EG,EJG = prob.eval_EFG(x,samples=samples)
                    print '{0:^12.5e}'.format(EF),
                    print '{0:^12.5e}'.format(np.max(EG))
                    t0 += time.time()-t1
                else:
                    print ''
            
            # Update
            alpha = theta/(k0+k+1.)
            x = prob.project_x(x - alpha*gL)
            lam = prob.project_lam(lam + alpha*G)

        return x


        
    
            
