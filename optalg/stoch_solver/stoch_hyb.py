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
from solver import StochasticSolver

class StochasticHybrid(StochasticSolver):

    def solve(self,maxiters=1001,period=50,quiet=True,samples=500,warm_start=False):

        # Local vars
        prob = self.problem

        # Header
        if not quiet:
            print '\nStochastic Hybrid'
            print '-----------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^10s}'.format('prop'),
            print '{0:^12s}'.format('EF')

        # Init
        sol_data = None
        t0 = time.time()
        g = np.zeros(prob.get_size_x())
        
        # Loop
        for k in range(maxiters):

            # Solve approx
            if warm_start:
                x,sol_data = prob.solve_approx(g_corr=g,quiet=True,init_data=sol_data)
            else:
                x,sol_data = prob.solve_approx(g_corr=g,quiet=True)                
            
            # Sample
            w = prob.sample_w()
            
            # Eval
            F,gF = prob.eval_F(x,w)

            # Eval approx
            F_approx,gF_approx = prob.eval_F_approx(x)
            
            # Output
            if not quiet:
                t1 = time.time()
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(t1-t0),
                print '{0:^10.2f}'.format(prob.get_prop_x(x)),
                if k % period == 0:
                    print '{0:^12.5e}'.format(prob.eval_EF(x,samples=samples)[0])
                    t0 += time.time()-t1
                else:
                    print ''

            # Update
            g += (gF-gF_approx-g)/(k+1.)

        return x
        
    
            
