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

class AdaCE(StochasticSolver):

    def solve(self,maxiters=1001,period=50,quiet=True,samples=500):

        # Header
        if not quiet:
            print '\nAdaCE'
            print '-----'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^10s}'.format('prop'),
            print '{0:^12s}'.format('EF')

        # Init
        t0 = time.time()
        g = np.zeros(self.problem.get_size_x())
        Ew = self.problem.get_Ew(samples=samples)
        
        # Iterations
        for k in range(maxiters):

            # Solve approx
            x = self.problem.solve_certainty_equivalent(g_corr=g,Ew=Ew,quiet=True)
            
            # Sample
            w = self.problem.sample_w()
            
            # Eval
            F,gF = self.problem.eval_F(x,w)

            # Eval CE
            Fce,gFce = self.problem.eval_F(x,Ew)
            
            # Output
            if not quiet:
                t1 = time.time()
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(t1-t0),
                print '{0:^10.2f}'.format(self.problem.get_prop_x(x)),
                if k % period == 0:
                    print '{0:^12.5e}'.format(self.problem.eval_EF(x,samples=samples)[0])
                    t0 += time.time()-t1
                else:
                    print ''

            # Update
            g += (gF-gFce-g)/(k+1.)

        return x
        
    
            
