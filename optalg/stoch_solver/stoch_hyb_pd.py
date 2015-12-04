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

class AdaCE_PrimalDual(StochasticSolver):

    def solve(self,maxiters=1001,period=50,quiet=True,samples=500):

        # Local vars
        prob = self.problem

        # Header
        if not quiet:
            print '\Primal-Dual AdaCE'
            print '-----------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^10s}'.format('prop'),
            print '{0:^12s}'.format('EF'),
            print '{0:^12s}'.format('EGmax')

        # Init
        t0 = time.time()
        g = np.zeros(prob.get_size_x())
        J = coo_matrix((prob.get.size_lam(),prob.get_size_x()))
        Ew = prob.get_Ew(samples=samples)
        
        # Iterations
        for k in range(maxiters):

            # Solve approx
            x = prob.solve_certainty_equivalent(g_corr=g,Ew=Ew,quiet=True)
            
            # Sample
            w = prob.sample_w()
            
            # Eval
            F,gF = prob.eval_F(x,w)

            # Eval CE
            Fce,gFce = prob.eval_F(x,Ew)
            
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
            g += (gF-gFce-g)/(k+1.)

        return x
        
    
            
