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

class PrimalDual_StochaticHybrid(StochasticSolver):

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
        lam = np.zeros(prob.get_size_lam())
        J = coo_matrix((prob.get.size_lam(),prob.get_size_x()))
        
        # Loop
        for k in range(maxiters):

            # Solve approx
            x = prob.solve_Lrelaxed_approx(lam,g_corr=g,J_corr=J,quiet=True)
            
            # Sample
            w = prob.sample_w()
            
            # Eval
            F,gF,G,JG = prob.eval_FG(x,w)

            # Eval approx
            F_approx,gF_approx,G_approx,JG_approx = prob.eval_FG_approx(x)
            
            # Output
            if not quiet:
                t1 = time.time()
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(t1-t0),
                print '{0:^10.2f}'.format(prob.get_prop_x(x)),
                if k % period == 0:
                    EF,EgF,EG,EJG = prob.eval_EFG(x,samples=samples)
                    print '{0:^12.5e}'.format(EF),
                    print '{0:^12.5e}'.format(np.max(EG))
                    t0 += time.time()-t1
                else:
                    print ''

            # Update
            lam = problem.project_lam(lam + alpha*G)
            g += (gF-gF_approx-g)/(k+1.)
            J = J + (JG-JG_approx-J)/(k+1.)

        return x
        
    
            
