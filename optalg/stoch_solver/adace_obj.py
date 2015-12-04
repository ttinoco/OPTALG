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

    # Constants
    RATE = 0.05

    def solve(self,maxiters=1001,period=50,quiet=True,samples=500):

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
                print '%d,%.2f,%.2e,%.2f,%.2f,' %(k,
                                                  t1-t0,
                                                  Fce+np.dot(g,x),
                                                  np.average(x/self.problem.p_max),
                                                  100.*(F-Fce-np.dot(g,x))/F),
                if k % period == 0:
                    EF,EgF = self.problem.eval_EF(x,samples=samples)
                    print '%.5e,%.2f,%.2f' %(EF,
                                             100.*(EF-Fce-np.dot(g,x))/EF,
                                             np.dot(EgF,gFce+g)/(norm(EgF)*norm(gFce+g)))
                    t0 += time.time()-t1
                else:
                    print ''

            # Update
            g += (1./(k+1.))*(gF-gFce-g)

        return x
        
    
            
