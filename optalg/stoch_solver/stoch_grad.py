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

class StochasticGradient(StochasticSolver):

    # Constants
    RATE = 0.08

    def solve(self,x=None,maxiters=1001,period=50,quiet=True,theta=1.,samples=300,k0=0):
                
        EF = 0.
        t0 = time.time()
        for k in range(maxiters):

            w = self.problem.sample_w()
            
            F,gF = self.problem.eval_F(x,w)
            
            EF += self.RATE*(F-EF)

            if not quiet:
                t1 = time.time()
                print '%d,%.2f,%.5e,%.2f' %(k,
                                            t1-t0,
                                            EF,
                                            np.average(x/self.problem.p_max)),
                if k % period == 0:
                    print ',%.5e' %(self.problem.eval_EF(x,samples=samples)[0])
                    t0 += time.time()-t1
                else:
                    print ''
            
            alpha = theta/(k0+k+1.)
            
            xtemp = x - alpha*gF
            
            x = self.problem.project_on_X(xtemp)
            
        return x
        
    
            
