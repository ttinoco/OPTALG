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

class StochasticGradientMB(StochasticSolver):
    """
    Model-based stochastic gradient.
    """

    # Constants
    RATE = 0.08

    def solve(self,x=None,maxiters=1001,period=50,quiet=True,theta=1.,samples=300,k0=0):
                
        EF = 0.
        t0 = time.time()
        for k in range(maxiters):

            w = self.problem.sample_w()
            
            F,gF = self.problem.eval_F(x,w,approx=False)
            Fa,gFa = self.problem.eval_F(x,w,approx=True)
            
            self.problem.update_Fapprox(F,gF,Fa,gFa,k)
            
            EF += self.RATE*(F-EF)

            if not quiet:
                t1 = time.time()
                print '%d,%.2f,%.5e,%.2f,%.2f,%.2e' %(k,
                                                      t1-t0,
                                                      EF,
                                                      np.average(x/self.problem.p_max),
                                                      100.*(F-Fa)/F,
                                                      self.problem.Qapprox_sigma),
                if k % period == 0:
                    
                    EF,EgF = self.problem.eval_EF(x,samples=samples,approx=False)
                    EFa,EgFa = self.problem.eval_EF(x,samples=samples,approx=True)
                    
                    print '\t,%.5e,%.2f,%.2f' %(EF,
                                                100.*(EF-EFa)/EF,
                                                100.*norm(EgF-EgFa-self.problem.Qapprox_g)/norm(EgF))

                    t0 += time.time()-t1
                else:
                    print ''
            
            alpha = theta/(k0+k+1.)
            
            xtemp = x - alpha*gF
            
            x = self.problem.project_on_X(xtemp)
            
        return x
        
    
            
