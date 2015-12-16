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

    def solve(self,x=None,maxiters=1001,period=50,quiet=True,theta=1.,samples=500,k0=0,tol=1e-4):

        # Local vars
        prob = self.problem

        # Header
        if not quiet:
            print '\nStochastic Gradient'
            print '-------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^10s}'.format('prop'),
            print '{0:^12s}'.format('EF')

        # Init
        t0 = time.time()
        
        # Loop
        for k in range(maxiters):
            
            # Sample
            w = prob.sample_w()
            
            # Eval
            F,gF = prob.eval_F(x,w,tol=tol)
            
            # Show progress
            if not quiet:
                t1 = time.time()
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(t1-t0),
                print '{0:^10.2e}'.format(prob.get_prop_x(x)),
                if k % period == 0:
                    print '{0:^12.5e}'.format(prob.eval_EF(x,samples=samples,tol=tol)[0])
                    t0 += time.time()-t1
                else:
                    print ''
            
            # Update
            alpha = theta/(k0+k+1.)
            x = prob.project_x(x - alpha*gF)
            
        return x


        
    
            
