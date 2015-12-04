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

    def solve(self,x=None,maxiters=1001,period=50,quiet=True,theta=1.,samples=500,k0=0):

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
            w = self.problem.sample_w()
            
            # Eval
            F,gF = self.problem.eval_F(x,w)
            
            # Show progress
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
            alpha = theta/(k0+k+1.)
            xtemp = x - alpha*gF
            x = self.problem.project_on_X(xtemp)
            
        return x


        
    
            
