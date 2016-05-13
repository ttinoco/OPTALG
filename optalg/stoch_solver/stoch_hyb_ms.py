#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import time
import numpy as np
from numpy.linalg import norm
from collections import deque
from scipy.sparse import coo_matrix
from stoch_solver import StochSolver

class MultiStage_StochHybrid(StochSolver):

    parameters = {'maxiters': 1000,
                  'msize': 100,
                  'quiet' : True,
                  'theta': 1.,
                  'warm_start': False,
                  'callback': None,
                  'debug': False,
                  'k0': 0,
                  'tol': 1e-4}

    def __init__(self):
        """
        Multi-Stage Stochastic Hybrid Approximation Algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = MultiStage_StochHybrid.parameters.copy()

    def solve(self,problem):
        
        # Local vars
        params = self.parameters
        T = problem.get_num_stages()
        n = problem.get_size_x()
       
        # Parameters
        maxiters = params['maxiters']
        msize = params['msize']
        quiet = params['quiet']
        theta = params['theta']
        warm_start = params['warm_start']
        callback = params['callback']
        debug = params['debug']
        k0 = params['k0']
        tol = params['tol']
 
        # Header
        if not quiet:
            print '\nMulti-Stage Stochastic Hybrid'
            print '-----------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^12s}'.format('dx0'),
            print '{0:^12s}'.format('cost0')

        # Init
        t0 = time.time()
        x0_prev = np.zeros(n)
        samples = deque(maxlen=msize)                       # sampled realizations of uncertainty
        dslopes = [deque(maxlen=msize) for i in range(T-1)] # slope corrections (includes steplengths)
        gammas = [1./(t+1.) for t in range(T-1)]            # scaling factors

        # Slope correction function
        def g(t,Wt,samples,dslopes):

            if t == T-1:
                return np.zeros(n)

            assert(0 <= t < T-1)
            assert(len(Wt) == t+1)
            assert(len(samples) == len(dslopes[t]))

            corr = np.zeros(n)
            for i in range(len(samples)):

                W = samples[i][:t+1]
                assert(len(W) == t+1)

                dslope = dslopes[t][i]

                corr += dslope*np.exp(-gammas[t]*(norm(np.array(Wt)-np.array(W))**2.))
                
            return corr

        # Loop
        for k in range(maxiters+1):
            
            # Sample uncertainty
            sample = []
            for t in range(T):
                sample.append(problem.sample_w(t,sample))
            assert(len(sample) == T)

            # Slope corrections
            g_corr = []
            for t in range(T):
                Wt = sample[:t+1]
                g_corr.append(g(t,Wt,samples,dslopes))

            # Solve subproblems
            costs = []
            xi_vecs = {}
            et_vecs = {}
            solutions = {-1 : problem.get_x_prev()}
            for t in range(T):
                w_list = sample[:t+1]
                g_corr_pr = [g_corr[t]]
                for tau in range(t+1,T):
                    w_list.append(problem.predict_w(tau,w_list))
                    g_corr_pr.append(g(tau,w_list,samples,dslopes))
                xt,Qt,gQt,gQtt = problem.eval_stage_approx(t,
                                                           w_list[t:],
                                                           solutions[t-1],
                                                           g_corr=g_corr_pr,
                                                           quiet=not debug)
                solutions[t] = xt
                xi_vecs[t-1] = gQt
                et_vecs[t] = gQtt
                costs.append(Qt)
            self.x = solutions[0]

            # Update samples
            samples.append(sample)

            # Update slopes
            for t in range(T-1):
                alpha = theta/(k0+k+1.)
                dslopes[t-1].append(alpha*(xi_vecs[t]-et_vecs[t]-g_corr[t-1]))
                
            # Output
            if not quiet:
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(time.time()-t0),
                print '{0:^12.5e}'.format(norm(solutions[0]-x0_prev)/norm(solutions[0])),
                print '{0:^12.5e}'.format(costs[0])
                
            # Hold
            if debug:
                raw_input()
                
            # Update
            x0_prev = solutions[0]
