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
from solver import StochasticSolver

class MultiStage_StochasticHybrid(StochasticSolver):

    def solve(self,maxiters=1001,msize=100,quiet=True,k0=0,theta=1.,warm_start=False,tol=1e-4,callback=None,debug=False):
        """
        Solves multi stage stochastic optimization problem

        Parameters
        ----------
        a bunch

        Returns
        -------
        policy : function
        """
        
        # Local vars
        prob = self.problem
        T = prob.get_num_stages()
        n = prob.get_size_x()
        
        # Time
        t0 = time.time()

        # Header
        if not quiet:
            print '\nMulti-Stage Stochastic Hybrid'
            print '-----------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^12s}'.format('dx0'),
            print '{0:^12s}'.format('cost0')

        # Init
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
        for k in range(maxiters):
            
            # Sample uncertainty
            sample = []
            for t in range(T):
                sample.append(prob.sample_w(t,sample))
            assert(len(sample) == T)

            # Slope corrections
            g_corr = []
            for t in range(T):
                Wt = sample[:t+1]
                g_corr.append(g(t,Wt,samples,dslopes))

            # Solve subproblems
            costs = []
            solutions = {-1 : prob.get_x_prev()}
            for t in range(T):
                w_list = sample[:t+1]
                g_corr_pr = [g_corr[t]]
                for tau in range(t+1,T):
                    w_list.append(prob.predict_w(tau,w_list))
                    g_corr_pr.append(g(tau,w_list,samples,dslopes))
                xt,Qt,gQt = prob.eval_stage_approx(t,
                                                   w_list[t:],
                                                   solutions[t-1],
                                                   g_corr=g_corr_pr,
                                                   quiet=not debug)
                solutions[t] = xt
                costs.append(Qt)

            # Update samples
            samples.append(sample)

            # Update slopes
            for t in range(T-1,0,-1):
                w_list_xi = sample[:t+1]
                w_list_et = sample[:t]+[prob.predict_w(t,sample[:t])]
                g_corr_xi = [g(t,w_list_xi,samples,dslopes)]
                g_corr_et = [g(t,w_list_et,samples,dslopes)]
                for tau in range(t+1,T):
                    w_list_xi.append(prob.predict_w(tau,w_list_xi))
                    w_list_et.append(prob.predict_w(tau,w_list_et))
                    g_corr_xi.append(g(tau,w_list_xi,samples,dslopes))
                    g_corr_et.append(g(tau,w_list_et,samples,dslopes))
                xt_xi,Qt_xi,gQt_xi = prob.eval_stage_approx(t,
                                                            w_list_xi[t:],
                                                            solutions[t-1],
                                                            g_corr=g_corr_xi,
                                                            quiet=not debug)
                xt_et,Qt_et,gQt_et = prob.eval_stage_approx(t,
                                                            w_list_et[t:],
                                                            solutions[t-1],
                                                            g_corr=g_corr_et,
                                                            quiet=not debug)
                alpha = theta/(k0+k+1.)
                dslopes[t-1].append(alpha*(gQt_xi-gQt_et-g_corr[t-1]))
                
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
