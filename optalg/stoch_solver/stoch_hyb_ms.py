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

    def solve(self,maxiters=1001,msize=100,quiet=True,k0=0,theta=1.,warm_start=False,tol=1e-4,callback=None):
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

        # Header
        if not quiet:
            print 'Multi-Stage Stochastic Hybrid'
            print '-----------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time(s)'),
            print '{0:^12s}'.format('prop'),
            print '{0:^12s}'.format('cost'),
            print '{0:^12s}'.format('info')

        # Init
        samples = deque(maxlen=msize)           # sampled realizations of uncertainty
        dslopes = deque(maxlen=msize)           # slope corrections (includes steplengths)
        gammas = [1./(t+1.) for t in range(T)]  # scaling factors

        # Slope correction function
        def g(t,Wt,samples,dslopes):
            
            assert(0 <= t < T)
            assert(len(Wt) == t+1)
            assert(len(samples) == len(dslopes))

            corr = np.zeros(n)
            for i in range(len(samples)):

                W = samples[i][:t+1]
                assert(len(W) == t+1)

                dslope = dslopes[i]

                corr += dslope*np.exp(-gammas[t]*(norm(Wt-W)**2.))
                
            return corr

        # Loop
        for k in range(maxiters):
            
            # Sample uncertainty
            sample = []
            for t in range(T):
                sample.append(prob.sample_w(t,sample))
            assert(len(sample) == T)

            # Compute slope corrections
            g_corr = []
            for t in range(T):
                Wt = sample[:t+1]
                assert(len(Wt) == t+1)
                g_corr.append(g(t,Wt,samples,dslopes))
            assert(len(g_corr) == T)

            # Solve subproblems
            solutions = {-1 : prob.x_prev.copy()}
            for t in range(T):
                w_list = sample[:t+1]
                for tau in range(t+1,T):
                    w_list.append(prob.predict_w(tau,w_list))
                xt,Qt,gQt = prob.eval_stage_approx(t,
                                                   w_list[t:],
                                                   solutions[t-1],
                                                   g_corr=g_corr[t:])
                solutions[t] = xt

            # Compute subgradients
            eta_list = []
            mu_list = []
            for t in range(T-1,-1,-1):
                w_list_eta = sample[:t+1]
                w_list_mu = sample[:t]+[prob.predict_w(t,sample[:t])]
                for tau in range(t+1,T):
                    w_list_eta.append(prob.predict_w(tau,w_list_eta))
                    w_list_mu.append(prob.predict_w(tau,w_list_mu))
                    
                    
                    
                
                
            break
                                                   
                
