#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import time
import numpy as np
from types import MethodType
from numpy.linalg import norm
from collections import deque
from scipy.sparse import coo_matrix
from stoch_solver import StochSolver
from stoch_obj_ms_policy import StochObjMS_Policy

class MultiStage_StochHybrid(StochSolver):

    parameters = {'maxiters': 1000,
                  'msize': 100,
                  'quiet' : True,
                  'theta': 1.,
                  'warm_start': False,
                  'callback': None,
                  'debug': False,
                  'k0': 0,
                  'gamma': 1e0,
                  'tol': 1e-4}

    def __init__(self):
        """
        Multi-Stage Stochastic Hybrid Approximation Algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = MultiStage_StochHybrid.parameters.copy()
        
        self.T = 0
        self.n = 0
        self.samples = None
        self.dslopes = None 
        self.gammas = None
        self.problem = None

    def g(self,t,Wt):
        """
        Slope correction function.
        
        Parameters
        ----------
        t : int
        Wt : list

        Returns
        -------
        g_corr : vector
        """
        
        T = self.T
        n = self.n
        samples = self.samples
        dslopes = self.dslopes
        gammas = self.gammas
        theta = self.parameters['theta']

        if t == T-1:
            return np.zeros(n)

        assert(0 <= t < T-1)
        assert(len(Wt) == t+1)
        assert(len(samples) == len(dslopes[t]))

        sum_phi = 0.
        corr = np.zeros(n)
        for i in range(len(samples)):

            W = samples[i][:t+1]
            assert(len(W) == t+1)

            dslope = dslopes[t][i]
            
            phi = np.exp(-gammas[t]*norm(np.hstack(Wt)-np.hstack(W))/np.sqrt(float(np.hstack(W).size)))
            sum_phi += phi

            alpha = theta*phi/max([1.,sum_phi])

            corr += alpha*dslope
        
        return corr

    def solve(self,problem):
        
        # Local vars
        params = self.parameters
        self.problem = problem
        self.T = problem.get_num_stages()
        self.n = problem.get_size_x()
        
        # Parameters
        maxiters = params['maxiters']
        msize = params['msize']
        quiet = params['quiet']
        warm_start = params['warm_start']
        callback = params['callback']
        debug = params['debug']
        k0 = params['k0']
        gamma = params['gamma']
        tol = params['tol']
 
        # Header
        if not quiet:
            print '\nMulti-Stage Stochastic Hybrid'
            print '-----------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time'),
            print '{0:^12s}'.format('dx'),            
            print '{0:^12s}'.format('gc'),
            print '{0:^12s}'.format('cost')

        # Init
        t0 = time.time()
        sol_data = self.T*[None]
        self.samples = deque(maxlen=msize)                            # sampled realizations of uncertainty
        self.dslopes = [deque(maxlen=msize) for i in range(self.T-1)] # slope corrections (no steplengths)
        self.gammas = [gamma for t in range(self.T-1)]                # scaling factors

        # Loop
        for k in range(maxiters+1):
            
            # Sample uncertainty
            sample = problem.sample_W(self.T-1)
            assert(len(sample) == self.T)

            # Slope corrections
            g_corr = []
            for t in range(self.T):
                Wt = sample[:t+1]
                g_corr.append(self.g(t,Wt))

            # Solve subproblems
            costs = []
            xi_vecs = {}
            et_vecs = {}
            solutions = {-1 : problem.get_x_prev()}
            for t in range(self.T):
                w_list = sample[:t+1]
                g_corr_pr = [g_corr[t]]
                for tau in range(t+1,self.T):
                    w_list.append(problem.predict_w(tau,w_list))
                    g_corr_pr.append(self.g(tau,w_list))
                x_list,Q_list,gQ_list,results = problem.eval_stage_approx(t,
                                                                          w_list[t:],
                                                                          solutions[t-1],
                                                                          g_corr=g_corr_pr,
                                                                          quiet=not debug,
                                                                          tol=tol,
                                                                          init_data=sol_data[t] if warm_start else None)
                if k == 0:
                    sol_data[t] = results
                solutions[t] = x_list[0]
                xi_vecs[t-1] = gQ_list[0]
                if t < self.T-1:
                    et_vecs[t] = gQ_list[1]
                costs.append(Q_list[0])
            self.x = solutions[0]
            
            # Reference
            if k == 0:
                x0_ce = self.x.copy()

            # Update samples
            self.samples.append(sample)

            # Update slopes
            for t in range(self.T-1):
                self.dslopes[t].append(np.zeros(self.n))#xi_vecs[t]-et_vecs[t]-g_corr[t])
                
            # Output
            if not quiet:
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(time.time()-t0),
                print '{0:^12.5e}'.format(norm(self.x-x0_ce)),
                print '{0:^12.5e}'.format(np.average(map(norm,g_corr))),
                print '{0:^12.5e}'.format(sum(costs))
                
            # Hold
            if debug:
                raw_input()
                
    def get_policy(self):
        """
        Gets operation policy.
        
        Returns
        -------
        policy : 
        """

        # Construct policy
        def apply(cls,t,x_prev,Wt):
            
            assert(0 <= t < cls.problem.T)
            assert(len(Wt) == t+1)

            solver = cls.data
            
            w_list = list(Wt)
            g_corr_pr = [solver.g(t,Wt)]
            for tau in range(t+1,self.T):
                w_list.append(cls.problem.predict_w(tau,w_list))
                g_corr_pr.append(self.g(tau,w_list))
            x_list,Q_list,gQ_list,results = cls.problem.eval_stage_approx(t,
                                                                          w_list[t:],
                                                                          x_prev,
                                                                          g_corr=g_corr_pr,
                                                                          quiet=True)
            
            # Check feasibility
            if not cls.problem.is_point_feasible(t,x_list[0],x_prev,Wt[-1]):
                raise ValueError('point not feasible')
            
            # Return
            return x_list[0]
            
        policy = StochObjMS_Policy(self.problem,data=self,name='Multi-Stage Stochastic Hybrid')
        policy.apply = MethodType(apply,policy)
        
        # Return
        return policy
