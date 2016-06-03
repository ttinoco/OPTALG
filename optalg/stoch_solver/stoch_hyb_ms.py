#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import time
import numpy as np
from utils import ApplyFunc
from types import MethodType
from numpy.linalg import norm
from collections import deque
from multiprocess import Pool
from scipy.sparse import coo_matrix
from stoch_solver import StochSolver
from stoch_obj_ms_policy import StochObjMS_Policy

class MultiStage_StochHybrid(StochSolver):

    parameters = {'maxiters': 1000,
                  'num_procs': 1,
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
        k0 = self.parameters['k0']

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

            alpha = theta*phi/max([1.,k0+sum_phi])

            corr += alpha*dslope
        
        return corr

    def solve_subproblems(self,sample,g_corr,save_sol_data=False):
        
        # Local variables
        problem = self.problem
        warm_start = self.parameters['warm_start']
        debug = self.parameters['debug']
        tol = self.parameters['tol']

        # Sol data
        sol_data = self.sol_data
        if sol_data is None:
            sol_data = self.T*[None]

        # Solve subproblems
        xi_vecs = {}
        et_vecs = {}
        solutions = {-1 : problem.get_x_prev()}
        for t in range(self.T):
                
            w_list_xi = sample[:t+1]
            g_corr_xi = [g_corr[t]]
            for tau in range(t+1,self.T):
                w_list_xi.append(problem.predict_w(tau,w_list_xi))
                g_corr_xi.append(self.g(tau,w_list_xi))

            x_xi,Q_xi,gQ_xi,results_xi = problem.eval_stage_approx(t,
                                                                   w_list_xi[t:],
                                                                   solutions[t-1],
                                                                   g_corr=g_corr_xi,
                                                                   quiet=not debug,
                                                                   tol=tol,
                                                                   init_data=sol_data[t] if warm_start else None,
                                                                   next_stage=True)

            solutions[t] = x_xi
            xi_vecs[t-1] = gQ_xi
            et_vecs[t] = results_xi['gQn']
            Q_et = results_xi['Qn']
            sol_data[t] = results_xi if save_sol_data else None

            # DEBUG: Check xi
            #****************
            if debug:
                for i in range(10):
                    d = np.random.randn(self.n)*1e-3
                    xper = solutions[t-1]+d
                    x1,Q1,gQ1,results1 = problem.eval_stage_approx(t,
                                                                   w_list_xi[t:],
                                                                   xper,
                                                                   g_corr=g_corr_xi,
                                                                   quiet=True,
                                                                   tol=tol,
                                                                   init_data=sol_data[t] if warm_start else None)
                    assert(Q1 >= Q_xi+np.dot(xi_vecs[t-1],d))
                    print 'xi vec ok'

            # DEBUG: Check eta
            #*****************
            if debug and t < self.T-1:
                for i in range(10):
                    d = np.random.randn(self.n)*1e-3
                    xper = solutions[t]+d
                    x1,Q1,gQ1,results1 = problem.eval_stage_approx(t+1,
                                                                   w_list_xi[t+1:],
                                                                   xper,
                                                                   g_corr=g_corr_xi[1:],
                                                                   quiet=True,
                                                                   tol=tol,
                                                                   init_data=sol_data[t+1] if warm_start else None)
                    assert(Q1 >= Q_et+np.dot(et_vecs[t],d))
                    print 'et vec ok'

        # Return
        if save_sol_data:
            return solutions,xi_vecs,et_vecs,sol_data
        else:
            return solutions,xi_vecs,et_vecs,None
        
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
        callback = params['callback']
        num_procs = params['num_procs']
        gamma = params['gamma']

        # Pool
        pool = Pool(num_procs)
 
        # Header
        if not quiet:
            print '\nMulti-Stage Stochastic Hybrid'
            print '-----------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time'),
            print '{0:^12s}'.format('dx'),            
            print '{0:^12s}'.format('gc'),
            print '{0:^10s}'.format('samples')

        # Init
        t0 = time.time()
        self.sol_data = None
        self.samples = deque(maxlen=msize)                            # sampled realizations of uncertainty
        self.dslopes = [deque(maxlen=msize) for i in range(self.T-1)] # slope corrections (no steplengths)
        self.gammas = [gamma for t in range(self.T-1)]                # scaling factors

        # Loop
        for k in range(maxiters+1):
            
            # Subroblems data
            sample = {}
            g_corr = {}
            for i in range(num_procs):
            
                # Sample uncertainty
                sample[i] = problem.sample_W(self.T-1)
                assert(len(sample[i]) == self.T)

                # Slope corrections
                g_corr[i] = []
                for t in range(self.T):
                    Wt = sample[i][:t+1]
                    g_corr[i].append(self.g(t,Wt))

            # Solve subproblems
            tasks = [(self,
                      'solve_subproblems',
                      sample[i],
                      g_corr[i],
                      k == 0 and i == 0) for i in range(num_procs)]
            if num_procs > 1:
                results = pool.map(ApplyFunc,tasks)
            else:
                results = map(ApplyFunc,tasks)

            assert(len(results) == num_procs)

            # Save sol
            self.x = results[0][0][0]
            for i in range(num_procs):
                assert(norm(results[i][0][0]-self.x) < (1e-8/norm(self.x)))
           
            # Reference
            if k == 0:
                x0_ce = self.x.copy()
                
            # Update
            for i in range(num_procs):
                
                sol,xi_vecs,et_vecs,sol_data = results[i]

                # Sol data
                if k == 0 and i == 0:
                    self.sol_data = sol_data

                # Update samples
                self.samples.append(sample[i])
 
                # Update slopes
                for t in range(self.T-1):
                    self.dslopes[t].append(xi_vecs[t]-et_vecs[t]-g_corr[i][t])
 
            # Output
            if not quiet:
                print '{0:^8d}'.format(k),
                print '{0:^10.2f}'.format(time.time()-t0),
                print '{0:^12.5e}'.format(norm(self.x-x0_ce)),
                print '{0:^12.5e}'.format(norm(g_corr[0][0])),
                print '{0:^10d}'.format(len(self.samples))

            # Checks
            for t in range(self.T-1):
                assert(len(self.dslopes[t]) == len(self.samples))
 
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
            x,Q,gQ,results = cls.problem.eval_stage_approx(t,
                                                           w_list[t:],
                                                           x_prev,
                                                           g_corr=g_corr_pr,
                                                           quiet=True)
            
            # Check feasibility
            if not cls.problem.is_point_feasible(t,x,x_prev,Wt[-1]):
                raise ValueError('point not feasible')
            
            # Return
            return x
            
        policy = StochObjMS_Policy(self.problem,data=self,name='Multi-Stage Stochastic Hybrid')
        policy.apply = MethodType(apply,policy)
        
        # Return
        return policy
