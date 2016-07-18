#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from __future__ import print_function
import time
import dill
import numpy as np
from .utils import ApplyFunc
from types import MethodType
from numpy.linalg import norm
from collections import deque
from scipy.sparse import coo_matrix
from .stoch_solver import StochSolver
from .problem_ms_policy import StochProblemMS_Policy

class StochHybridMS(StochSolver):

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
                  'key_iters': None,
                  'outdir': '',
                  'tol': 1e-4}

    def __init__(self):
        """
        Multi-Stage stochastic hybrid approximation algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochHybridMS.parameters.copy()
        
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
        """ 
        Solves subproblems.

        Parameters
        ----------
        sampled : list of uncertainy relizations for each stage
        g_corr : list of slope corrections for each stage
        save_sol_data : flag for saving solution data

        Results
        -------
        solutions : dict (stage solutions)
        xi_vects : dict (noisy subgradients)
        et_vects : dict (model subgradients)
        sol_data : list (solution data)
        """ 
        
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

            x_xi,Q_xi,gQ_xi,results_xi = problem.solve_stages(t,
                                                              w_list_xi[t:],
                                                              solutions[t-1],
                                                              g_corr=g_corr_xi,
                                                              quiet=True,
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
                    x1,Q1,gQ1,results1 = problem.solve_stages(t,
                                                              w_list_xi[t:],
                                                              xper,
                                                              g_corr=g_corr_xi,
                                                              quiet=True,
                                                              tol=tol,
                                                              init_data=sol_data[t] if warm_start else None)
                    assert(Q1+1e-8 >= Q_xi+np.dot(xi_vecs[t-1],d))
                    print('xi vec ok')

            # DEBUG: Check eta
            #*****************
            if debug and t < self.T-1:
                for i in range(10):
                    d = np.random.randn(self.n)*1e-3
                    xper = solutions[t]+d
                    x1,Q1,gQ1,results1 = problem.solve_stages(t+1,
                                                              w_list_xi[t+1:],
                                                              xper,
                                                              g_corr=g_corr_xi[1:],
                                                              quiet=True,
                                                              tol=tol,
                                                              init_data=sol_data[t+1] if warm_start else None)
                    assert(Q1+1e-8 >= Q_et+np.dot(et_vecs[t],d))
                    print('et vec ok')

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
        key_iters = params['key_iters']
        outdir = params['outdir']

        # Pool
        from multiprocess import Pool
        pool = Pool(num_procs)
 
        # Header
        if not quiet:
            print('\nMulti-Stage Stochastic Hybrid Approximation')
            print('---------------------------------------------')
            print('{0:^8s}'.format('iter'), end=' ')
            print('{0:^12s}'.format('time (min)'), end=' ')
            print('{0:^12s}'.format('dx'), end=' ')            
            print('{0:^12s}'.format('gc'), end=' ')
            print('{0:^10s}'.format('samples'))

        # Init
        self.k = 0
        t0 = time.time()
        x_prev = np.zeros(self.n)
        self.sol_data = None
        self.time = 0.                                                # construction time (min)
        self.samples = deque(maxlen=msize)                            # sampled realizations of uncertainty
        self.dslopes = [deque(maxlen=msize) for i in range(self.T-1)] # slope corrections (no steplengths)
        self.gammas = [gamma for t in range(self.T-1)]                # scaling factors

        # Loop
        while True:
            
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
                      self.k == 0 and i == 0) for i in range(num_procs)]
            if num_procs > 1:
                results = pool.map(ApplyFunc,tasks)
            else:
                results = list(map(ApplyFunc,tasks))
            assert(len(results) == num_procs)

            # Save sol
            self.x = results[0][0][0]
            for i in range(num_procs):
                assert(norm(results[i][0][0]-self.x) < (1e-8/norm(self.x)))
                           
            # Update approximations
            for i in range(num_procs):
                
                sol,xi_vecs,et_vecs,sol_data = results[i]

                # Sol data (CE solution)
                if self.k == 0 and i == 0:
                    self.sol_data = sol_data

                # Update samples
                self.samples.append(sample[i])
 
                # Update slopes
                for t in range(self.T-1):
                    self.dslopes[t].append(xi_vecs[t]-et_vecs[t]-g_corr[i][t])

            # Update time
            self.time = (time.time()-t0)/60.
 
            # Output
            if not quiet:
                print('{0:^8d}'.format(self.k), end=' ')
                print('{0:^12.2f}'.format(self.time), end=' ')
                print('{0:^12.5e}'.format(norm(self.x-x_prev)), end=' ')
                print('{0:^12.5e}'.format(norm(g_corr[0][0])), end=' ')
                print('{0:^10d}'.format(len(self.samples)))

            # Checks
            for t in range(self.T-1):
                assert(len(self.dslopes[t]) == len(self.samples))

            # Update
            x_prev = self.x.copy()

            # Update
            self.k += 1

            # Key iter
            if key_iters is not None and self.k in key_iters:
                policy = self.get_policy()
                f = open(outdir+'/'+'sh'+str(self.k)+'.policy','w')
                dill.dump(policy,f)
                f.close()
            
            # Maxiter
            if self.k >= maxiters:
                break
 
    def get_policy(self):
        """
        Gets operation policy.
        
        Returns
        -------
        policy : 
        """

        # Construct policy
        def apply(cls,t,x_prev,Wt):

            solver = cls.data
            problem = cls.problem
            
            assert(0 <= t < problem.T)
            assert(len(Wt) == t+1)

            w_list = list(Wt)
            g_corr_pr = [solver.g(t,Wt)]
            for tau in range(t+1,problem.T):
                w_list.append(problem.predict_w(tau,w_list))
                g_corr_pr.append(solver.g(tau,w_list))
            x,Q,gQ,results = problem.solve_stages(t,
                                                  w_list[t:],
                                                  x_prev,
                                                  g_corr=g_corr_pr,
                                                  quiet=True)
            
            # Check feasibility
            if not problem.is_point_feasible(t,x,x_prev,Wt[-1]):
                raise ValueError('point not feasible')
            
            # Return
            return x
            
        policy = StochProblemMS_Policy(self.problem,
                                       data=self,
                                       name='Stochastic Hybrid Approximation (%d)' %self.k,
                                       construction_time=self.time)
        policy.apply = MethodType(apply,policy)
        
        # Return
        return policy
