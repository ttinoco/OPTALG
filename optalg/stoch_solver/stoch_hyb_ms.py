#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
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
from .problem_ms import StochProblemMS
from .problem_ms_policy import StochProblemMS_Policy

class StochHybridMS(StochSolver):
    """
    Multi-Stage stochastic hybrid approximation algorithm.
    """

    parameters = {'maxiters': 1000,
                  'num_procs': 1,
                  'msize': 500,
                  'quiet' : True,
                  'theta': 1.,
                  'k0': 0,
                  'gamma': 1e0,
                  'key_iters': None,
                  'outdir': '',
                  'model': 'dynamic',
                  'tol': 1e-4}

    def __init__(self):
        """
        Multi-Stage stochastic hybrid approximation algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochHybridMS.parameters.copy()
        self.samples = None
        self.deltags = None 
        self.gammas = None
        self.problem = None

    def g(self,t,W):
        """
        Slope correction function.
        
        Parameters
        ----------
        t : {0,...,T-1}
        W : list (length t+1)

        Returns
        -------
        g : vector
        """
        
        # Local vars
        n = self.problem.get_size_x(t)
        T = self.problem.get_num_stages()
        samples = self.samples
        deltags = self.deltags
        gammas = self.gammas
        theta = self.parameters['theta']
        k0 = self.parameters['k0']

        # Last stage
        if t == T-1:
            return np.zeros(n)

        # Checks
        assert(0 <= t < T-1)
        assert(len(W) == t+1)
        assert(len(samples) == len(deltags[t]))

        # Correction
        corr = np.zeros(n)
        for k in range(len(samples)):

            Wk = samples[k][:t+1]
            assert(len(Wk) == t+1)

            deltag = deltags[t][k]
            
            phi = np.exp(-gammas[t]*norm(np.hstack(W)-np.hstack(Wk))/np.sqrt(float(np.hstack(Wk).size)))

            beta = theta/max([1.,float(k0+k)])

            alpha = phi*beta

            corr += alpha*deltag
        
        return corr

    def solve_subproblems(self,sample,corrections):
        """ 
        Solves subproblems.
        
        Parameters
        ----------
        sample : list of uncertainy relizations for each stage
        corrections : list of slope corrections for each stage

        Results
        -------
        solutions : dict (stage solutions)
        xi_vects : dict (noisy subgradients)
        et_vects : dict (model subgradients)
        """ 
        
        # Local variables
        problem = self.problem
        T = self.problem.get_num_stages()
        model = self.parameters['model']
        tol = self.parameters['tol']

        # Check
        assert(len(sample) == T)
        assert(len(corrections) == T)

        # Solve subproblems
        xi_vecs = {}
        et_vecs = {}
        solutions = {-1 : problem.get_x_prev()}
        for t in range(T):

            # Zeros
            z = np.zeros(problem.get_size_x(t))
            
            # Dynamic
            if model == 'dynamic':
                w_list = sample[:t+1]
                g_list = [corrections[t]]
                for tau in range(t+1,T):
                    w_list.append(problem.predict_w(tau,w_list))
                    g_list.append(self.g(tau,w_list))
            
            # Static
            elif model == 'static':
                w_list = sample[:t+1]
                g_list = [corrections[t]]
                for tau in range(t+1,T):
                    w_list.append(problem.predict_w(tau,w_list))
                    g_list.append(z)

            # Error
            else:
                raise ValueError('invalid model')

            # Solve
            x,H,gH,gHnext = problem.solve_stages(t,
                                                 w_list[t:],
                                                 solutions[t-1],
                                                 g_list=g_list,
                                                 quiet=True,
                                                 tol=tol,
                                                 next=True)

            # Save results
            solutions[t] = x
            xi_vecs[t-1] = gH
            et_vecs[t] = gHnext

        # Return
        return solutions,xi_vecs,et_vecs
        
    def solve(self,problem):
               
        # Check
        assert(isinstance(problem,StochProblemMS))
        
        # Save 
        self.problem = problem

        #  Local vars
        T = problem.get_num_stages()
        params = self.parameters
        maxiters = params['maxiters']
        msize = params['msize']
        quiet = params['quiet']
        num_procs = params['num_procs']
        gamma = params['gamma']
        key_iters = params['key_iters']
        outdir = params['outdir']
        model = self.parameters['model']

        # Name
        if model == 'dynamic':
            name = 'shD'
        elif model == 'static':
            name = 'shS'
        else:
            raise ValueError('invalid model')

        # Pool
        from multiprocess import Pool
        pool = Pool(num_procs)
 
        # Header
        if not quiet:
            print('\nMulti-Stage Stochastic Hybrid Approximation (%s)' %name)
            print('--------------------------------------------------------')
            print('{0:^8s}'.format('iter'), end=' ')
            print('{0:^12s}'.format('time (min)'), end=' ')
            print('{0:^12s}'.format('dx'), end=' ')            
            print('{0:^12s}'.format('gc'), end=' ')
            print('{0:^10s}'.format('samples'))

        # Init
        self.k = 0
        t0 = time.time()
        x_prev = np.zeros(problem.get_size_x(0))
        self.time = 0.                                           # construction time (min)
        self.samples = deque(maxlen=msize)                       # sampled realizations of uncertainty
        self.deltags = [deque(maxlen=msize) for t in range(T-1)] # slope correction vectors (no steplengths)
        self.gammas = [gamma for t in range(T-1)]                # scaling factors for radial basis functions

        # Loop
        while True:

            # Key iter
            if key_iters is not None and self.k in key_iters:
                policy = self.get_policy()
                f = open(outdir+'/'+name+str(self.k)+'.policy','w')
                dill.dump(policy,f)
                f.close()
            
            # Maxiter
            if self.k >= maxiters:
                break
            
            # Subroblems data
            sample = {}
            corrections = {}
            for i in range(num_procs):
            
                # Sample uncertatinty
                sample[i] = problem.sample_W(T-1)
                assert(len(sample[i]) == T)

                # Slope corrections
                corrections[i] = [self.g(t,sample[i][:t+1]) for t in range(T)]

            # Solve subproblems
            tasks = [(self,'solve_subproblems',sample[i],corrections[i]) for i in range(num_procs)]
            if num_procs > 1:
                results = pool.map(ApplyFunc,tasks)
            else:
                results = list(map(ApplyFunc,tasks))
            assert(len(results) == num_procs)

            # Update solution
            self.x = results[0][0][0]
            for i in range(num_procs):
                assert(norm(results[i][0][0]-self.x) < (1e-8/norm(self.x)))
                           
            # Update approximations
            for i in range(num_procs):

                # Get resutls
                sol,xi_vecs,et_vecs = results[i]

                # Update samples
                self.samples.append(sample[i])
 
                # Update slope correction vectors
                for t in range(T-1):
                    self.deltags[t].append(xi_vecs[t]-et_vecs[t]-corrections[i][t])

            # Update time
            self.time = (time.time()-t0)/60.
 
            # Output
            if not quiet:
                print('{0:^8d}'.format(self.k), end=' ')
                print('{0:^12.2f}'.format(self.time), end=' ')
                print('{0:^12.5e}'.format(norm(self.x-x_prev)), end=' ')
                print('{0:^12.5e}'.format(norm(corrections[0][0])), end=' ')
                print('{0:^10d}'.format(len(self.samples)))

            # Checks
            for t in range(T-1):
                assert(len(self.deltags[t]) == len(self.samples))

            # Update previos solution
            x_prev = self.x.copy()

            # Update iters
            self.k += 1
 
    def get_policy(self):
        """
        Gets operation policy.
        
        Returns
        -------
        policy : StochProblemMS_Policy
        """

        model = self.parameters['model']
        
        # Construct policy
        def apply(cls,t,x_prev,W):

            solver = cls.data
            problem = cls.problem

            T = problem.get_num_stages()
            z = np.zeros(problem.get_size_x(t))
            
            assert(0 <= t < T)
            assert(len(W) == t+1)
            
            # Dynamic
            if model == 'dynamic':
                w_list = list(W)
                g_list = [solver.g(t,W)]
                for tau in range(t+1,T):
                    w_list.append(problem.predict_w(tau,w_list))
                    g_list.append(solver.g(tau,w_list))

            # Static
            elif model == 'static':
                w_list = list(W)
                g_list = [solver.g(t,W)]
                for tau in range(t+1,T):
                    w_list.append(problem.predict_w(tau,w_list))
                    g_list.append(z)

            # Error
            else:
                raise ValueError('invalid model')
            
            x,H,gH,gHnext = problem.solve_stages(t,
                                                 w_list[t:],
                                                 x_prev,
                                                 g_list=g_list,
                                                 quiet=True)
            
            # Check feasibility
            if not problem.is_point_feasible(t,x,x_prev,W[-1]):
                raise ValueError('point not feasible')
            
            # Return
            return x
            
        policy = StochProblemMS_Policy(self.problem,
                                       data=self,
                                       name='Stochastic Hybrid Approximation (%s) %d' %(model,self.k),
                                       construction_time=self.time)
        policy.apply = MethodType(apply,policy)
        
        # Return
        return policy
