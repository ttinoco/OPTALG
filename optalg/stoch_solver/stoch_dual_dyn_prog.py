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
from scipy.sparse import coo_matrix
from .stoch_solver import StochSolver
from .problem_ms import StochProblemMS
from .problem_ms_policy import StochProblemMS_Policy

class StochDualDynProg(StochSolver):
    """
    Multi-stage stochastic dual dynamic programming algorithm.
    """

    parameters = {'maxiters': 1000,
                  'num_procs': 1,
                  'quiet' : True,
                  'warm_start': False,
                  'bounds': False,
                  'period': 10,
                  'key_iters': None,
                  'outdir': '',
                  'tol': 1e-4}

    def __init__(self):
        """
        Multi-stage stochastic dual dynamic programming algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochDualDynProg.parameters.copy()

    def compute_bounds(self):
        """
        Computes lower and upper bounds
        for optimal objective value.

        Returns
        -------
        lbound : float
        ubound : float
        """

        # Local vars
        problem = self.problem
        tree = self.tree
        T = self.T
        tol = self.parameters['tol']
        
        # Solve tree
        id2x = {} # id -> x
        id2F = {} # id -> stage current cost
        id2Ha = {}
        for t in range(T):
            nodes = tree.get_stage_nodes(t)
            for node in nodes:
                parent = node.get_parent()
                if parent:
                    x_prev = id2x[parent.get_id()]
                else:
                    x_prev = problem.get_x_prev() 
                    assert(t == 0)
                x,H,gH,results = problem.solve_stage_with_cuts(t,
                                                               node.get_w(),
                                                               x_prev,
                                                               self.cuts[node.get_id()][0], # A
                                                               self.cuts[node.get_id()][1], # b
                                                               quiet=True,
                                                               init_data=node.get_data(),
                                                               tol=tol)
                id2x[node.get_id()] = x
                id2F[node.get_id()] = problem.eval_F(t,x,node.get_w())
                id2Ha[node.get_id()] = H # total cost using cuts
        
        # Get upper bound
        id2H = {} # id -> stage total cost (current + cost to go)
        for t in range(T-1,-1,-1):
            nodes = tree.get_stage_nodes(t)
            for node in nodes:
                if node.get_children():
                    cost_to_go = sum([id2H[n.get_id()]*n.get_p() for n in node.get_children()])
                else:
                    cost_to_go = 0.
                    assert(t == T-1)
                id2H[node.get_id()] = id2F[node.get_id()]+cost_to_go
        
        # Return
        return id2Ha,id2H

    def solve(self,problem,tree):
        """
        Solves problem.

        Parameters
        ----------
        problem : StochProblemMS
        tree : StochProbleMS_Tree
        """

        # Check
        assert(isinstance(problem,StochProblemMS))
        
        # Local vars
        self.problem = problem
        self.tree = tree
        self.T = problem.get_num_stages()

        # Check tree
        nodes = tree.get_nodes()
        assert(len(set([n.get_id() for n in nodes])) == len(nodes))
        
        # Parameters
        params = self.parameters
        maxiters = params['maxiters']
        quiet = params['quiet']
        warm_start = params['warm_start']
        num_procs = params['num_procs']
        bounds = params['bounds']
        period = params['period']
        tol = params['tol']
        key_iters = params['key_iters']
        outdir = params['outdir']
 
        # Header
        if not quiet:
            print('\nMulti-stage Stochastic Dual Dynamic Programming')
            print('-----------------------------------------------')
            print('{0:^8s}'.format('iter'), end=' ')
            print('{0:^12s}'.format('time (min)'), end=' ')
            print('{0:^12s}'.format('dx'), end=' ')
            print('{0:^12s}'.format('lbound'), end=' ')
            print('{0:^12s}'.format('ubound'))

        # Init
        self.k = 0
        t0 = time.time()
        self.time = 0.
        x_prev = np.zeros(problem.get_size_x(0))
        lbound = {tree.root.get_id() : -np.inf}
        ubound = {tree.root.get_id() : np.inf}
        self.cuts = dict([(n.get_id(),
                           [np.zeros((0,problem.get_size_x(n.get_stage()))),np.zeros(0)]) # (A,b) 
                          for n in tree.get_nodes()])

        # Loop
        while True:

            # Key iter
            if key_iters is not None and self.k in key_iters:
                policy = self.get_policy()
                f = open(outdir+'/'+'sddp'+str(self.k)+'.policy','w')
                dill.dump(policy,f)
                f.close()
            
            # Maxiter
            if self.k >= maxiters:
                break
 
            # Sample tree branch
            branch = tree.sample_branch(self.T-1)
            assert(len(branch) == self.T)

            # Forward pass
            solutions = {-1 : problem.get_x_prev()}
            for t in range(self.T):
                node = branch[t]
                x,H,gh,results = problem.solve_stage_with_cuts(t,
                                                               node.get_w(),
                                                               solutions[t-1],
                                                               self.cuts[node.get_id()][0], # A
                                                               self.cuts[node.get_id()][1], # b
                                                               quiet=True,
                                                               init_data=node.get_data() if warm_start else None,
                                                               tol=tol)
                solutions[t] = x
                node.set_data(results)

            # Update solution
            self.x = solutions[0]

            # Backward pass
            for t in range(self.T-2,-1,-1):
                node = branch[t]
                assert(node.get_stage() == t)
                x = solutions[t]
                H = 0
                gH = np.zeros(problem.get_size_x(t))
                for n in node.get_children():
                    xn,Hn,gHn,results = problem.solve_stage_with_cuts(t+1,
                                                                      n.get_w(),
                                                                      x,
                                                                      self.cuts[n.get_id()][0], # A
                                                                      self.cuts[n.get_id()][1], # b
                                                                      quiet=True,
                                                                      init_data=n.get_data() if warm_start else None,
                                                                      tol=tol)
                    H += Hn*n.get_p()
                    gH += gHn*n.get_p()
                    n.set_data(results)
                a = -gH
                b = -H + np.dot(gH,x)
                self.cuts[node.get_id()][0] = np.vstack((self.cuts[node.get_id()][0],a)) # A
                self.cuts[node.get_id()][1] = np.hstack((self.cuts[node.get_id()][1],b)) # b

            # Update time
            self.time = (time.time()-t0)/60.

            # Output
            if not quiet and self.k % period == 0:
                if bounds:
                    lbound,ubound = self.compute_bounds()
                print('{0:^8d}'.format(self.k), end=' ')
                print('{0:^12.2f}'.format(self.time), end=' ')
                print('{0:^12.5e}'.format(norm(self.x-x_prev)), end=' ')
                print('{0:^12.5e}'.format(lbound[tree.root.get_id()]), end=' ')
                print('{0:^12.5e}'.format(ubound[tree.root.get_id()]))

            # Update previous solution
            x_prev = self.x.copy()

            # Update iters
            self.k += 1

    def get_policy(self):
        """
        Gets operation policy.
        
        Returns
        -------
        policy : StochProbleMS_Policy 
        """

        # Local vars
        maxiters = self.parameters['maxiters']

        # Construct policy
        def apply(cls,t,x_prev,W):
            
            solver = cls.data
            problem = cls.problem

            T = problem.get_num_stages()

            assert(0 <= t < T)
            assert(len(W) == t+1)
            
            branch = solver.tree.get_closest_branch(W) # check
            assert(len(branch) == len(W))
           
            node = branch[-1]
 
            x,H,gH,results = problem.solve_stage_with_cuts(t,
                                                           W[-1],                        # actual realization, not node
                                                           x_prev,
                                                           solver.cuts[node.get_id()][0], # A
                                                           solver.cuts[node.get_id()][1], # b
                                                           quiet=True)
            
            # Check feasibility
            if not problem.is_point_feasible(t,x,x_prev,W[-1]):
                raise ValueError('point not feasible')
            
            # Return
            return x
            
        policy = StochProblemMS_Policy(self.problem,
                                       data=self,
                                       name='Stochastic Dual Dynamic Programming %d' %self.k,
                                       construction_time=self.time)
        policy.apply = MethodType(apply,policy)
        
        # Return
        return policy
