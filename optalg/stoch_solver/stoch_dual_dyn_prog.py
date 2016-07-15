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
from scipy.sparse import coo_matrix
from stoch_solver import StochSolver
from problem_ms_policy import StochProblemMS_Policy

class StochDualDynProg(StochSolver):

    parameters = {'maxiters': 1000,
                  'num_procs': 1,
                  'quiet' : True,
                  'warm_start': False,
                  'callback': None,
                  'bounds': False,
                  'period': 10,
                  'debug': False,
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
        n = self.n
        tol = self.parameters['tol']
        
        # Solve tree
        id2x = {} # id -> x
        id2F = {} # id -> stage current cost
        id2Qa = {}
        for t in range(T):
            nodes = tree.get_stage_nodes(t)
            for node in nodes:
                parent = node.get_parent()
                if parent:
                    x_prev = id2x[parent.get_id()]
                else:
                    x_prev = problem.get_x_prev() 
                    assert(t == 0)
                x,Q,gQ,results = problem.solve_stage_with_cuts(t,
                                                               node.get_w(),
                                                               x_prev,
                                                               self.cuts[node.get_id()][0], # A
                                                               self.cuts[node.get_id()][1], # b
                                                               quiet=True,
                                                               init_data=node.get_data(),
                                                               tol=tol)
                id2x[node.get_id()] = x
                id2F[node.get_id()] = problem.eval_F(t,x,node.get_w())
                id2Qa[node.get_id()] = Q # total cost using cuts
        
        # Get upper bound
        id2Q = {} # id -> stage total cost (current + cost to go)
        for t in range(T-1,-1,-1):
            nodes = tree.get_stage_nodes(t)
            for node in nodes:
                if node.get_children():
                    cost_to_go = np.average(map(lambda n: id2Q[n.get_id()],node.get_children()))
                else:
                    cost_to_go = 0.
                    assert(t == T-1)
                id2Q[node.get_id()] = id2F[node.get_id()]+cost_to_go
        
        # Return
        return id2Qa,id2Q

    def solve(self,problem,tree):
        """
        Solves problem.

        Parameters
        ----------
        problem : StochProblemMS
        tree : StochProbleMS_Tree
        """
                
        # Local vars
        params = self.parameters
        self.problem = problem
        self.tree = tree
        self.T = problem.get_num_stages()
        self.n = problem.get_size_x()

        # Check tree
        nodes = tree.get_nodes()
        assert(len(set(map(lambda n: n.get_id(),nodes))) == len(nodes))
        
        # Parameters
        maxiters = params['maxiters']
        quiet = params['quiet']
        warm_start = params['warm_start']
        callback = params['callback']
        num_procs = params['num_procs']
        bounds = params['bounds']
        period = params['period']
        debug = params['debug']
        tol = params['tol']
 
        # Header
        if not quiet:
            print '\nMulti-stage Stochastic Dual Dynamic Programming'
            print '-----------------------------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^12s}'.format('time (min)'),
            print '{0:^12s}'.format('dx'),
            print '{0:^12s}'.format('lbound'),
            print '{0:^12s}'.format('ubound')

        # Init
        t0 = time.time()
        x_prev = np.zeros(self.n)
        lbound = {tree.root.get_id() : -np.inf}
        ubound = {tree.root.get_id() : np.inf}
        self.cuts = dict([(node.get_id(),[np.zeros((0,self.n)),np.zeros(0)]) # (A,b) 
                          for node in tree.get_nodes()])

        # Loop
        for k in range(maxiters):
 
            # Sample tree branch
            branch = tree.sample_branch(self.T-1)
            assert(len(branch) == self.T)

            # Forward pass
            solutions = {-1 : problem.get_x_prev()}
            for t in range(self.T):
                node = branch[t]
                x,Q,gQ,results = problem.solve_stage_with_cuts(t,
                                                               node.get_w(),
                                                               solutions[t-1],
                                                               self.cuts[node.get_id()][0], # A
                                                               self.cuts[node.get_id()][1], # b
                                                               quiet=True,
                                                               init_data=node.get_data() if warm_start else None,
                                                               tol=tol)
                solutions[t] = x
                node.set_data(results)

                # DEBUG: Check gQ
                #****************
                if debug:
                    for i in range(10):
                        d = np.random.randn(self.n)*1e-2
                        xper = solutions[t-1]+d
                        x1,Q1,gQ1,results = problem.solve_stage_with_cuts(t,
                                                                          node.get_w(),
                                                                          xper,
                                                                          self.cuts[node.get_id()][0], # A
                                                                          self.cuts[node.get_id()][1], # b
                                                                          quiet=True,
                                                                          tol=tol)
                        assert(Q1+1e-8 >= Q+np.dot(gQ,d))
                        print 'gQ ok'

            # Save sol
            self.x = solutions[0]

            # Backward pass
            for t in range(self.T-2,-1,-1):
                node = branch[t]
                x = solutions[t]
                Q = 0
                gQ = np.zeros(self.n)
                for i in range(node.get_num_children()):
                    n = node.get_child(i)
                    xn,Qn,gQn,results = problem.solve_stage_with_cuts(t+1,
                                                                      n.get_w(),
                                                                      x,
                                                                      self.cuts[n.get_id()][0], # A
                                                                      self.cuts[n.get_id()][1], # b
                                                                      quiet=True,
                                                                      init_data=n.get_data() if warm_start else None,
                                                                      tol=tol)
                    Q *= float(i)/float(i+1)
                    Q += Qn/float(i+1)
                    gQ *= float(i)/float(i+1)
                    gQ += gQn/float(i+1)
                    n.set_data(results)
                a = -gQ
                b = -Q + np.dot(gQ,x)
                self.cuts[node.get_id()][0] = np.vstack((self.cuts[node.get_id()][0],a)) # A
                self.cuts[node.get_id()][1] = np.hstack((self.cuts[node.get_id()][1],b)) # b

            # Output
            if not quiet and k % period == 0:
                if bounds:
                    lbound,ubound = self.compute_bounds()
                print '{0:^8d}'.format(k),
                print '{0:^12.2f}'.format((time.time()-t0)/60.),
                print '{0:^12.5e}'.format(norm(self.x-x_prev)),
                print '{0:^12.5e}'.format(lbound[tree.root.get_id()]),
                print '{0:^12.5e}'.format(ubound[tree.root.get_id()])

            # Update
            x_prev = self.x.copy()

    def get_policy(self):
        """
        Gets operation policy.
        
        Returns
        -------
        policy : 
        """

        # Local vars
        maxiters = self.parameters['maxiters']

        # Construct policy
        def apply(cls,t,x_prev,Wt):
            
            solver = cls.data
            problem = cls.problem

            assert(0 <= t < problem.T)
            assert(len(Wt) == t+1)
            
            branch = solver.tree.get_closest_branch(Wt)
            assert(len(branch) == len(Wt))
           
            node = branch[-1]
 
            x,Q,gQ,results = problem.solve_stage_with_cuts(t,
                                                           Wt[-1],                        # actual realization, not node
                                                           x_prev,
                                                           solver.cuts[node.get_id()][0], # A
                                                           solver.cuts[node.get_id()][1], # b
                                                           quiet=True)
            
            # Check feasibility
            if not problem.is_point_feasible(t,x,x_prev,Wt[-1]):
                raise ValueError('point not feasible')
            
            # Return
            return x
            
        policy = StochProblemMS_Policy(self.problem,
                                       data=self,
                                       name='Stochastic Dual Dynamic Programming (%d)' %maxiters)
        policy.apply = MethodType(apply,policy)
        
        # Return
        return policy
