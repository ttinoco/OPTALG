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
                  'callback': None,
                  'debug': False,
                  'tol': 1e-4}

    def __init__(self):
        """
        Multi-stage stochastic dual dynamic programming algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochDualDynProg.parameters.copy()
        
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
        callback = params['callback']
        num_procs = params['num_procs']
        debug = params['debug']
 
        # Header
        if not quiet:
            print '\nMulti-Stage Stochastic Dual Dynamic Programming'
            print '-------------------------------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^12s}'.format('time (min)'),
            print '{0:^12s}'.format('dx')

        # Init
        t0 = time.time()
        x_prev = np.zeros(self.n)
        self.cuts = dict([(node.get_id(),(np.zeros((0,self.n)),np.zeros(0))) # (A,b) 
                          for node in tree.get_nodes()])

        # Loop
        for k in range(maxiters+1):
 
            # Sample tree branch
            sample = problem.sample_branch(self.T-1)
            assert(len(sample) == self.T)

            # Forward pass
            solutions = {-1 : problem.get_x_prev()}
            for t in range(self.T):
                node = sample[t]
                x,Q,gQ = problem.solve_stage_with_cuts(t,
                                                       node.get_w(),
                                                       solutions[t-1],
                                                       self.cuts[node.get_id()][0], # A
                                                       self.cuts[node.get_id()][1], # b
                                                       quiet=not debug,
                                                       tol=tol)
                solutions[t] = x

            # Save sol
            self.x = solutions[0]

            # Backward pass
            for t in range(self.T-1,-1,-1):
                node = sample[t]
                x = solutions[t]
                Q = 0
                gQ = np.zeros(self.n)
                for i in range(len(node.get_children())):
                    n = node.get_child(i)
                    xn,Qn,gQn = problem.solve_stage_with_cuts(t,
                                                              n.get_w(),
                                                              self.cuts[n.get_id()][0], # A
                                                              self.cuts[n.get_id()][1], # b
                                                              quiet=not debug,
                                                              tol=tol)
                    Q *= float(i)/float(i+1)
                    Q += Qn/float(i+1)
                    gQ *= float(i)/float(i+1)
                    gQ += gQn/float(i+1)
                a = -gQ
                b = -Q + np.dot(gQ,x)
                self.cuts[node.get_id()][0] = np.vstack((self.cuts[node.get_id()][0],a)) # A
                self.cuts[node.get_id()][1] = np.hstack((self.cuts[node.get_id()][1],b)) # b

            # Output
            if not quiet:
                print '{0:^8d}'.format(k),
                print '{0:^12.2f}'.format((time.time()-t0)/60.),
                print '{0:^12.5e}'.format(norm(self.x-x_prev)),

            # Update
            x_prev = self.x.copy()

    def get_policy(self):
        """
        Gets operation policy.
        
        Returns
        -------
        policy : 
        """

        # Construct policy
        def apply(cls,t,x_prev,Wt):
            
            solver = cls.solver
            problem = cls.problem

            assert(0 <= t < problem.T)
            assert(len(Wt) == t+1)
            
            branch = solver.tree.get_closest_branch(Wt)
            assert(len(branch) == len(Wt))
           
            node = branch[-1]
 
            x,Q,gQ = problem.solve_stage_with_cuts(t,
                                                   Wt[-1],                        # actual realization, not node
                                                   solver.cuts[node.get_id()][0], # A
                                                   solver.cuts[node.get_id()][1], # b
                                                   quiet=not debug,
                                                   tol=tol)
            
            # Check feasibility
            if not problem.is_point_feasible(t,x,x_prev,Wt[-1]):
                raise ValueError('point not feasible')
            
            # Return
            return x
            
        policy = StochProblemMS_Policy(self.problem,data=self,name='Stochastic Dual Dynamic Programming')
        policy.apply = MethodType(apply,policy)
        
        # Return
        return policy
