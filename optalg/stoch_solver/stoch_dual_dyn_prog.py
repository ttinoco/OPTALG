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
            print '{0:^10s}'.format('time'),
            print '{0:^12s}'.format('dx')

        # Init
        t0 = time.time()
        cuts = dict([(node.get_id(),(np.zeros((0,self.n)),np.zeros(0))) # (A,b) 
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
                                                       cuts[node.get_id()][0], # A
                                                       cuts[node.get_id()][1], # b
                                                       quiet=not debug,
                                                       tol=tol)
                solutions[t] = x

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
                                                              cuts[n.get_id()][0], # A
                                                              cuts[n.get_id()][1], # b
                                                              quiet=not debug,
                                                              tol=tol)
                    Q *= float(i)/float(i+1)
                    Q += Qn/float(i+1)
                    gQ *= float(i)/float(i+1)
                    gQ += gQn/float(i+1)
                a = -gQ
                b = -Q + np.dot(gQ,x)
                cuts[node.get_id()][0] = np.vstack((cuts[node.get_id()][0],a)) # A
                cuts[node.get_id()][1] = np.hstack((cuts[node.get_id()][1],b)) # b

    def get_policy(self):
        """
        Gets operation policy.
        
        Returns
        -------
        policy : 
        """

        pass
