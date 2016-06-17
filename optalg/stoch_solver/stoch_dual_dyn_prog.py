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
        Stochastic Dual Dynamic Programming algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochDualDynProg.parameters.copy()
        
    def solve(self,problem,tree):
        """
        Solves problem.

        Parameters
        ----------
        problem : 
        tree :
        """
                
        # Local vars
        params = self.parameters
        self.problem = problem
        self.tree = tree
 
        # Parameters
        maxiters = params['maxiters']
        quiet = params['quiet']
        callback = params['callback']
        num_procs = params['num_procs']
 
        # Header
        if not quiet:
            print '\nMulti-Stage Stochastic Dual Dynamic Programming'
            print '-------------------------------------------------'
            print '{0:^8s}'.format('iter'),
            print '{0:^10s}'.format('time')

        # Init
 
    def get_policy(self):
        """
        Gets operation policy.
        
        Returns
        -------
        policy : 
        """

        pass
