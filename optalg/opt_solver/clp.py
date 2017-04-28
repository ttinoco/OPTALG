#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from __future__ import print_function
import numpy as np
from .opt_solver_error import *
from .opt_solver import OptSolver
from .problem_lin import LinProblem

class OptSolverClp(OptSolver):

    parameters = {'quiet' : False}

    def __init__(self):
        """
        Linear programming solver from COIN-OR.
        """
        
        OptSolver.__init__(self)
        self.parameters = OptSolverClp.parameters.copy()
        
    def solve(self,problem):

        # Import
        from ._clp import ClpContext

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']

        # Problem
        if not isinstance(problem,LinProblem):
            raise OptSolverError_BadProblemType(self)
        self.problem = problem

        # Clp context
        self.clp_context = ClpContext()
        self.clp_context.loadProblem(problem.get_num_primal_variables(),
                                     problem.A,
                                     problem.l,
                                     problem.u,
                                     problem.c,
                                     problem.b,
                                     problem.b)
        
        # Reset
        self.reset()

        # Options
        if quiet:
            self.clp_context.setlogLevel(0)

        # Solve
        self.clp_context.initialSolve()

        # Save
        self.x = self.clp_context.primalColumnSolution()
        self.lam = self.clp_context.dualRowSolution()
        self.pi = np.maximum(self.clp_context.dualColumnSolution(),0)
        self.mu = -np.minimum(self.clp_context.dualColumnSolution(),0)
        if self.clp_context.status() == 0:
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise OptSolverError_Clp(self)
            
