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
from .problem_mixintlin import MixIntLinProblem

class OptSolverCbc(OptSolver):

    parameters = {'quiet' : False}

    def __init__(self):
        """
        Mixed integer linear "branch and cut" sovler from COIN-OR.
        """
        
        OptSolver.__init__(self)
        self.parameters = OptSolverCbc.parameters.copy()
        
    def solve(self,problem):

        # Import
        from ._cbc import CbcContext

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']

        # Problem
        if not isinstance(problem,MixIntLinProblem):
            raise OptSolverError_BadProblemType(self)
        self.problem = problem

        # Cbc context
        self.cbc_context = CbcContext()
        self.cbc_context.loadProblem(problem.get_num_primal_variables(),
                                     problem.A,
                                     problem.l,
                                     problem.u,
                                     problem.c,
                                     problem.b,
                                     problem.b)
        self.cbc_context.copyInIntegerInformation(problem.P)
        
        # Reset
        self.reset()

        # Options
        if quiet:
            self.cbc_context.setlogLevel(0)

        # Solve
        self.cbc_context.branchAndBound()

        # Save
        self.x = self.cbc_context.getColSolution()
        if self.cbc_context.status() == 0:
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise OptSolverError_Cbc(self)
