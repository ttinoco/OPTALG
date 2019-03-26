#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2019, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from __future__ import print_function
import numpy as np
from .opt_solver_error import *
from .opt_solver import OptSolver
from .problem import OptProblem

class OptSolverCbc(OptSolver):

    parameters = {'quiet' : False}

    def __init__(self):
        """
        Mixed integer linear "branch and cut" solver from COIN-OR.
        """

        # Import
        from ._cbc import CbcContext
        
        OptSolver.__init__(self)
        self.parameters = OptSolverCbc.parameters.copy()

    def supports_properties(self, properties):

        for p in properties:
            if p not in [OptProblem.PROP_CURV_LINEAR,
                         OptProblem.PROP_VAR_CONTINUOUS,
                         OptProblem.PROP_VAR_INTEGER,
                         OptProblem.PROP_TYPE_FEASIBILITY,
                         OptProblem.PROP_TYPE_OPTIMIZATION]:
                return False
        return True
        
    def solve(self, problem):

        # Import
        from ._cbc import CbcContext

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']

        # Problem
        try:
            self.problem = problem.to_mixintlin()
        except:
            raise OptSolverError_BadProblemType(self)

        # Cbc context
        self.cbc_context = CbcContext()
        self.cbc_context.loadProblem(self.problem.get_num_primal_variables(),
                                     self.problem.A,
                                     self.problem.l,
                                     self.problem.u,
                                     self.problem.c,
                                     self.problem.b,
                                     self.problem.b)
        self.cbc_context.setInteger(self.problem.P)
        
        # Reset
        self.reset()

        # Options
        if quiet:
            self.cbc_context.setParameter("loglevel", 0)

        # Solve
        self.cbc_context.solve()
        
        # Save
        self.x = self.cbc_context.getColSolution()
        if self.cbc_context.status() == 0:
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise OptSolverError_Cbc(self)
