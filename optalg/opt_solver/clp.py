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

class OptSolverClp(OptSolver):

    parameters = {'quiet' : False}

    def __init__(self):
        """
        Linear programming solver from COIN-OR.
        """

        # Import
        from ._clp import ClpContext
        
        OptSolver.__init__(self)
        self.parameters = OptSolverClp.parameters.copy()
        
    def supports_properties(self, properties):

        for p in properties:
            if p not in [OptProblem.PROP_CURV_LINEAR,
                         OptProblem.PROP_VAR_CONTINUOUS,
                         OptProblem.PROP_TYPE_FEASIBILITY,
                         OptProblem.PROP_TYPE_OPTIMIZATION]:
                return False
        return True
        
    def solve(self, problem):

        # Import
        from ._clp import ClpContext

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']

        # Problem
        try:
            self.problem = problem.to_lin()
        except:
            raise OptSolverError_BadProblemType(self)

        # Clp context
        self.clp_context = ClpContext()
        self.clp_context.loadProblem(self.problem.get_num_primal_variables(),
                                     self.problem.A,
                                     self.problem.l,
                                     self.problem.u,
                                     self.problem.c,
                                     self.problem.b,
                                     self.problem.b)
        
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
            
