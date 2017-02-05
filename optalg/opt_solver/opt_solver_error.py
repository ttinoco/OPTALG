#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class OptSolverError(Exception):
    
    def __init__(self,solver,value):
        if solver:
            solver.set_status(solver.STATUS_ERROR)
            solver.set_error_msg(value)
        self.value = value
        
    def __str__(self):
        return str(self.value)

class OptSolverError_LineSearch(OptSolverError):    
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'line search failed')

class OptSolverError_BadLinSolver(OptSolverError):
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'invalid linear solver')

class OptSolverError_BadSearchDir(OptSolverError_LineSearch):
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'bad search direction')

class OptSolverError_BadLinSystem(OptSolverError):
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'bad linear system')

class OptSolverError_LinFeasLost(OptSolverError):
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'linear equality constraint feasibility lost')

class OptSolverError_Infeasibility(OptSolverError):
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'infeasible problem')

class OptSolverError_NoInterior(OptSolverError):
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'empty interior')

class OptSolverError_MaxIters(OptSolverError):
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'maximum number of iterations')

class OptSolverError_SmallPenalty(OptSolverError):
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'penalty parameter too small')

class OptSolverError_BadInitPoint(OptSolverError):
    def __init__(self,solver=None):
        OptSolverError.__init__(self,solver,'bad initial point')

