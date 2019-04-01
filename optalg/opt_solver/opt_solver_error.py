#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2019, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class OptSolverError(Exception):
    
    def __init__(self, solver, value):
        if solver:
            solver.set_status(solver.STATUS_ERROR)
            solver.set_error_msg(value)
        self.value = value
        
    def __str__(self):
        return str(self.value)
    
class OptSolverError_Cbc(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'cbc solver failed')

class OptSolverError_CbcCMD(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'cbc command-line solver failed')

class OptSolverError_CbcCMDCall(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'error while calling cbc command-line solver')
        
class OptSolverError_Clp(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'clp solver failed')

class OptSolverError_ClpCMD(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'clp command-line solver failed')

class OptSolverError_ClpCMDCall(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'error while calling clp command-line solver')

class OptSolverError_CplexCMD(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'cplex command-line solver failed')

class OptSolverError_CplexCMDCall(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'error while calling cplex command-line solver')

class OptSolverError_Ipopt(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'ipopt solver failed')

class OptSolverError_NumProblems(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'numerical problems')

class OptSolverError_LineSearch(OptSolverError):    
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'line search failed')

class OptSolverError_BadProblemType(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'invalid problem type')

class OptSolverError_BadLinSolver(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'invalid linear solver')

class OptSolverError_BadSearchDir(OptSolverError_LineSearch):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'bad search direction')

class OptSolverError_BadLinSystem(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'bad linear system')

class OptSolverError_LinFeasLost(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'linear equality constraint feasibility lost')

class OptSolverError_Infeasibility(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'problem appears infeasible')

class OptSolverError_NoInterior(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'empty interior')

class OptSolverError_MaxIters(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'maximum number of iterations')

class OptSolverError_SmallPenalty(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'penalty parameter too small')

class OptSolverError_BadInitPoint(OptSolverError):
    def __init__(self, solver=None):
        OptSolverError.__init__(self, solver, 'bad initial point')

