#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import sys
sys.path.append('.')

import numpy as np
from optalg.opt_solver import OptSolverClp, LinProblem

A = np.array([[6.,1.,1.,0.,0.],
              [3.,1.,0.,1.,0.],
              [4.,6.,0.,0.,1.]])
b = np.array([12.,8.,24.])

l = np.array([0.,0.,-1e8,-1e8,-1e8])
u = np.array([5.,5.,0.,0.,0.])
              
c = np.array([180.,160.,0.,0.,0.])

problem = LinProblem(c,A,b,l,u)

solver = OptSolverClp()
solver.set_parameters({'quiet':False})

solver.solve(problem)

print(solver.get_primal_variables())


