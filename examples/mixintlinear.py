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
from optalg.opt_solver import OptSolverCbc, MixIntLinProblem

A = np.array([[-2.,2. ,1.,0.],
              [-8.,10.,0.,1.]])
b = np.array([1.,13.])

l = np.array([-1e8,-1e8,-1e8,0.])
u = np.array([1e8,1e8,0.,1e8])

c = np.array([-1.,-1.,0.,0.])

P = np.array([True,True,False,False])

problem = MixIntLinProblem(c,A,b,l,u,P)

solver = OptSolverCbc()
solver.set_parameters({'quiet':False})

solver.solve(problem)

print(solver.get_primal_variables())

