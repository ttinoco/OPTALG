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
from optalg.opt_solver import OptSolverIQP, QuadProblem

g = np.array([3.,-6.])
H = np.array([[10.,-2],
               [-2.,10]])

A = np.array([[1.,1.]])
b = np.array([1.])

u = np.array([0.8,0.8])
l = np.array([0.2,0.2])

problem = QuadProblem(H,g,A,b,l,u)

solver = OptSolverIQP()

solver.set_parameters({'quiet': True,
                       'tol': 1e-6})

solver.solve(problem)

print solver.get_status()

x = solver.get_primal_variables()
lam,nu,mu,pi = solver.get_dual_variables()

print x

print x[0] + x[1]

print l <= x

print x <= u

print pi

print mu

print np.linalg.norm(g+np.dot(H,x)-np.dot(A.T,lam)+mu-pi)

print np.dot(mu,u-x)

print np.dot(pi,x-l)

