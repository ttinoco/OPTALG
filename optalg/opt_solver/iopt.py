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
from scipy.sparse import bmat

class OptSolverIPOPT(OptSolver):
    
    parameters = {'tol': 1e-7,
                  'quiet':False} # flag for omitting output
    
    def __init__(self):
        """
        Interior point nonlinear optimization algorithm.
        """
        
        OptSolver.__init__(self)
        self.parameters = OptSolverIPOPT.parameters.copy()
        self.problem = None

    def create_ipopt_problem(self,problem):

        # Imports
        import ipopt

        class P:

            def __init__(self,prob):                
                self.prob = prob
                self.x_last = np.NaN*np.ones(prob.get_num_primal_variables())

            def objective(self,x):
                if not np.all(x == self.x_last):
                    self.prob.eval(x)
                    self.x_last = x.copy()
                return self.prob.phi

            def gradient(self,x):
                if not np.all(x == self.x_last):
                    self.prob.eval(x)
                    self.x_last = x.copy()                    
                return self.prob.gphi
            
            def constraints(self,x):
                if not np.all(x == self.x_last):
                    self.prob.eval(x)
                    self.x_last = x.copy()                    
                return np.hstack((self.prob.A*x-self.prob.b,self.prob.f))
            
            def jacobian(self,x):
                if not np.all(x == self.x_last):
                    self.prob.eval(x)
                    self.x_last = x.copy()
                return bmat([[self.prob.A],[self.prob.J]],format='coo').data
            
            def jacobianstructure(self):
                J = bmat([[self.prob.A],[self.prob.J]],format='coo')
                return J.row,J.col
            
            def hessian(self,x,lam,obj_factor):
                if not np.all(x == self.x_last):
                    self.prob.eval(x)
                    self.x_last = x.copy()
                lamA = lam[:self.prob.get_num_linear_equality_constraints()]
                lamf = lam[self.prob.get_num_linear_equality_constraints():]
                self.prob.combine_H(lamf)
                return np.concatenate(((obj_factor*self.prob.Hphi).data,self.prob.H_combined.data))

            def hessianstructure(self):
                self.prob.combine_H(np.zeros(self.prob.get_num_linear_equality_constraints()))
                return (np.concatenate((self.prob.Hphi.row,self.prob.H_combined.row)),
                        np.concatenate((self.prob.Hphi.col,self.prob.H_combined.col)))

        n = problem.get_num_primal_variables()
        m = problem.get_num_linear_equality_constraints()+problem.get_num_nonlinear_equality_constraints()

        return ipopt.problem(n=n,
                             m=m,
                             problem_obj=P(problem),
                             lb=problem.l,
                             ub=problem.u,
                             cl=np.zeros(m),
                             cu=np.zeros(m))
                
    def solve(self,problem):
        
        # Local vars
        params = self.parameters
        
        # Parameters
        quiet = params['quiet']
        tol = params['tol']

        # Problem
        self.problem = problem
        self.ipopt_problem = self.create_ipopt_problem(problem)

        # Options
        self.ipopt_problem.addOption('tol',tol)
        self.ipopt_problem.addOption('print_level',0 if quiet else 5)

        # Reset
        self.reset()

        # Init point
        if problem.x is not None:
            x0 = problem.x
        else:
            x0 = (problem.u+problem.l)/2
                
        # Solve
        x,info = self.ipopt_problem.solve(x0)

        # Save
        self.x = x
        self.lam = -info['mult_g'][:problem.get_num_linear_equality_constraints()]
        self.nu = -info['mult_g'][problem.get_num_linear_equality_constraints():]
        self.pi = info['mult_x_L']
        self.mu = info['mult_x_U']
        if info['status'] == 0:
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise OptSolverError_IPOPT(self)
            
