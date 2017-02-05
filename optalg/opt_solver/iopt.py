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

    def create_problem_obj(self,problem):

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
                
            #def intermediate(self,alg_mod,iter_count,obj_value,inf_pr,inf_du,mu,d_norm,reg_size,alpha_du,alpha_pr,ls_trials):

            #pass

        return P(problem)
        
    def solve(self,problem):

        # Imports
        import ipopt
        
        # Local vars
        params = self.parameters
        
        # Parameters
        quiet = params['quiet']
        tol = params['tol']

        # Problem
        self.problem = problem
        n = problem.get_num_primal_variables()
        m = problem.get_num_linear_equality_constraints()+problem.get_num_nonlinear_equality_constraints()
        ipopt_problem = ipopt.problem(n=n,
                                      m=m,
                                      problem_obj=self.create_problem_obj(problem),
                                      lb=problem.l,
                                      ub=problem.u,
                                      cl=np.zeros(m),
                                      cu=np.zeros(m))

        # Options
        ipopt_problem.addOption('tol',tol)
        ipopt_problem.addOption('print_level',0 if quiet else 1)

        # Reset
        self.reset()

        # Init poin
        if problem.x is not None:
            x0 = problem.x
        else:
            x0 = (problem.u+problem.l)/2
                
        # Solve
        x,info = ipopt_problem.solve(x0)

        #print(info['status'])
        #print(info['status_msg'])
        #print(info['obj_val'])
        
        
