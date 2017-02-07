#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from __future__ import print_function
import os
import sys
import numpy as np
from .opt_solver_error import *
from .opt_solver import OptSolver
from scipy.sparse import bmat

class OptSolverIpopt(OptSolver):
    
    parameters = {'tol': 1e-7,
                  'quiet':False} # flag for omitting output
    
    def __init__(self):
        """
        Interior point nonlinear optimization algorithm.
        """
        
        OptSolver.__init__(self)
        self.parameters = OptSolverIpopt.parameters.copy()
        self.problem = None

    def create_ipopt_context(self,problem):

        # Imports
        import _ipopt

        def eval_f(x):
            problem.eval(x)
            return problem.phi

        def eval_grad_f(x):
            problem.eval(x)
            return problem.gphi

        def eval_g(x):
            problem.eval(x)
            return np.hstack((problem.A*x-problem.b,problem.f))

        def eval_jac_g(x,flag):
            if flag:
                J = bmat([[problem.A],[problem.J]],format='coo')
                return J.row,J.col
            else:
                problem.eval(x)
                return bmat([[problem.A],[problem.J]],format='coo').data

        def eval_h(x,lam,obj_factor,flag):
            if flag:
                problem.combine_H(np.zeros(problem.get_num_linear_equality_constraints()))
                return (np.concatenate((problem.Hphi.row,problem.H_combined.row)),
                        np.concatenate((problem.Hphi.col,problem.H_combined.col)))
            else:
                problem.eval(x)
                lamA = lam[:problem.get_num_linear_equality_constraints()]
                lamf = lam[problem.get_num_linear_equality_constraints():]
                problem.combine_H(lamf)
                return np.concatenate((obj_factor*(problem.Hphi.data),problem.H_combined.data))

        n = problem.get_num_primal_variables()
        m = problem.get_num_linear_equality_constraints()+problem.get_num_nonlinear_equality_constraints()

        return _ipopt.IpoptContext(n,
                                   m,
                                   problem.l,
                                   problem.u,
                                   np.zeros(m),
                                   np.zeros(m),
                                   eval_f,
                                   eval_g,
                                   eval_grad_f,
                                   eval_jac_g,
                                   eval_h)
                
    def solve(self,problem):
        
        # Local vars
        params = self.parameters
        
        # Parameters
        quiet = params['quiet']
        tol = params['tol']

        # Problem
        self.problem = problem
        self.ipopt_context = self.create_ipopt_context(problem)

        # Options
        self.ipopt_context.add_option('tol',tol)
        self.ipopt_context.add_option('print_level',0 if quiet else 5)

        # Reset
        self.reset()

        # Init point
        if problem.x is not None:
            x0 = problem.x
        else:
            x0 = (problem.u+problem.l)/2
                
        # Solve
        if quiet:
            stdout = os.dup(1)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.close(devnull)
        results = self.ipopt_context.solve(x0)
        if quiet: 
            os.dup2(stdout, 1)
            os.close(stdout)

        # Save
        self.x = results['x']
        self.lam = -results['lam'][:problem.get_num_linear_equality_constraints()]
        self.nu = -results['lam'][problem.get_num_linear_equality_constraints():]
        self.pi = results['pi']
        self.mu = results['mu']
        if results['status'] == 0:
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise OptSolverError_Ipopt(self)
            
