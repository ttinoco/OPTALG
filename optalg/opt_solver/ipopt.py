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
from .problem import cast_problem, OptProblem
from .opt_solver import OptSolver
from scipy.sparse import bmat

class OptSolverIpopt(OptSolver):
    
    parameters = {'tol': 1e-7,
                  'inf': 1e8,
                  'derivative_test': 'none',
                  'hessian_approximation': 'exact',
                  'linear_solver': 'mumps',
                  'print_level': 5,
                  'max_iter': 1000,
                  'mu_init': 1e-1,
                  'sb': 'yes',
                  'expect_infeasible_problem': 'no',
                  'check_derivatives_for_naninf': 'no',
                  'diverging_iterates_tol': 1e20,
                  'max_cpu_time': 1e6,
                  'quiet': False}
    
    def __init__(self):
        """
        Interior point nonlinear optimization algorithm from COIN-OR.
        """

        # Import
        from ._ipopt import IpoptContext
        
        OptSolver.__init__(self)
        self.parameters = OptSolverIpopt.parameters.copy()

    def supports_properties(self, properties):

        for p in properties:
            if p not in [OptProblem.PROP_CURV_LINEAR,
                         OptProblem.PROP_CURV_QUADRATIC,
                         OptProblem.PROP_CURV_NONLINEAR,
                         OptProblem.PROP_VAR_CONTINUOUS,
                         OptProblem.PROP_TYPE_FEASIBILITY,
                         OptProblem.PROP_TYPE_OPTIMIZATION]:
                return False
        return True

    def create_ipopt_context(self):
        
        # Import
        from ._ipopt import IpoptContext

        # Problem
        problem = self.problem
        
        # Parameters
        inf = self.parameters['inf']

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
                J = bmat([[problem.A],[problem.J]],format='coo')
                return J.data

        def eval_h(x,lam,obj_factor,flag):
            if flag:
                problem.combine_H(np.zeros(problem.get_num_nonlinear_equality_constraints()))
                return (np.concatenate((problem.Hphi.row,problem.H_combined.row)),
                        np.concatenate((problem.Hphi.col,problem.H_combined.col)))
            else:
                problem.eval(x)
                lamf = lam[problem.get_num_linear_equality_constraints():]
                problem.combine_H(lamf)
                return np.concatenate((obj_factor*(problem.Hphi.data),problem.H_combined.data))

        n = problem.get_num_primal_variables()
        m = problem.get_num_linear_equality_constraints()+problem.get_num_nonlinear_equality_constraints()

        return IpoptContext(n,
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
                
    def solve(self, problem):
        
        # Local vars
        params = self.parameters
        
        # Parameters
        quiet = params['quiet']
        tol = params['tol']
        der_test = params['derivative_test']
        mu_init = params['mu_init']
        h_approx = params['hessian_approximation']
        lin_solver = params['linear_solver']
        print_level = params['print_level']
        max_iter = params['max_iter']
        sb = params['sb']
        exp_infeasible = params['expect_infeasible_problem']
        d_check_naninf = params['check_derivatives_for_naninf']
        div_iters_tol = params['diverging_iterates_tol']
        max_cpu_time = params['max_cpu_time']

        # Problem
        problem = cast_problem(problem)
        self.problem = problem

        # Ipopt context
        self.ipopt_context = self.create_ipopt_context()

        # Options
        self.ipopt_context.add_option('sb', sb)
        self.ipopt_context.add_option('tol', tol)
        self.ipopt_context.add_option('print_level', 0 if quiet else print_level)
        self.ipopt_context.add_option('mumps_mem_percent', 1000)
        self.ipopt_context.add_option('derivative_test', der_test)
        self.ipopt_context.add_option('mu_init', mu_init)
        self.ipopt_context.add_option('max_iter', max_iter)
        self.ipopt_context.add_option('expect_infeasible_problem', exp_infeasible)
        self.ipopt_context.add_option('check_derivatives_for_naninf', d_check_naninf)
        self.ipopt_context.add_option('diverging_iterates_tol', div_iters_tol)
        self.ipopt_context.add_option('max_cpu_time', float(max_cpu_time))
        self.ipopt_context.add_option('hessian_approximation', h_approx)
        self.ipopt_context.add_option('linear_solver', lin_solver)

        # Reset
        self.reset()

        # Init point
        if problem.x is not None:
            x0 = problem.x.copy()
        else:
            x0 = (problem.u+problem.l)/2
        
        # Solve
        results = self.ipopt_context.solve(x0)
        
        # Save
        self.k = results['k']
        self.x = results['x'].copy()
        self.lam = -results['lam'][:problem.get_num_linear_equality_constraints()].copy()
        self.nu = -results['lam'][problem.get_num_linear_equality_constraints():].copy()
        self.pi = results['pi'].copy()
        self.mu = results['mu'].copy()
        if results['status'] == 0:
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise OptSolverError_Ipopt(self)
            
