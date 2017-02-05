#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from .opt_solver_error import *
from .opt_solver import OptSolver
from scipy.sparse import bmat,triu,eye,coo_matrix,tril

class OptSolverLCCP_CVXOPT(OptSolver):
    """
    Interior-point linearly-constrained convex program solver (CVXOPT).
    """
    
    # Solver parameters
    parameters = {'tol': 1e-4,        # optimality tolerance
                  'maxiter': 1000,    # max iterations
                  'eps': 1e-5,        # boundary proximity factor 
                  'quiet': False}     # quiet flag

    def __init__(self):
        """
        This algorithm solves problems of the form
        
        minimize    varphi(x)
        subject to  Ax = b
                    l <= x <= u
        
        using an interior point method, where
        varphi is convex.
        """
        
        # Init
        OptSolver.__init__(self)
        self.parameters = OptSolverLCCP_CVXOPT.parameters.copy()
        self.problem = None

    def solve(self,problem):
        """
        Solves optimization problem.
        
        Parameters
        ----------
        problem : OptProblem
        """

        # Imports
        from cvxopt import matrix,spmatrix
        from cvxopt.solvers import cp,options
        
        # Local vars
        parameters = self.parameters
        
        # Parameters
        tol = parameters['tol']
        maxiter = parameters['maxiter']
        quiet = parameters['quiet']
        eps = parameters['eps']
                
        # Problem
        self.problem = problem

        # Reset
        self.reset()

        # Data
        self.A = problem.A.tocoo()
        self.b = problem.b
        self.l = problem.l
        self.u = problem.u
        self.n = self.A.shape[1]
        self.m = self.A.shape[0]
        self.I = eye(self.n,format='coo')
    
        # Check limits
        assert(np.all(self.l < self.u))
        
        # Initial point
        if problem.x is None:
            self.x = (self.u + self.l)/2.
        else:
            dul = eps*(self.u-self.l)
            self.x = np.maximum(np.minimum(problem.x,self.u-dul),self.l+dul)
        self.x0 = self.x.copy()

        # Check interior
        assert(np.all(self.l < self.x)) 
        assert(np.all(self.x < self.u))

        # Construct A
        A = spmatrix(self.A.data.tolist(),
                     self.A.row.tolist(),
                     self.A.col.tolist(),
                     self.A.shape)

        # Construct b
        b = matrix(self.b,
                   (self.b.size,1))

        # Construct G
        G = bmat([[self.I],
                  [-self.I]],
                 format='coo')
        G = spmatrix(G.data.tolist(),
                     G.row.tolist(),
                     G.col.tolist(),
                     G.shape)

        # Construct h
        h = matrix(np.hstack((self.u,-self.l)),
                   (2*self.n,1))

        # Construct F
        def F(x=None,z=None):

            if x is None and z is None:
                return (0,
                        matrix(self.x0,(self.n,1)))

            else:

                self.x = np.array(x).flatten()

                problem.eval(self.x)

                f = matrix(problem.phi,
                           (1,1))

                Df = matrix(problem.gphi,
                            (1,self.n))

                if z is None:
                    
                    return (f,Df)

                else:
                    
                    H = (z[0]*problem.Hphi).tocoo() # lower triangular
                    H = spmatrix(H.data.tolist(),
                                 H.row.tolist(),
                                 H.col.tolist(),
                                 H.shape)
                    
                    return (f,Df,H)
                    
        # Solve
        options['show_progress'] = not quiet
        results = cp(F,G=G,h=h,A=A,b=b)                    
        
        # Extract data
        self.x = np.array(results['x']).flatten()
        self.k = 0
        self.lam = -np.array(results['y']).flatten()
        self.mu = np.array(results['zl']).flatten()[:self.n]
        self.pi = np.array(results['zl']).flatten()[self.n:]
        status = results['status']
        
        # Done
        if status == 'optimal':
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        
        # Error
        else:
            raise OptSolverError(self,'cvxopt error - %s' %status) 
            
        

            
        
            
